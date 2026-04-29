"""
Prep Time Prediction API
Endpoints:
  GET  /          → health check
  POST /predict   → prediksi waktu masak
  POST /retrain   → retrain model pakai data real dari Supabase
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prep Time Prediction API", version="1.0.0")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model/prep_time_model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "model/features.pkl")

# ── Load Model ────────────────────────────────────────────────────────────────
model    = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEAT_PATH)


# ── Request & Response Schema ─────────────────────────────────────────────────
class OrderItem(BaseModel):
    menu_item_name:           str
    quantity:                 int
    preparation_time_minutes: int
    special_requests:         str | None = None


class PredictRequest(BaseModel):
    items:        list[OrderItem]
    hour_of_day:  int   # jam saat order dibuat (0-23)
    queue_length: int   # jumlah order status 'preparing' saat ini


class PredictResponse(BaseModel):
    estimated_minutes: int
    breakdown:         dict


class RetrainResponse(BaseModel):
    status:        str
    samples_used:  int
    mae_minutes:   float
    r2_score:      float
    model_saved:   bool
    message:       str


# ── Helper: ambil Supabase client ─────────────────────────────────────────────
def _get_supabase():
    """Buat Supabase client dari env vars. Lazy import supaya tidak crash saat test lokal."""
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")  # pakai service key, bukan anon key
    if not url or not key:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL atau SUPABASE_SERVICE_KEY belum diset di environment."
        )
    return create_client(url, key)


# ── Helper: query & build training dataframe dari data real ──────────────────
def _build_training_df(supabase) -> pd.DataFrame:
    """
    Query order_items yang sudah punya sent_to_kitchen_at DAN prepared_at.
    Kemudian hitung fitur yang sama dengan yang dipakai saat training synthetic.
    
    Minimal data: kedua timestamp harus ada, dan prep time harus > 0.
    """
    # Query order_items join ke orders dan menu_items
    res = (
        supabase
        .from_("order_items")
        .select(
            "order_id, quantity, special_requests, "
            "sent_to_kitchen_at, prepared_at, "
            "menu_items(preparation_time_minutes), "
            "orders(created_at, branch_id)"
        )
        .not_.is_("sent_to_kitchen_at", "null")
        .not_.is_("prepared_at", "null")
        .execute()
    )

    rows = res.data
    if not rows:
        return pd.DataFrame()

    df_raw = pd.DataFrame(rows)

    # ── Flatten nested join ───────────────────────────────────────────────────
    df_raw["preparation_time_minutes"] = df_raw["menu_items"].apply(
        lambda x: x.get("preparation_time_minutes", 15) if isinstance(x, dict) else 15
    )
    df_raw["order_created_at"] = df_raw["orders"].apply(
        lambda x: x.get("created_at") if isinstance(x, dict) else None
    )
    df_raw = df_raw.drop(columns=["menu_items", "orders"])

    # ── Parse timestamps ──────────────────────────────────────────────────────
    df_raw["sent_to_kitchen_at"] = pd.to_datetime(df_raw["sent_to_kitchen_at"], utc=True)
    df_raw["prepared_at"]        = pd.to_datetime(df_raw["prepared_at"],        utc=True)
    df_raw["order_created_at"]   = pd.to_datetime(df_raw["order_created_at"],   utc=True)

    # ── Hitung actual_prep_minutes per item ───────────────────────────────────
    df_raw["actual_prep_minutes"] = (
        (df_raw["prepared_at"] - df_raw["sent_to_kitchen_at"])
        .dt.total_seconds() / 60
    ).round(1)

    # Buang data yang tidak masuk akal (< 1 menit atau > 120 menit)
    df_raw = df_raw[
        (df_raw["actual_prep_minutes"] >= 1) &
        (df_raw["actual_prep_minutes"] <= 120)
    ]

    if df_raw.empty:
        return pd.DataFrame()

    # ── Aggregate per order_id ────────────────────────────────────────────────
    # Fitur dihitung per order (sama persis dengan saat predict)
    def agg_order(grp):
        return pd.Series({
            "total_quantity":        grp["quantity"].sum(),
            "total_item_types":      len(grp),
            "weighted_prep_time":    (grp["preparation_time_minutes"] * grp["quantity"]).sum(),
            "special_request_count": grp["special_requests"].apply(
                lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0
            ).sum(),
            "hour_of_day":           grp["order_created_at"].iloc[0].hour
                                     if grp["order_created_at"].iloc[0] is not pd.NaT else 12,
            # actual = waktu sejak item pertama masuk dapur sampai semua siap
            "actual_prep_minutes":   (
                grp["prepared_at"].max() - grp["sent_to_kitchen_at"].min()
            ).total_seconds() / 60,
        })

    df_orders = df_raw.groupby("order_id").apply(agg_order).reset_index(drop=True)

    # ── Tambah queue_length per order dari data historis ─────────────────────
    # Approximasi: query berapa order lain yang sedang 'preparing' saat order ini masuk.
    # Untuk retrain kita pakai nilai median sebagai fallback karena
    # data historis queue tidak tersimpan per-order.
    df_orders["queue_length"] = 3  # median approximation untuk data historis

    # Buang outlier final
    df_orders = df_orders[
        (df_orders["actual_prep_minutes"] >= 1) &
        (df_orders["actual_prep_minutes"] <= 120)
    ]

    return df_orders


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Prep Time Prediction API"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="Items tidak boleh kosong")

    total_quantity        = sum(i.quantity for i in req.items)
    total_item_types      = len(req.items)
    weighted_prep_time    = sum(i.preparation_time_minutes * i.quantity for i in req.items)
    special_request_count = sum(
        1 for i in req.items
        if i.special_requests and i.special_requests.strip() != ""
    )

    features = {
        "total_quantity":        total_quantity,
        "total_item_types":      total_item_types,
        "weighted_prep_time":    weighted_prep_time,
        "special_request_count": special_request_count,
        "hour_of_day":           req.hour_of_day,
        "queue_length":          req.queue_length,
    }

    X = pd.DataFrame([features])[FEATURES]
    predicted         = model.predict(X)[0]
    estimated_minutes = max(1, round(predicted))

    return PredictResponse(
        estimated_minutes=estimated_minutes,
        breakdown={**features},
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain():
    """
    Retrain model menggunakan data real dari Supabase.
    
    Syarat data yang dipakai:
    - order_items.sent_to_kitchen_at  → tidak null (diisi saat dapur tekan 'Mulai Masak')
    - order_items.prepared_at         → tidak null (diisi saat dapur tekan 'Siap Saji')
    - actual_prep_minutes antara 1–120 menit
    
    Jika data real < 30 order, retrain ditolak — model synthetic lebih baik
    daripada model yang ditraining data terlalu sedikit.
    
    Jika data real >= 30 order, model baru akan menggantikan model lama
    HANYA jika MAE model baru <= MAE model lama + 2 menit (tidak lebih buruk).
    """
    global model, FEATURES

    # ── 1. Ambil & build training data dari Supabase ──────────────────────────
    supabase = _get_supabase()

    try:
        df = _build_training_df(supabase)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal query Supabase: {str(e)}")

    if df.empty or len(df) < 30:
        n = len(df) if not df.empty else 0
        raise HTTPException(
            status_code=422,
            detail=(
                f"Data real tidak cukup untuk retrain: {n} order tersedia, "
                f"minimal 30 dibutuhkan. Kumpulkan lebih banyak data dulu."
            )
        )

    # ── 2. Split train/test ───────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    X = df[FEATURES]
    y = df["actual_prep_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 3. Evaluasi model LAMA dulu sebagai baseline ──────────────────────────
    old_mae = mean_absolute_error(y_test, model.predict(X_test))

    # ── 4. Train model BARU ───────────────────────────────────────────────────
    new_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        random_state=42,
    )
    new_model.fit(X_train, y_train)

    y_pred  = new_model.predict(X_test)
    new_mae = mean_absolute_error(y_test, y_pred)
    new_r2  = r2_score(y_test, y_pred)

    # ── 5. Simpan model baru hanya jika tidak lebih buruk dari model lama ─────
    # Toleransi 2 menit: model baru boleh sedikit lebih buruk di test set
    # karena data real lebih representatif dari operasi dapur sesungguhnya.
    model_saved = False
    if new_mae <= old_mae + 2.0:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(FEATURES,  FEAT_PATH)

        # Reload model aktif di memory
        model = new_model
        model_saved = True
        msg = (
            f"✅ Model berhasil diretrain dengan {len(df)} order real. "
            f"MAE turun dari {old_mae:.1f} → {new_mae:.1f} menit."
            if new_mae < old_mae
            else
            f"✅ Model berhasil diretrain dengan {len(df)} order real. "
            f"MAE: {new_mae:.1f} menit (model lama: {old_mae:.1f} menit)."
        )
    else:
        msg = (
            f"⚠️ Model baru tidak disimpan — MAE model baru ({new_mae:.1f} menit) "
            f"lebih buruk dari model lama ({old_mae:.1f} menit) melebihi toleransi 2 menit. "
            f"Cek kualitas data (apakah ada timestamp yang salah?)."
        )

    return RetrainResponse(
        status      = "saved" if model_saved else "rejected",
        samples_used= len(df),
        mae_minutes = round(new_mae, 2),
        r2_score    = round(new_r2,  4),
        model_saved = model_saved,
        message     = msg,
    )