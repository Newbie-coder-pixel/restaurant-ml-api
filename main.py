"""
Prep Time Prediction API
Endpoints:
  GET  /                        → health check
  POST /predict                 → prediksi waktu masak (+ equipment_factor + adaptive buffer)
  POST /retrain                 → retrain model global atau per-branch dari data real Supabase
  GET  /branches/equipment      → lihat semua konfigurasi equipment_factor
  POST /branches/equipment      → set equipment_factor untuk satu branch
  GET  /branches/models         → lihat status model semua branch
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Prep Time Prediction API", version="1.3.0")

# ── Adaptive Buffer Config (Skenario A — Underpromise, Overdeliver) ───────────
# Buffer diterapkan SETELAH prediksi model, sebelum dikembalikan ke customer.
# Tujuan: estimasi yang ditampilkan selalu sedikit lebih lama dari prediksi murni,
# sehingga makanan hampir selalu datang lebih cepat dari yang dijanjikan.
#
# Landasan teori: Expectation-Disconfirmation Theory (Oliver, 1980) —
# kepuasan pelanggan terbentuk dari selisih POSITIF antara ekspektasi dan aktual.
#
# Tiga kondisi yang mempengaruhi besar buffer:
#   JAM PEAK   : jam 12–13 dan 18–20 → dapur lebih sibuk → uncertainty lebih tinggi
#   QUEUE PANJANG: queue_length > 5   → risiko antrian meleset lebih besar
#   NORMAL     : di luar kondisi di atas → buffer minimal
#
# Buffer dinyatakan sebagai MULTIPLIER (bukan penambahan flat):
#   1.10 = tampilkan 10% lebih lama dari prediksi model
#   1.20 = tampilkan 20% lebih lama dari prediksi model
#   1.25 = tampilkan 25% lebih lama dari prediksi model

PEAK_HOURS   = {12, 13, 18, 19, 20}   # jam-jam ramai makan siang & makan malam
BUFFER_PEAK_ONLY     = 1.15            # jam peak, queue normal  → +15%
BUFFER_QUEUE_ONLY    = 1.10            # jam normal, queue panjang → +10%
BUFFER_PEAK_AND_QUEUE = 1.25           # jam peak DAN queue panjang → +25%
BUFFER_NORMAL        = 1.05            # kondisi tenang → +5% (safety minimal)
QUEUE_THRESHOLD      = 5               # batas "queue panjang"

# ── Equipment Factor Config ────────────────────────────────────────────────────
# Nama tabel di Supabase: branch_equipment_config
# Kolom: branch_id (text, PK), equipment_factor (float, default 1.0), notes (text)
#
# Nilai equipment_factor:
#   1.0  → dapur manual / baseline (default)
#   0.8  → ada 1-2 alat modern (air fryer, mesin kopi otomatis)
#   0.65 → mayoritas alat modern (pressure cooker, combi oven, dll)
#   0.5  → dapur penuh teknologi (fully automated kitchen)
#
# Semakin kecil nilai, semakin cepat dapur tersebut relatif terhadap baseline.
EQUIPMENT_FACTOR_DEFAULT = 1.0
EQUIPMENT_TABLE          = "branch_equipment_config"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://restaurantmanagement-weld.vercel.app",
        "https://restaurant-qr-code-ten.vercel.app",
        "https://restaurant-staff-topaz.vercel.app",
        "https://restaurant-customer-two.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")

# Model global (fallback) — ditraining dari semua data / synthetic
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global", "prep_time_model.pkl")
GLOBAL_FEAT_PATH  = os.path.join(MODEL_DIR, "global", "features.pkl")

# Model lama (backward compat) — dipakai jika folder global/ belum ada
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, "prep_time_model.pkl")
LEGACY_FEAT_PATH  = os.path.join(MODEL_DIR, "features.pkl")

def _branch_model_path(branch_id: str) -> tuple[str, str]:
    """Kembalikan path (model, features) untuk model spesifik satu branch."""
    d = os.path.join(MODEL_DIR, f"branch_{branch_id}")
    return os.path.join(d, "prep_time_model.pkl"), os.path.join(d, "features.pkl")

# ── Load Model Global ──────────────────────────────────────────────────────────
# Prioritas: global/ → legacy flat file (backward compat saat pertama deploy)
def _load_global_model():
    if os.path.exists(GLOBAL_MODEL_PATH):
        return joblib.load(GLOBAL_MODEL_PATH), joblib.load(GLOBAL_FEAT_PATH)
    # Fallback ke model lama (flat file dari sebelum Tahap 3)
    return joblib.load(LEGACY_MODEL_PATH), joblib.load(LEGACY_FEAT_PATH)

model, FEATURES = _load_global_model()


# ── Request & Response Schema ─────────────────────────────────────────────────
class OrderItem(BaseModel):
    menu_item_name:           str
    quantity:                 int
    preparation_time_minutes: int
    special_requests:         str | None = None


class PredictRequest(BaseModel):
    items:        list[OrderItem]
    hour_of_day:  int            # jam saat order dibuat (0-23)
    queue_length: int            # jumlah order status 'preparing' saat ini
    branch_id:    str | None = None  # ID cabang — untuk ambil equipment_factor


class EquipmentConfigRequest(BaseModel):
    branch_id:        str
    equipment_factor: float  # 0.5 – 1.0
    notes:            str | None = None  # opsional, misal "punya air fryer & pressure cooker"


class PredictResponse(BaseModel):
    estimated_minutes: int
    breakdown:         dict


class RetrainRequest(BaseModel):
    branch_id: str | None = None  # None = retrain model global dari semua data


class RetrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status:        str
    scope:         str   # "global" atau "branch:{branch_id}"
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


# ── Helper: ambil equipment_factor untuk satu branch ─────────────────────────
def _get_equipment_factor(branch_id: str | None) -> float:
    """
    Ambil equipment_factor dari tabel branch_equipment_config di Supabase.

    - Jika branch_id None  → return 1.0 (tidak ada info cabang)
    - Jika branch belum ada di tabel → return 1.0 (belum dikonfigurasi = baseline)
    - Jika Supabase tidak tersedia   → return 1.0 (graceful degradation)
    - Nilai diklem antara 0.3–1.0 untuk mencegah konfigurasi ekstrem yang tidak masuk akal
    """
    if not branch_id:
        return EQUIPMENT_FACTOR_DEFAULT

    try:
        supabase = _get_supabase()
        res = (
            supabase
            .from_(EQUIPMENT_TABLE)
            .select("equipment_factor")
            .eq("branch_id", branch_id)
            .maybe_single()
            .execute()
        )
        if res.data and "equipment_factor" in res.data:
            raw = float(res.data["equipment_factor"])
            # Klem ke rentang yang masuk akal (min 0.3 = 70% lebih cepat dari baseline)
            return max(0.3, min(1.0, raw))
    except Exception:
        # Gagal ambil config → fallback ke default, jangan crash prediksi
        pass

    return EQUIPMENT_FACTOR_DEFAULT


# ── Helper: hitung adaptive buffer multiplier ─────────────────────────────────
def _compute_adaptive_buffer(hour_of_day: int, queue_length: int) -> tuple[float, str]:
    """
    Tentukan buffer multiplier berdasarkan jam dan panjang antrian.

    Mengembalikan tuple (multiplier, label) supaya breakdown bisa menjelaskan
    alasan buffer ke Flutter / untuk keperluan debugging dan dokumentasi.

    Contoh output:
      (1.25, "peak_and_queue") → jam ramai + antrian panjang
      (1.15, "peak_only")      → jam ramai, antrian normal
      (1.10, "queue_only")     → jam normal, antrian panjang
      (1.05, "normal")         → kondisi tenang
    """
    is_peak  = hour_of_day in PEAK_HOURS
    is_queue = queue_length > QUEUE_THRESHOLD

    if is_peak and is_queue:
        return BUFFER_PEAK_AND_QUEUE, "peak_and_queue"
    elif is_peak:
        return BUFFER_PEAK_ONLY, "peak_only"
    elif is_queue:
        return BUFFER_QUEUE_ONLY, "queue_only"
    else:
        return BUFFER_NORMAL, "normal"


# ── Helper: ambil model yang tepat untuk satu branch ─────────────────────────
def _get_model_for_branch(branch_id: str | None):
    """
    Cek apakah branch punya model terlatih sendiri.
    Kalau ada → pakai model branch (lebih akurat untuk dapur itu).
    Kalau tidak → fallback ke model global (synthetic / semua data).

    Mengembalikan tuple (model, features, model_scope).
    model_scope = "branch:{id}" atau "global" — untuk breakdown response.
    """
    if branch_id:
        branch_model_path, branch_feat_path = _branch_model_path(branch_id)
        if os.path.exists(branch_model_path) and os.path.exists(branch_feat_path):
            try:
                branch_model    = joblib.load(branch_model_path)
                branch_features = joblib.load(branch_feat_path)
                return branch_model, branch_features, f"branch:{branch_id}"
            except Exception:
                pass  # file korup / tidak terbaca → fallback ke global

    return model, FEATURES, "global"


# ── Helper: query & build training dataframe dari data real ──────────────────
def _build_training_df(supabase, branch_id: str | None = None) -> pd.DataFrame:
    """
    Query order_items yang sudah punya sent_to_kitchen_at DAN prepared_at.
    Kemudian hitung fitur yang sama dengan yang dipakai saat training synthetic.

    branch_id → None  : ambil semua data (untuk retrain model global)
    branch_id → "xyz" : hanya ambil data dari cabang tersebut (untuk retrain model branch)
    """
    query = (
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
    )

    # Filter per branch jika diminta
    if branch_id:
        query = query.eq("orders.branch_id", branch_id)

    res = query.execute()

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
            # Rentang waktu order ini benar-benar ada di dapur — dipakai untuk
            # menghitung queue_length historis (order lain yang overlap rentang ini).
            "order_start":           grp["sent_to_kitchen_at"].min(),
            "order_end":             grp["prepared_at"].max(),
            # actual = waktu sejak item pertama masuk dapur sampai semua siap
            "actual_prep_minutes":   (
                grp["prepared_at"].max() - grp["sent_to_kitchen_at"].min()
            ).total_seconds() / 60,
        })

    df_orders = df_raw.groupby("order_id").apply(agg_order).reset_index(drop=True)

    # ── Hitung queue_length historis yang sebenarnya ──────────────────────────
    # queue_length = jumlah order LAIN yang sudah masuk dapur (order_start <= t)
    # tapi belum selesai (order_end > t) tepat saat order ini mulai dimasak.
    # Ini meniru definisi queue_length yang sama persis dipakai saat /predict
    # ("jumlah order berstatus 'preparing' saat ini"). Sebelumnya nilai ini
    # di-hardcode ke 3 untuk semua baris, sehingga fitur ini konstan dan model
    # tidak pernah bisa belajar efek antrian dari data real sama sekali.
    starts = df_orders["order_start"].to_numpy()
    ends   = df_orders["order_end"].to_numpy()
    # matrix[i, j] = True jika order j masih 'preparing' saat order i mulai dimasak
    still_preparing = (starts[None, :] <= starts[:, None]) & (ends[None, :] > starts[:, None])
    queue_lengths = still_preparing.sum(axis=1) - 1  # -1 = jangan hitung order itu sendiri
    df_orders["queue_length"] = np.clip(queue_lengths, 0, None)
    df_orders = df_orders.drop(columns=["order_start", "order_end"])

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
    special_request_count = sum(
        1 for i in req.items
        if i.special_requests and i.special_requests.strip() != ""
    )
    raw_weighted_prep = sum(i.preparation_time_minutes * i.quantity for i in req.items)

    # ── Pilih model: branch-specific jika ada, fallback ke global ────────────
    active_model, active_features, model_scope = _get_model_for_branch(req.branch_id)

    # ── Terapkan equipment_factor HANYA saat memakai model global ────────────
    # Kalau branch ini sudah punya model sendiri (dilatih dari actual_prep_minutes
    # riil cabang tsb via /retrain), kecepatan dapur cabang itu SUDAH otomatis
    # terkandung dalam model — actual_prep_minutes historisnya memang sudah
    # mencerminkan dapur cepat/lambat itu. Mengalikan equipment_factor lagi di
    # sini akan menghitung efek kecepatan dapur DUA KALI dan membuat prediksi
    # bias (under-predict untuk dapur cepat, over-predict untuk dapur lambat).
    # equipment_factor jadi relevan hanya sebagai adjustment manual saat masih
    # memakai model global generik (belum ada cukup data real cabang tsb).
    if model_scope == "global":
        equipment_factor   = _get_equipment_factor(req.branch_id)
        weighted_prep_time = round(raw_weighted_prep * equipment_factor, 2)
    else:
        equipment_factor   = 1.0
        weighted_prep_time = round(raw_weighted_prep, 2)

    features = {
        "total_quantity":        total_quantity,
        "total_item_types":      total_item_types,
        "weighted_prep_time":    weighted_prep_time,
        "special_request_count": special_request_count,
        "hour_of_day":           req.hour_of_day,
        "queue_length":          req.queue_length,
    }

    X = pd.DataFrame([features])[active_features]
    predicted = active_model.predict(X)[0]

    # ── Terapkan adaptive buffer (Skenario A — Underpromise, Overdeliver) ────
    # model_prediction = hasil murni dari ML model
    # estimated_minutes = yang ditampilkan ke customer (sudah dibuffer)
    # Prinsip: lebih baik customer senang karena makanan datang lebih cepat
    # dari estimasi, daripada kecewa karena datang lebih lama.
    buffer_multiplier, buffer_label = _compute_adaptive_buffer(
        req.hour_of_day, req.queue_length
    )
    model_prediction  = max(1, round(predicted))
    estimated_minutes = max(1, round(predicted * buffer_multiplier))

    return PredictResponse(
        estimated_minutes=estimated_minutes,
        breakdown={
            **features,
            # Info tambahan untuk debugging / transparansi di Flutter
            "equipment_factor":          equipment_factor,
            "equipment_factor_applied":  model_scope == "global",
            "raw_weighted_prep":         raw_weighted_prep,
            "branch_id":                 req.branch_id,
            # Info buffer — berguna untuk UI "kenapa estimasi ini?" dan dokumentasi
            "model_prediction":     model_prediction,
            "buffer_multiplier":    buffer_multiplier,
            "buffer_label":         buffer_label,
            "buffer_added_minutes": estimated_minutes - model_prediction,
            # Info model yang dipakai — "branch:xyz" atau "global"
            "model_scope":          model_scope,
        },
    )


@app.get("/branches/equipment")
def get_equipment_configs():
    """
    Lihat semua konfigurasi equipment_factor yang sudah diset.
    Berguna untuk Owner melihat profil alat dapur semua cabang sekaligus.
    """
    try:
        supabase = _get_supabase()
        res = (
            supabase
            .from_(EQUIPMENT_TABLE)
            .select("branch_id, equipment_factor, notes")
            .order("branch_id")
            .execute()
        )
        return {
            "configs": res.data or [],
            "default_factor": EQUIPMENT_FACTOR_DEFAULT,
            "note": "Branch yang tidak ada di list menggunakan default factor 1.0 (dapur manual/baseline).",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ambil konfigurasi: {str(e)}")


@app.post("/branches/equipment")
def set_equipment_config(req: EquipmentConfigRequest):
    """
    Set atau update equipment_factor untuk satu branch.

    equipment_factor yang disarankan:
      1.0  → dapur manual / baseline
      0.8  → ada 1-2 alat modern (air fryer, mesin kopi otomatis)
      0.65 → mayoritas alat modern (pressure cooker, combi oven)
      0.5  → dapur penuh teknologi

    Endpoint ini melakukan upsert — aman dipanggil berulang kali.
    """
    if not (0.3 <= req.equipment_factor <= 1.0):
        raise HTTPException(
            status_code=422,
            detail="equipment_factor harus antara 0.3 dan 1.0."
        )

    try:
        supabase = _get_supabase()
        supabase.from_(EQUIPMENT_TABLE).upsert({
            "branch_id":        req.branch_id,
            "equipment_factor": req.equipment_factor,
            "notes":            req.notes or "",
        }).execute()

        speed_pct = round((1.0 - req.equipment_factor) * 100)
        return {
            "status":  "ok",
            "branch_id": req.branch_id,
            "equipment_factor": req.equipment_factor,
            "effect":  f"Prediksi prep time cabang ini akan {speed_pct}% lebih cepat dari baseline."
                       if speed_pct > 0 else "Cabang ini menggunakan baseline (dapur manual).",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal simpan konfigurasi: {str(e)}")


@app.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest = RetrainRequest()):
    """
    Retrain model menggunakan data real dari Supabase.

    req.branch_id = None  → retrain MODEL GLOBAL dari semua data semua cabang.
                            Model disimpan ke model/global/
    req.branch_id = "xyz" → retrain MODEL BRANCH dari data cabang tersebut saja.
                            Model disimpan ke model/branch_xyz/
                            Tidak mengganggu model global maupun model branch lain.

    Syarat data yang dipakai:
    - order_items.sent_to_kitchen_at  → tidak null
    - order_items.prepared_at         → tidak null
    - actual_prep_minutes antara 1–120 menit

    Minimum 30 order per scope sebelum retrain diizinkan.
    Model baru hanya disimpan jika MAE ≤ MAE model lama + 2 menit.
    """
    global model, FEATURES

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    branch_id = req.branch_id
    scope     = f"branch:{branch_id}" if branch_id else "global"

    # ── 1. Ambil & build training data dari Supabase ──────────────────────────
    supabase = _get_supabase()

    try:
        df = _build_training_df(supabase, branch_id=branch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal query Supabase: {str(e)}")

    if df.empty or len(df) < 30:
        n = len(df) if not df.empty else 0
        scope_label = f"cabang '{branch_id}'" if branch_id else "semua cabang"
        raise HTTPException(
            status_code=422,
            detail=(
                f"Data real tidak cukup untuk retrain {scope_label}: "
                f"{n} order tersedia, minimal 30 dibutuhkan."
            )
        )

    # ── 2. Tentukan model referensi (baseline untuk perbandingan MAE) ─────────
    # Untuk branch: bandingkan dengan model branch lama (jika ada), lalu global.
    # Untuk global: bandingkan dengan model global saat ini.
    if branch_id:
        ref_model, ref_features, _ = _get_model_for_branch(branch_id)
    else:
        ref_model, ref_features = model, FEATURES

    # ── 3. Split train/test ───────────────────────────────────────────────────
    X = df[ref_features]
    y = df["actual_prep_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    old_mae = mean_absolute_error(y_test, ref_model.predict(X_test))

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

    # ── 5. Simpan model baru hanya jika tidak lebih buruk ────────────────────
    model_saved = False
    if new_mae <= old_mae + 2.0:
        if branch_id:
            # Simpan sebagai model branch — tidak ganggu model global
            save_model_path, save_feat_path = _branch_model_path(branch_id)
        else:
            # Simpan sebagai model global baru
            save_model_path = GLOBAL_MODEL_PATH
            save_feat_path  = GLOBAL_FEAT_PATH

        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        joblib.dump(new_model,    save_model_path)
        joblib.dump(ref_features, save_feat_path)

        # Reload model global di memory jika yang diretrain adalah global
        if not branch_id:
            model   = new_model
            FEATURES = ref_features

        model_saved = True
        direction   = f"MAE turun dari {old_mae:.1f} → {new_mae:.1f} menit." \
                      if new_mae < old_mae \
                      else f"MAE: {new_mae:.1f} menit (model lama: {old_mae:.1f} menit)."
        scope_label = f"cabang '{branch_id}'" if branch_id else "global (semua cabang)"
        msg = f"✅ Model {scope_label} berhasil diretrain dengan {len(df)} order. {direction}"
    else:
        scope_label = f"cabang '{branch_id}'" if branch_id else "global"
        msg = (
            f"⚠️ Model {scope_label} tidak disimpan — "
            f"MAE baru ({new_mae:.1f} menit) lebih buruk dari model lama "
            f"({old_mae:.1f} menit) melebihi toleransi 2 menit. "
            f"Cek kualitas data (timestamp salah / data terlalu sedikit?)."
        )

    return RetrainResponse(
        status       = "saved" if model_saved else "rejected",
        scope        = scope,
        samples_used = len(df),
        mae_minutes  = round(new_mae, 2),
        r2_score     = round(new_r2,  4),
        model_saved  = model_saved,
        message      = msg,
    )


@app.get("/branches/models")
def get_branch_models():
    """
    Lihat status model semua branch — apakah sudah punya model sendiri atau masih pakai global.
    Berguna untuk Owner mengetahui cabang mana yang sudah cukup data untuk retrain.
    """
    result = []

    # Scan folder model/ untuk direktori branch_*
    if os.path.isdir(MODEL_DIR):
        for entry in sorted(os.listdir(MODEL_DIR)):
            if entry.startswith("branch_"):
                branch_id = entry[len("branch_"):]
                m_path, f_path = _branch_model_path(branch_id)
                has_model = os.path.exists(m_path) and os.path.exists(f_path)
                result.append({
                    "branch_id":   branch_id,
                    "has_model":   has_model,
                    "model_path":  m_path if has_model else None,
                    "model_scope": f"branch:{branch_id}" if has_model else "global (fallback)",
                })

    has_global = os.path.exists(GLOBAL_MODEL_PATH)
    return {
        "global_model": {
            "exists":     has_global,
            "path":       GLOBAL_MODEL_PATH if has_global else LEGACY_MODEL_PATH,
            "note":       "Dipakai sebagai fallback untuk branch yang belum punya model sendiri.",
        },
        "branch_models": result,
        "tip": (
            "Panggil POST /retrain dengan branch_id untuk melatih model spesifik per cabang. "
            "Minimal 30 order dengan timestamp sent_to_kitchen_at & prepared_at dibutuhkan."
        ),
    }