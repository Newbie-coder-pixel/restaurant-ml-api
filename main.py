"""
Prep Time Prediction API
Endpoints:
  GET  /                        -> health check
  POST /predict                 -> prediksi waktu masak (+ equipment_factor + adaptive buffer)
  POST /retrain                 -> retrain model global atau per-branch dari data real Supabase
  GET  /branches/equipment      -> lihat equipment_factor yang sudah dipelajari dari data per branch
  GET  /branches/models         -> lihat status model semua branch

Tidak ada lagi data sintetis dan tidak ada lagi konstanta bisnis yang di-hardcode:
equipment_factor dan adaptive buffer sekarang dihitung dari order real (lihat calibration.py),
dan diperbarui otomatis setiap kali /retrain dipanggil. Endpoint POST /branches/equipment
(input manual admin) sudah dihapus karena itu justru sumber "rekayasa" yang harus dihilangkan.
"""
import math
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from calibration import (
    compute_learned_buffer_table,
    compute_learned_equipment_factor,
    load_table,
    lookup_buffer,
    save_table,
)
from metrics_log import append_retrain_log
from model_fit import chronological_split, evaluate, fit_best_model
from paths import (
    GLOBAL_BUFFER_TABLE_PATH,
    GLOBAL_FEAT_PATH,
    GLOBAL_MODEL_PATH,
    MODEL_DIR,
    RETRAIN_HISTORY_PATH,
    branch_buffer_table_path,
    branch_dir,
    branch_model_path,
)
from training_data import FEATURES, MIN_SAMPLES_FOR_MODEL, TARGET, build_training_df, get_supabase

load_dotenv()

app = FastAPI(title="Prep Time Prediction API", version="2.0.0")

EQUIPMENT_FACTOR_DEFAULT = 1.0  # neutral identity value — used only when there isn't
                                # yet enough real branch data to estimate a factor.
EQUIPMENT_TABLE = "branch_equipment_config"
EQUIPMENT_FACTOR_READ_CLIP = (0.5, 2.0)  # matches calibration.EQUIPMENT_FACTOR_CLIP

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


# ── Load Model Global ──────────────────────────────────────────────────────────
def _load_global_model():
    """Returns (model, features). model is None if no real-data model has been
    trained yet (fresh deploy / not enough real orders) — this is a normal state,
    not an error, and predict() below handles it explicitly."""
    if os.path.exists(GLOBAL_MODEL_PATH):
        return joblib.load(GLOBAL_MODEL_PATH), joblib.load(GLOBAL_FEAT_PATH)
    return None, FEATURES


model, _loaded_features = _load_global_model()
if model is not None:
    FEATURES = _loaded_features


# ── Request & Response Schema ─────────────────────────────────────────────────
class OrderItem(BaseModel):
    menu_item_name: str
    quantity: int
    preparation_time_minutes: int
    special_requests: str | None = None


class PredictRequest(BaseModel):
    items: list[OrderItem]
    hour_of_day: int
    queue_length: int
    branch_id: str | None = None


class PredictResponse(BaseModel):
    estimated_minutes: int
    breakdown: dict


class RetrainRequest(BaseModel):
    branch_id: str | None = None


class RetrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    scope: str
    samples_used: int
    mae_minutes: float
    r2_score: float
    cv_mae_mean: float
    cv_mae_std: float
    model_saved: bool
    message: str


# ── Helper: model resolution per branch, fallback ke global ──────────────────
def _get_model_for_branch(branch_id: str | None):
    """Returns (model, features, model_scope). model_scope is "branch:{id}",
    "global", or "baseline_no_ml" if literally no real-data model exists yet
    anywhere for this request (a valid cold-start state, not an error)."""
    if branch_id:
        m_path, f_path = branch_model_path(branch_id)
        if os.path.exists(m_path) and os.path.exists(f_path):
            try:
                return joblib.load(m_path), joblib.load(f_path), f"branch:{branch_id}"
            except Exception:
                pass

    if model is not None:
        return model, FEATURES, "global"

    return None, FEATURES, "baseline_no_ml"


def _buffer_table_path_for_scope(model_scope: str, branch_id: str | None) -> str:
    if model_scope.startswith("branch:") and branch_id:
        return branch_buffer_table_path(branch_id)
    return GLOBAL_BUFFER_TABLE_PATH


# ── Helper: ambil equipment_factor (dipelajari dari data, bukan input manual) ─
def _get_equipment_factor(branch_id: str | None) -> float:
    """Reads the equipment_factor last computed by /retrain for this branch from
    Supabase. There is no longer a way to set this by hand — see
    calibration.compute_learned_equipment_factor for how it's derived from real
    order history. Returns the neutral 1.0 if branch_id is None, the branch has
    never been retrained, or Supabase is unavailable."""
    if not branch_id:
        return EQUIPMENT_FACTOR_DEFAULT

    try:
        supabase = get_supabase()
        res = (
            supabase.from_(EQUIPMENT_TABLE)
            .select("equipment_factor")
            .eq("branch_id", branch_id)
            .maybe_single()
            .execute()
        )
        if res.data and "equipment_factor" in res.data:
            raw = float(res.data["equipment_factor"])
            return max(EQUIPMENT_FACTOR_READ_CLIP[0], min(EQUIPMENT_FACTOR_READ_CLIP[1], raw))
    except Exception:
        pass

    return EQUIPMENT_FACTOR_DEFAULT


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Prep Time Prediction API", "global_model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="Items tidak boleh kosong")

    total_quantity = sum(i.quantity for i in req.items)
    total_item_types = len(req.items)
    special_request_count = sum(
        1 for i in req.items if i.special_requests and i.special_requests.strip() != ""
    )
    raw_weighted_prep = sum(i.preparation_time_minutes * i.quantity for i in req.items)

    active_model, active_features, model_scope = _get_model_for_branch(req.branch_id)

    # equipment_factor hanya relevan saat masih pakai model global generik — kalau
    # branch sudah punya model sendiri, kecepatan dapurnya sudah otomatis
    # terkandung dalam actual_prep_minutes historis yang melatih model itu.
    if model_scope == "global":
        equipment_factor = _get_equipment_factor(req.branch_id)
        weighted_prep_time = round(raw_weighted_prep * equipment_factor, 2)
    else:
        equipment_factor = 1.0
        weighted_prep_time = round(raw_weighted_prep, 2)

    features = {
        "total_quantity": total_quantity,
        "total_item_types": total_item_types,
        "weighted_prep_time": weighted_prep_time,
        "special_request_count": special_request_count,
        "hour_of_day": req.hour_of_day,
        "queue_length": req.queue_length,
    }

    if active_model is None:
        # Belum ada model ML terlatih dari data real untuk scope mana pun.
        # Estimasi jujur: jumlah waktu masak menu tanpa penyesuaian ML, tanpa buffer.
        model_prediction = max(1, round(raw_weighted_prep))
        buffer_multiplier, buffer_label = 1.0, "no_model_yet"
        estimated_minutes = model_prediction
    else:
        X = pd.DataFrame([features])[active_features]
        predicted = active_model.predict(X)[0]

        buffer_table = load_table(_buffer_table_path_for_scope(model_scope, req.branch_id))
        buffer_multiplier, buffer_label = lookup_buffer(req.hour_of_day, req.queue_length, buffer_table)

        model_prediction = max(1, round(predicted))
        estimated_minutes = max(1, round(predicted * buffer_multiplier))

    return PredictResponse(
        estimated_minutes=estimated_minutes,
        breakdown={
            **features,
            "equipment_factor": equipment_factor,
            "equipment_factor_applied": model_scope == "global",
            "raw_weighted_prep": raw_weighted_prep,
            "branch_id": req.branch_id,
            "model_prediction": model_prediction,
            "buffer_multiplier": buffer_multiplier,
            "buffer_label": buffer_label,
            "buffer_added_minutes": estimated_minutes - model_prediction,
            "model_scope": model_scope,
        },
    )


@app.get("/branches/equipment")
def get_equipment_configs():
    """Lihat equipment_factor yang sudah DIPELAJARI dari data real untuk tiap
    branch (diperbarui otomatis setiap /retrain dipanggil untuk branch itu).
    Tidak ada lagi jalur untuk menuliskan angka ini secara manual."""
    try:
        supabase = get_supabase()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        res = (
            supabase.from_(EQUIPMENT_TABLE)
            .select("branch_id, equipment_factor, notes")
            .order("branch_id")
            .execute()
        )
        return {
            "configs": res.data or [],
            "default_factor": EQUIPMENT_FACTOR_DEFAULT,
            "note": (
                "equipment_factor dihitung otomatis dari rasio actual_prep_minutes / "
                "weighted_prep_time branch tsb dibanding baseline global (lihat "
                "calibration.compute_learned_equipment_factor), diperbarui setiap POST /retrain "
                "untuk branch itu. Branch yang belum pernah diretrain tidak ada di list dan "
                "memakai default netral 1.0 (tidak ada penyesuaian)."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ambil konfigurasi: {str(e)}")


@app.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest = RetrainRequest()):
    """
    Retrain model menggunakan data real dari Supabase.

    - Split evaluasi KRONOLOGIS (holdout = order paling baru), bukan random split,
      supaya metrik yang dilaporkan tidak "melihat masa depan".
    - Hyperparameter GBR dipilih via GridSearchCV, bukan angka hardcoded.
    - Kalau branch_id diisi, equipment_factor branch itu ikut dihitung ulang dari
      data real dan disimpan ke Supabase — tidak pernah diinput manual.
    - Buffer adaptif branch/global itu ikut dikalibrasi ulang dari distribusi
      error out-of-fold model (lihat calibration.compute_learned_buffer_table).
    - Model baru hanya disimpan jika MAE holdout-nya tidak lebih buruk dari model
      lama + 2 menit. Setiap percobaan (diterima atau ditolak) dicatat ke
      model/retrain_history.jsonl untuk audit.

    Minimum MIN_SAMPLES_FOR_MODEL order per scope sebelum retrain diizinkan —
    dinaikkan dari batas lama supaya holdout evaluasi tidak dihitung dari
    segelintir baris yang mudah didominasi noise.
    """
    global model, FEATURES

    branch_id = req.branch_id
    scope = f"branch:{branch_id}" if branch_id else "global"

    try:
        supabase = get_supabase()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        df = build_training_df(supabase, branch_id=branch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal query Supabase: {str(e)}")

    if len(df) < MIN_SAMPLES_FOR_MODEL:
        scope_label = f"cabang '{branch_id}'" if branch_id else "semua cabang"
        raise HTTPException(
            status_code=422,
            detail=(
                f"Data real tidak cukup untuk retrain {scope_label}: "
                f"{len(df)} order tersedia, minimal {MIN_SAMPLES_FOR_MODEL} dibutuhkan "
                "(supaya holdout evaluasi kronologis punya cukup sampel dan tidak "
                "didominasi noise satu-dua order)."
            ),
        )

    # ── Equipment factor: dihitung ulang dari data setiap kali, bukan input manual ──
    equipment_result = None
    if branch_id:
        df_global_for_factor = build_training_df(supabase, branch_id=None)
        factor, factor_n = compute_learned_equipment_factor(df, df_global_for_factor)
        try:
            supabase.from_(EQUIPMENT_TABLE).upsert({
                "branch_id": branch_id,
                "equipment_factor": factor,
                "notes": f"auto-learned dari {factor_n} order real (di-refresh tiap /retrain)",
            }).execute()
        except Exception:
            pass
        equipment_result = {"equipment_factor": factor, "samples_used_for_factor": factor_n}

    # ── Model referensi (baseline yang harus dikalahkan) ──────────────────────
    if branch_id:
        ref_model, ref_features, _ = _get_model_for_branch(branch_id)
    else:
        ref_model, ref_features = model, FEATURES

    # ── Split KRONOLOGIS + hyperparameter search ──────────────────────────────
    train_df, test_df = chronological_split(df)
    best_model, best_params, cv_mae_mean, cv_mae_std = fit_best_model(
        train_df[FEATURES], train_df[TARGET]
    )
    new_metrics = evaluate(best_model, test_df[FEATURES], test_df[TARGET])

    if ref_model is not None:
        old_metrics = evaluate(ref_model, test_df[ref_features], test_df[TARGET])
        old_mae = old_metrics["mae"]
    else:
        old_mae = float("inf")  # tidak ada model lama → apa pun yang real "menang"

    model_saved = False
    if new_metrics["mae"] <= old_mae + 2.0:
        # Refit di SELURUH data real (train+holdout) dgn hyperparameter yang sudah
        # dipilih — holdout di atas hanya dipakai untuk metrik yang jujur.
        final_model, _, _, _ = fit_best_model(df[FEATURES], df[TARGET])
        buffer_table = compute_learned_buffer_table(df, best_params | {"random_state": 42})

        if branch_id:
            save_model_path, save_feat_path = branch_model_path(branch_id)
            buffer_path = branch_buffer_table_path(branch_id)
            os.makedirs(branch_dir(branch_id), exist_ok=True)
        else:
            save_model_path, save_feat_path = GLOBAL_MODEL_PATH, GLOBAL_FEAT_PATH
            buffer_path = GLOBAL_BUFFER_TABLE_PATH
            os.makedirs(os.path.dirname(GLOBAL_MODEL_PATH), exist_ok=True)

        joblib.dump(final_model, save_model_path)
        joblib.dump(FEATURES, save_feat_path)
        save_table(buffer_path, buffer_table)

        if not branch_id:
            model = final_model

        model_saved = True
        scope_label = f"cabang '{branch_id}'" if branch_id else "global (semua cabang)"
        direction = (
            f"MAE turun/stabil dari {old_mae:.1f} -> {new_metrics['mae']:.1f} menit."
            if old_mae != float("inf")
            else f"Model pertama untuk scope ini, MAE holdout {new_metrics['mae']:.1f} menit."
        )
        msg = f"Model {scope_label} berhasil diretrain dengan {len(df)} order real. {direction}"
    else:
        scope_label = f"cabang '{branch_id}'" if branch_id else "global"
        msg = (
            f"Model {scope_label} TIDAK disimpan — MAE baru ({new_metrics['mae']:.1f} menit) "
            f"lebih buruk dari model lama ({old_mae:.1f} menit) melebihi toleransi 2 menit."
        )

    r2 = new_metrics["r2"]
    r2_clean = 0.0 if isinstance(r2, float) and math.isnan(r2) else r2

    append_retrain_log(RETRAIN_HISTORY_PATH, {
        "scope": scope,
        "samples_used": len(df),
        "train_samples": len(train_df),
        "holdout_samples": len(test_df),
        "best_params": best_params,
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,
        "holdout_mae": new_metrics["mae"],
        "holdout_r2": r2_clean,
        "old_mae": None if old_mae == float("inf") else old_mae,
        "model_saved": model_saved,
        "equipment_factor": equipment_result,
    })

    return RetrainResponse(
        status="saved" if model_saved else "rejected",
        scope=scope,
        samples_used=len(df),
        mae_minutes=round(new_metrics["mae"], 2),
        r2_score=round(r2_clean, 4),
        cv_mae_mean=round(cv_mae_mean, 2),
        cv_mae_std=round(cv_mae_std, 2),
        model_saved=model_saved,
        message=msg,
    )


@app.get("/branches/models")
def get_branch_models():
    """Lihat status model semua branch — apakah sudah punya model sendiri
    (hasil retrain dari data real) atau masih fallback ke global."""
    result = []

    if os.path.isdir(MODEL_DIR):
        for entry in sorted(os.listdir(MODEL_DIR)):
            if entry.startswith("branch_"):
                branch_id = entry[len("branch_"):]
                m_path, f_path = branch_model_path(branch_id)
                has_model = os.path.exists(m_path) and os.path.exists(f_path)
                has_buffer = os.path.exists(branch_buffer_table_path(branch_id))
                result.append({
                    "branch_id": branch_id,
                    "has_model": has_model,
                    "model_path": m_path if has_model else None,
                    "buffer_calibrated": has_buffer,
                    "model_scope": f"branch:{branch_id}" if has_model else "global (fallback)",
                })

    return {
        "global_model": {
            "exists": model is not None,
            "path": GLOBAL_MODEL_PATH if model is not None else None,
            "note": (
                "Dipakai sebagai fallback untuk branch yang belum punya model sendiri. "
                "Jika 'exists' false, belum ada cukup order real untuk melatih model apa "
                "pun — /predict akan melayani estimasi non-ML (baseline_no_ml)."
            ),
        },
        "branch_models": result,
        "tip": (
            f"Panggil POST /retrain dengan branch_id untuk melatih model spesifik per cabang. "
            "Minimal order real dengan timestamp sent_to_kitchen_at & prepared_at dibutuhkan "
            "(lihat training_data.MIN_SAMPLES_FOR_MODEL)."
        ),
    }
