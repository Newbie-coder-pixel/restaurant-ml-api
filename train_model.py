"""Bootstrap training for the global model — from REAL Supabase order history only.

There used to be a synthetic-data generator here (a hand-tuned formula fed into a
Gradient Boosting model, which just re-learned the formula). That's gone. This
script now does exactly what POST /retrain does for scope=global: pull real
completed orders, chronologically hold out the most recent slice, grid-search
hyperparameters, and only replace the saved model if the new one isn't worse.

Cold start: if there isn't enough real data yet (a fresh deploy with an empty
database), this exits 0 WITHOUT writing any model files, so
`train_model.py && uvicorn ...` in the Procfile still starts the API — main.py
serves a clearly-labeled non-ML baseline (raw menu prep time, no buffer, no
equipment factor) until enough real orders exist.
"""
import os
import sys

import joblib

from calibration import compute_learned_buffer_table, save_table
from model_fit import chronological_split, evaluate, fit_best_model
from paths import GLOBAL_BUFFER_TABLE_PATH, GLOBAL_FEAT_PATH, GLOBAL_MODEL_PATH
from training_data import FEATURES, MIN_SAMPLES_FOR_MODEL, TARGET, build_training_df, get_supabase


def main() -> int:
    try:
        supabase = get_supabase()
    except RuntimeError as e:
        print(f"⚠️  {e}")
        print("   Tidak ada koneksi Supabase → tidak ada model yang bisa dilatih dari data real.")
        print("   API akan tetap jalan dan melayani estimasi non-ML (baseline) sampai ini diatur.")
        return 0

    df = build_training_df(supabase, branch_id=None)
    if len(df) < MIN_SAMPLES_FOR_MODEL:
        print(f"⚠️  Baru ada {len(df)} order real dengan timestamp lengkap "
              f"(butuh minimal {MIN_SAMPLES_FOR_MODEL} untuk model global yang bisa dipercaya).")
        print("   Tidak menulis model apa pun — API akan melayani estimasi non-ML (baseline) untuk sementara.")
        return 0

    train_df, test_df = chronological_split(df)
    best_model, best_params, cv_mae_mean, cv_mae_std = fit_best_model(train_df[FEATURES], train_df[TARGET])
    metrics = evaluate(best_model, test_df[FEATURES], test_df[TARGET])

    print(f"✅ Melatih model global dari {len(df)} order real "
          f"({len(train_df)} train / {len(test_df)} holdout kronologis)")
    print(f"   Hyperparameter terbaik (GridSearchCV): {best_params}")
    print(f"   CV MAE saat training : {cv_mae_mean:.2f} ± {cv_mae_std:.2f} menit")
    print(f"   Holdout MAE (terbaru): {metrics['mae']:.2f} menit")
    print(f"   Holdout R²            : {metrics['r2']:.4f}")

    # Fit final model on ALL real data (train+holdout) with the tuned hyperparameters
    # — the holdout above is only used to report an honest, look-ahead-free metric.
    final_model, _, _, _ = fit_best_model(df[FEATURES], df[TARGET])

    os.makedirs(os.path.dirname(GLOBAL_MODEL_PATH), exist_ok=True)
    joblib.dump(final_model, GLOBAL_MODEL_PATH)
    joblib.dump(FEATURES, GLOBAL_FEAT_PATH)

    buffer_table = compute_learned_buffer_table(df, best_params | {"random_state": 42})
    save_table(GLOBAL_BUFFER_TABLE_PATH, buffer_table)
    print(f"   Buffer terkalibrasi dari data disimpan ({buffer_table['samples_used']} sampel, "
          f"{len(buffer_table['per_hour'])} jam punya bucket sendiri).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
