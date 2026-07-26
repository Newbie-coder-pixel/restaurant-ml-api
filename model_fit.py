"""Shared fitting/evaluation logic for train_model.py and /retrain.

Two rigor fixes live here that used to be missing:
  1. chronological_split — real orders have real timestamps, so the holdout used
     to accept/reject a retrain is the most RECENT slice of data, never a random
     shuffle. A random split let the model "see the future" during evaluation.
  2. fit_best_model — hyperparameters are chosen by GridSearchCV against held-out
     folds instead of being hardcoded numbers nobody tuned.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
    "min_samples_split": [10],
}


def chronological_split(df: pd.DataFrame, min_test: int = 15, max_test_frac: float = 0.4):
    """Sort by real order_time and hold out the most recent slice — this is how the
    model will actually be used (trained on the past, evaluated on what comes next)."""
    df_sorted = df.sort_values("order_time").reset_index(drop=True)
    n = len(df_sorted)
    n_test = max(min_test, round(n * 0.2))
    n_test = min(n_test, max(1, int(n * max_test_frac)))
    n_test = max(1, min(n_test, n - 1))
    split_idx = n - n_test
    return (
        df_sorted.iloc[:split_idx].reset_index(drop=True),
        df_sorted.iloc[split_idx:].reset_index(drop=True),
    )


def fit_best_model(X_train: pd.DataFrame, y_train: pd.Series):
    """GridSearchCV over PARAM_GRID; cv folds shrink automatically for small datasets.
    Returns (best_model, best_params, cv_mae_mean, cv_mae_std) — the mean/std let a
    retrain report a confidence range instead of a single noisy point estimate."""
    n = len(X_train)
    cv = max(2, min(3, n))
    search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=cv,
    )
    search.fit(X_train, y_train)
    best_idx = search.best_index_
    cv_mae_mean = float(-search.cv_results_["mean_test_score"][best_idx])
    cv_mae_std = float(search.cv_results_["std_test_score"][best_idx])
    return search.best_estimator_, search.best_params_, cv_mae_mean, cv_mae_std


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan"),
    }
