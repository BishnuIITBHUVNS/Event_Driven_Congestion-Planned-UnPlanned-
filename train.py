"""
train.py  —  Step 2 of the pipeline.

Run:
    python src/train.py

Reads  : data/processed/events_features.csv
Saves  : models/xgb_clf.pkl
         models/lgb_clf.pkl
         models/xgb_reg.pkl
         models/lgb_reg.pkl
         models/feature_cols.pkl

Two tasks:
    Task A — Classification:  predict priority_class (High / Low)
    Task B — Regression:      predict impact_score (0–10)

Both use XGBoost + LightGBM with early stopping, then ensemble the two.
Evaluation: time-aware split (train on older events, test on recent ones).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, classification_report,
)

from config import DATA_PROC, MODEL_DIR, FEATURE_COLS, TARGET_CLS, TARGET_REG


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def load_processed(path=DATA_PROC):
    df = pd.read_csv(path)
    available = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available].fillna(0)
    y_cls = df[TARGET_CLS]                         # road_closure_class
    # Regression: only rows where duration was observed
    reg_mask = df[TARGET_REG].notna()
    X_reg  = X[reg_mask]
    y_reg  = df.loc[reg_mask, TARGET_REG]
    print(f"Loaded {len(df):,} rows | {len(available)} features")
    print(f"  Classification rows : {len(X):,}  positive={y_cls.mean():.1%}")
    print(f"  Regression rows     : {len(X_reg):,}  (events with closed_datetime)")
    return X, y_cls, X_reg, y_reg, df, available


def time_aware_split(df_path=DATA_PROC):
    """
    Split chronologically — train on the first 80% of time, test on the last 20%.
    Better than random split for time-series-flavoured data.
    """
    df = pd.read_csv(df_path, parse_dates=["start_datetime"])
    df = df.sort_values("start_datetime").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_idx = df.index[:split_idx]
    test_idx  = df.index[split_idx:]
    return train_idx.tolist(), test_idx.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# TASK A — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(X_tr, X_val, y_tr, y_val):
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = n_neg / max(n_pos, 1)           # handle class imbalance

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_clf = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    # ── Ensemble predict ─────────────────────────────────────────────────────
    xgb_p = xgb_clf.predict_proba(X_val)[:, 1]
    lgb_p = lgb_clf.predict_proba(X_val)[:, 1]
    ens_p = 0.5 * xgb_p + 0.5 * lgb_p
    ens_y = (ens_p >= 0.5).astype(int)

    metrics = {
        "F1":        f1_score(y_val, ens_y),
        "Precision": precision_score(y_val, ens_y),
        "Recall":    recall_score(y_val, ens_y),
        "ROC-AUC":   roc_auc_score(y_val, ens_p),
    }
    return xgb_clf, lgb_clf, metrics


# ─────────────────────────────────────────────────────────────────────────────
# TASK B — REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def train_regressor(X_tr, X_val, y_tr, y_val):

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_reg = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
        verbosity=0,
    )
    xgb_reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_reg.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(40, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    # ── Ensemble predict ─────────────────────────────────────────────────────
    ens_pred = np.clip(
        0.5 * xgb_reg.predict(X_val) + 0.5 * lgb_reg.predict(X_val),
        0, 10,
    )

    mape = np.mean(np.abs((y_val - ens_pred) / (y_val + 1e-8))) * 100
    metrics = {
        "MAE":      mean_absolute_error(y_val, ens_pred),
        "RMSE":     np.sqrt(mean_squared_error(y_val, ens_pred)),
        "R²":       r2_score(y_val, ens_pred),
        "MAPE (%)": mape,
    }
    return xgb_reg, lgb_reg, metrics


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance_report(xgb_model, lgb_model, feature_cols: list) -> pd.DataFrame:
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols, name="xgb")
    lgb_imp = pd.Series(lgb_model.feature_importances_, index=feature_cols, name="lgb")
    fi = pd.concat([xgb_imp, lgb_imp], axis=1)
    fi["mean_importance"] = fi.mean(axis=1)
    return fi.sort_values("mean_importance", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y_cls, X_reg, y_reg, df_full, feature_cols = load_processed()

    # ── Chronological split (classification) ─────────────────────────────────
    train_idx, test_idx = time_aware_split()
    X_tr,  X_val  = X.iloc[train_idx],     X.iloc[test_idx]
    yc_tr, yc_val = y_cls.iloc[train_idx], y_cls.iloc[test_idx]

    # Chronological split (regression — subset with valid duration)
    df_full_sorted = df_full.sort_values("start_datetime").reset_index(drop=True)
    reg_rows  = df_full_sorted[df_full_sorted[TARGET_REG].notna()].reset_index()
    r_split   = int(len(reg_rows) * 0.8)
    reg_tr_idx = reg_rows.index[:r_split]
    reg_val_idx = reg_rows.index[r_split:]
    X_r_tr  = X_reg.iloc[:r_split]
    X_r_val = X_reg.iloc[r_split:]
    yr_tr   = y_reg.iloc[:r_split]
    yr_val  = y_reg.iloc[r_split:]

    print(f"\nClassification split  — train: {len(X_tr):,}  test: {len(X_val):,}")
    print(f"Regression split      — train: {len(X_r_tr):,}  test: {len(X_r_val):,}")

    # ── Task A ────────────────────────────────────────────────────────────────
    print_section("Task A — Road closure classification (requires_road_closure)")
    xgb_clf, lgb_clf, cls_m = train_classifier(X_tr, X_val, yc_tr, yc_val)
    for k, v in cls_m.items():
        print(f"  {k:<15} {v:.4f}")

    print("\n  Full classification report:")
    ens_proba = 0.5 * xgb_clf.predict_proba(X_val)[:, 1] + \
                0.5 * lgb_clf.predict_proba(X_val)[:, 1]
    ens_pred_cls = (ens_proba >= 0.5).astype(int)
    print(classification_report(yc_val, ens_pred_cls,
                                target_names=["No closure", "Road closure"]))

    # ── Task B ────────────────────────────────────────────────────────────────
    print_section("Task B — Event duration regression (log_duration_min)")
    xgb_reg, lgb_reg, reg_m = train_regressor(X_r_tr, X_r_val, yr_tr, yr_val)
    for k, v in reg_m.items():
        print(f"  {k:<15} {v:.4f}")

    # ── Feature importance ────────────────────────────────────────────────────
    print_section("Top 15 features (road closure classifier)")
    fi = feature_importance_report(xgb_clf, lgb_clf, feature_cols)
    print(fi.head(15).to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(xgb_clf,      MODEL_DIR / "xgb_clf.pkl")
    joblib.dump(lgb_clf,      MODEL_DIR / "lgb_clf.pkl")
    joblib.dump(xgb_reg,      MODEL_DIR / "xgb_reg.pkl")
    joblib.dump(lgb_reg,      MODEL_DIR / "lgb_reg.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")
    print(f"\n  Models saved to {MODEL_DIR}/")

    return cls_m, reg_m


if __name__ == "__main__":
    run()
