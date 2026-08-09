"""
training.py

Reads features from the Hopsworks Feature Store, trains 72 per-horizon
XGBoost models, backtests them, and pushes the model bundle to the
Hopsworks Model Registry.

Usage:
    python code/training.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


import hopsworks
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 6
HORIZON = 72
MODEL_DIR = "models"
MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, "aqi_xgb_72h.joblib")


# ============================================================
# 1. Load features from Hopsworks
# ============================================================
def load_features():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    df = fg.read(read_options={"use_hive": True})

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, project

# ============================================================
# 2. Build targets + feature columns
# ============================================================
def build_targets(df):
    df = df.copy()
    for h in range(1, HORIZON + 1):
        df[f"target_t+{h}"] = df["us_aqi"].shift(-h)
    df = df.dropna().reset_index(drop=True)

    target_cols = [f"target_t+{h}" for h in range(1, HORIZON + 1)]
    feature_cols = [c for c in df.columns if c not in ["timestamp"] + target_cols]
    return df, feature_cols, target_cols


# ============================================================
# 3. Train/val/backtest split
# ============================================================
def split_data(df, backtest_hours=24 * 30, val_frac=0.1):
    split_idx = len(df) - backtest_hours
    train_val_df = df.iloc[:split_idx]
    backtest_df = df.iloc[split_idx:].reset_index(drop=True)

    val_split = int(len(train_val_df) * (1 - val_frac))
    train_df = train_val_df.iloc[:val_split]
    val_df = train_val_df.iloc[val_split:]

    return train_df, val_df, backtest_df


# ============================================================
# 4. Train 72 per-horizon models
# ============================================================
def train_models(train_df, val_df, feature_cols):
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    models = {}
    val_scores = []

    for h in range(1, HORIZON + 1):
        target = f"target_t+{h}"
        y_train = train_df[target]
        y_val = val_df[target]

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=40,
            eval_metric="mae",
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        models[h] = model
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        val_scores.append(mae)

        if h % 12 == 0:
            print(f"Horizon t+{h:2d} | Val MAE: {mae:.2f}")

    print(f"\nAverage Val MAE across all horizons: {np.mean(val_scores):.2f}")
    return models


# ============================================================
# 5. Walk-forward backtest
# ============================================================
def predict_from_origin(origin_idx, models, feature_cols, df, horizon=HORIZON):
    feats = df.iloc[origin_idx][feature_cols].values.reshape(1, -1)
    return np.array([models[h].predict(feats)[0] for h in range(1, horizon + 1)])


def walk_forward_backtest(models, feature_cols, backtest_df, n_origins=30):
    max_origin = len(backtest_df) - HORIZON - 1
    if max_origin < 1:
        print("Backtest pool too small, skipping walk-forward backtest.")
        return {}

    origin_indices = np.linspace(0, max_origin, min(n_origins, max_origin)).astype(int)
    all_errors = np.zeros((len(origin_indices), HORIZON))

    for i, origin in enumerate(origin_indices):
        preds = predict_from_origin(origin, models, feature_cols, backtest_df)
        actual = backtest_df["us_aqi"].iloc[origin + 1: origin + 1 + HORIZON].values
        all_errors[i, :] = np.abs(preds - actual)

    mae_per_horizon = all_errors.mean(axis=0)

    metrics = {
        "day1_mae": float(mae_per_horizon[:24].mean()),
        "day2_mae": float(mae_per_horizon[24:48].mean()),
        "day3_mae": float(mae_per_horizon[48:].mean()),
        "overall_mae": float(mae_per_horizon.mean()),
        "n_origins": len(origin_indices),
    }

    print("\n===== Walk-Forward Backtest =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


# ============================================================
# 6. Save locally + push to Model Registry
# ============================================================
def save_and_register(models, feature_cols, metrics, project):
    os.makedirs(MODEL_DIR, exist_ok=True)

    bundle = {
        "models": models,
        "feature_cols": feature_cols,
        "horizon": HORIZON,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"\nSaved local bundle: {MODEL_BUNDLE_PATH}")

    mr = project.get_model_registry()

    model = mr.python.create_model(
        name="aqi_forecast_lahore",
        metrics={
            "overall_mae": metrics.get("overall_mae"),
            "day1_mae": metrics.get("day1_mae"),
            "day2_mae": metrics.get("day2_mae"),
            "day3_mae": metrics.get("day3_mae"),
        },
        description="72 per-horizon XGBoost models predicting hourly US AQI for Lahore, 1-72h ahead.",
    )
    model.save(MODEL_DIR)  # uploads everything in models/, including the joblib bundle
    print(f"Registered model in Hopsworks Model Registry: {model.name} v{model.version}")


# ============================================================
# Entry point
# ============================================================
def main():
    print("Loading features from Hopsworks...")
    df, project = load_features()
    print(f"Loaded {df.shape}")

    df, feature_cols, target_cols = build_targets(df)
    print(f"Features: {len(feature_cols)} | Targets: {len(target_cols)}")

    train_df, val_df, backtest_df = split_data(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Backtest pool: {len(backtest_df)}")

    print("\nTraining 72 per-horizon models...")
    models = train_models(train_df, val_df, feature_cols)

    print("\nRunning walk-forward backtest...")
    metrics = walk_forward_backtest(models, feature_cols, backtest_df)

    print("\nSaving and registering model...")
    save_and_register(models, feature_cols, metrics, project)


if __name__ == "__main__":
    main()