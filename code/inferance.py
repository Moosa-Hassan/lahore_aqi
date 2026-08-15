"""
inference.py

Loads the latest registered model from Hopsworks Model Registry,
pulls the most recent feature row from the Feature Store, and
produces a 72-hour ahead AQI forecast.

Usage:
    python code/inference.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import shutil
from datetime import datetime, timezone
import hopsworks

load_dotenv()

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 6
MODEL_NAME = "aqi_forecast_lahore"
LOCAL_MODEL_DIR = "models/downloaded"


# ============================================================
# 1. Load latest model from registry
# ============================================================
import shutil

def load_latest_model(project, max_retries=3):
    mr = project.get_model_registry()
    model = mr.get_best_model(MODEL_NAME, "overall_mae", "min")

    # wipe any leftover partial downloads from previous failed attempts
    if os.path.exists(LOCAL_MODEL_DIR):
        shutil.rmtree(LOCAL_MODEL_DIR)
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            model_dir = model.download(LOCAL_MODEL_DIR)
            break
        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            print("Retrying...")
            # clean up before retrying, same reason as above
            if os.path.exists(LOCAL_MODEL_DIR):
                shutil.rmtree(LOCAL_MODEL_DIR)
            os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    else:
        raise RuntimeError("Model download failed after retries")

    bundle_path = os.path.join(model_dir, "aqi_xgb_72h.joblib")
    bundle = joblib.load(bundle_path)

    print(f"Loaded model '{MODEL_NAME}' v{model.version}, trained_at={bundle['trained_at']}")
    return bundle


def read_feature_group_with_retry(feature_group, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            return feature_group.read()
        except Exception as e:
            print(f"Feature group read attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            print("Retrying...")


# ============================================================
# 2. Load latest feature row from feature store
# ============================================================
def load_latest_features(project, feature_cols):
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = read_feature_group_with_retry(fg)   # <-- added
    df = df.sort_values("timestamp").reset_index(drop=True)

    latest_row = df.iloc[-1]
    latest_timestamp = latest_row["timestamp"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature store is missing columns the model expects: {missing}")

    feats = latest_row[feature_cols].values.reshape(1, -1)
    return feats, latest_timestamp

# ============================================================
# 3. Predict 72h forward
# ============================================================
def forecast_next_72h(bundle, feats, latest_timestamp):
    models = bundle["models"]
    horizon = bundle["horizon"]

    preds = np.array([models[h].predict(feats)[0] for h in range(1, horizon + 1)])

    future_timestamps = pd.date_range(
        start=latest_timestamp + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )

    forecast_df = pd.DataFrame({
        "timestamp": future_timestamps,
        "predicted_aqi": preds,
    })
    # Store the actual pipeline generation time so the dashboard can report
    # freshness instead of estimating it from the first forecast timestamp.
    forecast_df["forecast_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    forecast_df["data_as_of"] = latest_timestamp
    return forecast_df


# ============================================================
# 4. Entry point
# ============================================================
def main():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])

    bundle = load_latest_model(project)
    feature_cols = bundle["feature_cols"]

    feats, latest_timestamp = load_latest_features(project, feature_cols)
    forecast_df = forecast_next_72h(bundle, feats, latest_timestamp)

    print(f"\nForecast generated from data as of {latest_timestamp}")
    print(forecast_df.head(10))

    # save locally so the dashboard can read it without re-running inference
    os.makedirs("predictions", exist_ok=True)
    out_path = "predictions/latest_forecast.csv"
    forecast_df.to_csv(out_path, index=False)
    print(f"\nSaved forecast to {out_path}")

    return forecast_df


if __name__ == "__main__":
    main() 