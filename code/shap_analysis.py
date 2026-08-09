"""
shap_analysis.py

Generates SHAP feature importance plots for representative forecast
horizons (short, mid, long) from the trained 72-model XGBoost bundle.

Usage:
    python code/shap_analysis.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

MODEL_BUNDLE_PATH = "models/aqi_xgb_72h.joblib"
OUTPUT_DIR = "reports/shap"
HORIZONS_TO_EXPLAIN = [1, 24, 72]   # t+1h, t+24h, t+72h
N_SAMPLES = 500                      # subsample for speed; SHAP on full data is slow


def load_bundle():
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    return bundle


def load_background_data(feature_cols):
    """
    Uses the same feature bundle's training data if available;
    otherwise expects a local df snapshot. Simplest: re-read from
    Hopsworks feature group so SHAP explains on real recent data.
    """
    import hopsworks
    from dotenv import load_dotenv
    load_dotenv()

    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=6)
    df = fg.read(read_options={"use_hive": True})
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature store missing columns model expects: {missing}")

    X = df[feature_cols].dropna()
    if len(X) > N_SAMPLES:
        X = X.sample(N_SAMPLES, random_state=42)
    return X


def explain_horizon(model, X, horizon_label, output_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 1. Summary plot (beeswarm) — overall feature impact + direction
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title(f"SHAP Summary — Horizon t+{horizon_label}h")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_summary_t{horizon_label}.png"), dpi=150)
    plt.close()

    # 2. Bar plot — mean absolute SHAP value per feature (cleaner for report)
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP Feature Importance (mean |SHAP|) — Horizon t+{horizon_label}h")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_bar_t{horizon_label}.png"), dpi=150)
    plt.close()

    # 3. Top-N features table (for quoting numbers directly in report)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df.to_csv(
        os.path.join(output_dir, f"shap_importance_t{horizon_label}.csv"), index=False
    )

    print(f"\n=== Horizon t+{horizon_label}h — Top 10 features ===")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model bundle...")
    bundle = load_bundle()
    models = bundle["models"]
    feature_cols = bundle["feature_cols"]

    print("Loading background data from feature store...")
    X = load_background_data(feature_cols)
    print(f"Using {len(X)} rows for SHAP explanation")

    results = {}
    for h in HORIZONS_TO_EXPLAIN:
        print(f"\nExplaining horizon t+{h}h...")
        model = models[h]
        results[h] = explain_horizon(model, X, h, OUTPUT_DIR)

    # ============================================================
    # Comparison: how importance shifts from short to long horizon
    # ============================================================
    print("\n=== Comparing top-5 features across horizons ===")
    for h in HORIZONS_TO_EXPLAIN:
        top5 = results[h].head(5)["feature"].tolist()
        print(f"t+{h}h: {top5}")

    print(f"\nAll plots and CSVs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()