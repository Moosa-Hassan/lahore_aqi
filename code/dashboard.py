"""
dashboard.py

Streamlit dashboard displaying the latest 72-hour AQI forecast for Lahore.
Reads from predictions/latest_forecast.csv (produced by inference.py).

Usage:
    streamlit run code/dashboard.py
"""

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

FORECAST_PATH = "predictions/latest_forecast.csv"

st.set_page_config(page_title="Lahore AQI Forecast", layout="wide")


# ============================================================
# AQI category helper (US AQI breakpoints)
# ============================================================
def aqi_category(value):
    if value <= 50:
        return "Good", "#00e400"
    elif value <= 100:
        return "Moderate", "#ffff00"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif value <= 200:
        return "Unhealthy", "#ff0000"
    elif value <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


# ============================================================
# Load forecast
# ============================================================
def load_forecast():
    if not os.path.exists(FORECAST_PATH):
        return None
    df = pd.read_csv(FORECAST_PATH, parse_dates=["timestamp"])
    return df


forecast_df = load_forecast()

st.title("🌫️ Lahore AQI Forecast — Next 72 Hours")

if forecast_df is None:
    st.error(
        f"No forecast found at `{FORECAST_PATH}`. "
        f"Run `python code/inferance.py` first to generate one."
    )
    st.stop()

generated_at = forecast_df["timestamp"].iloc[0] - pd.Timedelta(hours=1)
st.caption(f"Forecast generated from data as of **{generated_at}**")

# ============================================================
# Current / next-hour AQI summary card
# ============================================================
next_hour_aqi = forecast_df["predicted_aqi"].iloc[0]
peak_aqi = forecast_df["predicted_aqi"].max()
peak_time = forecast_df.loc[forecast_df["predicted_aqi"].idxmax(), "timestamp"]

category, color = aqi_category(next_hour_aqi)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Next Hour AQI", f"{next_hour_aqi:.0f}", category)

with col2:
    st.metric("72h Peak AQI", f"{peak_aqi:.0f}",
              f"at {peak_time.strftime('%a %I %p')}")

with col3:
    day1_avg = forecast_df["predicted_aqi"].iloc[:24].mean()
    day3_avg = forecast_df["predicted_aqi"].iloc[48:].mean()
    trend = "↑ worsening" if day3_avg > day1_avg else "↓ improving"
    st.metric("3-Day Trend", trend, f"{day3_avg - day1_avg:+.0f} avg AQI")

# ============================================================
# Hazardous AQI alert
# ============================================================
HAZARD_THRESHOLD = 200  # "Unhealthy" and above

hazardous_hours = forecast_df[forecast_df["predicted_aqi"] >= HAZARD_THRESHOLD]

if len(hazardous_hours) > 0:
    first_hazard = hazardous_hours.iloc[0]
    st.error(
        f"⚠️ **Hazardous AQI alert:** AQI is predicted to reach "
        f"**{first_hazard['predicted_aqi']:.0f}** "
        f"({aqi_category(first_hazard['predicted_aqi'])[0]}) "
        f"around **{first_hazard['timestamp'].strftime('%A %I %p')}**. "
        f"{len(hazardous_hours)} hour(s) in the next 72h cross this threshold."
    )
else:
    st.success(f"✅ No hours in the next 72h are predicted to reach hazardous levels (AQI ≥ {HAZARD_THRESHOLD}).")

# ============================================================
# Forecast chart
# ============================================================
st.subheader("72-Hour Forecast")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(forecast_df["timestamp"], forecast_df["predicted_aqi"],
        color="crimson", linewidth=2, marker="o", markersize=3)

# shade AQI category bands
bands = [(0, 50, "#00e400"), (50, 100, "#ffff00"), (100, 150, "#ff7e00"),
         (150, 200, "#ff0000"), (200, 300, "#8f3f97"), (300, 500, "#7e0023")]
for low, high, c in bands:
    ax.axhspan(low, high, color=c, alpha=0.08)

ax.axvline(forecast_df["timestamp"].iloc[23], color="gray", linestyle="--", alpha=0.5)
ax.axvline(forecast_df["timestamp"].iloc[47], color="gray", linestyle="--", alpha=0.5)

ax.set_xlabel("Time")
ax.set_ylabel("US AQI")
ax.set_ylim(0, max(300, forecast_df["predicted_aqi"].max() * 1.15))
ax.grid(alpha=0.3)
fig.autofmt_xdate()

st.pyplot(fig)

# ============================================================
# Day-by-day breakdown
# ============================================================
st.subheader("Day-by-Day Breakdown")

day_labels = ["Day 1 (next 24h)", "Day 2 (24-48h)", "Day 3 (48-72h)"]
day_slices = [forecast_df.iloc[:24], forecast_df.iloc[24:48], forecast_df.iloc[48:]]

cols = st.columns(3)
for col, label, day_df in zip(cols, day_labels, day_slices):
    avg = day_df["predicted_aqi"].mean()
    peak = day_df["predicted_aqi"].max()
    cat, _ = aqi_category(avg)
    with col:
        st.markdown(f"**{label}**")
        st.write(f"Avg: {avg:.0f} ({cat})")
        st.write(f"Peak: {peak:.0f}")

# ============================================================
# Raw data (expandable)
# ============================================================
with st.expander("Raw forecast data"):
    st.dataframe(forecast_df, use_container_width=True)