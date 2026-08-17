"""
features.py

Fetches AQI + weather data from Open-Meteo, computes engineered features,
and writes the result to the Hopsworks Feature Store.

Usage:
    python scripts/feature_pipeline.py --mode backfill --days 700
    python scripts/feature_pipeline.py --mode hourly
"""

import os
import argparse
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import hopsworks

load_dotenv()

CITY = "Lahore"
FEATURE_GROUP_NAME = "aqi_features_lahore"
FEATURE_GROUP_VERSION = 1


# ============================================================
# 1. Data fetching
# ============================================================
def get_city_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    response = requests.get(url, timeout=30)
    data = response.json()
    if "results" not in data:
        raise ValueError(f"City '{city_name}' not found.")
    r = data["results"][0]
    return r["latitude"], r["longitude"], r["name"]


def fetch_aqi_data(lat, lon, start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    if "hourly" not in data:
        raise ValueError(f"Open-Meteo AQI API error: {data}")

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "us_aqi": data["hourly"]["us_aqi"],
        "pm25": data["hourly"]["pm2_5"],
        "pm10": data["hourly"]["pm10"],
        "co": data["hourly"]["carbon_monoxide"],
        "no2": data["hourly"]["nitrogen_dioxide"],
        "so2": data["hourly"]["sulphur_dioxide"],
        "o3": data["hourly"]["ozone"],
        "dust": data["hourly"]["dust"],
    })


def fetch_weather_data(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    if "hourly" not in data:
        raise ValueError(f"Open-Meteo weather API error: {data}")

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"],
        "humidity_pct": data["hourly"]["relative_humidity_2m"],
        "wind_speed_kmh": data["hourly"]["wind_speed_10m"],
        "precipitation_mm": data["hourly"]["precipitation"],
    })


def fetch_live_aqi_data(lat, lon, past_hours=240, forecast_hours=1):
    """Fetch recent/current AQI data from the live Air Quality Forecast API.

    Unlike the archive API, this endpoint is continuously updated and supports
    past_hours/forecast_hours, so the latest feature row can stay close to now.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust",
        "past_hours": past_hours,
        "forecast_hours": forecast_hours,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "hourly" not in data:
        raise ValueError(f"Open-Meteo live AQI API error: {data}")

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "us_aqi": data["hourly"]["us_aqi"],
        "pm25": data["hourly"]["pm2_5"],
        "pm10": data["hourly"]["pm10"],
        "co": data["hourly"]["carbon_monoxide"],
        "no2": data["hourly"]["nitrogen_dioxide"],
        "so2": data["hourly"]["sulphur_dioxide"],
        "o3": data["hourly"]["ozone"],
        "dust": data["hourly"]["dust"],
    })


def fetch_live_weather_data(lat, lon, past_hours=240, forecast_hours=1):
    """Fetch recent/current weather from the live Weather Forecast API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "past_hours": past_hours,
        "forecast_hours": forecast_hours,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "hourly" not in data:
        raise ValueError(f"Open-Meteo live weather API error: {data}")

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"],
        "humidity_pct": data["hourly"]["relative_humidity_2m"],
        "wind_speed_kmh": data["hourly"]["wind_speed_10m"],
        "precipitation_mm": data["hourly"]["precipitation"],
    })


def fetch_raw_data(days, lag_days=5):
    """
    lag_days: archive APIs need a few days' lag before data is finalized,
    """
    lat, lon, city = get_city_coordinates(CITY)
    print(f"Coordinates for {city}: lat={lat}, lon={lon}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df_aqi = fetch_aqi_data(lat, lon, start_date, end_date)
    df_weather = fetch_weather_data(lat, lon, start_date, end_date)

    df_raw = pd.merge(df_aqi, df_weather, on="timestamp", how="inner")
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
    return df_raw


def create_features(df, target_col="us_aqi"):
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -------------------------------------------------
    # 1. Calendar / Time features
    # -------------------------------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # -------------------------------------------------
    # 2. Columns we want to create lags / rolling for
    # -------------------------------------------------
    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3", "dust"]
    weather_cols   = ["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm"]
    all_cols       = pollutant_cols + weather_cols + [target_col]

    # -------------------------------------------------
    # 3. Lag 1, Lag 24 and Trend (exactly as you wanted)
    # -------------------------------------------------
    for col in all_cols:
        df[f"{col}_lag_1"]  = df[col].shift(1)
        df[f"{col}_lag_24"] = df[col].shift(24)
        df[f"{col}_trend"]  = df[f"{col}_lag_1"] - df[f"{col}_lag_24"]

    # -------------------------------------------------
    # 4. Extra useful lags (recommended)
    # -------------------------------------------------
    extra_lags = [2, 3, 6, 12, 48, 72]
    for col in [target_col] + weather_cols:          # mainly for AQI + weather
        for lag in extra_lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # -------------------------------------------------
    # 5. Rolling means (24h) - as you requested
    # -------------------------------------------------
    for col in all_cols:
        df[f"{col}_rolling_24h"] = df[col].rolling(window=24, min_periods=1).mean()

    # -------------------------------------------------
    # 6. Extra strong features (help with under-prediction)
    # -------------------------------------------------
    # Rolling max (captures recent peaks)
    for window in [6, 12, 24]:
        df[f"{target_col}_max_{window}"] = df[target_col].rolling(window).max()

    # Short-term differences
    df[f"{target_col}_diff_1"] = df[target_col].diff(1)
    df[f"{target_col}_diff_3"] = df[target_col].diff(3)
    df[f"{target_col}_diff_6"] = df[target_col].diff(6)

    # Rolling std (volatility)
    df[f"{target_col}_std_24"] = df[target_col].rolling(24).std()

    # -------------------------------------------------
    # 7. Drop original current-hour columns
    #    (they are not available at prediction time)
    # -------------------------------------------------
    cols_to_drop = pollutant_cols + weather_cols
    df = df.drop(columns=cols_to_drop)

    # Remove rows with NaN created by lags/rolling
    df = df.dropna().reset_index(drop=True)

    return df

# ============================================================
# 3. Hopsworks write
# ============================================================
def write_to_feature_store(df):
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name="aqi_features",
        version=6,
        description="Hourly AQI, weather, and engineered features for ML training",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )

    feature_group.insert(df, write_options={"wait_for_job": True})
    print(f"Wrote {len(df)} rows to feature group 'aqi_features' v6")

# ============================================================
# 4. Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backfill", "hourly"], default="hourly")
    parser.add_argument("--days", type=int, default=700, help="only used in backfill mode")
    args = parser.parse_args()

    if args.mode == "backfill":
        print(f"Running backfill for {args.days} days...")
        df_raw = fetch_raw_data(days=args.days)
    else:
        # Live mode: use forecast APIs with recent history instead of archive APIs.
        # This avoids the multi-day lag of finalized archive/reanalysis data.
        print("Running live hourly fetch...")
        lat, lon, city = get_city_coordinates(CITY)
        print(f"Coordinates for {city}: lat={lat}, lon={lon}")

        df_aqi = fetch_live_aqi_data(lat, lon, past_hours=240, forecast_hours=1)
        df_weather = fetch_live_weather_data(lat, lon, past_hours=240, forecast_hours=1)

        df_raw = pd.merge(df_aqi, df_weather, on="timestamp", how="inner")
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
        print(f"Latest live source timestamp: {df_raw['timestamp'].iloc[-1]}")

    print(f"Raw data: {df_raw.shape}")
    print(f"Latest 10 data points:\n{df_raw.tail(10)}")
    df_features = create_features(df_raw)
    print(f"Feature data: {df_features.shape}")
    print(df_features.tail(5))

    write_to_feature_store(df_features)


if __name__ == "__main__":
    main()