"""
script.py

Retrieves the latest data from the Hopsworks Feature Store and displays it.

Usage:
    python script.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv()

# Feature Store configuration
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 6

def get_latest_data(project, feature_group, n_rows=10):
    """
    Retrieve the most recent data from the feature group.

    Args:
        project: Hopsworks project connection
        feature_group: FeatureGroup object
        n_rows: Number of most recent rows to return

    Returns:
        DataFrame with the most recent data
    """
    for attempt in range(1, 4):  # 3 retries
        try:
            df = feature_group.read()
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"Feature group read attempt {attempt} failed: {e}")
            if attempt == 3:
                raise
            print("Retrying...")
    
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Return the most recent n_rows
    return df.tail(n_rows)

def main():
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])

    print(f"Retrieving latest data from feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}...")
    fs = project.get_feature_store()
    feature_group = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    latest_data = get_latest_data(project, feature_group)

    print(f"\nMost recent {len(latest_data)} data points:")
    print("=" * 60)
    print(latest_data.to_string(index=False))
    print("=" * 60)

    print(f"\nData summary:")
    print(f"  - Latest timestamp: {latest_data['timestamp'].max()}")
    print(f"  - Earliest timestamp: {latest_data['timestamp'].min()}")
    print(f"  - Total rows retrieved: {len(latest_data)}")

if __name__ == "__main__":
    main()
