"""
preprocessing.py
================
Person 1 – Data & Feature Engineer
Flipkart Gridlock Hackathon 2.0: Traffic Demand Prediction

Steps covered:
  1. Spatial Extraction  – decode geohash → lat/lon (using a pure-Python decoder)
  2. Temporal Features   – hour, minute, day_of_week + cyclical (sin/cos)
  3. Categorical Encoding – Weather, RoadType (target-enc + OHE fallback),
                            LargeVehicles / Landmarks → binary
  4. Lag / Rolling Features – demand rolling stats per geohash × time-slot
  5. Hand-off              – export enriched train/test CSVs + encoder artifacts

Usage:
    python preprocessing.py               # uses default paths
    python preprocessing.py --train data/train.csv --test data/test.csv
"""

import argparse
import os
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. Config
# ──────────────────────────────────────────────

DATA_DIR   = Path("dataset")
OUT_DIR    = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

TARGET      = "demand"
N_LAG_DAYS  = 3          # how many previous day-slots to use for lag features
ROLLING_WIN = [3, 7]     # rolling window sizes (in sorted day-order per geohash)

# ──────────────────────────────────────────────
# 1. Helpers
# ──────────────────────────────────────────────

def decode_geohash_pure(geohash_str: str) -> tuple[float, float, float, float]:
    """
    Pure Python geohash decoder.
    Decodes a geohash string into (latitude, longitude, lat_err, lon_err).
    Avoids requiring a C/Rust compiler to install binary geohash libraries.
    """
    if not isinstance(geohash_str, str) or not geohash_str:
        return np.nan, np.nan, np.nan, np.nan

    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    base32_map = {char: i for i, char in enumerate(base32)}

    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    is_even = True

    for char in geohash_str.lower():
        if char not in base32_map:
            return np.nan, np.nan, np.nan, np.nan
        val = base32_map[char]
        for mask in [16, 8, 4, 2, 1]:
            bit = 1 if (val & mask) else 0
            if is_even:
                # Longitude
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if bit:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                # Latitude
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if bit:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            is_even = not is_even

    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    lat_err = (lat_interval[1] - lat_interval[0]) / 2.0
    lon_err = (lon_interval[1] - lon_interval[0]) / 2.0
    return lat, lon, lat_err, lon_err


def decode_geohash_column(series: pd.Series) -> pd.DataFrame:
    """
    Decode a Series of geohash strings into a DataFrame with
    columns: latitude, longitude, lat_err, lon_err.
    """
    decoded = series.apply(decode_geohash_pure)
    out = pd.DataFrame(
        decoded.tolist(),
        columns=["latitude", "longitude", "lat_err", "lon_err"],
        index=series.index,
    )
    return out


def parse_timestamp(series: pd.Series) -> pd.DataFrame:
    """
    Parse HH:MM or HH:MM:SS timestamp strings into:
        hour, minute, second, time_minutes (total mins from midnight)
    """
    parsed = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")
    fallback_mask = parsed.isna()
    parsed[fallback_mask] = pd.to_datetime(
        series[fallback_mask], format="%H:%M", errors="coerce"
    )

    return pd.DataFrame(
        {
            "hour":         parsed.dt.hour,
            "minute":       parsed.dt.minute,
            "second":       parsed.dt.second.fillna(0).astype(int),
            "time_minutes": parsed.dt.hour * 60 + parsed.dt.minute,
        },
        index=series.index,
    )


def cyclical_encode(df: pd.DataFrame, col: str, period: float) -> pd.DataFrame:
    """
    Add sin/cos cyclical encoding for a column with given period.
    E.g. period=24 for hours, period=7 for day_of_week.
    """
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df


# ──────────────────────────────────────────────
# 2. Main preprocessing function
# ──────────────────────────────────────────────

def build_features(
    train_path: Path = TRAIN_PATH,
    test_path:  Path = TEST_PATH,
    out_dir:    Path = OUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline.
    Returns (train_enriched, test_enriched).
    """

    print("=" * 60)
    print("  Flipkart Gridlock 2.0 – Feature Engineering Pipeline")
    print("=" * 60)

    # ── Load ──────────────────────────────────
    print("\n[1/5] Loading datasets …")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"  train shape: {train.shape}")
    print(f"  test  shape: {test.shape}")

    # Mark splits so we can process together, then re-split
    train["_split"] = "train"
    test["_split"]  = "test"
    if TARGET not in test.columns:
        test[TARGET] = np.nan

    df = pd.concat([train, test], axis=0, ignore_index=True)

    # ── STEP 1: Spatial Features ──────────────
    print("\n[2/5] Step 1 – Spatial extraction (geohash decode) …")
    geo_df = decode_geohash_column(df["geohash"])
    df = pd.concat([df, geo_df], axis=1)
    print(f"  lat range  : {df['latitude'].min():.4f} – {df['latitude'].max():.4f}")
    print(f"  lon range  : {df['longitude'].min():.4f} – {df['longitude'].max():.4f}")
    print(f"  null lats  : {df['latitude'].isna().sum()}")

    # Geohash precision (string length) as a feature
    df["geohash_precision"] = df["geohash"].str.len()

    # ── STEP 2: Temporal Features ─────────────
    print("\n[3/5] Step 2 – Temporal feature extraction …")
    time_df = parse_timestamp(df["timestamp"])
    df = pd.concat([df, time_df], axis=1)

    # Day-of-week proxy: `day` column is a sequential integer (not a calendar date),
    # so we create a modular day_of_week (0-6).
    df["day_of_week"] = df["day"] % 7

    # Cyclical encodings
    df = cyclical_encode(df, "hour",        period=24)
    df = cyclical_encode(df, "minute",      period=60)
    df = cyclical_encode(df, "time_minutes",period=1440)
    df = cyclical_encode(df, "day_of_week", period=7)
    df = cyclical_encode(df, "day",         period=df["day"].max() + 1)  # full cycle

    # Time-of-day bins (rush-hour, night, etc.)
    df["time_bin"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 9, 12, 17, 20, 23],
        labels=["night", "morning_rush", "midday", "afternoon_rush", "evening", "late_night"],
    ).astype(str)

    print(f"  hours extracted: {df['hour'].nunique()} unique values")

    # ── STEP 3: Categorical Encoding ──────────
    print("\n[4/5] Step 3 – Categorical & binary encoding …")

    # Binary flags
    for col in ["LargeVehicles", "Landmarks"]:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )
        else:
            df[col] = df[col].fillna(0).astype(int)
    print(f"  LargeVehicles unique: {sorted(df['LargeVehicles'].unique())}")
    print(f"  Landmarks unique    : {sorted(df['Landmarks'].unique())}")

    # RoadType – LabelEncode (ordinal proxy; target-encode done later with OOF)
    le_road = LabelEncoder()
    df["RoadType_raw"] = df["RoadType"].astype(str).str.strip()
    df["RoadType_le"]  = le_road.fit_transform(df["RoadType_raw"])

    # Weather – LabelEncode
    le_weather = LabelEncoder()
    df["Weather_raw"] = df["Weather"].astype(str).str.strip()
    df["Weather_le"]  = le_weather.fit_transform(df["Weather_raw"])

    # One-Hot Encoding for Weather and RoadType (low-cardinality)
    df = pd.get_dummies(df, columns=["RoadType_raw", "Weather_raw"], drop_first=False, dtype=int)

    # Temperature – fill missing
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    temp_median = df.loc[df["_split"] == "train", "Temperature"].median()
    df["Temperature"] = df["Temperature"].fillna(temp_median)

    # Geohash Target Encoding (mean demand per geohash, using TRAIN only to avoid leakage)
    print("  Computing geohash target encoding (train-only mean) …")
    train_mask = df["_split"] == "train"
    geo_mean   = df.loc[train_mask].groupby("geohash")[TARGET].mean().rename("geohash_demand_mean")
    geo_std    = df.loc[train_mask].groupby("geohash")[TARGET].std().rename("geohash_demand_std")
    geo_count  = df.loc[train_mask].groupby("geohash")[TARGET].count().rename("geohash_count")
    df = df.join(geo_mean,  on="geohash").join(geo_std, on="geohash").join(geo_count, on="geohash")
    # Fill unseen test geohashes with global train mean
    global_mean = df.loc[train_mask, TARGET].mean()
    global_std  = df.loc[train_mask, TARGET].std()
    df["geohash_demand_mean"].fillna(global_mean, inplace=True)
    df["geohash_demand_std"].fillna(global_std,   inplace=True)
    df["geohash_count"].fillna(0,                 inplace=True)

    # RoadType × Weather interaction feature
    df["road_weather_interaction"] = df["RoadType_le"].astype(str) + "_" + df["Weather_le"].astype(str)
    df["road_weather_le"] = LabelEncoder().fit_transform(df["road_weather_interaction"])

    # ── STEP 4: Lag & Rolling Features ────────
    print("\n[5/5] Step 4 – Lag and rolling demand features …")

    # Sort by geohash, day, time for rolling to be meaningful
    df = df.sort_values(["geohash", "day", "time_minutes"]).reset_index(drop=True)

    # We only have demand for train rows; test demand is NaN.
    # We create lag features by shifting within each geohash group.
    # This captures historical demand for the same location at earlier time steps.
    train_df_sorted = df[df["_split"] == "train"].copy()

    # Group-level lag features (within geohash, sorted by day + time)
    print(f"  Creating {N_LAG_DAYS} lag features …")
    for lag in range(1, N_LAG_DAYS + 1):
        col_name = f"demand_lag_{lag}"
        train_df_sorted[col_name] = (
            train_df_sorted.groupby("geohash")[TARGET].shift(lag)
        )

    # Rolling mean/std features
    for w in ROLLING_WIN:
        print(f"  Rolling window = {w} …")
        train_df_sorted[f"demand_roll_mean_{w}"] = (
            train_df_sorted.groupby("geohash")[TARGET]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        train_df_sorted[f"demand_roll_std_{w}"] = (
            train_df_sorted.groupby("geohash")[TARGET]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
        )

    # Hour-of-day × geohash mean demand (a proxy for typical rush-hour demand)
    geo_hour_mean = (
        train_df_sorted.groupby(["geohash", "hour"])[TARGET]
        .mean()
        .rename("geo_hour_demand_mean")
        .reset_index()
    )

    # Merge lag features back – test rows get NaN (no historical demand available)
    lag_cols = (
        [f"demand_lag_{i}"         for i in range(1, N_LAG_DAYS + 1)]
        + [f"demand_roll_mean_{w}" for w in ROLLING_WIN]
        + [f"demand_roll_std_{w}"  for w in ROLLING_WIN]
    )
    df = df.merge(
        train_df_sorted[["Index"] + lag_cols],
        on="Index",
        how="left",
    )

    # Merge geo_hour_demand_mean
    df = df.merge(geo_hour_mean, on=["geohash", "hour"], how="left")
    df["geo_hour_demand_mean"].fillna(global_mean, inplace=True)

    # Fill lag NaNs with the geohash mean (safe fallback for test rows)
    for col in lag_cols:
        df[col].fillna(df["geohash_demand_mean"], inplace=True)

    # ── STEP 5: Export ────────────────────────
    print("\n[6/6] Step 5 – Exporting enriched datasets …")

    # Columns to drop before hand-off (raw / intermediate)
    drop_cols = [
        "timestamp", "geohash",           # raw strings replaced by features
        "road_weather_interaction",        # temp string, already label-encoded
        "time_bin",                        # can keep if needed; drop to stay numeric
        "_split",
        "lat_err", "lon_err",             # error bounds, rarely useful
    ]
    df_export = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    train_out = df_export[df_export.index.isin(
        df[df["_split"] == "train"].index
    )].copy()
    test_out  = df_export[df_export.index.isin(
        df[df["_split"] == "test"].index
    )].copy()

    # Alternatively split by original indices (more robust)
    train_out = df_export.loc[df["_split"] == "train"].reset_index(drop=True)
    test_out  = df_export.loc[df["_split"] == "test"].reset_index(drop=True)

    # Drop target from test
    test_out.drop(columns=[TARGET], inplace=True, errors="ignore")

    train_out.to_csv(out_dir / "train_features.csv", index=False)
    test_out.to_csv( out_dir / "test_features.csv",  index=False)
    print(f"  ✅  train_features.csv  →  {train_out.shape}")
    print(f"  ✅  test_features.csv   →  {test_out.shape}")

    # Save encoder artifacts for Person 2 (reproducibility)
    encoders = {
        "le_road":    le_road,
        "le_weather": le_weather,
        "geo_mean":   geo_mean,
        "geo_std":    geo_std,
        "temp_median": temp_median,
        "global_mean": global_mean,
    }
    with open(out_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print(f"  ✅  encoders.pkl saved  →  {out_dir / 'encoders.pkl'}")

    # Print feature summary
    print("\n── Feature Summary ──────────────────────────────────────")
    print(f"  Total features in train : {train_out.shape[1]}")
    print(f"  Total features in test  : {test_out.shape[1]}")
    feature_cols = [c for c in train_out.columns if c not in [TARGET, "Index"]]
    print("\n  Feature list:")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:>3}. {col}")

    print("\n" + "=" * 60)
    print("  Hand-off complete.")
    print("=" * 60)

    return train_out, test_out


# ──────────────────────────────────────────────
# 3. Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline – Person 1")
    parser.add_argument("--train", type=str, default=str(TRAIN_PATH), help="Path to train.csv")
    parser.add_argument("--test",  type=str, default=str(TEST_PATH),  help="Path to test.csv")
    parser.add_argument("--out",   type=str, default=str(OUT_DIR),    help="Output directory")
    args = parser.parse_args()

    train_out, test_out = build_features(
        train_path=Path(args.train),
        test_path =Path(args.test),
        out_dir   =Path(args.out),
    )
