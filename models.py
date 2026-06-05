"""
models.py
=========
Person 2 (Modeling) & Person 3 (MLOps)
Flipkart Gridlock Hackathon 2.0: Traffic Demand Prediction

This script:
  1. Loads processed training and test features.
  2. Prepares features by dropping non-numeric/raw ID columns.
  3. Sets up a 5-Fold Cross-Validation scheme.
  4. Trains an ensemble of advanced regressors (scikit-learn HistGradientBoosting Regressors).
  5. Computes the Out-Of-Fold R-squared (R2) score and the Hackerrank metric.
  6. Predicts on the test set and performs automated QA checks on the output submission CSV.

Usage:
    python models.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------
# 0. Config
# ----------------------------------------------

DATA_DIR = Path("data/processed")
TRAIN_FEAT_PATH = DATA_DIR / "train_features.csv"
TEST_FEAT_PATH = DATA_DIR / "test_features.csv"
SUB_OUTPUT_PATH = Path("submission.csv")

TARGET = "demand"

# ----------------------------------------------
# 1. Main Pipeline
# ----------------------------------------------

def run_pipeline(
    train_path: Path = TRAIN_FEAT_PATH,
    test_path: Path = TEST_FEAT_PATH,
    sub_path: Path = SUB_OUTPUT_PATH
):
    print("=" * 60)
    print("  Flipkart Gridlock 2.0 - Model Training & Inference")
    print("=" * 60)

    # --- Load Features ------------------------
    print("\n[1/5] Loading engineered features ...")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find processed features at {DATA_DIR}.\n"
            "Please run 'python preprocessing.py' first to generate them."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"  train shape: {train.shape}")
    print(f"  test shape : {test.shape}")

    # --- Feature Selection --------------------
    print("\n[2/5] Selecting training features ...")
    
    # Drop index and target, as well as raw string/categorical columns (non-numeric)
    ignore_cols = ["Index", TARGET, "RoadType", "Weather"]
    feature_cols = [c for c in train.columns if c not in ignore_cols]
    
    # Ensure all selected features exist in test set
    feature_cols = [c for c in feature_cols if c in test.columns]
    
    X = train[feature_cols].copy()
    y = train[TARGET].copy()
    X_test = test[feature_cols].copy()
    
    print(f"  Using {len(feature_cols)} features for model training.")
    print("  Sample features:", feature_cols[:10])

    # --- Cross-Validation Setup ---------------
    print("\n[3/5] Setting up 5-Fold Cross-Validation ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Arrays to store out-of-fold and test predictions
    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    
    # Define two complementary regressor configurations to ensemble
    # (Using HistGradientBoostingRegressor which is fast and handles missing values natively)
    model_configs = [
        {
            "name": "Model_A_Base",
            "params": {"learning_rate": 0.08, "max_iter": 200, "max_leaf_nodes": 31, "random_state": 42}
        },
        {
            "name": "Model_B_Deep",
            "params": {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 47, "random_state": 2026}
        }
    ]

    # --- Model Training Loop ------------------
    print("\n[4/5] Training models across folds ...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        fold_oof = np.zeros(len(val_idx))
        fold_test = np.zeros(len(test))
        
        # Train and average the configurations
        for config in model_configs:
            print(f"  Training {config['name']} ...")
            model = HistGradientBoostingRegressor(**config['params'])
            model.fit(X_train, y_train)
            
            # Predict
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)
            
            # Aggregate predictions (simple average)
            fold_oof += val_preds / len(model_configs)
            fold_test += test_preds / len(model_configs)
            
        fold_r2 = r2_score(y_val, fold_oof)
        fold_score = max(0, 100 * fold_r2)
        print(f"  Fold {fold} Validation R2 Score: {fold_r2:.5f} (Hackerrank Score: {fold_score:.2f})")
        
        oof_predictions[val_idx] = fold_oof
        test_predictions += fold_test / 5.0  # Average test predictions across folds

    overall_r2 = r2_score(y, oof_predictions)
    overall_score = max(0, 100 * overall_r2)
    print("\n" + "=" * 45)
    print(f"  Overall OOF R2 Score: {overall_r2:.5f}")
    print(f"  Overall Hackerrank Score: {overall_score:.2f} / 100.0")
    print("=" * 45)

    # --- Submission & QA Checks ---------------
    print("\n[5/5] Creating submission file and running QA Checks ...")
    
    # Clip predictions to target logical bounds [0, 1] (since demand is typically in [0.0, 1.0])
    test_predictions = np.clip(test_predictions, 0.0, 1.0)
    
    sub_df = pd.DataFrame({
        "Index": test["Index"].astype(int),
        "demand": test_predictions
    })
    
    # Load original test set to restore original row order
    original_test = pd.read_csv(Path("dataset/test.csv"))
    
    # Reindex to match the exact sequential order of the original test.csv
    sub_df = sub_df.set_index("Index").reindex(original_test["Index"]).reset_index()
    
    # QA Check 1: Row count
    expected_rows = 41778
    assert len(sub_df) == expected_rows, f"QA FAILED: Expected {expected_rows} rows, got {len(sub_df)}"
    print(f"  [PASS] QA Pass: Row count is exactly {expected_rows}.")
    
    # QA Check 2: Column Names
    expected_cols = ["Index", "demand"]
    assert list(sub_df.columns) == expected_cols, f"QA FAILED: Expected columns {expected_cols}, got {list(sub_df.columns)}"
    print(f"  [PASS] QA Pass: Column names are exactly {expected_cols}.")
    
    # QA Check 3: Check for NaNs/null values
    null_count = sub_df.isnull().sum().sum()
    assert null_count == 0, f"QA FAILED: Found {null_count} null values in predictions."
    print("  [PASS] QA Pass: Zero null values found.")
    
    # QA Check 4: Check Index matching test.csv
    assert (sub_df["Index"].values == original_test["Index"].values).all(), "QA FAILED: Index alignment mismatch with test.csv"
    print("  [PASS] QA Pass: Index matches test.csv index exactly.")


    # Save to CSV
    sub_df.to_csv(sub_path, index=False)
    print(f"\n[SUCCESS] Final predictions saved to: {sub_path.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training and Prediction Pipeline")
    parser.add_argument("--train", type=str, default=str(TRAIN_FEAT_PATH), help="Path to train_features.csv")
    parser.add_argument("--test", type=str, default=str(TEST_FEAT_PATH), help="Path to test_features.csv")
    parser.add_argument("--sub", type=str, default=str(SUB_OUTPUT_PATH), help="Path to save submission.csv")
    args = parser.parse_args()

    run_pipeline(
        train_path=Path(args.train),
        test_path=Path(args.test),
        sub_path=Path(args.sub)
    )
