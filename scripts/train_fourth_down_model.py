"""End-to-end training script for the 4th-down decision model.

Usage:
    uv run python scripts/train_fourth_down_model.py \
        --train-seasons 2019 2020 2021 2022 \
        --test-seasons 2023 \
        --output-dir models/latest \
        --cache-data

Workflow: ingest NFL PBP data → Ibis feature pipeline → train XGBoost → evaluate → save
"""

import argparse
import sys
from pathlib import Path

import ibis
import pandas as pd

from nfl.ingest import load_pbp_data, pbp_to_parquet, load_pbp_from_parquet
from ml.dataset import build_training_dataset, split_features_target, train_test_split_by_season
from ml.model import FourthDownModel
from ml.evaluate import evaluate_model
from ml.serialize import save_model


def main():
    parser = argparse.ArgumentParser(description="Train the 4th-down decision model")
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        type=int,
        default=[2021, 2022],
        help="Seasons to train on (default: 2021 2022)",
    )
    parser.add_argument(
        "--test-seasons",
        nargs="+",
        type=int,
        default=[2023],
        help="Seasons to test on (default: 2023)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/latest",
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--cache-data",
        action="store_true",
        help="Cache downloaded PBP data to data/pbp/ as Parquet",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of XGBoost trees (default: 200)",
    )
    args = parser.parse_args()

    all_seasons = sorted(set(args.train_seasons + args.test_seasons))
    print(f"Seasons: train={args.train_seasons}, test={args.test_seasons}")

    # Step 1: Load data
    cache_path = Path("data/pbp/pbp_cache.parquet")
    if args.cache_data and cache_path.exists():
        print(f"Loading cached PBP data from {cache_path}")
        raw_df = load_pbp_from_parquet(cache_path)
    else:
        print(f"Downloading PBP data for seasons {all_seasons}...")
        raw_df = load_pbp_data(all_seasons)
        if args.cache_data:
            pbp_to_parquet(raw_df, cache_path)
            print(f"Cached to {cache_path}")

    print(f"Raw data: {len(raw_df):,} plays")

    # Step 2: Build training dataset via Ibis
    print("Building training dataset via Ibis pipeline...")
    con = ibis.duckdb.connect()
    pbp_table = ibis.memtable(raw_df)
    dataset_expr = build_training_dataset(pbp_table)
    dataset = con.execute(dataset_expr)
    print(f"4th-down dataset: {len(dataset):,} plays")

    # Step 3: Temporal split
    train_df, test_df = train_test_split_by_season(dataset, test_seasons=args.test_seasons)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    print(f"Train: {len(X_train):,} plays | Test: {len(X_test):,} plays")

    if len(X_train) == 0:
        print("ERROR: No training data. Check --train-seasons.")
        sys.exit(1)
    if len(X_test) == 0:
        print("=" * 60)
        print("WARNING: No test data available for the specified test seasons.")
        print("Evaluation metrics will reflect TRAINING data — not held-out data.")
        print("Results may be overly optimistic. Use --test-seasons to specify")
        print("a held-out season for honest evaluation.")
        print("=" * 60)
        X_test, y_test = X_train, y_train

    # Step 4: Train
    print(f"Training XGBoost ({args.n_estimators} trees)...")
    model = FourthDownModel(hyperparams={"n_estimators": args.n_estimators})
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Step 5: Evaluate
    print("Evaluating on test set...")
    report = evaluate_model(model, X_test, y_test)
    print(report.summary())

    # Step 6: Save
    save_model(model, args.output_dir, report=report)
    print(f"\nModel saved to {args.output_dir}/")
    print(f"  model.joblib    — serialized XGBoost model")
    print(f"  metadata.json   — version, features, evaluation metrics")
    print(f"\nTo serve: MODEL_DIR={args.output_dir} uv run uvicorn api.main:app")


if __name__ == "__main__":
    main()
