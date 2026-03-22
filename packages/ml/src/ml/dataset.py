"""Training dataset builder — composes the full Ibis pipeline."""

import ibis
import ibis.expr.types as ir
import pandas as pd

from nfl.fourth_down_filter import filter_fourth_downs, classify_decision
from nfl.features import build_fourth_down_features, MODEL_FEATURE_COLUMNS
from nfl.target import add_target_label, TARGET_COLUMN


def build_training_dataset(pbp_table: ir.Table) -> ir.Table:
    """Build the full training dataset from raw PBP data.

    Pipeline: filter 4th downs → classify decision → engineer features
    → add target label → drop rows with null targets.

    This is an Ibis expression — backend-agnostic until executed.
    """
    table = filter_fourth_downs(pbp_table)
    table = classify_decision(table)
    table = build_fourth_down_features(table)
    table = add_target_label(table)
    # Drop rows where target is null (unmapped decisions)
    table = table.filter(table[TARGET_COLUMN].notnull())
    return table


def split_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a materialized dataset into features (X) and target (y)."""
    X = dataset[MODEL_FEATURE_COLUMNS].copy()
    y = dataset[TARGET_COLUMN].copy()
    return X, y


def train_test_split_by_season(
    dataset: pd.DataFrame,
    test_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: train on earlier seasons, test on specified seasons.

    Prevents data leakage from future seasons.
    """
    test_mask = dataset["season"].isin(test_seasons)
    train = dataset[~test_mask].copy()
    test = dataset[test_mask].copy()
    return train, test
