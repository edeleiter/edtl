"""NFL-specific feature engineering — all as Ibis expressions.

These transforms are backend-agnostic: they compile to SQL on Snowflake
and execute natively on DuckDB. The validation harness proves parity.
"""

import ibis
import ibis.expr.types as ir

# Note: quarter_seconds_remaining excluded — redundant with game_seconds_remaining and half_seconds_remaining
# The 17 features the model expects, in exact order.
# First 8 are raw game-state fields, last 9 are engineered.
MODEL_FEATURE_COLUMNS = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "qtr",
    "goal_to_go",
    "wp",
    # Engineered features
    "is_opponent_territory",
    "is_fg_range",
    "is_short_yardage",
    "log_ydstogo",
    "is_trailing",
    "is_two_score_game",
    "abs_score_diff",
    "is_second_half",
    "is_late_and_trailing",
]


def add_field_position_features(table: ir.Table) -> ir.Table:
    """Add field-position-derived features."""
    return table.mutate(
        is_opponent_territory=(table["yardline_100"] <= 50).cast("int64"),
        is_fg_range=(table["yardline_100"] <= 40).cast("int64"),
        is_short_yardage=(table["ydstogo"] <= 2).cast("int64"),
        log_ydstogo=ibis.greatest(table["ydstogo"], 1).log(),
    )


def add_game_state_features(table: ir.Table) -> ir.Table:
    """Add game-state-derived features."""
    return table.mutate(
        is_trailing=(table["score_differential"] < 0).cast("int64"),
        is_two_score_game=(table["score_differential"].abs() >= 9).cast("int64"),
        abs_score_diff=table["score_differential"].abs(),
    )


def add_time_features(table: ir.Table) -> ir.Table:
    """Add time-derived features."""
    is_second_half = (table["qtr"] >= 3).cast("int64")
    is_late_and_trailing = (
        (table["qtr"] == 4)
        & (table["game_seconds_remaining"] <= 300)
        & (table["score_differential"] < 0)
    ).cast("int64")
    return table.mutate(
        is_second_half=is_second_half,
        is_late_and_trailing=is_late_and_trailing,
    )


def build_fourth_down_features(table: ir.Table) -> ir.Table:
    """Apply all NFL feature transforms in sequence.

    Input table must have the 8 raw game-state columns.
    Output table has all 17 MODEL_FEATURE_COLUMNS.
    """
    table = add_field_position_features(table)
    table = add_game_state_features(table)
    table = add_time_features(table)
    return table
