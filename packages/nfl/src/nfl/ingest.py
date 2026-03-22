"""NFL play-by-play data ingestion via nfl-data-py."""

from pathlib import Path

import pandas as pd

PBP_MINIMUM_COLUMNS = [
    "game_id", "play_type", "down", "ydstogo", "yardline_100",
    "epa", "posteam", "defteam", "score_differential",
    "half_seconds_remaining", "game_seconds_remaining",
    "quarter_seconds_remaining", "qtr", "goal_to_go", "wp",
    "season", "week",
]


def load_pbp_data(
    years: list[int],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Download NFL play-by-play data for given years.

    Uses nfl-data-py (nflverse). First call downloads from GitHub;
    subsequent calls may be cached by the library.
    """
    import nfl_data_py as nfl

    cols = columns or PBP_MINIMUM_COLUMNS
    df = nfl.import_pbp_data(years, columns=cols)
    return df


def pbp_to_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Cache a PBP DataFrame to a local Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_pbp_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load PBP data from a cached Parquet file."""
    return pd.read_parquet(path)
