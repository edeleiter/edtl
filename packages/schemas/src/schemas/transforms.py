"""Transform pipeline configuration models."""

import datetime
import json
import warnings
from pathlib import Path
from typing import Any

from pydantic import Field

from schemas._base import MutableModel, StrictModel


class TransformStep(StrictModel):
    """A single transform step in a pipeline."""

    name: str = Field(description="Registered transform function name")
    params: dict[str, Any] = Field(default_factory=dict, description="Transform parameters")


class PipelineDefinition(StrictModel):
    """An ordered sequence of transforms that defines a feature pipeline."""

    name: str
    steps: list[TransformStep]
    version: str = Field(default="0.1.0")
    description: str = ""


class TransformRequest(MutableModel):
    """API request to apply transforms to input data."""

    data: list[dict[str, Any]]
    transforms: list[str]
    transform_params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TransformResponse(StrictModel):
    """API response with transformed data."""

    data: list[dict[str, Any]]
    transforms_applied: list[str]


class ReferenceDataManifest(StrictModel):
    """Manifest of available reference data Parquet files.

    Tables are snapshotted from Snowflake at training time and loaded
    into DuckDB at inference API startup.
    """

    tables: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of table_name -> file path",
    )
    snapshot_timestamp: datetime.datetime | None = Field(default=None)
    source_query_hash: str | None = Field(default=None)

    def staleness_days(self) -> float | None:
        """Return how many days old the snapshot is, or None if unknown."""
        if self.snapshot_timestamp is None:
            return None
        delta = datetime.datetime.now(datetime.timezone.utc) - self.snapshot_timestamp
        return delta.total_seconds() / 86400

    def warn_if_stale(self, max_days: int = 30) -> None:
        """Issue a warning if the snapshot is older than max_days."""
        days = self.staleness_days()
        if days is not None and days > max_days:
            warnings.warn(
                f"Reference data snapshot is {days:.1f} days old "
                f"(threshold: {max_days} days). Consider re-snapshotting.",
                stacklevel=2,
            )

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ReferenceDataManifest":
        """Build manifest by scanning a directory for .parquet files."""
        directory = Path(directory)
        tables = {}
        for path in sorted(directory.glob("*.parquet")):
            tables[path.stem] = str(path)

        snapshot_timestamp = None
        source_query_hash = None
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            meta = json.loads(manifest_path.read_text())
            if "snapshot_timestamp" in meta:
                snapshot_timestamp = datetime.datetime.fromisoformat(
                    meta["snapshot_timestamp"]
                )
            source_query_hash = meta.get("source_query_hash")

        return cls(
            tables=tables,
            snapshot_timestamp=snapshot_timestamp,
            source_query_hash=source_query_hash,
        )
