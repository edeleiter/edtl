"""Feature analysis result models."""

from enum import Enum

from pydantic import Field

from schemas._base import StrictModel, Probability


class ImportanceMethod(str, Enum):
    GINI = "gini"
    PERMUTATION = "permutation"
    SHAP = "shap"


class ImportanceEntry(StrictModel):
    """A single feature's importance score."""

    feature: str
    importance: float
    std: float | None = None


class ImportanceResult(StrictModel):
    """Feature importance scores from a single method."""

    method: ImportanceMethod
    entries: list[ImportanceEntry]


class CorrelationPair(StrictModel):
    """A pair of correlated features."""

    feature_a: str
    feature_b: str
    correlation: float = Field(ge=-1.0, le=1.0)


class VIFEntry(StrictModel):
    """Variance Inflation Factor for a single feature."""

    feature: str
    vif: float = Field(ge=1.0, description="VIF >= 1.0 (1.0 = no collinearity)")


class DistributionStats(StrictModel):
    """Statistical profile of a single feature column."""

    column: str
    count: int
    null_count: int
    null_pct: float = Field(ge=0.0, le=100.0)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    unique_count: int | None = None
    histogram_counts: list[int] | None = None
    histogram_bin_edges: list[float] | None = None


class SubsetEvaluation(StrictModel):
    """Evaluation results for a feature subset."""

    name: str
    features: list[str]
    n_features: int
    accuracy: Probability
    cv_mean: float
    cv_std: float = Field(ge=0.0)
    cv_scores: list[float]


class FeatureMetadata(StrictModel):
    """Metadata about a single feature in the model."""

    name: str
    dtype: str
    description: str = ""
    source: str = ""
    is_engineered: bool = False
    transform_name: str | None = None
