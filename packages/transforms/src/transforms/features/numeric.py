"""Numeric feature transforms — all defined as Ibis expressions."""

import ibis
import ibis.expr.types as ir


def zscore_normalize(table: ir.Table, column: str) -> ir.Table:
    """Add a z-score normalized column: (x - mean) / stddev."""
    col = table[column]
    mean = col.mean()
    std = col.std()
    return table.mutate(**{f"{column}_zscore": (col - mean) / std})


def log_transform(table: ir.Table, column: str) -> ir.Table:
    """Add a natural log transformed column. Assumes positive values."""
    col = table[column]
    return table.mutate(**{f"{column}_log": col.log()})


def clip_outliers(
    table: ir.Table, column: str, lower: float, upper: float
) -> ir.Table:
    """Add a clipped column bounded by [lower, upper]."""
    col = table[column]
    clipped = ibis.greatest(ibis.least(col, upper), lower)
    return table.mutate(**{f"{column}_clipped": clipped})


def ratio(
    table: ir.Table, numerator_col: str, denominator_col: str
) -> ir.Table:
    """Add a ratio column. Returns null when denominator is zero."""
    num = table[numerator_col]
    den = table[denominator_col]
    safe_ratio = ibis.cases((den != 0, num / den), else_=ibis.null())
    return table.mutate(
        **{f"{numerator_col}_{denominator_col}_ratio": safe_ratio}
    )
