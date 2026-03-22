"""Compose multiple transforms into a pipeline."""

from typing import Callable

import ibis.expr.types as ir


def apply_pipeline(
    table: ir.Table,
    transforms: list[Callable[..., ir.Table]],
) -> ir.Table:
    """Apply a sequence of transforms to a table expression."""
    result = table
    for transform in transforms:
        result = transform(result)
    return result
