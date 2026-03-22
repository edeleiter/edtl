"""Categorical feature transforms as Ibis expressions."""

import ibis
import ibis.expr.types as ir


def one_hot_flag(table: ir.Table, column: str, value: str) -> ir.Table:
    """Add a binary flag column: 1 if column == value, else 0."""
    flag = (table[column] == value).ifelse(1, 0)
    return table.mutate(**{f"{column}_is_{value}": flag})


def label_encode_from_map(
    table: ir.Table, column: str, mapping: dict[str, int]
) -> ir.Table:
    """Encode a categorical column using an explicit mapping dict."""
    col = table[column]
    branches = [(col == label, code) for label, code in mapping.items()]
    encoded = ibis.cases(*branches, else_=ibis.null())
    return table.mutate(**{f"{column}_encoded": encoded})
