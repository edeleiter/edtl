"""Temporal feature transforms as Ibis expressions."""

import datetime

import ibis
import ibis.expr.types as ir


def extract_dow(table: ir.Table, column: str) -> ir.Table:
    """Extract day of week (Monday=0, Sunday=6)."""
    dow = table[column].day_of_week.index()
    return table.mutate(**{f"{column}_dow": dow})


def extract_hour(table: ir.Table, column: str) -> ir.Table:
    """Extract hour of day (0-23)."""
    hour = table[column].hour()
    return table.mutate(**{f"{column}_hour": hour})


def days_since(
    table: ir.Table, column: str, reference_date: datetime.date
) -> ir.Table:
    """Compute days elapsed since a reference date."""
    delta = table[column].date() - reference_date
    return table.mutate(
        **{f"{column}_days_since": delta.cast("int64")}
    )
