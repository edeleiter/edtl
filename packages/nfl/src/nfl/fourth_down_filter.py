"""Filter play-by-play data to fourth-down decisions."""

import ibis
import ibis.expr.types as ir

from schemas._base import Decision

DECISION_GO = Decision.GO_FOR_IT.value
DECISION_PUNT = Decision.PUNT.value
DECISION_FG = Decision.FIELD_GOAL.value

_GO_PLAY_TYPES = ("pass", "run", "qb_kneel", "qb_spike")


def filter_fourth_downs(table: ir.Table) -> ir.Table:
    """Filter to 4th-down plays with valid play types."""
    valid_types = (*_GO_PLAY_TYPES, "punt", "field_goal")
    return table.filter(
        (table["down"] == 4) & table["play_type"].isin(valid_types)
    )


def classify_decision(table: ir.Table) -> ir.Table:
    """Add a 'decision' column classifying the play type.

    Maps play_type to one of: go_for_it, punt, field_goal.
    """
    decision = ibis.cases(
        (table["play_type"].isin(_GO_PLAY_TYPES), DECISION_GO),
        (table["play_type"] == "punt", DECISION_PUNT),
        (table["play_type"] == "field_goal", DECISION_FG),
        else_=ibis.null(),
    )
    return table.mutate(decision=decision)
