"""Target variable engineering for the 4th-down model.

The model predicts what coaches DO (actual historical decisions),
not what they SHOULD do. This is intentional for the prediction
use case. The EPA comparison in evaluation measures whether the
model's recommended decision has higher expected EPA.
"""

import ibis
import ibis.expr.types as ir

TARGET_COLUMN = "decision_label"

LABEL_MAP = {
    "go_for_it": 0,
    "punt": 1,
    "field_goal": 2,
}

INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def add_target_label(table: ir.Table) -> ir.Table:
    """Map the decision column to integer labels for classification.

    Rows with unmapped decision values get NULL target. These should be
    filtered out before training — log a warning with the count of
    excluded rows rather than silently dropping them.
    """
    label_expr = ibis.cases(
        *[(table["decision"] == label, code) for label, code in LABEL_MAP.items()],
        else_=ibis.null(),
    )
    return table.mutate(**{TARGET_COLUMN: label_expr})
