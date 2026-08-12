#!/usr/bin/env python3
"""Record the predeclared decision about adding a GNN to the paper."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "ml_results"
    / "baseline_comparison"
    / "gnn_gate_decision.json"
)


def main() -> None:
    decision = {
        "status": "deferred",
        "decision": "do_not_add_to_primary_paper",
        "gate": {
            "required": [
                "AST plus call/control/data-flow graph representation",
                "duplicate-safe repeated LODO evaluation",
                "improvement on at least 3 of 4 held-out datasets",
                "improvement in mean PR-AUC or ROC-AUC across repeated seeds",
                "no dependence on unavailable statement-level labels",
            ],
            "failure_action": "omit GNN from primary claims",
        },
        "evidence": {
            "node_or_statement_labels_available": False,
            "retained_graph_representation_available": False,
            "current_source_only_semantic_features_validated": True,
            "current_primary_bottleneck": "dataset shift and label policy",
        },
        "rationale": (
            "Adding a shallow graph model without a defensible graph construction "
            "or node-level supervision would add complexity without addressing "
            "the reviewer-identified leakage and dataset-shift problems."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(decision, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
