"""
reporter.py — Export to graph.json

Serializes GraphResult + MappingResult into a
human-readable, machine-parseable graph.json file.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


def export_graph_json(
    graph_result,
    mapping_result=None,
    output_path: str = "graph.json",
    model_name: str  = "",
) -> dict:
    """
    Export the full analysis to graph.json.

    Returns the dict (also writes to disk).
    """
    doc = {
        "openaibox": {
            "version":    "1.0.0b1",
            "generated":  datetime.utcnow().isoformat() + "Z",
            "model":      model_name or graph_result.model_name,
            "package_url": "https://github.com/tryboy869/openaibox",
        },

        # ── Architecture overview ─────────────────────────────────────────
        "architecture": {
            "class":        graph_result.architecture,
            "total_params": graph_result.total_params,
            "num_layers":   graph_result.num_layers,
            "hidden_dim":   graph_result.hidden_dim,
            "vocab_size":   graph_result.vocab_size,
        },

        # ── Injection points ──────────────────────────────────────────────
        "injection_points": [
            {
                "name":        p.name,
                "role":        p.role,
                "layer":       p.layer_name,
                "in_shape":    p.in_shape,
                "out_shape":   p.out_shape,
                "description": p.description,
            }
            for p in graph_result.injection_points
        ],

        # ── Layer flow ────────────────────────────────────────────────────
        "layer_flow": [
            {
                "order":    l.order,
                "name":     l.name,
                "type":     l.layer_type,
                "in":       l.in_shapes,
                "out":      l.out_shapes,
                "params":   l.params,
            }
            for l in graph_result.layers
        ],
    }

    # ── Dimension map (optional) ──────────────────────────────────────────
    if mapping_result is not None:
        doc["dimension_map"] = {
            "hidden_dim":  mapping_result.dim,

            "contrast_groups": {
                name: {
                    "question":      data["question"],
                    "cos_similarity": round(data["cos_sim"], 4),
                    "separability":  data["separability"],
                    "mean_diff":     round(data["mean_diff"], 4),
                    "top_dimensions": data["top_dims"][:10],
                }
                for name, data in mapping_result.groups.items()
            },

            "top_dimensions": [
                {
                    "index":       d.index,
                    "roles":       d.roles,
                    "score":       round(d.score, 4),
                    "description": d.description,
                }
                for d in mapping_result.top_dimensions[:20]
            ],

            "multi_role_dimensions": [
                {
                    "index": d.index,
                    "roles": d.roles,
                    "score": round(d.score, 4),
                }
                for d in mapping_result.multi_role_dims[:15]
            ],

            "specialist_dimensions": mapping_result.specialist_dims,
        }

    Path(output_path).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return doc


def print_summary(graph_result, mapping_result=None):
    """Pretty-print a human-readable summary to stdout."""
    g = graph_result
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  GRAPHRUNTIME — Analysis Report")
    print(f"  Model : {g.model_name}")
    print(sep)
    print(f"\n  Architecture : {g.architecture}")
    print(f"  Parameters   : {g.total_params:,}")
    print(f"  Layers       : {g.num_layers}")
    print(f"  Hidden dim   : {g.hidden_dim}")
    print(f"  Vocabulary   : {g.vocab_size:,}")

    print(f"\n  {'─'*50}")
    print(f"  INJECTION POINTS")
    print(f"  {'─'*50}")
    for p in g.injection_points:
        print(f"\n  [{p.role.upper()}]  {p.layer_name}")
        print(f"  Shape  : {p.in_shape} → {p.out_shape}")
        print(f"  Detail : {p.description}")

    if mapping_result:
        print(f"\n  {'─'*50}")
        print(f"  DIMENSION MAP — Top Roles")
        print(f"  {'─'*50}")
        for d in mapping_result.multi_role_dims[:10]:
            roles_str = ", ".join(d.roles)
            print(f"  dim_{d.index:<4}  {len(d.roles)} roles  →  {roles_str}")

        print(f"\n  SEPARABILITY PER CONTRAST")
        for name, data in sorted(
            mapping_result.groups.items(),
            key=lambda x: x[1]["cos_sim"]
        ):
            bar = "█" * int((1 - data["cos_sim"]) * 20)
            print(f"  {name:<24} cos={data['cos_sim']:.4f}  [{data['separability']}] {bar}")

    print(f"\n{sep}\n")
