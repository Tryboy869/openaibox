"""
analyzer.py — Open AI Box Public API

Single entry point for all analysis operations.

    from openaibox import OpenAIBox

    oaib = OpenAIBox("HuggingFaceTB/SmolLM-360M")
    gr.discover()
    gr.map_dimensions()
    gr.export("graph.json")
    gr.print_summary()
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from openaibox.discovery import Discoverer, GraphResult
from openaibox.mapping   import DimensionMapper, MappingResult, DEFAULT_GROUPS
from openaibox.reporter  import export_graph_json, print_summary


class OpenAIBox:
    """
    Universal LLM introspection.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path.
        Example: "HuggingFaceTB/SmolLM-360M"

    model : nn.Module (optional)
        If you already have the model loaded, pass it directly.

    tokenizer : optional
        If you already have the tokenizer, pass it directly.

    dtype : torch.dtype
        Default float32 for CPU analysis.
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        model:      nn.Module | None = None,
        tokenizer               = None,
        dtype: torch.dtype      = torch.float32,
    ):
        self.model_name = model_name_or_path

        if model is not None and tokenizer is not None:
            # Pre-loaded model
            self._model     = model
            self._tokenizer = tokenizer
        else:
            self._model, self._tokenizer = self._load(model_name_or_path, dtype)

        self._model.eval()

        self._graph_result:   GraphResult   | None = None
        self._mapping_result: MappingResult | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def discover(self, prompt: str = "Hello world") -> "OpenAIBox":
        """
        Run an inference pass and build the full architecture graph.

        Parameters
        ----------
        prompt : str
            Short prompt used for the discovery pass.
            Complexity doesn't matter — architecture is invariant.

        Returns self for chaining.
        """
        disc = Discoverer(
            model      = self._model,
            tokenizer  = self._tokenizer,
            model_name = self.model_name,
        )
        self._graph_result = disc.run(prompt)
        return self

    def map_dimensions(self, groups=None) -> "OpenAIBox":
        """
        Map which dimensions carry which type of information.

        Requires discover() to have been called first.

        Parameters
        ----------
        groups : list[ContrastGroup] | None
            Custom contrast groups. Defaults to built-in 6 groups.

        Returns self for chaining.
        """
        if self._graph_result is None:
            self.discover()

        decision = self._get_decision_point()
        if decision is None:
            raise RuntimeError(
                "No decision_point detected in the model. "
                "The model architecture may not be supported yet. "
                "Please open an issue at github.com/tryboy869/openaibox"
            )

        mapper = DimensionMapper(
            model                = self._model,
            tokenizer            = self._tokenizer,
            decision_layer_name  = decision.layer_name,
            dim                  = self._graph_result.hidden_dim,
            groups               = groups,
        )
        self._mapping_result = mapper.run()
        return self

    def export(self, output_path: str = "graph.json") -> dict:
        """
        Export analysis to graph.json.

        Returns the dict.
        """
        if self._graph_result is None:
            raise RuntimeError("Call discover() before export().")

        return export_graph_json(
            graph_result   = self._graph_result,
            mapping_result = self._mapping_result,
            output_path    = output_path,
            model_name     = self.model_name,
        )

    def print_summary(self) -> "OpenAIBox":
        """Print human-readable summary to stdout."""
        if self._graph_result is None:
            raise RuntimeError("Call discover() before print_summary().")
        print_summary(self._graph_result, self._mapping_result)
        return self

    def injection_points(self) -> list:
        """Return the detected injection points as a list of dicts."""
        if self._graph_result is None:
            raise RuntimeError("Call discover() first.")
        return [
            {
                "name":        p.name,
                "role":        p.role,
                "layer":       p.layer_name,
                "in_shape":    p.in_shape,
                "out_shape":   p.out_shape,
                "description": p.description,
            }
            for p in self._graph_result.injection_points
        ]

    def explain_dimension(self, dim_index: int) -> dict:
        """
        Return everything known about a specific dimension.

        Parameters
        ----------
        dim_index : int
            0-based index of the dimension.
        """
        if self._mapping_result is None:
            raise RuntimeError("Call map_dimensions() before explain_dimension().")

        m = self._mapping_result

        # Find roles from groups
        roles = []
        scores = {}
        for gname, data in m.groups.items():
            if dim_index in data["top_dims"]:
                rank = data["top_dims"].index(dim_index) + 1
                roles.append(gname)
                scores[gname] = rank

        group_details = {}
        for gname, data in m.groups.items():
            group_details[gname] = {
                "question":     data["question"],
                "separability": data["separability"],
            }

        return {
            "dimension":     dim_index,
            "roles":         roles,
            "rank_in_groups": scores,
            "global_score":  float(
                next((d.score for d in m.top_dimensions if d.index == dim_index), 0.0)
            ),
            "description":   next(
                (d.description for d in m.top_dimensions if d.index == dim_index),
                "No specific role detected for this dimension."
            ),
            "group_context": group_details,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_decision_point(self):
        if self._graph_result is None:
            return None
        for p in self._graph_result.injection_points:
            if p.role == "decision":
                return p
        return None

    @staticmethod
    def _load(model_name_or_path: str, dtype: torch.dtype):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers torch"
            )

        print(f"⬇  Loading {model_name_or_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model     = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype
        )
        print(f"✅ Loaded — {sum(p.numel() for p in model.parameters()):,} params")
        return model, tokenizer
