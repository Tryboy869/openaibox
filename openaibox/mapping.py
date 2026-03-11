"""
mapping.py — Dimension Role Mapping

Analyzes which dimensions of the decision vector carry
specific semantic, syntactic, causal, or temporal information.

Works on any model whose decision_point has been identified.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DimensionRole:
    index:       int
    roles:       list[str]       # categories where this dim is discriminant
    score:       float           # global importance score (0–1)
    description: str


@dataclass
class ContrastGroup:
    name:        str
    prompts_a:   list[str]
    prompts_b:   list[str]
    label_a:     str
    label_b:     str
    question:    str


@dataclass
class MappingResult:
    dim:              int
    groups:           dict               # group_name → {cos_sim, top_dims, separability}
    top_dimensions:   list[DimensionRole]
    multi_role_dims:  list[DimensionRole]
    specialist_dims:  dict               # role → [dim indices]


# ── Built-in contrast groups ────────────────────────────────────────────────

DEFAULT_GROUPS = [
    ContrastGroup(
        name    = "syntax_semantics",
        label_a = "grammatical",
        label_b = "absurd",
        question = "Which dimensions detect semantic absurdity?",
        prompts_a = [
            "The cat sat on the mat",
            "The dog ran in the park",
            "The bird flew over the lake",
        ],
        prompts_b = [
            "The mat sat on the cat",
            "The park ran in the dog",
            "The lake flew over the bird",
        ],
    ),
    ContrastGroup(
        name    = "causality",
        label_a = "causal",
        label_b = "correlational",
        question = "Which dimensions carry causality?",
        prompts_a = [
            "Because it rained, the ground is wet",
            "Since she studied, she passed the exam",
        ],
        prompts_b = [
            "The ground is wet and it rained",
            "She studied and she passed the exam",
        ],
    ),
    ContrastGroup(
        name    = "certainty",
        label_a = "certain",
        label_b = "uncertain",
        question = "Which dimensions carry certainty?",
        prompts_a = [
            "The sun rises every morning",
            "Water freezes at 0 degrees",
            "2 plus 2 equals 4",
        ],
        prompts_b = [
            "Maybe the sun will rise tomorrow",
            "Water might freeze around 0 degrees",
            "2 plus 2 could equal 4",
        ],
    ),
    ContrastGroup(
        name    = "concrete_abstract",
        label_a = "concrete",
        label_b = "abstract",
        question = "Which dimensions carry abstraction?",
        prompts_a = [
            "The red apple is on the wooden table",
            "The cold water fills the glass",
        ],
        prompts_b = [
            "Justice is the foundation of society",
            "Freedom defines the human spirit",
        ],
    ),
    ContrastGroup(
        name    = "temporality",
        label_a = "past",
        label_b = "future",
        question = "Which dimensions carry temporality?",
        prompts_a = [
            "Yesterday she walked to the market",
            "Last year the company grew rapidly",
        ],
        prompts_b = [
            "Tomorrow she will walk to the market",
            "Next year the company will grow rapidly",
        ],
    ),
    ContrastGroup(
        name    = "emotion",
        label_a = "emotional",
        label_b = "neutral",
        question = "Which dimensions carry emotion?",
        prompts_a = [
            "I love the beautiful sunset",
            "The child cried with happiness",
        ],
        prompts_b = [
            "The sun sets at a specific angle",
            "The child produced vocal sounds",
        ],
    ),
]


class DimensionMapper:
    """
    Maps dimension roles by contrasting hidden state vectors.

    Parameters
    ----------
    model       : nn.Module
    tokenizer   : tokenizer with __call__
    hook_target : callable that receives (model, name) and registers a hook
                  returning the hidden state tensor for the decision point.
                  If None, auto-detected from GraphResult.
    dim         : hidden dimension size
    groups      : list of ContrastGroup (defaults to DEFAULT_GROUPS)
    """

    def __init__(
        self,
        model,
        tokenizer,
        decision_layer_name: str,
        dim: int,
        groups: list[ContrastGroup] | None = None,
    ):
        self.model               = model
        self.tokenizer           = tokenizer
        self.decision_layer_name = decision_layer_name
        self.dim                 = dim
        self.groups              = groups or DEFAULT_GROUPS

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, top_n: int = 20) -> MappingResult:
        from collections import defaultdict

        group_data: dict = {}

        for group in self.groups:
            vecs_a = [self._get_vector(p) for p in group.prompts_a]
            vecs_b = [self._get_vector(p) for p in group.prompts_b]

            mean_a = torch.stack(vecs_a).mean(0)
            mean_b = torch.stack(vecs_b).mean(0)
            diff   = (mean_a - mean_b).abs()
            diff_n = diff / (diff.max() + 1e-8)

            top_dims = diff_n.argsort(descending=True)[:top_n].tolist()
            cos_sim  = float(F.cosine_similarity(
                mean_a.unsqueeze(0), mean_b.unsqueeze(0)
            ))

            group_data[group.name] = {
                "cos_sim":    cos_sim,
                "top_dims":   top_dims,
                "diff_norm":  diff_n,
                "mean_diff":  float(diff.mean()),
                "max_diff":   float(diff.max()),
                "active_a":   float((mean_a.abs() > 0.1).float().mean()),
                "active_b":   float((mean_b.abs() > 0.1).float().mean()),
                "separability": self._label_separability(cos_sim),
                "question":   group.question,
            }

        # Aggregate dimension roles
        dim_roles: defaultdict = defaultdict(list)
        for gname, data in group_data.items():
            for d in data["top_dims"]:
                dim_roles[d].append(gname)

        # Global importance score
        global_scores = torch.zeros(self.dim)
        for data in group_data.values():
            global_scores += data["diff_norm"]
        global_scores /= (global_scores.max() + 1e-8)

        # Build DimensionRole objects
        all_dims = [
            DimensionRole(
                index       = idx,
                roles       = roles,
                score       = float(global_scores[idx]),
                description = self._describe(roles),
            )
            for idx, roles in dim_roles.items()
        ]
        all_dims.sort(key=lambda d: len(d.roles), reverse=True)

        top_dimensions  = sorted(all_dims, key=lambda d: d.score, reverse=True)[:top_n]
        multi_role_dims = [d for d in all_dims if len(d.roles) >= 2]

        specialist_dims: dict = {}
        for d in all_dims:
            if len(d.roles) == 1:
                role = d.roles[0]
                specialist_dims.setdefault(role, []).append(d.index)

        return MappingResult(
            dim             = self.dim,
            groups          = group_data,
            top_dimensions  = top_dimensions,
            multi_role_dims = multi_role_dims,
            specialist_dims = specialist_dims,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_vector(self, prompt: str) -> torch.Tensor:
        captured = {}

        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                captured["h"] = out[:, -1, :].detach().squeeze()

        # Navigate to target layer
        target = self._get_module(self.decision_layer_name)
        handle = target.register_forward_hook(hook)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            self.model(**inputs)
        handle.remove()

        if "h" not in captured:
            raise RuntimeError(
                f"Hook on '{self.decision_layer_name}' did not capture output. "
                "The layer may return a tuple — check your model architecture."
            )
        return captured["h"]

    def _get_module(self, name: str):
        parts = name.split(".")
        module = self.model
        for p in parts:
            module = getattr(module, p)
        return module

    @staticmethod
    def _label_separability(cos_sim: float) -> str:
        if cos_sim < 0.90:
            return "VERY_SEPARABLE"
        elif cos_sim < 0.96:
            return "SEPARABLE"
        else:
            return "LOW_SEPARABILITY"

    @staticmethod
    def _describe(roles: list[str]) -> str:
        if not roles:
            return "Unknown role"
        if len(roles) == 1:
            desc_map = {
                "syntax_semantics":  "Semantic coherence detector",
                "causality":         "Causal reasoning carrier",
                "certainty":         "Certainty/confidence signal",
                "concrete_abstract": "Concreteness level",
                "temporality":       "Temporal orientation",
                "emotion":           "Emotional tone",
            }
            return desc_map.get(roles[0], f"Specialist: {roles[0]}")
        return f"Multi-role regulator ({len(roles)} categories)"
