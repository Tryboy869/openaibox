"""
discovery.py — Universal Layer Discovery

Auto-discovers the architecture of any HuggingFace model
by registering forward hooks and tracing live inference.

Supports: LlamaForCausalLM, Qwen2ForCausalLM, GPT2LMHeadModel,
          MistralForCausalLM, FalconForCausalLM, MPTForCausalLM,
          BloomForCausalLM, OPTForCausalLM, and any custom model.
"""

from __future__ import annotations

import re
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LayerInfo:
    name:       str
    layer_type: str
    in_shapes:  list
    out_shapes:  list
    order:      int
    params:     int = 0


@dataclass
class InjectionPoint:
    """A point in the model where behavior can be observed or analyzed."""
    name:         str
    role:         str          # "input" | "decision" | "memory" | "output"
    layer_name:   str
    in_shape:     list
    out_shape:    list
    description:  str


@dataclass
class GraphResult:
    """Full architecture graph from a single inference pass."""
    model_name:      str
    architecture:    str
    total_params:    int
    num_layers:      int
    hidden_dim:      int
    vocab_size:      int
    layers:          list[LayerInfo]  = field(default_factory=list)
    injection_points: list[InjectionPoint] = field(default_factory=list)
    flow_summary:    list[str]        = field(default_factory=list)


# ── Heuristics to find the decision point across architectures ──────────────

_DECISION_NORM_PATTERNS = [
    r"^model\.norm$",           # Llama, Qwen, Mistral, Falcon
    r"^transformer\.ln_f$",     # GPT-2, GPT-NeoX, CodeGen
    r"^model\.final_layer_norm$",  # BLOOM, OPT
    r"^transformer\.norm_f$",   # MPT, Mosaic
    r"^gpt_neox\.final_layer_norm$",
    r"^roberta\.encoder\.layer_norm$",
]

_LM_HEAD_PATTERNS = [
    r"^lm_head$",
    r"^embed_out$",       # Pythia / GPT-NeoX
    r"^output$",          # some RWKV variants
]

_EMBED_PATTERNS = [
    r"^model\.embed_tokens$",
    r"^transformer\.wte$",
    r"^model\.embed$",
    r"^gpt_neox\.embed_in$",
]


def _shape_list(tensor_or_tuple) -> list:
    """Convert any tensor / tuple of tensors to a list of shape lists."""
    if isinstance(tensor_or_tuple, torch.Tensor):
        return [list(tensor_or_tuple.shape)]
    if isinstance(tensor_or_tuple, (tuple, list)):
        shapes = []
        for t in tensor_or_tuple:
            if isinstance(t, torch.Tensor):
                shapes.append(list(t.shape))
        return shapes
    return []


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _matches(name: str, patterns: list[str]) -> bool:
    return any(re.match(p, name) for p in patterns)


class Discoverer:
    """
    Traces a live inference pass and builds a GraphResult.

    Usage
    -----
    disc   = Discoverer(model, model_name="SmolLM-360M")
    result = disc.run("Hello world")
    """

    def __init__(self, model: nn.Module, tokenizer=None, model_name: str = "unknown"):
        self.model      = model
        self.tokenizer  = tokenizer
        self.model_name = model_name

        self._hooks:   list = []
        self._records: list = []          # (order, name, type, in_shapes, out_shapes)
        self._counter: int  = 0

    # ── public ────────────────────────────────────────────────────────────────

    def run(self, prompt: str = "Hello world") -> GraphResult:
        """Run one inference pass and return the full GraphResult."""
        self._attach_hooks()

        try:
            self._infer(prompt)
        finally:
            self._detach_hooks()

        return self._build_result()

    # ── hooks ─────────────────────────────────────────────────────────────────

    def _attach_hooks(self):
        self._records.clear()
        self._counter = 0

        for name, module in self.model.named_modules():
            if name == "":
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

    def _detach_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._records.append((
                self._counter,
                name,
                type(module).__name__,
                _shape_list(inp),
                _shape_list(out),
            ))
            self._counter += 1
        return hook

    # ── inference ─────────────────────────────────────────────────────────────

    def _infer(self, prompt: str):
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for discovery inference.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            self.model(**inputs)

    # ── result builder ────────────────────────────────────────────────────────

    def _build_result(self) -> GraphResult:
        # Basic model metadata
        arch        = type(self.model).__name__
        total_params = sum(p.numel() for p in self.model.parameters())

        # Detect dims from embed layer
        hidden_dim = 0
        vocab_size = 0
        for order, name, ltype, in_sh, out_sh in self._records:
            if _matches(name, _EMBED_PATTERNS) and out_sh:
                hidden_dim = out_sh[0][-1] if len(out_sh[0]) >= 3 else 0
            if _matches(name, _LM_HEAD_PATTERNS) and out_sh:
                vocab_size = out_sh[0][-1] if out_sh[0] else 0
            if hidden_dim and vocab_size:
                break

        # Count transformer layers by counting unique layer indices
        layer_indices = set()
        for _, name, _, _, _ in self._records:
            m = re.search(r"\.layers?\.(\d+)\.", name)
            if m:
                layer_indices.add(int(m.group(1)))
        num_layers = len(layer_indices) if layer_indices else 0

        # Build LayerInfo list (deduplicated)
        seen_names: set = set()
        layers: list[LayerInfo] = []
        for order, name, ltype, in_sh, out_sh in self._records:
            if name in seen_names:
                continue
            seen_names.add(name)
            try:
                module = dict(self.model.named_modules())[name]
                nparams = _count_params(module)
            except KeyError:
                nparams = 0

            layers.append(LayerInfo(
                name       = name,
                layer_type = ltype,
                in_shapes  = in_sh,
                out_shapes  = out_sh,
                order      = order,
                params     = nparams,
            ))

        # Detect injection points
        injection_points = self._detect_injection_points(layers)

        # Build flow summary
        flow_summary = [
            f"{l.name} [{l.layer_type}] {l.in_shapes} → {l.out_shapes}"
            for l in layers
            if l.out_shapes
        ]

        return GraphResult(
            model_name       = self.model_name,
            architecture     = arch,
            total_params     = total_params,
            num_layers       = num_layers,
            hidden_dim       = hidden_dim,
            vocab_size       = vocab_size,
            layers           = layers,
            injection_points = injection_points,
            flow_summary     = flow_summary,
        )

    def _detect_injection_points(self, layers: list[LayerInfo]) -> list[InjectionPoint]:
        points: list[InjectionPoint] = []

        for l in layers:
            # Input embedding point
            if _matches(l.name, _EMBED_PATTERNS) and l.out_shapes:
                points.append(InjectionPoint(
                    name       = "input_point",
                    role       = "input",
                    layer_name = l.name,
                    in_shape   = l.in_shapes[0] if l.in_shapes else [],
                    out_shape  = l.out_shapes[0] if l.out_shapes else [],
                    description = (
                        "Token embeddings — where the prompt enters the model. "
                        "Each input token is converted to a vector here."
                    ),
                ))

            # Decision point (norm before lm_head)
            if _matches(l.name, _DECISION_NORM_PATTERNS) and l.out_shapes:
                points.append(InjectionPoint(
                    name       = "decision_point",
                    role       = "decision",
                    layer_name = l.name,
                    in_shape   = l.in_shapes[0] if l.in_shapes else [],
                    out_shape  = l.out_shapes[0] if l.out_shapes else [],
                    description = (
                        "Final normalization — the model's full understanding before "
                        "deciding which token to output. This is the richest representation "
                        "of the model's reasoning state."
                    ),
                ))

            # Output / lm_head
            if _matches(l.name, _LM_HEAD_PATTERNS) and l.out_shapes:
                points.append(InjectionPoint(
                    name       = "output_point",
                    role       = "output",
                    layer_name = l.name,
                    in_shape   = l.in_shapes[0] if l.in_shapes else [],
                    out_shape  = l.out_shapes[0] if l.out_shapes else [],
                    description = (
                        "Language model head — projects the hidden state to vocabulary "
                        "logits. The final token probabilities are computed here."
                    ),
                ))

            # Attention K/V (memory points) — only first occurrence
            if re.search(r"self_attn\.k_proj|self_attn\.v_proj|attn\.k_proj", l.name):
                if not any(p.role == "memory" for p in points):
                    points.append(InjectionPoint(
                        name       = "memory_point",
                        role       = "memory",
                        layer_name = l.name,
                        in_shape   = l.in_shapes[0] if l.in_shapes else [],
                        out_shape  = l.out_shapes[0] if l.out_shapes else [],
                        description = (
                            "Key/Value projection — where the model encodes information "
                            "for attention. Compressed shape reveals the KV cache ratio."
                        ),
                    ))

        return points
