# Understanding Open AI Box Results

## Overview

When you run `gr.export("graph.json")`, Open AI Box produces a structured JSON file
with four main sections. This document explains every field and what it means concretely.

---

## 1. `openaibox` — Package metadata

```json
"openaibox": {
  "version": "1.0.0b1",
  "generated": "2025-03-11T12:00:00Z",
  "model": "HuggingFaceTB/SmolLM-360M",
  "package_url": "https://github.com/tryboy869/openaibox"
}
```

| Field | Meaning |
|-------|---------|
| `version` | Version of Open AI Box that produced this analysis |
| `generated` | UTC timestamp of the analysis |
| `model` | Model identifier used |

---

## 2. `architecture` — Model structure

```json
"architecture": {
  "class": "LlamaForCausalLM",
  "total_params": 361821120,
  "num_layers": 30,
  "hidden_dim": 960,
  "vocab_size": 49152
}
```

### `class`
The Python class name of the model. Tells you which transformer family it belongs to:
- `LlamaForCausalLM` → Llama / SmolLM / Mistral family
- `Qwen2ForCausalLM` → Qwen 2/2.5 family
- `GPT2LMHeadModel` → GPT-2 family

### `total_params`
Total number of parameters in the model.
- 100M–500M → Small/mobile model
- 1B–7B → Standard consumer model
- 70B+ → Large frontier model

### `num_layers`
Number of transformer blocks (decoder layers). Each layer performs:
self-attention + normalization + feed-forward projection.

### `hidden_dim`
The size of the main hidden state vector.
This is the dimension that Open AI Box maps in `dimension_map`.
Every token's internal representation is a vector of this size.

**Key insight**: The `decision_point` output is `[1, 1, hidden_dim]`.
The hidden_dim dimensions are what carry all the model's reasoning.

### `vocab_size`
Number of tokens the model knows.
The `lm_head` projects `hidden_dim → vocab_size` to produce output probabilities.

---

## 3. `injection_points` — Where to observe the model

This is the most actionable section.

```json
"injection_points": [
  {
    "name": "input_point",
    "role": "input",
    "layer": "model.embed_tokens",
    "in_shape": [1, 5],
    "out_shape": [1, 5, 960],
    "description": "Token embeddings..."
  },
  {
    "name": "decision_point",
    "role": "decision",
    "layer": "model.norm",
    "in_shape": [1, 1, 960],
    "out_shape": [1, 1, 960],
    "description": "Final normalization..."
  }
]
```

### `role: "input"` — The embedding layer

The prompt enters the model here as token IDs `[1, seq_len]`
and is converted to dense vectors `[1, seq_len, hidden_dim]`.

**Shape reading**: `[1, 5, 960]` means:
- Batch size = 1
- Sequence length = 5 tokens
- Hidden dimension = 960

### `role: "decision"` — ⭐ The most important point

This is the final normalization layer — after all 30 transformer blocks
have processed the input, but before the model decides what to say next.

The output shape `[1, 1, 960]` represents:
- The last token position only (`seq=1` at inference)
- All 960 dimensions encoding the model's complete understanding

**This is where the model "knows" everything it knows, but hasn't spoken yet.**

### `role: "memory"` — Key/Value projections

KV projections compress context for the attention mechanism.
The shape ratio tells you about Grouped Query Attention:
- `Q: [1, 5, 1536]` and `KV: [1, 5, 256]` → ratio 6:1 (Qwen2.5 style)
- `Q: [1, 5, 960]` and `KV: [1, 5, 320]` → ratio 3:1 (SmolLM style)

A higher GQA ratio means more memory-efficient inference.

### `role: "output"` — The language model head

`lm_head` is a linear projection: `hidden_dim → vocab_size`.
The output `[1, 1, 49152]` contains the raw logit score for every token.

---

## 4. `layer_flow` — Complete execution trace

```json
"layer_flow": [
  {"order": 0, "name": "model.embed_tokens", "type": "Embedding",
   "in": [[1, 5]], "out": [[1, 5, 960]], "params": 47185920},
  {"order": 1, "name": "model.layers.0.input_layernorm", "type": "LlamaRMSNorm",
   "in": [[1, 5, 960]], "out": [[1, 5, 960]], "params": 960},
  ...
]
```

Each entry shows:
- `order` — Execution order during inference (0 = first)
- `name` — PyTorch module path (matches `model.named_modules()`)
- `type` — Class name
- `in` / `out` — Tensor shapes at this layer
- `params` — Number of trainable parameters in this layer

**Reading the flow:**
1. `embed_tokens` → converts token IDs to vectors
2. `layers.0 → layers.N` → N transformer blocks process the vectors
3. `norm` → final normalization (the decision point)
4. `lm_head` → projects to vocabulary

---

## 5. `dimension_map` — What each dimension carries

Only present if you called `gr.map_dimensions()`.

```json
"dimension_map": {
  "hidden_dim": 960,
  "contrast_groups": {
    "syntax_semantics": {
      "question": "Which dimensions detect semantic absurdity?",
      "cos_similarity": 0.6451,
      "separability": "VERY_SEPARABLE",
      "top_dimensions": [696, 792, 766, ...]
    }
  },
  "multi_role_dimensions": [
    {"index": 696, "roles": ["syntax_semantics", "causality", ...], "score": 0.91}
  ],
  "specialist_dimensions": {
    "causality": [295, 157],
    "certainty": [32, 545, 702, 683]
  }
}
```

### `cos_similarity` — How different are the two groups?

| Value | Meaning |
|-------|---------|
| 0.60 | Very separable — model encodes clear difference |
| 0.85 | Separable — detectable difference |
| 0.98 | Low separability — model barely distinguishes these |

**Key findings for SmolLM-360M:**
- `syntax_semantics`: cos=0.6451 → model clearly distinguishes grammatical vs absurd
- `causality`: cos=0.9572 → model barely distinguishes cause from correlation
- This explains why small LLMs often confuse "because" with "and"

### `multi_role_dimensions` — Global regulators

Dimensions present in the top-20 of 3+ contrast groups are **global regulators**.
They modulate overall "signal strength" across many categories.

For SmolLM: `dim_696` and `dim_544` are in all 6 categories — they are
the most influential single dimensions in the model.

### `specialist_dimensions` — Precise carriers

Dimensions present in only one category are **specialists**.

For SmolLM:
- `[295, 157]` → carry **causal** information specifically
- `[32, 545, 702, 683]` → carry **certainty** information specifically
- `[164, 93, 395]` → carry **temporal** information specifically

These are the dimensions you would target for architecture-level analysis.

---

## Reading a result in 60 seconds

1. Check `architecture.hidden_dim` → this is the resolution of the model's reasoning
2. Find `injection_points` where `role == "decision"` → this is the main observation window
3. Check `dimension_map.contrast_groups` → find which contrasts have low `cos_similarity` (< 0.90)
4. Look at `specialist_dimensions` → these dimensions carry specific types of reasoning
5. Look at `multi_role_dimensions` → dim with 5+ roles are the model's most influential neurons

---

*Open AI Box v1.0.0-beta — [github.com/tryboy869/openaibox](https://github.com/tryboy869/openaibox)*
