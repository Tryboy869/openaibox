# Changelog

All notable changes to Open AI Box are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0-beta] — 2025-03-11

### Added
- `Open AI Box` main class — universal LLM introspection entry point
- `Discoverer` — live inference tracing with PyTorch forward hooks
- Auto-detection of injection points: `input_point`, `decision_point`, `memory_point`, `output_point`
- Support for Llama, Qwen2, GPT-2, BLOOM, Falcon, OPT, MPT, GPT-NeoX architectures
- `DimensionMapper` — contrast-based dimension role analysis
- 6 built-in contrast groups: syntax/semantics, causality, certainty, abstraction, temporality, emotion
- `export()` → `graph.json` with full architecture + injection points + dimension map
- `explain_dimension(idx)` → per-dimension role explanation
- `print_summary()` → human-readable console output
- Documentation: `understanding_results.md` + `understanding_results.fr.md`
- SVG animations: header, lock/Open AI Box, creator card, footer
- GitHub Actions: auto-release + PyPI publish on CHANGELOG section bump
- Shell scripts: `release.sh`, `publish_pypi.sh`
- Colab deployment script

### Architecture
- Discovered experimentally on SmolLM-360M (361M params, LlamaForCausalLM)
- Validated on Qwen2.5-Math-1.5B (1.5B params, Qwen2ForCausalLM)
- Decision point `norm → lm_head` confirmed universal across both families

### Known Limitations
- Requires local model weights (cloud APIs not supported — hidden states inaccessible)
- Auto-detection may miss injection points on highly custom architectures (manual fallback provided)
- Dimension mapping uses 3 prompts per group — increase for more robust results

---

*Next: [1.1.0] — Expanded architecture support, HTML report export, CLI interface*
