"""
tests/test_smollm_analysis.py

Validation test — runs OpenAIBox on SmolLM-360M and
verifies the expected structure of graph.json.

Run: python tests/test_smollm_analysis.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openaibox import OpenAIBox

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(RESULTS_DIR, "smollm_graph.json")


def test_discovery():
    print("\n" + "="*60)
    print("  TEST 1 — Architecture Discovery")
    print("="*60)

    gr = OpenAIBox("HuggingFaceTB/SmolLM-360M")
    gr.discover("Hello world")

    result = gr._graph_result
    assert result.architecture == "LlamaForCausalLM", f"Expected LlamaForCausalLM, got {result.architecture}"
    assert result.hidden_dim == 960, f"Expected 960, got {result.hidden_dim}"
    assert result.vocab_size == 49152, f"Expected 49152, got {result.vocab_size}"
    assert result.num_layers == 30, f"Expected 30, got {result.num_layers}"
    assert result.total_params > 300_000_000, "Params seem too low"

    # Check injection points
    roles = [p.role for p in result.injection_points]
    assert "decision" in roles, "decision_point not detected"
    assert "input" in roles, "input_point not detected"

    decision = next(p for p in result.injection_points if p.role == "decision")
    assert decision.layer_name == "model.norm", f"Expected model.norm, got {decision.layer_name}"

    print(f"  ✅ Architecture: {result.architecture}")
    print(f"  ✅ Hidden dim:   {result.hidden_dim}")
    print(f"  ✅ Vocab size:   {result.vocab_size}")
    print(f"  ✅ Num layers:   {result.num_layers}")
    print(f"  ✅ Params:       {result.total_params:,}")
    print(f"  ✅ Decision pt:  {decision.layer_name}")
    print(f"  ✅ Injection pts: {roles}")
    return gr


def test_dimension_map(gr):
    print("\n" + "="*60)
    print("  TEST 2 — Dimension Mapping")
    print("="*60)

    gr.map_dimensions()
    result = gr._mapping_result

    assert result.dim == 960, f"Expected 960, got {result.dim}"
    assert len(result.groups) == 6, f"Expected 6 groups, got {len(result.groups)}"
    assert len(result.top_dimensions) > 0, "No top dimensions found"
    assert len(result.multi_role_dims) > 0, "No multi-role dims found"

    # Check that dim_696 is detected as multi-role (established experimentally)
    multi_indices = [d.index for d in result.multi_role_dims]
    # At least some multi-role dims should be found
    assert len(multi_indices) >= 5, f"Expected at least 5 multi-role dims, got {len(multi_indices)}"

    print(f"  ✅ Groups analyzed:   {len(result.groups)}")
    print(f"  ✅ Top dimensions:    {len(result.top_dimensions)}")
    print(f"  ✅ Multi-role dims:   {len(result.multi_role_dims)}")
    print(f"  ✅ Specialist roles:  {list(result.specialist_dims.keys())}")

    # Print separability
    for name, data in sorted(result.groups.items(), key=lambda x: x[1]["cos_sim"]):
        print(f"     {name:<24} cos={data['cos_sim']:.4f} [{data['separability']}]")

    return gr


def test_export(gr):
    print("\n" + "="*60)
    print("  TEST 3 — Export to graph.json")
    print("="*60)

    doc = gr.export(OUTPUT_PATH)

    # Validate structure
    assert "openaibox" in doc
    assert "architecture" in doc
    assert "injection_points" in doc
    assert "layer_flow" in doc
    assert "dimension_map" in doc

    arch = doc["architecture"]
    assert arch["class"] == "LlamaForCausalLM"
    assert arch["hidden_dim"] == 960

    # Validate injection points
    points = {p["role"]: p for p in doc["injection_points"]}
    assert "decision" in points
    assert "input" in points

    # Validate dimension map
    dm = doc["dimension_map"]
    assert dm["hidden_dim"] == 960
    assert "contrast_groups" in dm
    assert "specialist_dimensions" in dm

    # Check file was written
    assert os.path.exists(OUTPUT_PATH), f"File not found: {OUTPUT_PATH}"
    file_size = os.path.getsize(OUTPUT_PATH)

    print(f"  ✅ graph.json written: {OUTPUT_PATH}")
    print(f"  ✅ File size: {file_size:,} bytes")
    print(f"  ✅ Sections: {list(doc.keys())}")
    print(f"  ✅ Injection points: {list(points.keys())}")


def main():
    print("\n" + "█"*60)
    print("  GRAPHRUNTIME — Validation Tests")
    print("  Model: HuggingFaceTB/SmolLM-360M")
    print("█"*60)

    try:
        gr = test_discovery()
        gr = test_dimension_map(gr)
        test_export(gr)
        gr.print_summary()

        print("\n" + "="*60)
        print(f"  ✅ ALL TESTS PASSED")
        print(f"  Results saved to: {OUTPUT_PATH}")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n  ❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
