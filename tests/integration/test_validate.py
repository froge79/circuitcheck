"""End-to-end validate() on the real model, plus incompatibility errors."""

import json
from dataclasses import replace

import pytest

from circuitcheck.adapters.circuit_tracer import IncompatibleCircuit
from circuitcheck.split import SplitViolation
from circuitcheck.types import Anchor, Node, PromptFamily
from circuitcheck.validate import validate

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def circuit(adapter, family, graph_path):
    return adapter.load_circuit(graph_path, family, node_threshold=0.8)


@pytest.fixture(scope="module")
def report(adapter, family, circuit):
    return validate(circuit, family, adapter=adapter, ablation="mean", n_prompts=4, seed=0)


def test_report_fields_and_consistency(report, capsys):
    d = report.data
    with capsys.disabled():
        print("\n[validate] headline:", d["headline"], "warnings:", d["warnings"])
        for p in d["per_prompt"]:
            fr = p["regimes"]["frozen"]
            print(f"  {p['instance_id']:8s} m_full={fr['m_full']:.4f} m_C={fr['m_c']:.4f} "
                  f"m_empty={fr['m_empty']:.4f} F={fr['f']} err={fr['err_bound']} "
                  f"cov={p['coverage']['fraction']:.2f} unfrozenF={p['regimes']['unfrozen']['f']}")
    assert d["split"]["excluded_discovery_instances"] == 1
    assert "france" not in {p["instance_id"] for p in d["per_prompt"]}
    assert len(d["per_prompt"]) == 4
    for p in d["per_prompt"]:
        for r in p["regimes"].values():
            for k in ["m_full", "m_c", "m_empty", "d", "f", "valid", "reason", "warnings",
                      "err_bound", "resolution", "n_intervened"]:
                assert k in r
    assert d["summary"]["frozen"]["n_valid"] >= 1
    assert d["summary"]["frozen"]["role"] == "headline"
    prov = d["provenance"]
    assert prov["model_revision"] and prov["numerics"]["deterministic"]
    assert prov["numerics"]["dtype"] == "bfloat16"
    assert json.loads(report.to_json())  # serializes


def test_repeat_run_identical(adapter, family, circuit, report):
    again = validate(circuit, family, adapter=adapter, ablation="mean", n_prompts=4, seed=0)
    a, b = dict(report.data), dict(again.data)
    a.pop("created_utc"), b.pop("created_utc")
    assert a == b


def test_empty_circuit_scores_zero(adapter, family, circuit):
    r = validate(replace(circuit, nodes=()), family, adapter=adapter, ablation="zero", n_prompts=2)
    for p in r.data["per_prompt"]:
        assert p["regimes"]["frozen"]["f"] == 0.0


def test_discovery_prompt_in_family_is_excluded_and_near_match_raises(adapter, family, circuit, tmp_path):
    fam = tmp_path / "f.yaml"
    fam.write_text(
        'name: near\ntemplate: "Fact: the capital of {country} is"\ninstances:\n'
        '  - {country: Frances, correct: " Paris", foil: " London"}\n'
    )
    with pytest.raises(SplitViolation, match="near"):
        validate(circuit, PromptFamily.from_yaml(fam), adapter=adapter)
    r = validate(circuit, PromptFamily.from_yaml(fam), adapter=adapter, allow_overlap=True,
                 ablation="zero")
    assert r.data["overlap_override"] is True and r.data["warnings"][0] == "OVERLAP_OVERRIDE"


def test_incompatible_configurations(adapter, family, circuit, tmp_path):
    with pytest.raises(IncompatibleCircuit, match="ablation"):
        validate(circuit, family, adapter=adapter, ablation="resample")
    with pytest.raises(IncompatibleCircuit, match="transcoders"):
        validate(replace(circuit, scan="other/transcoders"), family, adapter=adapter)
    with pytest.raises(IncompatibleCircuit, match="model"):
        validate(replace(circuit, model_name="gpt2"), family, adapter=adapter)
    with pytest.raises(IncompatibleCircuit, match="out of range"):
        validate(replace(circuit, nodes=(Node(26, 0, Anchor("_lit1", -1)),)), family, adapter=adapter)
    with pytest.raises(IncompatibleCircuit, match="out of range"):
        validate(replace(circuit, nodes=(Node(3, 16384, Anchor("_lit1", -1)),)), family, adapter=adapter)
    other = tmp_path / "o.yaml"
    other.write_text('name: o\ntemplate: "The {city} is in"\ninstances:\n'
                     '  - {city: Louvre, correct: " Paris", foil: " London"}\n')
    with pytest.raises(IncompatibleCircuit, match="anchors do not resolve"):
        validate(circuit, PromptFamily.from_yaml(other), adapter=adapter, ablation="zero")
