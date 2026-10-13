"""Sanity checks of intervention semantics on the real model (Gemma-2-2b, bf16, deterministic)."""

import torch
import pytest

pytestmark = pytest.mark.gpu


def logits_of(adapter, prep, interventions, freeze=True):
    logits, _ = adapter.run(prep, interventions, freeze)
    return logits[0]


def test_no_intervention_equals_plain_forward(adapter, france):
    plain = adapter.model(france.tokens)[0]
    assert torch.equal(logits_of(adapter, france, []), plain)
    assert torch.equal(logits_of(adapter, france, [], freeze=False), plain)
    # string path (TransformerLens to_tokens) and our token tensor agree
    assert torch.equal(adapter.model(france.prompt.text)[0], plain)
    assert adapter.measure(france, [], True).correct_is_argmax


def test_identity_intervention_through_freeze_path(adapter, france):
    acts = adapter.dense_activations(france)
    entries = []
    for layer in range(adapter.n_layers):
        pos, feat = acts[layer].nonzero(as_tuple=True)
        entries.append((layer, pos, feat, acts[layer, pos, feat]))
    assert adapter.n_entries(entries) > 1000
    assert torch.equal(logits_of(adapter, france, entries), adapter.model(france.tokens)[0])


def test_position_and_layer_mapping(adapter, france):
    layer, pos = 10, 4
    _, clean = adapter.run(france, [], True, return_activations=True)
    logits, acts = adapter.run(france, [(layer, pos, 123, 500.0)], True, return_activations=True)
    plain = adapter.model(france.tokens)[0]
    assert torch.equal(logits[0, :pos], plain[:pos])          # causal: earlier positions untouched
    assert not torch.equal(logits[0, pos:], plain[pos:])
    assert torch.equal(acts[: layer + 1], clean[: layer + 1])  # layers <= L read pre-delta inputs
    assert torch.equal(acts[layer + 1:, :pos], clean[layer + 1:, :pos])
    assert not torch.equal(acts[layer + 1:, pos:], clean[layer + 1:, pos:])


def test_circuit_nodes_map_to_graph_features(adapter, family, graph_path, france):
    from circuit_tracer.graph import Graph  # test-only read of the raw graph

    circuit = adapter.load_circuit(graph_path, family, top_k=50)
    graph = Graph.from_pt(str(graph_path))
    rows = {tuple(r) for r in graph.active_features[graph.selected_features].tolist()}
    acts = adapter.dense_activations(france)
    assert len(circuit.nodes) == 50
    for n in circuit.nodes:
        pos = n.anchor.resolve(france.span_map)
        assert (n.layer, pos, n.feature) in rows
        assert acts[n.layer, pos, n.feature] > 0
    assert torch.equal(france.tokens.cpu(), graph.input_tokens.cpu())


def test_relevant_intervention_moves_metric_and_irrelevant_does_not(adapter, france):
    base = adapter.measure(france, [], True).ld
    last = france.n_pos - 1
    acts = adapter.dense_activations(france)
    entries = []
    for layer in range(adapter.n_layers):
        feat = acts[layer, last].nonzero(as_tuple=True)[0]
        entries.append((layer, torch.full_like(feat, last), feat, torch.zeros_like(acts[layer, last, feat])))
    assert base - adapter.measure(france, entries, True).ld >= 1.0
    # zeroing an inactive feature changes nothing
    inactive = int((acts[5, last] == 0).nonzero()[0])
    assert adapter.measure(france, [(5, last, inactive, 0.0)], True).ld == base


@pytest.mark.parametrize("freeze", [True, False])
def test_invariants_full_and_empty_circuit(adapter, france, freeze):
    from circuitcheck.score import faithfulness, resolution

    values = adapter.ablation_values([france], "zero")
    all_keep = torch.ones_like(values, dtype=torch.bool)
    none_keep = torch.zeros_like(all_keep)
    full = adapter.measure(france, [], freeze)
    empty = adapter.measure(france, adapter.complement_interventions(none_keep, values), freeze)
    c_all = adapter.measure(france, adapter.complement_interventions(all_keep, values), freeze)
    c_none = adapter.measure(france, adapter.complement_interventions(none_keep, values), freeze)
    res = resolution(adapter.eps, 30.0)
    assert faithfulness(full.ld, c_all.ld, empty.ld, res).f == 1.0
    assert faithfulness(full.ld, c_none.ld, empty.ld, res).f == 0.0


def test_spike_regression_and_complement_leak(adapter, family, graph_path, france, capsys):
    """Spike 3, bf16 deterministic, frozen, zero-ablation, clean-active encoding:
    m_full 9.0, m_C 7.71875, m_empty 3.8125. The graph is regenerated here, so the top-50
    circuit may differ slightly from the spike's and m_C gets a tolerance."""
    circuit = adapter.load_circuit(graph_path, family, top_k=50)
    acts = adapter.dense_activations(france)
    keep = adapter.keep_mask(france, circuit.nodes)
    values = adapter.ablation_values([france], "zero")

    def active_only(keep_):
        out = []
        for layer in range(adapter.n_layers):
            pos, feat = (acts[layer].ne(0) & ~keep_[layer]).nonzero(as_tuple=True)
            out.append((layer, pos, feat, 0.0))
        return out

    none = torch.zeros_like(keep)
    m_full = adapter.measure(france, [], True).ld
    m_empty = adapter.measure(france, active_only(none), True).ld
    m_c = adapter.measure(france, active_only(keep), True).ld
    assert m_full == 9.0
    assert m_empty == 3.8125
    assert m_c == pytest.approx(7.71875, abs=0.25)

    dense_empty = adapter.measure(france, adapter.complement_interventions(none, values), True).ld
    dense_c = adapter.measure(france, adapter.complement_interventions(keep, values), True).ld
    with capsys.disabled():
        print(f"\n[spike regression] active-only: m_C={m_c} m_empty={m_empty}; "
              f"dense: m_C={dense_c} m_empty={dense_empty}; "
              f"complement_leak_delta: C={dense_c - m_c:+.4f} empty={dense_empty - m_empty:+.4f}")
    assert all(map(lambda x: x == x, [dense_c, dense_empty]))


def test_repeat_is_deterministic(adapter, france):
    values = adapter.ablation_values([france], "zero")
    lst = adapter.complement_interventions(torch.zeros_like(values, dtype=torch.bool), values)
    a, b = adapter.run(france, lst, True)[0], adapter.run(france, lst, True)[0]
    assert torch.equal(a, b)
