from pathlib import Path

import pytest

from circuitcheck.types import (
    Anchor,
    Node,
    PromptFamily,
    SpanError,
    span_layout,
    token_span_map,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "country_capital.yaml"
TEXT = "Fact: the capital of France is"
# Real Gemma-2 offsets for TEXT (add_special_tokens=False), recorded from the tokenizer.
OFFSETS = [(0, 4), (4, 5), (5, 9), (9, 17), (17, 20), (20, 27), (27, 30)]


def family():
    return PromptFamily.from_yaml(FIXTURE)


def test_family_spans():
    p = family().match(TEXT)
    assert p.char_spans == {"_lit0": (0, 21), "country": (21, 27), "_lit1": (27, 30)}
    assert TEXT[slice(*p.char_spans["country"])] == "France"


def test_token_span_map_assigns_leading_space_token_to_slot():
    spans = token_span_map(TEXT, family().match(TEXT).char_spans, OFFSETS)
    assert spans == {"_bos": (0, 1), "_lit0": (1, 6), "country": (6, 7), "_lit1": (7, 8)}


def test_anchor_round_trip_and_negative_offsets():
    spans = token_span_map(TEXT, family().match(TEXT).char_spans, OFFSETS)
    for pos in range(8):
        assert Anchor.from_position(pos, spans).resolve(spans) == pos
    assert Anchor("_lit1", -1).resolve(spans) == 7
    assert Anchor("_lit0", -2).resolve(spans) == 4


def test_missing_span_and_out_of_range_anchor_raise():
    spans = token_span_map(TEXT, family().match(TEXT).char_spans, OFFSETS)
    with pytest.raises(SpanError, match="not in prompt spans"):
        Anchor("city", 0).resolve(spans)
    with pytest.raises(SpanError, match="outside span"):
        Anchor("country", 1).resolve(spans)


def test_straddling_token_raises():
    # one token covering "of France" crosses _lit0 and country
    offsets = OFFSETS[:4] + [(17, 27), (27, 30)]
    with pytest.raises(SpanError, match="straddles"):
        token_span_map(TEXT, family().match(TEXT).char_spans, offsets)


def test_node_rejects_integer_positions():
    with pytest.raises(TypeError, match="Anchor"):
        Node(3, 17, 7)  # type: ignore[arg-type]


def test_span_layout_detects_misalignment():
    a = {"_bos": (0, 1), "_lit0": (1, 6), "country": (6, 7), "_lit1": (7, 8)}
    b = {"_bos": (0, 1), "_lit0": (1, 6), "country": (6, 8), "_lit1": (8, 9)}
    assert span_layout(a) != span_layout(b)


def test_family_missing_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text('name: x\ntemplate: "a {slot} b"\ninstances:\n  - {slot: q, correct: " c"}\n')
    with pytest.raises(ValueError, match="foil"):
        PromptFamily.from_yaml(bad)


def test_multi_token_task_token_is_a_clear_error():
    from transformers import AutoTokenizer

    from circuitcheck.adapters.circuit_tracer import IncompatibleCircuit, prepare_with_tokenizer
    import torch

    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    p = family().match(TEXT)
    tokens = torch.tensor([tok.bos_token_id, *tok(TEXT, add_special_tokens=False).input_ids])
    assert prepare_with_tokenizer(tok, p, tokens).correct_id == tok(" Paris", add_special_tokens=False).input_ids[0]
    from dataclasses import replace

    with pytest.raises(IncompatibleCircuit, match="exactly 1"):
        prepare_with_tokenizer(tok, replace(p, foil=" Washington D.C."), tokens)
