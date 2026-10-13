import pytest

from circuitcheck.split import SplitViolation, check_split, normalize, prompt_hash, similarity
from circuitcheck.types import Circuit

DISCOVERY = "Fact: the capital of France is"


def circuit(*texts):
    return Circuit(
        nodes=(), model_name="m", scan="s", source={}, selection={},
        discovery_hashes=tuple(prompt_hash(t) for t in texts),
        discovery_normalized=tuple(normalize(t) for t in texts),
    )


def test_normalize_and_hash():
    assert normalize("  Fact:\tthe   CAPITAL of France is \n") == "fact: the capital of france is"
    assert prompt_hash("FACT: the capital of  France is") == prompt_hash(DISCOVERY)


def test_identical_prompt_raises():
    with pytest.raises(SplitViolation, match="exact"):
        check_split([DISCOVERY], circuit(DISCOVERY))


@pytest.mark.parametrize("variant", ["fact: the capital of france is", "Fact:  the capital of France is  "])
def test_whitespace_and_case_variants_raise_as_exact(variant):
    with pytest.raises(SplitViolation, match="exact"):
        check_split([variant], circuit(DISCOVERY))


def test_one_character_variant_raises_as_near():
    with pytest.raises(SplitViolation, match="near"):
        check_split(["Fact: the capital of Frances is"], circuit(DISCOVERY))


def test_entity_swap_passes():
    rec = check_split(["Fact: the capital of Germany is"], circuit(DISCOVERY))
    assert rec["collisions"] == [] and rec["overlap_override"] is False
    assert similarity(normalize("Fact: the capital of Germany is"), normalize(DISCOVERY)) < 0.95


def test_allow_overlap_records_flag_and_collisions():
    rec = check_split([DISCOVERY, "Fact: the capital of Germany is"], circuit(DISCOVERY),
                      allow_overlap=True)
    assert rec["overlap_override"] is True
    assert [c["kind"] for c in rec["collisions"]] == ["exact"]
