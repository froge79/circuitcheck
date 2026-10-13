"""circuitcheck: validate a claimed circuit on prompts it was not discovered on."""

from circuitcheck.types import PromptFamily
from circuitcheck.validate import validate


def load_circuit(graph_path, family, discovery_prompts=(), **kwargs):
    from circuitcheck.adapters.circuit_tracer import load_circuit as _load

    return _load(graph_path, family, discovery_prompts, **kwargs)


__all__ = ["load_circuit", "PromptFamily", "validate"]
