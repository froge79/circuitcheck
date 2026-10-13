import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before anything initializes CUDA

from pathlib import Path

import pytest

from circuitcheck.types import PromptFamily

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "country_capital.yaml"
DISCOVERY = "Fact: the capital of France is"


@pytest.fixture(scope="session")
def adapter():
    from circuitcheck.adapters.circuit_tracer import CircuitTracerAdapter

    return CircuitTracerAdapter.load(seed=0)


@pytest.fixture(scope="session")
def family():
    return PromptFamily.from_yaml(FIXTURE)


@pytest.fixture(scope="session")
def graph_path(adapter, tmp_path_factory):
    return adapter.attribute_to_file(DISCOVERY, tmp_path_factory.mktemp("graph") / "france.pt")


@pytest.fixture(scope="session")
def france(adapter, family):
    return adapter.prepare(family.match(DISCOVERY))
