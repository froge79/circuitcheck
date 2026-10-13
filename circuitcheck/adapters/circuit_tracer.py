"""The only module that imports circuit_tracer or touches model internals.

Intervention semantics (circuit-tracer 0.5.0, TransformerLens backend, read from source):
``feature_intervention`` takes (layer, pos, feature, value) entries. At each layer it encodes
the *current* MLP input, computes ``value - current_activation`` for the listed entries
only, and adds ``delta * W_dec`` to the real MLP output. So error terms stay live,
token/embedding nodes cannot be addressed, unlisted features run free, and with
``constrained_layers=None`` ``freeze_attention=True`` freezes attention patterns only
(LayerNorm scales and MLPs stay live). An empty list skips the freeze path.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from circuitcheck.split import normalize, prompt_hash
from circuitcheck.types import (
    Anchor,
    Circuit,
    Node,
    Prompt,
    PromptFamily,
    SpanError,
    token_span_map,
)

CUBLAS_CONFIG = ":4096:8"
DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_TRANSCODERS = "gemma"
ERROR_NODE_CONVENTION = (
    "Error, embedding and token nodes are always retained and are not counted in the node "
    "budget. The library cannot address them: the real MLP output is kept and feature "
    "deltas are added to it, so reconstruction error is live (recomputed on the perturbed "
    "input), not frozen. Scores are not comparable to tools using other conventions."
)
REGIMES = {
    True: "attention patterns frozen at their clean-run values; LayerNorm scales, MLPs, "
          "transcoder features and error terms recomputed live (freeze_attention=True)",
    False: "nothing frozen; all effects propagate (freeze_attention=False)",
}


class IncompatibleCircuit(ValueError):
    """The circuit, model, family or intervention configuration do not fit together."""


def enable_determinism(seed: int) -> dict[str, Any]:
    """Deterministic mode per DESIGN.md. Must run before CUDA initializes cuBLAS."""
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA was initialized before CUBLAS_WORKSPACE_CONFIG was set; set it to "
                f"{CUBLAS_CONFIG!r} in the environment before importing the model"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CONFIG
    elif current != CUBLAS_CONFIG:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG is {current!r}, expected {CUBLAS_CONFIG!r}")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return {"deterministic": True, "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG, "seed": seed}


def _hf_revision(repo_id: str) -> str | None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    ref = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def _versions() -> dict[str, str | None]:
    out = {}
    for pkg in ["circuit-tracer", "transformer-lens", "torch", "transformers"]:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_id(scan: Any) -> str:
    return scan if isinstance(scan, str) else repr(scan)


@dataclass
class PreparedPrompt:
    prompt: Prompt
    tokens: torch.Tensor  # 1-D, BOS first; the one tensor every condition runs on
    span_map: dict[str, tuple[int, int]]
    correct_id: int
    foil_id: int
    prompt_hash: str

    @property
    def n_pos(self) -> int:
        return int(self.tokens.numel())


@dataclass
class Measurement:
    ld: float
    logit_correct: float
    logit_foil: float
    correct_is_argmax: bool
    n_entries: int


@dataclass
class CallRecord:
    prompt_hash: str
    tokens_sha: str
    correct_id: int
    foil_id: int
    freeze: bool
    n_entries: int


def _single_token(tokenizer, text: str, what: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise IncompatibleCircuit(f"{what} {text!r} tokenizes to {len(ids)} tokens; need exactly 1")
    return ids[0]


def prepare_with_tokenizer(tokenizer, prompt: Prompt, tokens: torch.Tensor) -> PreparedPrompt:
    """Span map and task-token ids for ``prompt``, checked against the model's own tokens."""
    enc = tokenizer(prompt.text, add_special_tokens=False, return_offsets_mapping=True)
    body = tokens[1:].tolist()
    if enc.input_ids != body:
        raise SpanError(
            f"offset tokenization {enc.input_ids} disagrees with model tokens {body} "
            f"for {prompt.text!r}"
        )
    spans = token_span_map(prompt.text, prompt.char_spans, enc.offset_mapping, n_prefix=1)
    return PreparedPrompt(
        prompt,
        tokens,
        spans,
        _single_token(tokenizer, prompt.correct, "correct token"),
        _single_token(tokenizer, prompt.foil, "foil token"),
        prompt_hash(prompt.text),
    )


@dataclass
class CircuitTracerAdapter:
    model: Any
    model_name: str
    transcoder_set: str
    dtype: torch.dtype
    numerics: dict[str, Any]
    call_log: list[CallRecord] = field(default_factory=list)

    @staticmethod
    def load(
        model_name: str = DEFAULT_MODEL,
        transcoder_set: str = DEFAULT_TRANSCODERS,
        dtype: torch.dtype = torch.bfloat16,
        deterministic: bool = True,
        seed: int = 0,
    ) -> CircuitTracerAdapter:
        if deterministic:
            numerics = enable_determinism(seed)
        else:
            torch.manual_seed(seed)
            numerics = {"deterministic": False,
                        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "seed": seed}
        numerics["dtype"] = str(dtype).removeprefix("torch.")
        from circuit_tracer import ReplacementModel

        model = ReplacementModel.from_pretrained(model_name, transcoder_set, dtype=dtype)
        return CircuitTracerAdapter(model, model_name, transcoder_set, dtype, numerics)

    # --- identity --------------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return int(self.model.cfg.n_layers)

    @property
    def tl_model_name(self) -> str:
        return str(self.model.cfg.model_name)

    @property
    def scan(self) -> str:
        return _scan_id(self.model.scan)

    @property
    def _width(self) -> int:
        """Transcoder width, read from a real activation tensor rather than assumed."""
        if not hasattr(self, "_cached_width"):
            _, acts = self.model.get_activations(self.model.ensure_tokenized("a"))
            self._cached_width = int(acts.shape[-1])
        return self._cached_width

    def provenance(self) -> dict[str, Any]:
        scan = self.scan
        return {
            "model_name": self.model_name,
            "model_revision": _hf_revision(self.model_name),
            "transcoder_set": self.transcoder_set,
            "scan": scan,
            "scan_revision": _hf_revision(scan) if "/" in scan else None,
            "backend": "transformerlens",
            "library_versions": _versions(),
            "numerics": dict(self.numerics),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "error_node_convention": ERROR_NODE_CONVENTION,
        }

    # --- circuits --------------------------------------------------------------------------

    def attribute_to_file(self, text: str, path: str | Path, max_feature_nodes: int = 2000) -> Path:
        """Discovery graph for ``text``, written with Graph.to_pt (test and fixture helper)."""
        from circuit_tracer import attribute

        graph = attribute(text, self.model, max_feature_nodes=max_feature_nodes)
        graph.to_pt(str(path))
        return Path(path)

    def load_circuit(
        self,
        graph_path: str | Path,
        family: PromptFamily,
        discovery_prompts: Sequence[str] = (),
        node_threshold: float | None = 0.8,
        edge_threshold: float = 0.98,
        top_k: int | None = None,
    ) -> Circuit:
        return load_circuit(graph_path, family, discovery_prompts, node_threshold,
                            edge_threshold, top_k, tokenizer=self.model.tokenizer)

    def check_compatible(self, circuit: Circuit) -> None:
        problems = []
        if circuit.model_name != self.tl_model_name:
            problems.append(f"circuit model {circuit.model_name!r} != loaded {self.tl_model_name!r}")
        if circuit.scan != self.scan:
            problems.append(f"circuit transcoders {circuit.scan!r} != loaded {self.scan!r}")
        for n in circuit.nodes:
            if not 0 <= n.layer < self.n_layers or not 0 <= n.feature < self._width:
                problems.append(f"node {n} out of range (layers {self.n_layers}, width {self._width})")
                break
        if problems:
            raise IncompatibleCircuit("; ".join(problems))

    # --- prompts and activations -----------------------------------------------------------

    def prepare(self, prompt: Prompt) -> PreparedPrompt:
        tokens = self.model.ensure_tokenized(prompt.text)
        return prepare_with_tokenizer(self.model.tokenizer, prompt, tokens)

    def dense_activations(self, prep: PreparedPrompt) -> torch.Tensor:
        """Clean-run transcoder activations, [n_layers, n_pos, width]; position 0 is zero."""
        _, acts = self.model.get_activations(prep.tokens)
        return acts

    def ablation_values(self, preps: Sequence[PreparedPrompt], mode: str) -> torch.Tensor:
        """Per-(layer, pos, feature) ablation values shared by every prompt in ``preps``.

        'mean': activation averaged over ``preps`` (inactive counts as zero), computed in fp32
        and cast to the model dtype. 'zero': zeros. Callers guarantee the prompts share one
        span layout, so a position here names the same anchor in every prompt.
        """
        if mode not in ("mean", "zero"):
            raise IncompatibleCircuit(f"unknown ablation mode {mode!r}; use 'mean' or 'zero'")
        n_pos = preps[0].n_pos
        if any(p.n_pos != n_pos for p in preps):
            raise IncompatibleCircuit("ablation values need prompts of equal token length")
        if mode == "zero":
            return torch.zeros(self.n_layers, n_pos, self._width, dtype=self.dtype,
                               device=self.model.cfg.device)
        total = None
        for p in preps:
            a = self.dense_activations(p).float()
            total = a if total is None else total + a
        return (total / len(preps)).to(self.dtype)

    # --- interventions ---------------------------------------------------------------------

    def keep_mask(self, prep: PreparedPrompt, nodes: Iterable[Node]) -> torch.Tensor:
        """Boolean [n_layers, n_pos, width] mask of the circuit's resolved entries."""
        mask = torch.zeros(self.n_layers, prep.n_pos, self._width, dtype=torch.bool,
                           device=self.model.cfg.device)
        for n in nodes:
            mask[n.layer, n.anchor.resolve(prep.span_map), n.feature] = True
        return mask

    def complement_interventions(self, keep: torch.Tensor, values: torch.Tensor) -> list[tuple]:
        """Set every (layer, pos >= 1, feature) outside ``keep`` to ``values`` there.

        Dense on purpose: features that only become active under the intervention are
        ablated too. Position 0 (BOS) is excluded; its activations are zeroed by the library.
        One entry per layer, with tensors for pos, feature and value.
        """
        if keep.shape != values.shape:
            raise IncompatibleCircuit(f"keep mask {tuple(keep.shape)} != values {tuple(values.shape)}")
        ablate = ~keep
        ablate[:, 0, :] = False
        out = []
        for layer in range(self.n_layers):
            pos, feat = ablate[layer].nonzero(as_tuple=True)
            if pos.numel():
                out.append((layer, pos, feat, values[layer, pos, feat]))
        return out

    @staticmethod
    def n_entries(interventions: Sequence[tuple]) -> int:
        n = 0
        for _, pos, _, _ in interventions:
            n += int(pos.numel()) if isinstance(pos, torch.Tensor) else 1
        return n

    def run(self, prep: PreparedPrompt, interventions: Sequence[tuple], freeze: bool,
            return_activations: bool = False):
        """Raw feature_intervention on the prepared token tensor; logits are [1, n_pos, vocab]."""
        return self.model.feature_intervention(
            prep.tokens, list(interventions), freeze_attention=freeze,
            return_activations=return_activations,
        )

    def measure(self, prep: PreparedPrompt, interventions: Sequence[tuple], freeze: bool) -> Measurement:
        """Correct-minus-foil logit difference at the last position, logged for consistency."""
        logits, _ = self.run(prep, interventions, freeze)
        last = logits[0, -1]
        lc, lf = float(last[prep.correct_id]), float(last[prep.foil_id])
        n = self.n_entries(interventions)
        self.call_log.append(CallRecord(
            prep.prompt_hash,
            hashlib.sha256(prep.tokens.cpu().numpy().tobytes()).hexdigest(),
            prep.correct_id, prep.foil_id, freeze, n,
        ))
        return Measurement(lc - lf, lc, lf, int(last.argmax()) == prep.correct_id, n)

    @property
    def eps(self) -> float:
        return float(torch.finfo(self.dtype).eps)


def load_circuit(
    graph_path: str | Path,
    family: PromptFamily,
    discovery_prompts: Sequence[str] = (),
    node_threshold: float | None = 0.8,
    edge_threshold: float = 0.98,
    top_k: int | None = None,
    tokenizer=None,
) -> Circuit:
    """Read a circuit from a circuit-tracer graph file.

    Nodes are the graph's feature nodes kept by ``prune_graph(node_threshold, edge_threshold)``,
    or, with ``top_k``, the k feature nodes of highest ``compute_node_influence``. Positions
    become Anchors through ``family``'s span map of the graph's own prompt. The graph prompt
    is always part of the discovery record.
    """
    from circuit_tracer.graph import Graph, compute_node_influence, prune_graph

    graph = Graph.from_pt(str(graph_path))
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)

    # attribute() stores input_string decoded from tokens, BOS included ("<bos>Fact: ..."), so
    # the prompt text is recovered from the tokens after the special prefix and checked by
    # re-tokenizing it in prepare_with_tokenizer.
    tokens = graph.input_tokens.cpu()
    if int(tokens[0]) not in tokenizer.all_special_ids:
        raise IncompatibleCircuit("graph tokens do not start with a special (BOS) token")
    text = tokenizer.decode(tokens[1:].tolist())
    discovery = family.match(text)
    prep = prepare_with_tokenizer(tokenizer, discovery, tokens)

    n_feat = len(graph.selected_features)
    if top_k is not None:
        n_logits = len(graph.logit_targets)
        weights = torch.zeros(graph.adjacency_matrix.shape[0])
        weights[-n_logits:] = graph.logit_probabilities.cpu().float()
        influence = compute_node_influence(graph.adjacency_matrix.cpu().float(), weights)[:n_feat]
        chosen = torch.topk(influence, top_k).indices.sort().values
        selection = {"method": "top_k by compute_node_influence", "top_k": top_k}
    elif node_threshold is not None:
        mask = prune_graph(graph, node_threshold, edge_threshold).node_mask[:n_feat].cpu()
        chosen = mask.nonzero(as_tuple=True)[0]
        selection = {"method": "prune_graph feature nodes", "node_threshold": node_threshold,
                     "edge_threshold": edge_threshold}
    else:
        raise IncompatibleCircuit("give node_threshold or top_k")

    rows = graph.active_features[graph.selected_features[chosen].cpu()].cpu().tolist()
    nodes = tuple(
        Node(int(layer), int(feat), Anchor.from_position(int(pos), prep.span_map))
        for layer, pos, feat in rows
    )
    selection = {**selection, "n_selected_in_graph": n_feat,
                 "n_active_in_graph": int(len(graph.active_features))}

    texts = [text, *discovery_prompts]
    return Circuit(
        nodes=nodes,
        model_name=str(graph.cfg.model_name),
        scan=_scan_id(graph.scan),
        source={"graph_path": str(graph_path), "graph_sha256": _file_sha256(graph_path),
                "graph_prompt_hash": prompt_hash(text)},
        selection=selection,
        discovery_hashes=tuple(dict.fromkeys(prompt_hash(t) for t in texts)),
        discovery_normalized=tuple(dict.fromkeys(normalize(t) for t in texts)),
    )
