"""Core types. Nothing here imports circuit_tracer or touches a model."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

BOS_SPAN = "_bos"

# span name -> half-open token range [start, stop) in one tokenized prompt
SpanMap = Mapping[str, tuple[int, int]]


class SpanError(ValueError):
    """A prompt's spans cannot be mapped onto its tokens, or an anchor does not resolve."""


@dataclass(frozen=True)
class Anchor:
    """A position named by span: offset >= 0 counts from the span start, < 0 from its end."""

    span: str
    offset: int

    def __post_init__(self):
        if not isinstance(self.span, str) or not isinstance(self.offset, int):
            raise TypeError("Anchor needs a span name and an integer offset")

    def resolve(self, span_map: SpanMap) -> int:
        if self.span not in span_map:
            raise SpanError(f"anchor span {self.span!r} not in prompt spans {sorted(span_map)}")
        start, stop = span_map[self.span]
        pos = start + self.offset if self.offset >= 0 else stop + self.offset
        if not start <= pos < stop:
            raise SpanError(f"anchor {self} falls outside span {self.span!r} = [{start}, {stop})")
        return pos

    @staticmethod
    def from_position(pos: int, span_map: SpanMap) -> Anchor:
        for name, (start, stop) in span_map.items():
            if start <= pos < stop:
                return Anchor(name, pos - start)
        raise SpanError(f"position {pos} is not covered by any span")


@dataclass(frozen=True)
class Node:
    """One transcoder feature of a circuit. Its position is an Anchor, never an index."""

    layer: int
    feature: int
    anchor: Anchor

    def __post_init__(self):
        if not isinstance(self.anchor, Anchor):
            raise TypeError(f"Node position must be an Anchor, got {type(self.anchor).__name__}")


@dataclass(frozen=True)
class Circuit:
    """A claimed circuit plus the provenance needed to keep validation honest.

    Built only by an adapter's load_circuit. ``discovery_normalized`` stays in memory for the
    near-match check; reports carry ``discovery_hashes`` only.
    """

    nodes: tuple[Node, ...]
    model_name: str
    scan: str
    source: Mapping[str, Any]
    selection: Mapping[str, Any]
    discovery_hashes: tuple[str, ...]
    discovery_normalized: tuple[str, ...] = field(repr=False)

    def describe(self) -> dict[str, Any]:
        return {
            "n_nodes": len(self.nodes),
            "model_name": self.model_name,
            "scan": self.scan,
            "source": dict(self.source),
            "selection": dict(self.selection),
            "discovery_prompt_hashes": list(self.discovery_hashes),
        }


@dataclass(frozen=True)
class Prompt:
    """One family instance: text, character spans per named span, and the task tokens."""

    instance_id: str
    text: str
    char_spans: Mapping[str, tuple[int, int]]
    correct: str
    foil: str


def _parse_template(template: str) -> list[tuple[str, str | None]]:
    """Split a template into (span_name, literal_or_None) parts; None marks a slot."""
    parts: list[tuple[str, str | None]] = []
    n_lit = 0
    for literal, slot, spec, conv in string.Formatter().parse(template):
        if literal:
            parts.append((f"_lit{n_lit}", literal))
            n_lit += 1
        if slot is not None:
            if not slot or spec or conv:
                raise ValueError(f"template slots must be plain named fields: {template!r}")
            parts.append((slot, None))
    return parts


@dataclass(frozen=True)
class PromptFamily:
    name: str
    template: str
    instances: tuple[Mapping[str, Any], ...]

    @staticmethod
    def from_yaml(path: str | Path) -> PromptFamily:
        d = yaml.safe_load(Path(path).read_text())
        fam = PromptFamily(d["name"], d["template"], tuple(d["instances"]))
        slots = fam.slots
        for i, inst in enumerate(fam.instances):
            missing = [k for k in [*slots, "correct", "foil"] if k not in inst]
            if missing:
                raise ValueError(f"instance {i} of family {fam.name!r} lacks {missing}")
        return fam

    @property
    def slots(self) -> list[str]:
        return [name for name, lit in _parse_template(self.template) if lit is None]

    def prompts(self) -> list[Prompt]:
        out = []
        for i, inst in enumerate(self.instances):
            text, spans = "", {}
            for name, lit in _parse_template(self.template):
                piece = lit if lit is not None else str(inst[name])
                spans[name] = (len(text), len(text) + len(piece))
                text += piece
            out.append(Prompt(str(inst.get("id", i)), text, spans, inst["correct"], inst["foil"]))
        return out

    def match(self, text: str) -> Prompt:
        """Recover slot fillers and spans for a prompt rendered from this template."""
        for p in self.prompts():
            if p.text == text:
                return p
        raise SpanError(f"{text!r} is not an instance of family {self.name!r}")


def token_span_map(
    text: str,
    char_spans: Mapping[str, tuple[int, int]],
    offsets: Sequence[tuple[int, int]],
    n_prefix: int = 1,
) -> dict[str, tuple[int, int]]:
    """Map named character spans onto token ranges.

    ``offsets`` are per-token character ranges for the text without special tokens;
    ``n_prefix`` special tokens (BOS) precede them and form the ``_bos`` span. A token
    belongs to the span holding its last non-space character; a token whose non-space
    characters cross a span boundary raises, as does a span with no tokens.
    """

    def span_of(ch: int) -> str:
        for name, (s, e) in char_spans.items():
            if s <= ch < e:
                return name
        raise SpanError(f"character {ch} of {text!r} lies in no span")

    assigned: list[str] = []
    for i, (s, e) in enumerate(offsets):
        chars = [c for c in range(s, e) if not text[c].isspace()]
        if not chars:
            raise SpanError(f"token {i} {text[s:e]!r} is whitespace only")
        names = {span_of(c) for c in chars}
        if len(names) > 1:
            raise SpanError(f"token {i} {text[s:e]!r} straddles spans {sorted(names)}")
        assigned.append(names.pop())

    spans: dict[str, tuple[int, int]] = {BOS_SPAN: (0, n_prefix)} if n_prefix else {}
    for i, name in enumerate(assigned):
        pos = i + n_prefix
        if name in spans:
            start, stop = spans[name]
            if stop != pos:
                raise SpanError(f"span {name!r} is not contiguous in tokens")
            spans[name] = (start, pos + 1)
        else:
            spans[name] = (pos, pos + 1)
    empty = [n for n in char_spans if n not in spans]
    if empty:
        raise SpanError(f"spans {empty} map to no tokens in {text!r}")
    return spans


def span_layout(span_map: SpanMap) -> tuple[tuple[str, int], ...]:
    """Span names with token lengths, in token order. Equal layouts mean anchors align."""
    return tuple((n, e - s) for n, (s, e) in sorted(span_map.items(), key=lambda kv: kv[1]))


@dataclass
class Report:
    data: dict[str, Any]

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.data, indent=2, sort_keys=False)
        if path is not None:
            Path(path).write_text(s + "\n")
        return s
