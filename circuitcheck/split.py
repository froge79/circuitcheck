"""Discovery/validation split guard: exact match on normalized hashes, then near match."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from circuitcheck.types import Circuit


class SplitViolation(RuntimeError):
    """A validation prompt matches, or nearly matches, a discovery prompt."""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """Normalized edit similarity of two already-normalized strings, in [0, 1]."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def check_split(
    held_out: Iterable[str],
    circuit: Circuit,
    near_threshold: float = 0.95,
    allow_overlap: bool = False,
) -> dict[str, Any]:
    """Raise SplitViolation on any exact or near collision unless allow_overlap is set.

    Returns a record for the report; with allow_overlap it lists the collisions and
    carries ``overlap_override: True``.
    """
    discovery = set(circuit.discovery_hashes)
    collisions = []
    for text in held_out:
        norm, h = normalize(text), prompt_hash(text)
        if h in discovery:
            collisions.append({"prompt_hash": h, "kind": "exact", "similarity": 1.0})
            continue
        for d_norm in circuit.discovery_normalized:
            sim = similarity(norm, d_norm)
            if sim >= near_threshold:
                collisions.append(
                    {"prompt_hash": h, "kind": "near", "similarity": round(sim, 4),
                     "discovery_hash": prompt_hash(d_norm)}
                )
                break
    if collisions and not allow_overlap:
        raise SplitViolation(
            f"{len(collisions)} validation prompt(s) collide with discovery prompts: {collisions}"
        )
    return {
        "method": "sha256 of normalized text (strip, collapse whitespace, lowercase); "
                  "near match by normalized Levenshtein similarity",
        "near_threshold": near_threshold,
        "n_discovery_prompts": len(circuit.discovery_hashes),
        "overlap_override": bool(allow_overlap),
        "collisions": collisions,
    }
