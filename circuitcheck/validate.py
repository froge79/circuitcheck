"""validate(circuit, family): the only scoring entry point. It draws its own held-out prompts."""

from __future__ import annotations

import random
import statistics
from datetime import datetime, timezone
from typing import Any

from circuitcheck import score as score_mod
from circuitcheck.adapters.circuit_tracer import (
    REGIMES,
    CircuitTracerAdapter,
    IncompatibleCircuit,
)
from circuitcheck.split import check_split, prompt_hash
from circuitcheck.types import Circuit, PromptFamily, Report, SpanError, span_layout

HEADLINE_FREEZE = True


def _draw(circuit: Circuit, family: PromptFamily, n_prompts: int | None, seed: int):
    prompts = family.prompts()
    discovery = set(circuit.discovery_hashes)
    held_out = [p for p in prompts if prompt_hash(p.text) not in discovery]
    n_excluded = len(prompts) - len(held_out)
    random.Random(seed).shuffle(held_out)
    if n_prompts is not None:
        held_out = held_out[:n_prompts]
    return held_out, n_excluded


def _check_consistent(adapter: CircuitTracerAdapter, start: int, prompt_hash_: str) -> None:
    calls = adapter.call_log[start:]
    keys = {(c.prompt_hash, c.tokens_sha, c.correct_id, c.foil_id) for c in calls}
    if len(keys) != 1 or next(iter(keys))[0] != prompt_hash_:
        raise RuntimeError(f"conditions for prompt {prompt_hash_} used inconsistent inputs: {keys}")


def validate(
    circuit: Circuit,
    family: PromptFamily,
    *,
    adapter: CircuitTracerAdapter | None = None,
    ablation: str = "mean",
    n_prompts: int | None = None,
    seed: int = 0,
    allow_overlap: bool = False,
    near_threshold: float = 0.95,
    k_min: float = score_mod.DEFAULT_K_MIN,
    f_tol: float = score_mod.DEFAULT_F_TOL,
    weak_rel: float = score_mod.DEFAULT_WEAK_REL,
) -> Report:
    if ablation not in ("mean", "zero"):
        raise IncompatibleCircuit(f"unknown ablation mode {ablation!r}; use 'mean' or 'zero'")
    held_out, n_excluded = _draw(circuit, family, n_prompts, seed)
    if not held_out:
        raise IncompatibleCircuit(f"family {family.name!r} has no prompts outside the discovery set")
    split = check_split([p.text for p in held_out], circuit, near_threshold, allow_overlap)
    split["excluded_discovery_instances"] = n_excluded

    if adapter is None:
        adapter = CircuitTracerAdapter.load(seed=seed)
    adapter.check_compatible(circuit)

    preps = [adapter.prepare(p) for p in held_out]
    layouts = {span_layout(p.span_map) for p in preps}
    if len(layouts) != 1:
        raise IncompatibleCircuit(
            f"held-out prompts have different span layouts {sorted(layouts)}; anchor-aligned "
            "ablation values need equal token counts per span in this milestone"
        )
    try:
        keeps = [adapter.keep_mask(p, circuit.nodes) for p in preps]
    except SpanError as e:
        raise IncompatibleCircuit(f"circuit anchors do not resolve on family {family.name!r}: {e}")
    values = adapter.ablation_values(preps, ablation)
    empty_keep = keeps[0] & False
    empty_list = adapter.complement_interventions(empty_keep, values)

    per_prompt = []
    for prep, keep in zip(preps, keeps):
        start = len(adapter.call_log)
        acts = adapter.dense_activations(prep)
        n_active = int(sum(
            bool(acts[n.layer, n.anchor.resolve(prep.span_map), n.feature] > 0) for n in circuit.nodes
        ))
        del acts
        c_list = adapter.complement_interventions(keep, values)
        full = adapter.measure(prep, [], freeze=HEADLINE_FREEZE)

        regimes = {}
        for freeze in (True, False):
            empty = adapter.measure(prep, empty_list, freeze=freeze)
            m_c = adapter.measure(prep, c_list, freeze=freeze)
            mag = max(abs(x) for m in (full, empty, m_c) for x in (m.logit_correct, m.logit_foil))
            s = score_mod.faithfulness(full.ld, m_c.ld, empty.ld,
                                       score_mod.resolution(adapter.eps, mag), k_min, f_tol, weak_rel)
            if full.ld <= 0 or not full.correct_is_argmax:
                s.warnings.append("TASK_NOT_PRESERVED")
            regimes["frozen" if freeze else "unfrozen"] = {
                **s.to_dict(),
                "logits": {
                    "full": [full.logit_correct, full.logit_foil],
                    "C": [m_c.logit_correct, m_c.logit_foil],
                    "empty": [empty.logit_correct, empty.logit_foil],
                },
                "n_intervened": {"C": m_c.n_entries, "empty": empty.n_entries},
            }
        _check_consistent(adapter, start, prep.prompt_hash)
        per_prompt.append({
            "instance_id": prep.prompt.instance_id,
            "prompt_hash": prep.prompt_hash,
            "correct": {"token": prep.prompt.correct, "id": prep.correct_id},
            "foil": {"token": prep.prompt.foil, "id": prep.foil_id},
            "n_tokens": prep.n_pos,
            "coverage": {"active_circuit_nodes": n_active, "circuit_nodes": len(circuit.nodes),
                         "fraction": n_active / len(circuit.nodes) if circuit.nodes else None},
            "full_correct_is_argmax": full.correct_is_argmax,
            "regimes": regimes,
        })

    summary = {}
    for name in ("frozen", "unfrozen"):
        fs = [p["regimes"][name]["f"] for p in per_prompt if p["regimes"][name]["valid"]]
        summary[name] = {
            "regime": REGIMES[name == "frozen"],
            "role": "headline" if (name == "frozen") == HEADLINE_FREEZE else "secondary",
            "n_prompts": len(per_prompt),
            "n_valid": len(fs),
            "mean_f": statistics.fmean(fs) if fs else None,
            "f_per_prompt": [p["regimes"][name]["f"] for p in per_prompt],
        }

    warnings = sorted({w for p in per_prompt for r in p["regimes"].values() for w in r["warnings"]})
    if split["overlap_override"]:
        warnings.insert(0, "OVERLAP_OVERRIDE")
    return Report({
        "circuitcheck_report": "vertical-slice-0",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headline": {"f_mean": summary["frozen"]["mean_f"], "regime": REGIMES[HEADLINE_FREEZE],
                     "n_valid": summary["frozen"]["n_valid"], "n_prompts": len(per_prompt)},
        "overlap_override": split["overlap_override"],
        "warnings": warnings,
        "summary": summary,
        "per_prompt": per_prompt,
        "config": {
            "family": family.name,
            "template": family.template,
            "ablation": ablation,
            "ablation_values": "mean over the drawn held-out prompts, per (layer, anchor, feature), "
                               "inactive counted as zero" if ablation == "mean" else "zero",
            "complement": "dense: every (layer, position >= 1, feature) outside the circuit "
                          "is set to its ablation value; BOS position excluded",
            "m_empty": "every (layer, position >= 1, feature) set to its ablation value",
            "m_full": "empty intervention list (unmodified replacement-model forward pass)",
            "circuit_features": "not clamped: recomputed live from the perturbed input",
            "metric": "logit(correct) - logit(foil) at the last position",
            "n_prompts_requested": n_prompts,
            "seed": seed,
            "score_policy": {"k_min": k_min, "f_tol": f_tol, "weak_rel": weak_rel,
                             "resolution": "dtype spacing at the largest |logit| used",
                             "clipping": "never"},
        },
        "split": split,
        "circuit": circuit.describe(),
        "provenance": adapter.provenance(),
        "null": "not computed in this milestone",
    })
