# CircuitCheck

## Problem

Interpretability work often claims that a small set of features (a "circuit")
explains how a model produces some output. That claim is typically checked on
the same prompt the circuit was found on--evidence that it fits the prompt,
not that it generalizes. CircuitCheck scores a circuit on prompts it has never
seen, so a reported result reflects generalization rather than curve-fitting.

## Method

For a prompt family (e.g. "the capital of France is" / "... Germany is" / ...),
the model is run three ways on each held-out prompt:

1. Unmodified.
2. With everything except the circuit ablated.
3. With everything ablated.

Faithfulness is `F = (circuit - nothing) / (normal - nothing)`: 1 means the
circuit alone reproduces the full behavior, 0 means it does no better than
nothing.

Guarantees:

- Scoring on the discovery prompt (or near-duplicates) is refused unless
  explicitly overridden; overrides are recorded in the output.
- Ablation behavior is verified against the underlying library's actual
  implementation, not just its documentation.
- Scores include error bars from the model's low-precision arithmetic.
  Undefined scores (near-zero denominator) are reported as such rather than
  as a number. Out-of-range scores are kept and flagged, not clipped.
- Reports pin model/library versions, precision settings, and seed.

## Status

Early--one circuit (Gemma-2-2b, found on the France prompt) scored on eight
held-out country prompts: mean F = 0.50, range 0.03–1.83. The high-end score
carries a wide error bar from a small denominator.

Open questions surfaced so far:

- Whether features that activate only as a side effect of ablation should
  count as "outside the circuit" — this choice swung one score from 7.7 to 0.9.
- Mean-ablation is a weak baseline on prompts that share most of their tokens.

## Next

Score circuits against size-matched random baselines so a result is judged
against chance, not read in isolation. Then expand prompt families beyond
single-token substitution.

## Quickstart

Requires an NVIDIA GPU (~22GB) and access to Gemma-2-2b on Hugging Face.

```
# build an attribution graph for the discovery prompt
circuit-tracer attribute -p "Fact: the capital of France is" -m google/gemma-2-2b \
    -t gemma --dtype bf16 --max_feature_nodes 2000 -o france.pt

# score the circuit on held-out prompts
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m circuitcheck.cli validate \
    --circuit france.pt --family fixtures/country_capital.yaml --out report.json
```

`pytest` runs unit tests; `pytest -m gpu` runs the model-dependent ones.

## Example output

`report.json`, trimmed to the headline and one held-out prompt:

```json
{
  "headline": { "f_mean": 0.50, "regime": "frozen", "n_valid": 8, "n_prompts": 8 },
  "per_prompt": [
    {
      "instance_id": "germany",
      "correct": { "token": " Berlin" },
      "foil": { "token": " London" },
      "regimes": {
        "frozen": { "f": 0.03, "valid": true, "warnings": [] }
      }
    }
  ]
}
```

`f` is the per-prompt score; `headline.f_mean` averages it over all valid,
held-out prompts. A prompt scores `valid: false` with no `f` when its
denominator is too small to trust.

Built on [circuit-tracer](https://github.com/safety-research/circuit-tracer).
