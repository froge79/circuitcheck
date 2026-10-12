# Attribution Graph Validation Harness — Final Design

Oct 12, 2026 · @Samuel

## Design lock

The tool takes one claimed circuit and one prompt family, and returns a faithfulness score on prompts the circuit was never discovered on, next to a size-matched null it has to beat.

Everything below is settled. Changes from here are implementation detail, not redesign.

| Question | Settled as |
| --- | --- |
| What the tool outputs | A JSON report with one headline number, its null distribution, and a full provenance block |
| How positions transfer across prompts | Circuit nodes are anchored to named template spans, never absolute token indices |
| What enforces the split | Validation prompts are hashed against discovery prompts and the run raises on collision; the override is recorded in the report |
| Ablation style | Mean-ablation over the validation family's own distribution, zero-ablation behind a flag |
| Error and token nodes | Always retained, never counted toward node budget, always logged |
| Null sampling | Matched on node count and attribution mass, layer-stratified by default |
| First perturbation family | Entity swap, because it is span-native; paraphrase ships second |
| Packaging | Separate package from the workbench, sharing the perturbation and gate layer |

The contribution is measurement, not rendering. No UI ships, at any point.

## Core objects

Four objects, each carrying one invariant the rest of the system can assume. The hard one is `Anchor`: an attribution graph is a per-prompt artifact, so a node discovered at token 7 of one prompt has no meaning at token 7 of another.

| Object | Holds | Invariant it enforces |
| --- | --- | --- |
| `Anchor` | A span name plus offset, or a negative offset from the end | No absolute token index survives construction, so a circuit is portable by default |
| `Circuit` | Nodes as (layer, feature index, anchor), source graph metadata, model and transcoder revision, and a frozen hash list of its discovery prompts | Discovery provenance cannot be edited after construction |
| `PromptFamily` | A template with named spans, filler sets, and a correct/foil token pair per instance | Every prompt it emits carries its own span map, so anchors always resolve |
| `Report` | Scores, the full null distribution, per-prompt gate outcomes, and a provenance block | Serializes to JSON that reproduces the run |

`Anchor` is what makes the rest work. Entity swap is span-native — refill a slot, spans move with it. Paraphrase must emit a span alignment alongside the new text or the prompt is rejected before scoring.

A circuit is read in through an adapter, never constructed by hand in a user's script. That keeps the discovery record honest.

## Public API and the enforced split

There is no public call that scores a circuit on prompts the caller chose. The only scoring entry point takes a family and draws held-out prompts from it.

```python
from circuitcheck import load_circuit, PromptFamily, validate

circuit = load_circuit("graph.pt", discovery_prompts=[...])
family  = PromptFamily.from_yaml("country_capital.yaml")

report = validate(circuit, family, perturbations=["entity_swap"], seed=0)
report.to_json("report.json")
```

Three layers of defence, in order:

1. Exact match. Validation prompts are normalized — stripped, whitespace-collapsed, lowercased — then hashed against the discovery record. A collision raises.
2. Near match. Any validation prompt within 0.95 normalized edit similarity of a discovery prompt also raises, since whitespace alone is trivial to vary.
3. Recorded override. `allow_overlap=True` exists, and setting it writes a flag into the report that cannot be removed by the caller.

The third point is the one that matters. A check a user can quietly disable is decoration; a check whose bypass prints itself in the output is a constraint.

The CLI is one command, because the success condition is someone else running this.

```
circuitcheck validate --circuit circuit.json --family family.yaml --out report.json
```

## The faithfulness metric

The headline number is normalized recovered logit difference on held-out prompts, at the circuit's own node count. The task metric is the logit of the correct token minus the logit of the foil, which entity-swap families supply for free.

```latex
F(C, x) = \frac{m_C(x) - m_\emptyset(x)}{m_{\text{full}}(x) - m_\emptyset(x)}
```

Here `m_C` runs the model with every feature outside the circuit mean-ablated, `m_full` is the unmodified model, and `m_empty` is the same family with all candidate features ablated. A score of 1 means the circuit reproduces the behaviour; 0 means it does no better than nothing.

Three reporting rules, each guarding against a way this number can mislead:

- Scores are never clipped to \[0, 1\]. Values above 1 or below 0 are real and usually mean a bug or a bad baseline, and hiding them hides the bug.
- Error nodes, embedding nodes and token nodes are always retained and never counted in the node budget. Ablating reconstruction error destroys the model, so retaining it is the only sane choice, but it inflates F and must be stated.
- Per-prompt scores ship alongside the mean. Aggregate faithfulness is known to mask per-input failure, so the distribution is part of the output, not a diagnostic extra.

Because the error-node convention is not standardized across the field, the report states it explicitly and the docs say these numbers are not comparable to published F values from other tools.

## The size-matched null

A claimed circuit has to beat a random subgraph drawn from the same attribution graph and matched on the things that would otherwise explain its score. One hundred null samples per run, seeded and logged.

Matching criteria, in the order they constrain the draw:

| Criterion | Setting | Why |
| --- | --- | --- |
| Node count | Exact match to the claimed circuit | The obvious confound: bigger subgraphs recover more |
| Attribution mass | Within 10 percent of the claimed circuit's total | A null of high-mass nodes is the honest comparison |
| Layer histogram | Stratified to match, on by default | An unstratified null drawn mostly from early layers is beatable by anything |

Layer stratification is the deviation worth defending. It is stricter than a plain random baseline, and it is the difference between a null that makes every circuit look significant and one that can actually fail a bad circuit. The tool's credibility rests on the null being hard.

The report carries the full null distribution, not only a z-score. A degenerate null — all 100 samples scoring near zero, or near the circuit — is the user's signal that the comparison was uninformative, and they can only see that if the numbers are there.

Headline significance is reported as the circuit's percentile against the null, plus the gap between its score and the null median.

## Perturbation suites and the gate

Two families ship, generated and versioned rather than hand-picked, so a run can be re-created from its report. Each suite has a version string that changes whenever its generator changes, and that string lands in the provenance block.

**Entity swap** refills the template's named slots from the family's filler sets. Spans move with the fillers, so anchors resolve with no extra work. This family ships first for exactly that reason.

**Paraphrase** rewrites the template around the slots and must emit a span alignment with the new text. A paraphrase that cannot say where its spans went is rejected at generation time, not silently scored. This is the harder family and it ships second.

The gate runs before any prompt is scored. A prompt must pass all three checks or it is excluded and logged with its failure reason:

1. Task preserved. The correct token is still argmax, and the logit difference is at least half the family's median on its seed prompts.
2. Spans resolve. Every anchor in the circuit maps to a real position in the perturbed prompt.
3. Length sanity. The perturbed prompt tokenizes within a configured factor of the seed, catching generators that run away.

The rejection rate is a headline number in the report, not a footnote. A suite that drops most of its prompts has told you its generator is broken, and a tool that buries that is reporting faithfulness on a sample it quietly chose.

This is the mitigation for the counterfactual-generation problem, not a solution to it. If a paraphrase changes the task in a way the gate cannot detect, the resulting instability is still blamed on the circuit.

## Module layout

One file imports `circuit_tracer`. Everything above it speaks core types, so an upstream API change costs one file rather than a rewrite, which is the only real answer to the external-dependency risk.

&#91;embedded content: module layout · 4 layers, one dependency boundary\]

The perturbation module never touches the model, which makes it the piece the workbench can share. Gate, null and score all need interventions, so all three sit on the adapter.

## Carried risks and deferred decisions

The main scientific risk is unchanged and now has a concrete shape: span alignment is where counterfactual generation fails. The gate catches paraphrases that break the task, not paraphrases that preserve the task while moving the computation.

| Risk | Carried as | Fallback |
| --- | --- | --- |
| Counterfactual generation is an open problem | Paraphrase ships second and can be cut | Entity-swap-only release, which is still an enforced split nobody has shipped |
| Model coverage bounded by public transcoder suites | Gemma-2-2b only, adapter isolated to one file | None needed; the boundary is the point |
| Null may be too easy or too hard | Full distribution reported, not just a z-score | Users can re-stratify; the tool does not hide the choice |
| Scores not comparable to published numbers | Error-node convention stated in every report | Documented as a limitation, not papered over |
| Adoption friction from breaking people's circuits | Framed as infrastructure, results reported on own circuits first | Lead the writeup with the tool, not the failures |

Deferred past v0.1, deliberately: distractor insertion, template change, translation, a second model family, CSV formatting, and the one-page report generator. Each is a nice-to-have that does not change whether a researcher can run the thing.

One decision stays open and is yours to make rather than mine: whether a circuit that fails the null gets a loud verdict in the report, or just numbers. A verdict is more useful and more likely to be argued with.
