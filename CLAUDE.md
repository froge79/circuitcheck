# circuitcheck

Takes a claimed circuit from a transcoder-based attribution graph and reports whether
it survives on prompts it was not discovered on.

Read `docs/DESIGN.md` before any non-trivial change. It is the spec; this file is the
set of rules that are easy to violate without noticing.

Model: Gemma-2-2b only. Upstream: `circuit-tracer`.

## Hard invariants

- **No absolute token indices, anywhere.** Positions are `Anchor`s: a span name plus
  offset, or a negative offset from the end. If you are about to write `tokens[7]` or
  store a bare `position: int` on a node, stop and raise it with me.
- **Only `adapters/` imports `circuit_tracer`** or touches model internals. Everything
  above it speaks core types. This boundary is the whole point of the layout.
- **Faithfulness scores are never clipped to [0, 1].** Values outside that range are
  real signal, usually a bug or a bad baseline. Clipping hides it.
- **Error nodes, embedding nodes and token nodes are always retained** during ablation
  and never counted toward the node budget. Record this in the report.
- **No public function scores a circuit on caller-supplied prompts.** The only scoring
  entry point is `validate(circuit, family)`, which draws held-out prompts itself.
- **`allow_overlap=True` writes a flag into the report** that the caller cannot remove.
  A check a user can quietly disable is decoration.
- **Every run reproduces from its report**: seed, perturbation suite version, ablation
  mode, gate threshold, model and transcoder revision, library versions.

## Out of scope — do not build these

UI or rendering of any kind, a second model family, distractor insertion, template
change, translation, CSV export, a one-page report generator, large-scale sweeps.

If a task seems to need one of these, say so rather than building it.

## Working rules

- Ask before adding a dependency.
- Tests alongside code, pytest. For anything numeric, write the invariant tests first
  (F = 1 when the circuit is the whole graph, F = 0 when it is empty).
- Do not create modules that are not in the module layout in `docs/DESIGN.md`.
- When an external API does not do what the design assumed, report that and stop.
  Never stub around it and carry on — the stub will be wrong in a way I cannot see.

## Exception

`spike.py` is exempt from all of the above. It is a throwaway probe of the
`circuit-tracer` API and is deleted at the end of session 1.