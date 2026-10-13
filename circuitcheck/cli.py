"""circuitcheck validate --circuit graph.pt --family family.yaml --out report.json"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="circuitcheck")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="Score a circuit on held-out prompts from a family.")
    v.add_argument("--circuit", required=True, help="circuit-tracer graph .pt file")
    v.add_argument("--family", required=True, help="prompt family YAML")
    v.add_argument("--out", required=True, help="report JSON path")
    v.add_argument("--discovery-prompt", action="append", default=[],
                   help="extra discovery prompt (the graph prompt is always included)")
    v.add_argument("--node-threshold", type=float, default=0.8)
    v.add_argument("--top-k", type=int, default=None)
    v.add_argument("--ablation", choices=["mean", "zero"], default="mean")
    v.add_argument("--n-prompts", type=int, default=None)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--allow-overlap", action="store_true")
    args = parser.parse_args(argv)

    from circuitcheck.adapters.circuit_tracer import CircuitTracerAdapter
    from circuitcheck.types import PromptFamily
    from circuitcheck.validate import validate

    adapter = CircuitTracerAdapter.load(seed=args.seed)
    family = PromptFamily.from_yaml(args.family)
    circuit = adapter.load_circuit(args.circuit, family, args.discovery_prompt,
                                   node_threshold=args.node_threshold, top_k=args.top_k)
    report = validate(circuit, family, adapter=adapter, ablation=args.ablation,
                      n_prompts=args.n_prompts, seed=args.seed, allow_overlap=args.allow_overlap)
    report.to_json(args.out)
    h = report.data["headline"]
    print(f"F (frozen, headline) = {h['f_mean']} on {h['n_valid']}/{h['n_prompts']} valid prompts; "
          f"warnings: {report.data['warnings']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
