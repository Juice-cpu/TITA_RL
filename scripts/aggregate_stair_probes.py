#!/usr/bin/env python3
"""Aggregate strict stair-probe successes across deterministic seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tita_uc_residual_rl.stair_validation import aggregate_strict_probe_results


parser = argparse.ArgumentParser()
parser.add_argument("--inputs", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def main() -> int:
    payloads = [
        json.loads(path.expanduser().read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    result = aggregate_strict_probe_results(payloads)
    encoded = json.dumps(result, indent=2)
    args.output.expanduser().write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["strict_multiseed_candidate_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
