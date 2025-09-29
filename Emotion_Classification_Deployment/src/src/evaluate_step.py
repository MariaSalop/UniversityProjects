# src/evaluate_step.py
"""CLI entry-point that turns the raw metrics produced during training
into (a) Azure ML run logs and (b) a standalone JSON file the next
pipeline step can consume.

Typical usage inside an AML pipeline:
python evaluate_step.py \
       --metrics_in  outputs/train/metrics.json \
       --metrics_out outputs/eval/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

# Azure ML Run is available *only* when executed inside AML;
# fall back to a stub for local `pytest` and dry-runs.
try:
    from azureml.core import Run  # type: ignore
except ModuleNotFoundError:                      # offline / unit-test
    class _OfflineRun:                            # minimal no-op stub
        id = "OfflineRun"
        @staticmethod
        def log(*_a, **_kw): ...
    Run = _OfflineRun  # type: ignore


def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path) as fp:
        return json.load(fp)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(metrics, fp, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Log and forward evaluation metrics")
    p.add_argument("--metrics_in", type=Path, required=True,
                   help="JSON produced by the training step")
    p.add_argument("--metrics_out", type=Path, required=True,
                   help="Where to write the metrics for downstream steps")
    args = p.parse_args()

    run = Run.get_context()
    metrics = load_metrics(args.metrics_in)

    # Log every scalar metric to the AML run dashboard
    for key, value in metrics.items():
        try:
            run.log(key, float(value))
        except Exception:  # keep pipeline alive even if value is not numeric
            pass

    save_metrics(metrics, args.metrics_out)


if __name__ == "__main__":
    main()
