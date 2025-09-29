"""
Stub-step for model registration.
Writes a JSON with a note, so that the pipeline can proceed.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Stub model registration")
    p.add_argument("--model_folder", type=str, required=True,
                   help="Path to retrained model folder")
    p.add_argument("--eval_metrics", type=str, required=True,
                   help="Path to evaluation metrics JSON")
    p.add_argument("--registry_out", type=str, required=True,
                   help="Where to write registry_info.json")
    args = p.parse_args()

    # Just create an output file so the pipeline doesn't crash
    out_path = Path(args.registry_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "stub registration successful"}, open(out_path, "w"))


if __name__ == "__main__":
    main()
