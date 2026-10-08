"""Reuse one MSA-server result for every PXR input.

Run one small Boltz prediction with ``--use_msa_server`` first. Boltz writes
the generated protein MSA as ``<output>/boltz_results_<input>/msa/*_A.csv``.
This script copies that CSV to a stable path and rewrites all input YAMLs to
reference it. Subsequent predictions can run without ``--use_msa_server``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def find_server_msa(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("boltz_results_*/msa/*_A.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No generated MSA CSV under {output_dir}/boltz_results_*/msa/"
        )
    return candidates[0]


def rewrite_inputs(input_dir: Path, shared_msa: Path) -> int:
    count = 0
    for path in sorted(input_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        changed = False
        for entry in data.get("sequences", []):
            protein = entry.get("protein")
            if protein is not None:
                if protein.get("msa") != str(shared_msa):
                    protein["msa"] = str(shared_msa)
                    changed = True
        if changed:
            path.write_text(yaml.safe_dump(data, sort_keys=False))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--shared-msa",
        type=Path,
        default=Path("structures/boltz2/msa/pxr_server.csv"),
    )
    args = parser.parse_args()

    source = find_server_msa(args.server_output)
    args.shared_msa.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.shared_msa)
    n = rewrite_inputs(args.inputs, args.shared_msa.resolve())
    print(f"source MSA: {source}")
    print(f"shared MSA: {args.shared_msa.resolve()}")
    print(f"rewrote YAML files: {n}")


if __name__ == "__main__":
    main()
