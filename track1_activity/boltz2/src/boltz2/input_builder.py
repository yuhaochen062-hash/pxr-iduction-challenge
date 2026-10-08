"""Build Boltz-2 YAML inputs for PXR + ligand complexes."""

from pathlib import Path
from typing import Any

import yaml

from .constants import (
    LIGAND_CHAIN_ID,
    MSA_PATH,
    POCKET_MAX_DISTANCE,
    PROTEIN_CHAIN_ID,
    PXR_CORE_POCKET_RESIDUES,
    PXR_SEQUENCE,
)


def build_yaml(
    smiles: str,
    msa_path: Path | None = MSA_PATH,
    use_pocket_constraint: bool = True,
    request_affinity: bool = True,
) -> dict[str, Any]:
    """Build the Boltz-2 YAML input dictionary for one PXR-ligand complex.

    Parameters
    ----------
    smiles
        Ligand SMILES string. Should be a standardised SMILES (e.g. from
        ``compounds.std_smiles`` produced by ChEMBL structure pipeline).
    msa_path
        Path to a precomputed PXR MSA in a3m format. If ``None``, omit the
        MSA field; Boltz can then generate it with ``--use_msa_server``.
    use_pocket_constraint
        Whether to add a soft pocket constraint anchoring the ligand to the
        core PXR pocket residues. ``force=False`` so the constraint is a
        hint, not a hard restraint.
    request_affinity
        Whether to request the Boltz-2 affinity head output.
    """
    protein: dict[str, Any] = {
        "id": PROTEIN_CHAIN_ID,
        "sequence": PXR_SEQUENCE,
    }
    if msa_path is not None:
        protein["msa"] = str(Path(msa_path).resolve())

    yaml_dict: dict[str, Any] = {
        "version": 1,
        "sequences": [
            {"protein": protein},
            {
                "ligand": {
                    "id": LIGAND_CHAIN_ID,
                    "smiles": smiles,
                }
            },
        ],
    }

    if use_pocket_constraint:
        yaml_dict["constraints"] = [
            {
                "pocket": {
                    "binder": LIGAND_CHAIN_ID,
                    "contacts": [
                        [PROTEIN_CHAIN_ID, residue]
                        for residue in PXR_CORE_POCKET_RESIDUES
                    ],
                    "max_distance": POCKET_MAX_DISTANCE,
                    "force": False,
                }
            }
        ]

    if request_affinity:
        yaml_dict["properties"] = [{"affinity": {"binder": LIGAND_CHAIN_ID}}]

    return yaml_dict


def write_yaml(yaml_dict: dict[str, Any], output_path: Path) -> None:
    """Serialise a Boltz-2 YAML dict to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False, default_flow_style=False)
