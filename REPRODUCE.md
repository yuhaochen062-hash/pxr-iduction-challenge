# Reproduce The Track 1 Result

This document is the recommended reproduction order for the OpenADMET PXR
Blind Challenge Track 1 result in this repository. It is written for a remote
Linux server with GPUs; the commands are not intended to be run on a laptop.

The reported reference result is:

| Phase | Rank | MAE | RAE | R2 | Spearman | Kendall |
|---|---:|---:|---:|---:|---:|---:|
| Phase 1 | 4 | 0.4059 | 0.5359 | 0.6496 | 0.8343 | 0.6459 |
| Phase 2 | 4 | 0.4113 | 0.5703 | 0.6008 | 0.8161 | 0.6225 |

The repository contains source code and a curated data bundle. It does **not**
contain every runtime artifact used by the final submission: large embedding
tables, model checkpoints, Boltz outputs, the local experiment database, and
submission credentials are intentionally outside Git. Therefore reproduction
has two levels:

1. Rebuild the environment/database and audit the tracked data.
2. Recreate the expensive feature/model artifacts, then replay the final
   ensemble and calibration.

Do not expect the final CSV from a fresh clone until level 2 is complete.

## 1. Server prerequisites

Use a Linux server with a CUDA-capable GPU. The original development setup used
Python 3.12, PostgreSQL 18 with the RDKit cartridge, and an RTX 5080. The
current `pixi.toml` declares CUDA 13.1 and PyTorch GPU dependencies. A newer
driver may report a different CUDA compatibility version; that is not itself a
problem.

```bash
cd ~/projects/pxr-iduction-challenge
git switch reproduce                 # or another non-main working branch
pixi install
pixi run python --version
pixi run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

Keep generated data, checkpoints, embeddings, and submissions outside Git. Do
not recreate or commit `track1_activity/scripts/api.py` or `submit.py`; they
may contain account-specific state.

## 2. Verify the tracked data first

This step is cheap and should be done before any GPU work.

```bash
sha256sum --check data/MANIFEST.sha256
pixi run python - <<'PY'
import pandas as pd
from pathlib import Path

for name in ["default_train.parquet", "default_test.parquet",
             "train_activity_db.parquet", "test_activity_db.parquet"]:
    p = Path("data") / name
    df = pd.read_parquet(p)
    print(f"{p}: {len(df)} rows, columns={len(df.columns)}")
PY
```

The upstream `default_train.parquet` and the historical local database can
have different row counts. When joining the published RDKit descriptor tables,
use `data/train_activity_db.parquet` and `data/test_activity_db.parquet`, which
preserve database activity ordering and `compound_id`.

## 3. Download/refresh the source data

The repository already includes a curated copy. To refresh it from the official
Hugging Face dataset, run:

```bash
pixi run python download_data.py
```

Record the dataset revision/date used for an experiment. Do not mix a fresh
upstream train file with an old database by row position.

## 4. Initialize PostgreSQL + RDKit

The project expects PostgreSQL on port `5433` and a Unix socket under `/tmp`.
The `db/pgdata` directory is local runtime state and is not source code.

```bash
pixi run db-start
pixi run db-status

# Create the database and enable the RDKit cartridge if this is a new cluster.
createdb -h /tmp -p 5433 pxr_challenge 2>/dev/null || true
pixi run db-psql -f /path/to/rdkit-cartridge.sql   # only if your PostgreSQL install requires it

# Apply the repository schema. If the database already has these tables,
# inspect before re-running rather than dropping data.
pixi run db-psql -f db/schema.sql
pixi run db-psql -f db/experiments_schema.sql
pixi run db-psql -f db/lb_submissions_schema.sql
pixi run db-psql -f db/add_std_columns.sql
```

The exact RDKit extension installation command depends on the PostgreSQL
package supplied by the server image. Confirm that `CREATE EXTENSION rdkit`
is available before loading data. The schema uses generated `mol` columns and
GiST molecular indexes, so a plain PostgreSQL instance is insufficient.

Load the source tables once:

```bash
pixi run python db/load_data.py
pixi run python db/standardize_compounds.py
```

The important invariant is that `compounds` has populated `std_smiles` and
`std_mol` before descriptor generation.

Load released labels and Phase 2 HTChem data when those files are available:

```bash
pixi run python db/load_test_activity_phase1_labels.py
pixi run python db/load_htchem_activity.py
```

## 5. Build the deterministic baseline features

Start with the small, reproducible 2D feature tables. Run these from the
repository root and check each command's row count before continuing:

```bash
pixi run install-jazzy
pixi run python db/compute_rdkit_descriptors_full.py
pixi run python db/compute_mordred.py
pixi run python db/compute_jazzy.py
```

Other feature builders are optional until the baseline works:

```bash
pixi run python db/compute_chemeleon.py
pixi run python db/compute_embeddings.py
pixi run python db/compute_chemfm.py
```

The latter commands can require substantial disk, RAM, GPU time, model
downloads, and database storage. Preserve their outputs and document model
versions. Do not recompute Boltz-2 features until the 2D baseline and database
joins are verified.

## 6. Understand the canonical split and feature contract

The production convention is a five-fold UMAP split:

- Morgan radius 2, 2048 bits as the UMAP input;
- Jaccard distance;
- UMAP seed 42;
- 50 KMeans clusters distributed across five folds.

Scaffold split is diagnostic, not the canonical production split. All loaders
must keep database ordering (`ORDER BY t.id`). The split implementation is in
`track1_activity/src/splits.py`; feature names and database mappings are in
`track1_activity/src/features.py` and `track1_activity/scripts/run_train.py`.

The important low-fidelity signal is **predicted** `log2_fc` at 8.25 and 33
microM. It is generated from SMILES by a ChemProp model and passed as two
features to a downstream pEC50 learner. It is not the final pEC50 target.

## 7. Recreate low-fidelity predictions and embeddings

For the final family, train ChemProp on the available single-concentration
`log2_fc` data, then predict both concentrations for train and test compounds.
The relevant entry points include:

```bash
pixi run python track1_activity/scripts/run_chemprop_pretrain.py --help
pixi run python track1_activity/scripts/run_chemprop_pretrain_optuna.py --help
pixi run python track1_activity/scripts/run_chemprop_predict_log2fc.py --help
```

Use the recorded experiment configuration/checkpoint names in
`docs/track1_explain/model_inventory.md` and the research logs (GitHub issues
#100 and #208) when reproducing a specific historical member. The production
recipe uses multi-seed/pretrained low-fidelity models and frozen embeddings;
do not silently substitute a direct pEC50 fine-tune.

The final pool also includes frozen ChemProp/other encoder embeddings and
Boltz-2 trunk representations. Those artifacts must be generated separately
and registered in PostgreSQL before a member can be trained. For Boltz-2, use
the resume-oriented scripts under `track1_activity/boltz2/scripts/`; full
coverage is multi-day work and should be run as a resumable job.

## 8. Train model members

Use `run_train.py` for a controlled single member. Always specify the split and
seed-sensitive settings explicitly, and keep the stdout/configuration with the
experiment record:

```bash
pixi run python track1_activity/scripts/run_train.py \
  --model tabpfn \
  --feature <feature-name> \
  --split umap \
  --umap-seed 42 \
  --umap-clusters 50 \
  --trials 0
```

Inspect supported feature names before launching a long job:

```bash
pixi run python track1_activity/scripts/run_train.py --help
```

Each intended ensemble member must write aligned OOF predictions and test
predictions and register an experiment in the database. A missing OOF row,
different row order, or a mixed split invalidates the ensemble comparison.

`run_all_models.sh` is a historical GPU pipeline, not a shortcut to the final
2026 ensemble. Use it only after checking its scripts and trial counts against
the target experiment record:

```bash
bash track1_activity/scripts/run_all_models.sh
```

## 9. Rebuild the canonical ensemble

The authoritative member allow-list is the `ENSEMBLE_MODELS` tuple in
`track1_activity/scripts/run_ensemble.py`. Do not replace it with an automatic
query over every experiment; stale or mixed-split experiments can contaminate
the pool.

Once all listed members exist in PostgreSQL with aligned OOF/test outputs:

```bash
pixi run python track1_activity/scripts/run_ensemble.py
```

The script writes several candidate submissions, including
`ens_caruana_bag20.csv`, and records their OOF metrics. The production default
is the Caruana forward-selection ensemble with bagging (`caruana_bag20`), but
inspect the printed strategy table rather than assuming the lowest OOF score
will win the public leaderboard.

Run both post-hoc calibrators after a material pool change:

```bash
pixi run python track1_activity/scripts/run_ensemble_calibrate.py
pixi run python track1_activity/scripts/run_ensemble_calibrate_importance.py
```

## 10. Validate locally and compare with released labels

Before treating a CSV as a candidate, compare it with the trusted anchor and
inspect distribution/shift diagnostics:

```bash
pixi run python track1_activity/scripts/submission_preflight.py \
  --candidate track1_activity/submissions/<candidate>.csv \
  --anchor track1_activity/submissions/ens_caruana_bag20.csv
```

Use Phase 1/Phase 2 unblinded files only for answer-checks and validation
design. They cover released subsets of the blinded test set; they are not a
replacement for the original blind evaluation. Recompute MAE, RAE, R2,
Spearman, and Kendall with the repository evaluation helpers and record the
exact subset and label source.

Do not submit a new variant solely because OOF improved by a tiny amount. The
project history contains repeated OOF/public-LB reversals, especially for
correlated ensemble additions.

## 11. Submission (only if needed)

Submission clients are local and ignored. Recreate them from the challenge
client instructions, authenticate on the remote server, run the preflight
check, and submit the final CSV through the local client. Never commit tokens
or `api.py`/`submit.py`.

Keep a timestamped note containing: candidate filename, source experiment IDs,
ensemble weights, calibration method, preflight result, and submission notes.
For retrospective leaderboard work, consult the latest files under
`docs/leaderboards/activity/` and the `lb_submissions` database tables.

## 12. First ablation: remove predicted `log2_fc`

After the reference path is reproduced, run the ablation as a controlled SWAP:

1. Keep the same UMAP folds, random seeds, TabPFN version, and training rows.
2. Remove only the two predicted `log2_fc` columns from the tabular feature
   matrix.
3. Train the same member(s), saving new experiment names and OOF predictions.
4. Compare OOF metrics, prediction correlation, and Phase 1/AS1 metrics.
5. Rebuild a separate ensemble pool; never overwrite the canonical allow-list.

The repository includes a focused entry point for this direction:

```bash
pixi run python track1_activity/scripts/run_admet_ai_tabpfn_no_log2fc_top500.py --help
```

For a general member, implement the feature exclusion in the feature assembly
path used by `run_train.py`, then verify that the saved feature count and
experiment metadata explicitly say `no_log2fc`. A valid ablation changes one
factor at a time and preserves the exact train/test compound ordering.

## 13. Stop conditions and expected outputs

You have a useful reproduction checkpoint when:

- the database has deterministic train/test rows and descriptors;
- at least one UMAP-split TabPFN baseline writes OOF/test predictions;
- `experiments` and `experiment_oof_predictions` contain those outputs;
- `run_ensemble.py` can load every member in its allow-list;
- `ens_caruana_bag20.csv` and calibration outputs pass preflight;
- metrics and artifacts are recorded with versions and commands.

Do not delete ignored runtime directories to save space without checking disk
usage and confirming that no checkpoint or embedding is still needed. The
repository's `AGENTS.md` is the durable operating guide; GitHub issues #100 and
#208 contain the detailed historical decisions behind the final pool.
