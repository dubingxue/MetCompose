# Reproducibility

## Environment

The tested environment is recorded in `environment.json`. Install the pinned dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Input records

Training accepts JSONL records with the following fields:

| Field | Description |
| --- | --- |
| `parent` | Canonical substrate SMILES |
| `product` | Reference product SMILES |
| `actions` | Reference graph edits |
| `primitive_program` | Multiset of operation tokens |
| `reaction_id` | Optional record identifier |
| `ec3` | Optional EC annotation; defaults to unknown |

Each action has the form `[kind, i, j, z1, z2, old, new]`. `kind` is `B`, `ADD`, `DEL` or `Q`; atom indices refer to the substrate graph. The corresponding operation token is `[kind, z1, z2, old, new]`. Reference actions must reconstruct the product exactly using `chemistry.execute`. Edit extraction is implemented in `chemistry.extract`.

## Training

```bash
python metcompose.py train --data train.jsonl --output trained_model --seed 17 --epochs 12
```

The neural objective combines separately averaged positive and negative action losses with edit-count cross entropy weighted by 0.3. Pair compatibility uses 103 symmetric features and a separately fitted, standardised, class-balanced logistic regression model with C=0.1.

## Checkpoint and prediction

```bash
python metcompose.py predict \
  --checkpoint checkpoints/fold0_seed17 \
  --smiles 'NC(=O)C1CCCCC1' \
  --output prediction.json
```

The included neural checkpoint is from internal combination-holdout fold 0, seed 17. Its pair scorer was reconstructed from the same training fold using the original deterministic procedure. External-transfer figures used different fits.

Prediction considers up to 64 local actions and three edits per program, attempts at most 128 executions and returns at most 64 unique products. Each product is accompanied by its generating edits. Search scores are not calibrated probabilities of metabolite formation. Checkpoints use PyTorch and pickle serialization and should only be loaded from trusted sources.

## Verification

```bash
python scripts/verify_release.py
```

This checks the SHA-256 file manifest and compares inference with the stored reference output. The reference output was checked against the original implementation: all 64 candidate products and edit programs matched.

## Evaluation files

- `manifests/`: record identifiers and hashes for the three train/test partitions.
- `results/`: internal combination-evaluation ranks and aggregate metrics.
- `experiments/`: original replication and aggregation scripts, which require the historical experiment directory structure and processed records.

The internal evaluation covers 133 substrates, three folds and five seeds. Mean Top-10 recall is 9.02% for MetCompose and 4.66% for hard filtering. Locality controls remain competitive. External comparisons use models with different training histories.

## Data access

Full EnzymeMap, MetXBioDB, GLORYx and Rhea exports are not included. Obtain these datasets under their respective terms. The manifests identify processed records but do not replace the reaction tables. Reproducing every reported evaluation also requires those tables, provenance reconciliation and the external baseline environments.
