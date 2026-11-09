# MetCompose

MetCompose prioritises compatible graph-edit combinations for metabolic reaction product prediction.

This repository contains the original flat-token graph-edit model, its separately fitted pair-compatibility scorer, a CPU training/inference entry point, one internal holdout checkpoint, frozen split identifiers and numerical results. It retains the edit-count prior used in the manuscript. It does not contain an end-to-end higher-order neural compatibility model.

## Install

The release was checked in the Python environment recorded in `environment.json`. Create a dedicated Python environment and install:

```bash
python -m pip install -r requirements.txt
```

## Predict

```bash
python metcompose.py predict \
  --checkpoint checkpoints/fold0_seed17 \
  --smiles 'NC(=O)C1CCCCC1' \
  --output prediction.json
```

The checkpoint is one original internal combination-holdout fit (fold 0, seed 17), not the frozen model used for external-transfer figures. The pair scorer was reconstructed using the original training fold and deterministic fitting procedure. Only load checkpoints you trust: PyTorch and pickle files can execute code when loaded.

Outputs contain ranked unique product SMILES and the edit programs that generated them. The search considers up to 64 local actions and three edits per program, attempts at most 128 executions and retains at most 64 products. Scores are search priorities, not calibrated probabilities of metabolite formation or therapeutic benefit. EC defaults to the unknown category.

## Train on prepared records

```bash
python metcompose.py train --data train.jsonl --output trained_model --seed 17 --epochs 12
```

Each JSONL record must contain `parent` (canonical substrate SMILES), `product`, `actions`, and `primitive_program`; optional fields include `reaction_id` and `ec3`. Each action is `[kind, i, j, z1, z2, old, new]`, where `kind` is `B`, `ADD`, `DEL` or `Q`, indices refer to the substrate graph, and `z1/z2` are element identifiers. `primitive_program` is the multiset of operation tokens `[kind, z1, z2, old, new]`. Input records must pass exact execution/reconstruction checks before training. See `chemistry.extract` and `chemistry.execute`.

The neural objective combines separately averaged positive/negative action softplus terms and count cross entropy weighted by 0.3. Pair compatibility uses 103 symmetric features and standardised class-balanced logistic regression (C=0.1). This classifier is fitted separately, not through the neural training loss.

## Reproduction scope

- Root Python files: original implementation modules. Use the release entry point above; legacy module `main` functions retain historical experiment paths.
- `checkpoints/fold0_seed17`: one stored neural checkpoint and reconstructed pair scorer.
- `manifests`: train/test record identifiers and SHA-256 fingerprints for three historical folds. These are identifiers, not redistributed full reaction datasets.
- `results`: substrate/seed/control ranks and aggregate internal results underlying the combination analysis.
- `experiments`: archived original replication/aggregation scripts for provenance. These require the original experiment directory structure and data; they are not standalone release commands.
- `scripts/verify_release.py`: validates included file hashes, executes inference, and compares ranks on a synthetic chemical input with the original implementation during release testing.

The 133-substrate, three-fold, five-seed internal analysis reports mean Top-10 recall of 9.02% for the full model and 4.66% for hard filtering. Locality controls remain competitive. External benchmark models have unequal training histories; these comparisons are not evidence of matched architectural superiority. The selected chemical case examples are illustrative and do not provide experimental validation.

## Data access

Full EnzymeMap, MetXBioDB, GLORYx and Rhea reaction exports are not bundled in this release. Obtain upstream datasets under their respective terms. The frozen manifests identify the processed records used here. Reconstructing an entire published evaluation additionally requires the original processed reaction records, provenance reconciliation and external baseline environments. This initial code release must not be described as an independently reproduced, complete benchmark archive.

## Authors

Bingxue Du, Bei Zhu, Zhenyi Wu, Pengzhen Hu, Hui Yu and Jian-Yu Shi.

Correspondence: huiyu@nwpu.edu.cn; jianyushi@nwpu.edu.cn.

## Citation and licensing

The manuscript is in preparation; no publication DOI is assigned here. No third-party baseline source code or proprietary dataset export is included. A software reuse licence has not yet been selected; public availability alone does not grant an open-source licence.
