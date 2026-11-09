# MetCompose: Compatibility-guided graph-edit composition for metabolic reaction product prediction

MetCompose combines local graph-edit predictions with learned pair compatibility to generate and rank candidate metabolic products. The original model includes a two-layer graph encoder, a separate logistic pair scorer and a substrate-conditioned edit-count prior.

# Requirements

- Python 3.10 (see `environment.json` for the tested version)
- PyTorch == 2.2.2
- scikit-learn == 1.4.2
- NumPy == 1.26.4
- RDKit == 2023.09.6

```bash
pip install -r requirements.txt
```

# Dataset

The study uses curated enzymatic reactions and external metabolic reaction records from EnzymeMap, MetXBioDB, GLORYx and Rhea. Frozen train/test identifiers and record hashes are provided in `manifests/`; internal combination-evaluation ranks and aggregate metrics are in `results/`.

Full third-party reaction exports are not included. Obtain upstream data under their respective terms. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for input schema, data requirements and the limits of this initial release.

# Model and training

- `model.py`: molecular graph encoder, local action scorer and count head.
- `chemistry.py`: action vocabulary, reaction edit extraction and product execution.
- `compatibility.py`: symmetric edit-pair features, compatibility fitting and search.
- `train.py`: original neural training utilities.
- `metcompose.py`: release command-line entry point.

```bash
python metcompose.py train --data train.jsonl --output trained_model --seed 17 --epochs 12
```

Training records must contain canonical substrate/product SMILES, executable reference actions and primitive edit multisets. The pair scorer is fitted separately from the graph encoder.

# Prediction

```bash
python metcompose.py predict \
  --checkpoint checkpoints/fold0_seed17 \
  --smiles 'NC(=O)C1CCCCC1' \
  --output prediction.json
```

One original internal holdout checkpoint is included (fold 0, seed 17). Its pair scorer was reconstructed deterministically from the original training fold. This is not the external-transfer checkpoint. Only load trusted checkpoint files.

The output contains at most 64 unique products and their generating edits. Predictions do not establish enzyme mechanisms or therapeutic efficacy.

# Verification

```bash
python scripts/verify_release.py
```

The release prediction fixture was checked against the original implementation: all 64 candidate products and edit programs matched. The verifier checks file hashes and deterministic inference. Archived experiment scripts in `experiments/` require the original data layout.
