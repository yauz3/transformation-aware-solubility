# Transformation-Aware Solubility

Transformation-aware ensemble modeling for temperature-dependent solubility prediction under solute–solvent generalization constraints.

This repository contains the data-processing scripts, molecular feature-generation workflow, leakage-aware splitting protocol, model-training pipeline, and transformation-aware ensemble analysis used for temperature-dependent molecular solubility prediction.

## Overview

Accurate prediction of molecular solubility is challenging because solubility depends on the combined effects of solute structure, solvent environment, and thermodynamic conditions such as temperature. Conventional random train/test splits can overestimate model performance when repeated or closely related solute–solvent pairs appear in both training and test sets.

This project evaluates solubility prediction under more realistic solute–solvent generalization settings and investigates whether target transformations and ensemble modeling can improve robustness.

The workflow includes:

- leakage-aware solute–solvent pair splitting;
- MACCS fingerprint generation;
- MAP4 fingerprint generation;
- PaDEL / physicochemical descriptor generation;
- engineered synthetic solute–solvent interaction features;
- neural-network-based model training;
- unseen-pair prediction;
- transformation-aware ensemble analysis.

## Repository Structure

```text
transformation-aware-solubility/
│
├── BigSolDBv2.0.csv
├── Step_0_group_pair_split_data.py
├── Step_1_1_maccs_fp_features.py
├── Step_1_2-MAP4.py
├── Step_1_3-chemprop.py
├── Step_1_4_sentetik_features.py
├── Step_2_stacking_unseen_pair.py
└── README.md


## Files

### `BigSolDBv2.0.csv`

Input dataset derived from BigSolDB 2.0. The dataset contains temperature-dependent solubility measurements, solute and solvent identifiers, SMILES representations, temperature values, solubility values, and LogS values.

### `Step_0_group_pair_split_data.py`

Creates leakage-aware train/test splits based on unique solute–solvent pairs. This prevents the same solute–solvent pair from appearing in both training and test sets.

### `Step_1_1_maccs_fp_features.py`

Generates MACCS fingerprint features for solute and solvent molecules.

### `Step_1_2-MAP4.py`

Generates MAP4 fingerprint features for solute and solvent molecules.

### `Step_1_3-chemprop.py`

Generates molecular descriptor features or ChemProp-related molecular representations, depending on the configured workflow.

### `Step_1_4_sentetik_features.py`

Generates engineered synthetic solute–solvent interaction features. These may include similarity-based, temperature-modulated, difference-based, product-based, and random-projection-based interaction descriptors.

### `Step_2_stacking_unseen_pair.py`

Runs model training and prediction under the unseen solute–solvent pair setting. This script also supports transformation-aware modeling and ensemble-style prediction depending on the selected configuration.

## Workflow

The recommended execution order is:

```bash
python Step_0_group_pair_split_data.py
python Step_1_1_maccs_fp_features.py
python Step_1_2-MAP4.py
python Step_1_3-chemprop.py
python Step_1_4_sentetik_features.py
python Step_2_stacking_unseen_pair.py
```

## Main Methodological Components

### 1. Leakage-Aware Generalization

The main evaluation protocol is based on an unseen solute–solvent pair split. Each unique solute–solvent pair is assigned exclusively to either the training or test set.

This setting evaluates whether a model can generalize to new combinations of known solutes and solvents rather than memorizing repeated or related measurements.

### 2. Molecular Representation

The feature space combines multiple molecular representations:

* MACCS fingerprints;
* MAP4 fingerprints;
* physicochemical descriptors;
* solute-side descriptors;
* solvent-side descriptors;
* temperature-derived variables;
* engineered solute–solvent interaction features.

### 3. Temperature-Dependent Solubility Prediction

Temperature is treated as an explicit thermodynamic input feature. Additional temperature-derived variables may be used to capture nonlinear temperature effects.

### 4. Transformation-Aware Modeling

Multiple target representations can be evaluated, including:

* direct LogS prediction;
* direct solubility prediction;
* root-based transformations;
* logarithmic transformations;
* inverse transformations;
* quantile-based transformations;
* Box–Cox-type transformations.

### 5. Ensemble Analysis

Predictions from different target transformations can be combined to evaluate whether complementary target spaces improve robustness in LogS space or raw solubility space.

## Citation

If you use this repository, please cite the associated manuscript:

```text
Ugurlu, S. Y.; He, S.
Transformation-Aware Ensemble Modeling for Temperature-Dependent Solubility Prediction under Solute–Solvent Generalization Constraints.
Manuscript under preparation / submitted.
```

Please also cite BigSolDB 2.0 if you use the raw solubility data.

## Data Availability

The raw experimental solubility data are derived from BigSolDB 2.0. Processed datasets, feature-generation scripts, model-training workflows, and analysis code are provided in this repository.

## Requirements

The workflow may require the following Python packages:

```text
pandas
numpy
scikit-learn
rdkit
torch
matplotlib
seaborn
scipy
joblib
```

Additional packages may be required depending on the descriptor-generation workflow, such as MAP4, PaDEL-related tools, or ChemProp-related dependencies.

## Notes

* The main recommended evaluation setting is the unseen solute–solvent pair split.
* Random splitting should be interpreted only as an interpolation-oriented reference.
* LogS is used as the primary modeling target because it provides a more stable regression space than raw solubility.
* Raw solubility predictions are evaluated after reverse transformation for physical interpretability.

## License

This repository is released for academic and research use. Please check the license file or contact the authors for additional usage conditions.



---

## Final dataset and reproducibility package

The final dataset and reproducibility materials used in the revised manuscript are available in this repository.

Key folders:

- `data/raw/`
- `data/processed_splits/`
- `results/split_audit/`
- `results/chemical_identity_audit/`
- `results/repeatability_ceiling/`
- `results/validation_rerun/`
- `code/audit/`
- `code/splits/`
- `manifests/SHA256SUMS.txt`

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for details.
