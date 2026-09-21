# Temperature-Dependent Solubility Prediction under Solute--Solvent Generalization Constraints

Reproducibility repository for the manuscript:

**Temperature-Dependent Solubility Prediction under Solute--Solvent Generalization Constraints: Diagnostic Evaluation and Target-Transformation Analysis**

This repository contains the data-processing scripts, molecular feature-generation workflow, leakage-aware splitting protocol, model-training pipeline, split-audit materials, repeatability analyses, target-transformation analyses, and post hoc transformation-ensemble diagnostics used in the study.

## Overview

Temperature-dependent molecular solubility depends jointly on solute structure, solvent environment, and thermodynamic conditions. This project evaluates predictive performance under progressively stricter solute--solvent generalization settings rather than relying only on random train/test splitting.

The study includes:

- random splitting as an interpolation-oriented reference;
- leakage-aware unseen solute--solvent pair evaluation;
- strict unseen-solute and unseen-solvent stress tests;
- a fully unseen solute--solvent diagnostic split;
- repeated outer component-selection analyses for strict splits;
- MACCS, MAP4, PaDEL, temperature-derived, and engineered interaction features;
- a fixed MLP training pipeline with train-only standardization and group-aware internal validation;
- repeatability-based empirical performance references;
- feature-source ablation and grouped permutation analyses;
- SHAP, LIME, VIF, t-SNE, and PCA diagnostics;
- systematic target-transformation screening;
- post hoc transformation-ensemble analysis interpreted as an empirical upper-envelope diagnostic rather than as a prospectively selected deployment model.

Direct LogS is retained as the primary prospective target for logarithmic or relative-solubility prediction. Direct raw-solubility prediction is retained as the principal single-target comparator when absolute mol/L accuracy is the intended endpoint.

## Repository Structure

```text
transformation-aware-solubility/
├── BigSolDBv2.0.csv
├── README.md
├── DATA_AVAILABILITY.md
├── data/
│   ├── raw/
│   └── processed_splits/
│       └── repeated_component_resampling/
│           ├── strict_unseen_solute_seed7/
│           ├── strict_unseen_solvent_seed7/
│           ├── strict_unseen_solute_seed123/
│           └── strict_unseen_solvent_seed123/
├── results/
│   ├── split_audit/
│   ├── chemical_identity_audit/
│   ├── repeatability_ceiling/
│   ├── validation_rerun/
│   └── repeated_strict_component_resampling/
├── code/
│   ├── audit/
│   └── splits/
├── manifests/
│   └── SHA256SUMS.txt
├── Step_0_group_pair_split_data.py
├── Step_1_1_maccs_fp_features.py
├── Step_1_2-MAP4.py
├── Step_1_3-chemprop.py
├── Step_1_4_sentetik_features.py
└── Step_2_stacking_unseen_pair.py
```

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the reproducibility package layout and reconstruction instructions for large split files.

## Main Evaluation Regimes

### 1. Random split

Random splitting is retained only as an interpolation-oriented reference. Solutes, solvents, and complete solute--solvent pairs can overlap between train and test partitions.

### 2. Unseen solute--solvent pair split

Each complete solute--solvent pair is assigned exclusively to either training or testing. Individual solutes and solvents can still appear in both partitions when paired with different counterpart components.

This is the primary application-relevant split used for the main baseline, feature ablation, target-transformation screening, and post hoc ensemble diagnostics.

### 3. Strict unseen-solute split

All test solutes are absent from training. Solvents may remain shared across partitions.

### 4. Strict unseen-solvent split

All test solvents are absent from training. Solutes may remain shared across partitions.

### 5. Fully unseen solute--solvent split

Both test solutes and test solvents are absent from training. Mixed-status records are excluded from model fitting and final testing for this diagnostic split and are retained separately for audit purposes.

## Repeated Strict Component Selection

The seed-42 strict unseen-solute and strict unseen-solvent splits are retained as the primary fixed stress tests. To assess sensitivity to the specific held-out components, the strict splits were additionally regenerated using outer component-selection seeds **7** and **123** while retaining the same source data, feature matrix, split definitions, architecture, optimization settings, and operational prediction procedure.

The repeated split CSVs are provided under:

```text
data/processed_splits/repeated_component_resampling/
```

with separate folders for:

```text
strict_unseen_solute_seed7/
strict_unseen_solvent_seed7/
strict_unseen_solute_seed123/
strict_unseen_solvent_seed123/
```

The Direct LogS repeated analysis produced valid operational predictions for all three unseen-solute selections and for unseen-solvent seeds 7 and 42. The seed-123 unseen-solvent partition exposed an extreme descriptor-space numerical out-of-distribution condition under the predefined preprocessing and mixed-precision inference pipeline. That replicate is retained as a documented numerical failure; no seed-specific clipping, descriptor deletion, or variance filtering was introduced after observing the held-out partition.

## Large Split Files

Several model-ready training CSVs exceed the practical per-object size used in this repository. Large training CSVs are therefore stored as ordered Git LFS chunks in a companion `.chunks/` directory.

For example:

```bash
cat \
  data/processed_splits/repeated_component_resampling/strict_unseen_solute_seed7/\
train_strict_unseen_solute_maccs_map4_padel_sen_features.csv.chunks/\
train_strict_unseen_solute_maccs_map4_padel_sen_features.csv.part-* \
> train_strict_unseen_solute_maccs_map4_padel_sen_features.csv
```

The reconstructed file can be verified against `manifests/SHA256SUMS.txt`.

Test CSVs are stored directly through Git LFS.

## Molecular Representation

The feature space combines:

- MACCS fingerprints;
- MAP4 fingerprints;
- PaDEL / physicochemical descriptors;
- solute-side descriptors;
- solvent-side descriptors;
- temperature and temperature-derived variables;
- engineered solute--solvent interaction features.

## Model Training

The canonical neural-learning pipeline uses a fully connected MLP with:

- hidden layers `[4096, 2048, 1024, 512, 256, 128, 64, 32]`;
- ReLU activations;
- batch normalization;
- dropout = 0.15;
- AdamW optimization;
- learning rate = `1e-3`;
- weight decay = `1e-5`;
- batch size = 1024;
- maximum 100 epochs;
- early stopping patience = 7;
- MAE training loss;
- train-only `StandardScaler`;
- 5-fold group-aware internal validation;
- CUDA AMP mixed precision in the canonical runs.

Principal metrics are computed from the final fold-averaged operational prediction vector. Fold-model mean ± SD values are retained only as diagnostics of model-to-model variability within one fixed outer split.

## Target-Transformation Analysis

Fourteen target representations were evaluated, including:

- Direct LogS;
- Direct solubility;
- square-root and cube-root transformations;
- logarithmic transformations;
- inverse and negative-log transformations;
- fractional and power-law transformations;
- asinh transformation;
- quantile-normal and quantile-uniform mappings;
- Box--Cox where valid.

Target representations are compared in their native spaces and, where defined, after mapping to common LogS and raw-solubility spaces.

## Post Hoc Ensemble Analysis

Transformation-specific operational prediction vectors can be averaged after mapping to a common evaluation space. Candidate transformation subsets were examined after evaluation on the fixed test set; therefore, these ensemble comparisons are **post hoc diagnostics** and should not be interpreted as independently selected deployment estimates.

A transformation subset intended for deployment should be selected using an independent validation procedure or nested cross-validation and then evaluated once on an untouched test set.

## Reproducibility Outputs

Important outputs include:

- split construction summaries;
- held-out component manifests;
- train--test leakage checks;
- target-distribution diagnostics;
- chemical-identity audits using raw SMILES, RDKit canonical SMILES, and InChIKey;
- InChIKey-overlap sensitivity analyses;
- repeatability-based ceiling analyses;
- repeated strict component-selection summaries;
- fold-specific and operational prediction outputs;
- target-transformation validation results.

File integrity checks are provided in:

```text
manifests/SHA256SUMS.txt
```

## Requirements

The workflow may require:

```text
pandas
numpy
scikit-learn
rdkit
torch
matplotlib
scipy
joblib
```

Additional packages may be required for MAP4-, PaDEL-, or ChemProp-related descriptor-generation steps.

## Citation

If you use this repository, please cite the associated manuscript:

```text
Ugurlu, S. Y.; He, S.
Temperature-Dependent Solubility Prediction under Solute--Solvent Generalization Constraints:
Diagnostic Evaluation and Target-Transformation Analysis.
```

Please also cite BigSolDB 2.0 when using the underlying experimental solubility data.

## Data Availability

The raw experimental solubility data are derived from BigSolDB 2.0. Processed splits, repeated component-selection partitions, split-audit outputs, validation summaries, prediction outputs, and analysis code used in the revised manuscript are provided in this repository.

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for details.
