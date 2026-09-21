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
└── manifests/
    └── SHA256SUMS.txt
```

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the complete package layout and reconstruction instructions for large split files.

## Evaluation Regimes

1. **Random split** — interpolation-oriented reference.
2. **Unseen solute--solvent pair split** — complete pairs are disjoint between train and test; this is the primary application-relevant setting.
3. **Strict unseen-solute split** — all test solutes are absent from training.
4. **Strict unseen-solvent split** — all test solvents are absent from training.
5. **Fully unseen solute--solvent split** — both test components are absent from training.

## Repeated Strict Component Selection

The seed-42 strict unseen-solute and strict unseen-solvent splits are retained as the primary fixed stress tests. Additional outer component-selection seeds **7** and **123** were used to assess sensitivity to the particular held-out components while keeping the source data, feature matrix, split definitions, architecture, optimization settings, preprocessing, and operational prediction procedure fixed.

Repeated split CSVs are available under:

```text
data/processed_splits/repeated_component_resampling/
```

The Direct LogS repeated analysis produced valid operational predictions for all three unseen-solute selections and for unseen-solvent seeds 7 and 42. The seed-123 unseen-solvent partition exposed an extreme descriptor-space numerical out-of-distribution condition under the predefined preprocessing and mixed-precision inference pipeline. The split is retained for reproducibility and no seed-specific corrective preprocessing was introduced after observing the held-out partition.

## Large Split Files

Large training matrices are stored as ordered Git LFS chunks in `<filename>.chunks/` directories. Reconstruct them by concatenating the chunk files in lexical order and verify the result against the accompanying SHA256 file.

Test CSVs are stored directly through Git LFS.

See `data/processed_splits/repeated_component_resampling/RECONSTRUCT.md`.

## Model Training

The canonical MLP pipeline uses:

- hidden layers `[4096, 2048, 1024, 512, 256, 128, 64, 32]`;
- ReLU activations;
- batch normalization;
- dropout = 0.15;
- AdamW;
- learning rate = `1e-3`;
- weight decay = `1e-5`;
- batch size = 1024;
- maximum 100 epochs;
- early stopping patience = 7;
- MAE loss;
- train-only `StandardScaler`;
- 5-fold group-aware internal validation;
- CUDA AMP mixed precision.

Principal metrics are computed from the final fold-averaged operational prediction vector. Fold-model mean ± SD values are secondary diagnostics of model-to-model variability within one fixed outer split.

## Target-Transformation and Ensemble Diagnostics

Fourteen target representations were screened. Transformation-specific predictions were mapped to common LogS and raw-solubility evaluation spaces where defined.

Transformation-ensemble subsets were evaluated post hoc on the fixed test set. They are therefore reported as empirical upper-envelope diagnostics, not as independently selected deployment models.

## Reproducibility Outputs

The repository includes split audits, held-out component manifests, target-distribution diagnostics, chemical-identity audits, repeatability analyses, repeated component-selection summaries, validation predictions, and target-transformation outputs.

File integrity information is provided in:

```text
manifests/SHA256SUMS.txt
```

## Citation

If you use this repository, please cite:

```text
Ugurlu, S. Y.; He, S.
Temperature-Dependent Solubility Prediction under Solute--Solvent Generalization Constraints:
Diagnostic Evaluation and Target-Transformation Analysis.
```

Please also cite BigSolDB 2.0 when using the underlying experimental solubility data.

## Data Availability

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).
