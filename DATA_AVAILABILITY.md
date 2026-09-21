# Dataset Availability

This repository contains the final dataset, processed split files, split-audit outputs, exclusion-accounting results, chemical-identity audit outputs, repeated strict component-selection materials, validation outputs, and reproducibility manifests used in the revised manuscript:

**Temperature-Dependent Solubility Prediction under Solute--Solvent Generalization Constraints: Diagnostic Evaluation and Target-Transformation Analysis**

## Included materials

- `data/raw/`  
  Raw BigSolDB-derived input file used as the starting dataset.

- `data/processed_splits/`  
  Final model-ready train/test split files used for random, unseen-pair, strict unseen-solute, strict unseen-solvent, and fully unseen solute--solvent evaluation.

- `data/processed_splits/repeated_component_resampling/`  
  Additional strict component-selection partitions for outer seeds 7 and 123:
  - `strict_unseen_solute_seed7/`
  - `strict_unseen_solvent_seed7/`
  - `strict_unseen_solute_seed123/`
  - `strict_unseen_solvent_seed123/`

  Large training CSVs are stored as ordered Git LFS chunks under `<filename>.chunks/`. The corresponding held-out test CSVs are stored directly through Git LFS.

- `results/split_audit/`  
  Split-construction audit outputs, including held-out component counts, train/test overlap checks, target-distribution diagnostics, and split manifests.

- `results/chemical_identity_audit/`  
  Exclusion accounting, chemical-identity audit outputs, RDKit canonical-SMILES/InChIKey overlap checks, solvent one-to-one audit, disconnected-structure summaries, and InChIKey-overlap sensitivity analysis.

- `results/repeatability_ceiling/`  
  Repeatability-based ceiling analyses where available.

- `results/validation_rerun/`  
  Validation summaries and prediction outputs required to reproduce the principal reported metrics. Trained model weight files are not required and are not included.

- `results/repeated_strict_component_resampling/`  
  Repeated strict component-selection outputs for Direct LogS, including per-seed summaries, fold metrics, held-out component manifests, operational prediction vectors where numerically valid, aggregate summaries, and the documented seed-123 unseen-solvent numerical out-of-distribution failure.

- `code/audit/` and `code/splits/`  
  Scripts used for audit, split construction, repeated component selection, operational-metric validation, and sensitivity checks.

- `manifests/SHA256SUMS.txt`  
  SHA256 checksums for reproducibility-package files.

## Repeated strict component-selection design

The seed-42 strict unseen-solute and strict unseen-solvent splits are retained as the primary fixed component-level stress tests. Additional outer component-selection seeds **7** and **123** were used to assess sensitivity to the particular solutes or solvents held out for testing.

Only the outer component-selection seed was changed. The source data, descriptor matrix, split definition, architecture, optimization settings, preprocessing rules, and operational prediction procedure were otherwise held fixed.

Direct LogS produced valid operational predictions for:

- strict unseen-solute: seeds 7, 42, and 123;
- strict unseen-solvent: seeds 7 and 42.

The strict unseen-solvent seed-123 partition generated non-finite held-out predictions under the predefined mixed-precision inference pipeline after an extreme descriptor-space out-of-distribution condition emerged under train-only standardization. The split itself is retained and distributed for reproducibility. No seed-specific clipping, feature deletion, or variance filtering was introduced after observing this partition.

## Reconstructing large training CSVs

For any repeated-split condition:

```bash
cd data/processed_splits/repeated_component_resampling/<condition>

NAME="train_strict_unseen_solute_maccs_map4_padel_sen_features.csv"
cat "$NAME.chunks/$NAME".part-* > "$NAME"

sha256sum -c "$NAME.sha256"
```

For unseen-solvent folders, replace `solute` with `solvent` in `NAME`.

Test split CSVs are stored directly through Git LFS.

## Notes

Direct LogS is retained as the primary prospective baseline for logarithmic or relative-solubility prediction. Direct raw-solubility prediction is the principal single-target comparator when absolute mol/L accuracy is the intended endpoint.

Post hoc transformation-ensemble results are diagnostic empirical upper-envelope analyses and should not be interpreted as independently selected deployment models.

The chemical-identity audit uses the SMILES identifiers present in the BigSolDB-derived modeling file as the original operational split identifiers and adds post hoc RDKit canonical-SMILES and InChIKey checks where possible.
