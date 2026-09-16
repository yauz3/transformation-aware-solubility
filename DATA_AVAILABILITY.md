# Dataset Availability

This repository contains the final dataset, processed split files, split-audit outputs, exclusion-accounting results, chemical-identity audit outputs, and reproducibility manifests used in the revised manuscript:

**Temperature-Dependent Solubility Prediction under Solute--Solvent Generalization Constraints: Diagnostic Evaluation and Target-Transformation Analysis**

## Included materials

- `data/raw/`  
  Raw BigSolDB-derived input file used as the starting dataset.

- `data/processed_splits/`  
  Final model-ready train/test split files used for random, unseen-pair, strict unseen-solute, strict unseen-solvent, and fully unseen solute--solvent evaluation.

- `results/split_audit/`  
  Split-construction audit outputs, including held-out component counts, train/test overlap checks, target-distribution diagnostics, and split manifests.

- `results/chemical_identity_audit/`  
  Exclusion accounting, chemical-identity audit outputs, RDKit canonical-SMILES/InChIKey overlap checks, solvent one-to-one audit, disconnected-structure summaries, and InChIKey-overlap sensitivity analysis.

- `results/repeatability_ceiling/`  
  Repeatability-based ceiling analyses where available.

- `results/validation_rerun/`  
  Validation summaries and prediction outputs required to reproduce reported metrics. Trained model weight files are not required and are not included.

- `code/audit/` and `code/splits/`  
  Scripts used for audit, split construction, and sensitivity checks.

- `manifests/SHA256SUMS.txt`  
  SHA256 checksums for repository files.

## Notes

The direct LogS model is retained as the primary prospective baseline. Post hoc transformation-ensemble results are provided as diagnostic empirical upper-envelope analyses and should not be interpreted as independently selected deployment models.

The chemical-identity audit uses the SMILES identifiers present in the BigSolDB-derived modeling file as the original operational split identifiers and adds post hoc RDKit canonical-SMILES and InChIKey checks where possible.
