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
