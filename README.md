# Taste prediction with SchNet-3D and physicochemical-property FiLM conditioning

Code, data splits, configuration files and result tables accompanying the manuscript
"Integrating physicochemical properties into an augmented 3D molecular learning framework for taste prediction"
(sweet / bitter / tasteless classification on ChemTastesDB).

## Contents

| Path | Description |
|---|---|
| `code/generate_sdf.ipynb` | Builds the multi-label SDF (3D conformers, PubChem properties) from the downloaded ChemTastesDB |
| `code/exclusion_counts.py` | Counts exclusions and final dataset sizes |
| `code/main_training_evaluation.ipynb` | Part A: preprocessing, SchNet / SchNet+FiLM training, test-time augmentation evaluation, interpretation (fixed split). Part B: 5-fold out-of-fold SchNet training that saves the fold checkpoints |
| `code/analyses/schnet_oof_probabilities.py` | SchNet out-of-fold class probabilities from the fold checkpoints saved by Part B of the main notebook |
| `code/analyses/morgan_rf_oof.py` | Morgan-fingerprint random forest out-of-fold probabilities |
| `code/analyses/schnet_fp_property_oof.py` | SchNet(3D+Property+FP) out-of-fold probabilities |
| `code/analyses/multiseed_training.py` | Repeats the SchNet(3D+Property+FP) training with several seeds |
| `code/analyses/ensemble_evaluation.ipynb` | Soft-vote ensemble, bootstrap and McNemar tests, tasteless-class analysis, multi-seed summary, fixed-weight sensitivity |
| `code/analyses/ablation.py` | Component ablation (property-only, SchNet-only, FiLM vs concatenation, fingerprint variants) |
| `code/analyses/property_diagnostics.ipynb` | Property correlation, VIF and correlation figure; performance by availability of measured properties |
| `code/analyses/scaffold_split_evaluation.ipynb` | Scaffold-grouped folds, random forest and SchNet(3D+Property+FP) on scaffold folds, comparison with SMILES-grouped folds |
| `code/analyses/scaffold_multiseed.ipynb` | Multi-seed SchNet(3D+Property+FP) on scaffold folds |
| `code/analyses/augmentation.ipynb` | Tasteless conformer augmentation with and without class-balanced sampling, multi-seed |
| `configs/hyperparameters.yaml` | All training, inference, split and augmentation settings |
| `data/identifiers_ternary.csv`, `data/identifiers_binary.csv` | Compound identifiers used in the study: SDF index, name, canonical SMILES, label (ternary: 2,359 molecules; binary: 2,157 molecules) |
| `data/indices_*.json` | Fixed train/validation/test split indices (binary and ternary) |
| `data/folds_*_scaffold_seed42.json` | Bemis-Murcko scaffold-grouped fold assignments |
| `data/exclusion_counts.csv` | Exclusion flow and final dataset sizes |
| `results/` | Result tables behind the supplementary tables, Table 2 and Figure 5 |
| `results/oof_probabilities/` | Out-of-fold class probabilities of each model |

## Data

The ChemTastesDB data are **not** redistributed in this repository. ChemTastesDB (v1.0; 2,944 molecular tastants) can be downloaded from Zenodo:

> Rojas, C., Ballabio, D., Pacheco Sarmiento, K., Pacheco Jaramillo, E., Mendoza, M., & García, F. (2021). *ChemTastesDB: A Curated Database of Molecular Tastants* (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5747393 (CC BY 4.0)

Download ChemTastesDB.rar (molecule structures) and ChemTastesDB_database.xlsx from that record and place them under data/ChemTastesDB/.
The multi-label SDF used by the experiments (3D conformers and PubChem-derived physicochemical properties) is not included in this repository; generate it with code/generate_sdf.ipynb. 
The scripts expect it at "data/ChemTastesDB/ChemTastesDB_multilabel_properties.sdf".

The molecules used in the study are listed in `data/identifiers_ternary.csv` and `data/identifiers_binary.csv` (columns: `sdf_index`, `name`, `canonical_smiles`, `label`, `label_id`).
The split files (`data/indices_*.json`, `data/folds_*.json`) refer to molecules by their 0-based row position in these two files.
Final dataset sizes after exclusions (`data/exclusion_counts.csv`): ternary 2,359 molecules (1,181 bitter, 976 sweet, 202 tasteless; 2,220 unique canonical SMILES), binary 2,157 molecules (2,028 unique canonical SMILES).

## Reproducing

1. Create a Python environment with the packages in `requirements.txt`. Install the PyTorch and PyTorch Geometric builds that match your hardware.
2. Prepare the data as described above (`generate_sdf.ipynb`).
3. Run Part A of `code/main_training_evaluation.ipynb` for the fixed-split results and Part B for the 5-fold out-of-fold SchNet models (fold checkpoints).
4. Run the scripts and notebooks in `code/analyses/` for the cross-validated analyses, in this order: `schnet_oof_probabilities.py`, `morgan_rf_oof.py`, `schnet_fp_property_oof.py`, `multiseed_training.py`, `ensemble_evaluation.ipynb`, `ablation.py`, then the property, scaffold and augmentation notebooks.
   Paths at the top of each file (`data/ChemTastesDB/...`, `analysis/...`) point to the folders created by earlier steps.
5. Results are stochastic; the multi-seed tables use training seeds 1, 2, 3, 4 and 42 with fixed fold assignments.

## Citation

See `CITATION.cff` and the associated manuscript.

## License

Code: MIT (see `LICENSE`). ChemTastesDB is subject to its own license.
