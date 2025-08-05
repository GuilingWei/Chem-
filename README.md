# LD50 Prediction from Molecular Representations

This repository contains code, data, and supporting materials for a research project aimed at predicting LD₅₀ (lethal dose, 50%) values for small molecules. Molecular features were derived by concatenating infrared (IR) spectral vectors with Extended Functional Class Fingerprints (EFCP), both based on original SMILES inputs.

The project explores several machine learning approaches to evaluate how well such representations can predict molecular toxicity. It includes preprocessing scripts, training routines, and evaluation workflows.

---

## Repository Contents

.
├── data/
│ ├── raw/ # Raw molecular data (e.g. SMILES, IR spectra)
│ ├── processed/ # Final input vectors and LD50 labels
│ └── missing/ # Molecules with missing/invalid features (optional)
├── src/
│ ├── preprocessing.py # Feature extraction and transformation scripts
│ ├── train_models.py # Main scripts for training regression models
│ └── evaluate.py # Metrics and prediction analysis
├── models/
│ └── saved/ # Serialized models (.pkl, .h5, etc.)
├── notebooks/
│ └── model_training.ipynb # Jupyter notebook for full training pipeline
├── slurm_scripts/
│ └── crest_array.slurm # Job array script for parallel molecular processing (optional)
├── requirements.txt # Python environment specification
└── README.md # Project overview and metadata


---

## Project Overview

Each molecule is encoded as a fixed-length (1,000-bit) vector derived from:

- A numerical representation of its IR spectrum (e.g., interpolated/transformed)
- An EFCP molecular fingerprint (e.g., RDKit-derived)

Regression models (including Random Forest, Support Vector Machine, and Multilayer Perceptron) are trained to predict LD₅₀ values, which serve as the ground truth labels. The project benchmarks model performance using standard metrics such as R², RMSE, and MAE.

---

## How to Use

1. Prepare your data in `data/processed/` as NumPy arrays or `.csv` files.
2. Train a model using `train_models.py`.
3. Evaluate the model using `evaluate.py` or `notebooks/model_training.ipynb`.
4. Models and logs will be saved in `models/`.

Python 3.8+ is recommended. Install dependencies using:

```bash
pip install -r requirements.txt
