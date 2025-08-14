# LD50 Prediction from Molecular IR Spectra

This repository contains code, data, and supporting materials for a research project aimed at predicting LD₅₀ (lethal dose, 50%) values for molecules. 

Molecular features are derived from **interpolated infrared (IR) spectra** generated from molecular structures (via SMILES → 3D conformers → vibrational analysis).  

The project evaluates several regression approaches, focusing on the predictive power of **spectral features** for molecular toxicity.

---


## Repository Contents

.
├── data/
│ ├── raw/ # Raw molecular data (e.g., SMILES, raw IR spectra files)
│ ├── processed/ # Preprocessed spectral vectors and LD50 labels 
│ └── missing/ # Molecules with missing/invalid features (optional)
├── src/
│ ├── preprocessing.py # Feature extraction, interpolation, zero-variance filtering
│ ├── train_models.py # Regression model training scripts (RF, XGBoost, CNN)
│ ├── evaluate.py # Model evaluation scripts
│ └── utils/ # Helper functions
├── models/
│ └── saved/ # Serialized trained models (.joblib, .h5, etc.)
├── notebooks/
│ └── model_training.ipynb # Jupyter notebook for exploratory training/testing
├── slurm_scripts/
│ └── crest_array.slurm # Example SLURM job array for vibrational analysis
├── requirements.txt # Python dependencies
└── README.md # Project overview (this file)


---


## License

This repository uses a **three-way licensing structure** to maximise re-use while preserving attribution where appropriate.

- **Code** (`src/`, `slurm_scripts/`):  
  Licensed under the **MIT License** — you may use, modify, and distribute, even commercially, with attribution.  

- **Research Data** (`data/`):  
  Released under **CC0 1.0 Universal (Public Domain Dedication)** — you may copy, modify, and use without any restrictions.  
  *(Note: In most jurisdictions, raw factual data such as experimental LD₅₀ values and computed spectra are not copyrightable.)*

- **Creative Content** (documentation, database schema, figures, descriptions):  
  Licensed under **CC BY 4.0** — you may share and adapt for any purpose, even commercially, but must give appropriate credit.  
