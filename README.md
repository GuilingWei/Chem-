# LD50 Prediction from Molecular IR Spectra

This repository contains code, data, and supporting materials for a research project aimed at predicting LD₅₀ (lethal dose, 50%) values for molecules. 

Molecular features are derived from **interpolated infrared (IR) spectra** generated from molecular structures (via SMILES → 3D conformers → vibrational analysis).  

The project evaluates several regression approaches, focusing on the predictive power of **spectral features** for molecular toxicity.

---


## Repository Contents

.
├── Data/
│ ├── IR_process.py # Script for processing IR spectra
│ ├── Inchikey_diagnose.py # Script for diagnosing InChIKey issues
│ ├── Chem_Broden_Group_Final.ipynb # Final notebook with broadening included
│ ├── Chem_Nobroadening_Final.ipynb # Final notebook without broadening
│ └── .DS_Store # macOS system file (can be ignored)
├── README.md 

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
