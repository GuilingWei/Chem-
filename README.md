# LD50 Prediction from Molecular IR Spectra

This repository contains code, data, and supporting materials for a research project aimed at predicting LD₅₀ (lethal dose, 50%) values for molecules. 

Molecular features are derived from **interpolated infrared (IR) spectra** generated from molecular structures (via SMILES → 3D conformers → vibrational analysis).  

The project evaluates several regression approaches, focusing on the predictive power of **spectral features** for molecular toxicity.

---


## Repository Contents

```
.
├── src/
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── generate_inchikey_mapping.py
│   │   └── Inchikey_diagnose.py         # Script for diagnosing InChIKey issues
│   ├── transform/
│   │   ├── __init__.py
│   │   ├── ir2grid_broadening.py        # Script for grid conversion with broadening
│   │   └── ir2grid.py                   # Script for grid conversion without broadening
│   ├── __init__.py
│   ├── ECFP.py                          # Script for generating Extended-Connectivity Fingerprints
│   └── feature_engineering.py           # Feature engineering utilities
├── Chem_Broden_Group_Final.ipynb        # Final notebook with broadening included
├── Chem_Nobroadening_Final.ipynb        # Final notebook without broadening
└── README.md                            # Repository documentation

```

## License

This repository uses a **three-way licensing structure** to maximise re-use while preserving attribution where appropriate.

- **Code** (`src/`, `slurm_scripts/`):  
  Licensed under the **MIT License** — you may use, modify, and distribute, even commercially, with attribution.  

- **Research Data** (`data/`):  
  Released under **CC0 1.0 Universal (Public Domain Dedication)** — you may copy, modify, and use without any restrictions.  
  *(Note: In most jurisdictions, raw factual data such as experimental LD₅₀ values and computed spectra are not copyrightable.)*

- **Creative Content** (documentation, database schema, figures, descriptions):  
  Licensed under **CC BY 4.0** — you may share and adapt for any purpose, even commercially, but must give appropriate credit.  
