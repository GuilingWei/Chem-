#!/usr/bin/env python3
import numpy as np
import os
from glob import glob
from pathlib import Path
import pandas as pd
import re

# -----------------------------
# Configuration
# -----------------------------
FIXED_LENGTH = 2520          # base grid length before optional downsampling
WN_MIN = 400.0
WN_MAX = 4000.0
FWHM = 10.0                  # Gaussian broadening width (cm^-1). Try 10, 20, 40
ADD_BAND_FEATURES = True     # append compact band features
DOWNSAMPLE_STRIDE = 1        # e.g., 2 or 4 to reduce dimensionality; 1 = no downsample
OUTPUT_CSV = "Broadening_Normalization.csv"  

# Bands for compact features (mean, max, area per band)
BANDS = [
    (600, 900),
    (1000, 1500),
    (1500, 1800),
    (2800, 3100),
    (3200, 3700),
]

# Regex for standard InChIKey format
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


# -----------------------------
# Utilities
# -----------------------------
def gaussian_broaden(wave_numbers, intensities, wn_grid, fwhm=20.0):
    """
    Sum of Gaussians centered at discrete modes to produce a smooth spectrum.
    wave_numbers: (N,)
    intensities:  (N,)
    wn_grid:      (G,)
    Returns:      (G,)
    """
    if wave_numbers.size == 0:
        return np.zeros_like(wn_grid, dtype=float)

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # (G,1) - (1,N) -> (G,N)
    diff = wn_grid[:, None] - wave_numbers[None, :]
    weights = np.exp(-0.5 * (diff / sigma) ** 2)
    spec = (weights * intensities[None, :]).sum(axis=1)
    return spec


def band_features(wn_grid, spec, bands):
    """
    For each band (lo, hi), compute mean, max, and area (trapezoid).
    Returns flat array of length 3*len(bands).
    """
    feats = []
    for lo, hi in bands:
        m = (wn_grid >= lo) & (wn_grid <= hi)
        if not m.any():
            feats.extend([0.0, 0.0, 0.0])
            continue
        seg = spec[m]
        feats.append(seg.mean())
        feats.append(seg.max())
        feats.append(np.trapz(seg, wn_grid[m]))
    return np.array(feats, dtype=float)


def process_spectrum_file(file_path, fixed_length=2520, wn_min=400, wn_max=4000, fwhm=20.0):
    """
    Reads crest.vibspectrum (columns: ... wn intensity) and returns
    a broadened, area-normalized spectrum on a uniform grid.
    """
    wave_numbers = []
    intensities = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                wn = float(parts[2])        # 3rd column: wavenumber
                it = float(parts[3])        # 4th column: intensity
            except ValueError:
                continue
            if wn > 0:
                wave_numbers.append(wn)
                intensities.append(it)

    if not wave_numbers:
        return np.zeros(fixed_length, dtype=float), np.linspace(wn_min, wn_max, fixed_length)

    wave_numbers = np.array(wave_numbers, dtype=float)
    intensities = np.array(intensities, dtype=float)

    wn_grid = np.linspace(wn_min, wn_max, fixed_length)
    spec = gaussian_broaden(wave_numbers, intensities, wn_grid, fwhm=fwhm)

    # Area normalization to reduce per-molecule scale differences
    area = np.trapz(spec, wn_grid)
    if area > 0:
        spec = spec / area

    return spec, wn_grid


def find_all_spectra(base_dir):
    """
    Find crest.vibspectrum files in:
    - combined_outputs/<InChIKey>/crest.vibspectrum
    - combined_outputs/<InChIKey>/xtb_results/crest.vibspectrum
    """
    pattern1 = os.path.join(base_dir, "*", "crest.vibspectrum")
    pattern2 = os.path.join(base_dir, "*", "xtb_results", "crest.vibspectrum")
    return glob(pattern1) + glob(pattern2)


# -----------------------------
# Main batching
# -----------------------------
def batch_process_all(base_dir, smiles_file, fixed_length=2520, wn_min=400, wn_max=4000,
                      fwhm=20.0, save_path=OUTPUT_CSV,
                      add_band_features=True, downsample_stride=1):
    """
    Create a CSV with:
      InChIKey, f0..f{M-1}, [band_mean/max/area ...], LD50
    """
    # Load mapping (must contain InChIKey, LD50)
    smiles_df = pd.read_csv(smiles_file)
    if "InChIKey" not in smiles_df.columns or "LD50" not in smiles_df.columns:
        raise ValueError("smiles_file must contain columns: InChIKey, LD50")

    ld50_map = dict(zip(smiles_df['InChIKey'], smiles_df['LD50']))

    all_files = find_all_spectra(base_dir)
    print(f"Found {len(all_files)} crest.vibspectrum files.")

    spectra_rows = []
    labels_list = []
    inchikey_list = []

    # For band feature naming
    band_cols = []
    if add_band_features:
        for (lo, hi) in BANDS:
            band_cols += [f"band_{lo}_{hi}_mean", f"band_{lo}_{hi}_max", f"band_{lo}_{hi}_area"]

    for i, fpath in enumerate(all_files, 1):
        p = Path(fpath)
        # InChIKey resolution from path
        if p.parent.name == "xtb_results":
            inchikey = p.parent.parent.name
        else:
            inchikey = p.parent.name

        if not INCHIKEY_PATTERN.match(inchikey):
            # keep quiet or print once:
            # print(f"Skipping invalid InChIKey: {inchikey}")
            continue

        if inchikey not in ld50_map:
            # print(f"Skipping: No LD50 for {inchikey}")
            continue

        try:
            spec, wn_grid = process_spectrum_file(
                fpath, fixed_length=fixed_length, wn_min=wn_min, wn_max=wn_max, fwhm=fwhm
            )
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

        # Optional downsample (uniform stride)
        if downsample_stride > 1:
            spec = spec[::downsample_stride]
            wn_grid = wn_grid[::downsample_stride]

        row = spec.astype(float)

        # Append band features
        if add_band_features:
            bf = band_features(wn_grid, spec, BANDS)
            row = np.concatenate([row, bf], axis=0)

        spectra_rows.append(row)
        labels_list.append(ld50_map[inchikey])
        inchikey_list.append(inchikey)

        if i % 100 == 0 or i == len(all_files):
            print(f"Processed {i}/{len(all_files)}")

    if not spectra_rows:
        print("No spectra processed. Exiting.")
        return

    X = np.vstack(spectra_rows)
    y = np.array(labels_list, dtype=float)

    # Build header
    n_spec = spec.shape[0]  # last processed length (after downsampling)
    feat_cols = [f"f{i}" for i in range(n_spec)]
    if add_band_features:
        feat_cols = feat_cols + band_cols

    # Compose DataFrame to ensure exact formatting
    out_df = pd.DataFrame(X, columns=feat_cols)
    out_df.insert(0, "InChIKey", inchikey_list)
    out_df["LD50"] = y

    out_df.to_csv(save_path, index=False)
    print(f"Saved {len(y)} rows to {save_path}, shape: {out_df.shape}")
    print(f"Grid length (after downsampling): {n_spec}")
    if add_band_features:
        print(f"Added band features: {len(band_cols)} columns")


if __name__ == "__main__":
    base_dir = "/nobackup/slsr63/combined_outputs/"
    smiles_file = "/nobackup/slsr63/InChIKey_LD50_SMILES.csv"
    batch_process_all(
        base_dir, smiles_file,
        fixed_length=FIXED_LENGTH,
        wn_min=WN_MIN, wn_max=WN_MAX,
        fwhm=FWHM,
        save_path=OUTPUT_CSV,
        add_band_features=ADD_BAND_FEATURES,
        downsample_stride=DOWNSAMPLE_STRIDE
    )

