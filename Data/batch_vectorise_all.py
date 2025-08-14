import numpy as np
import os
from glob import glob
from pathlib import Path
import pandas as pd

def process_spectrum_file(file_path, fixed_length=2520):
    """
    Reads crest.vibspectrum and returns padded/truncated wave numbers.
    """
    wave_numbers = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                wn = float(parts[2])
            except ValueError:
                continue
            if wn > 0:
                wave_numbers.append(wn)

    wave_numbers = np.array(wave_numbers)
    if len(wave_numbers) < fixed_length:
        padded = np.pad(wave_numbers, (0, fixed_length - len(wave_numbers)), mode='constant')
    else:
        padded = wave_numbers[:fixed_length]
    return padded

def find_all_spectra(base_dir):
    """
    Recursively find all crest.vibspectrum files in base_dir.
    """
    pattern = os.path.join(base_dir, "**", "crest.vibspectrum")
    all_files = glob(pattern, recursive=True)
    return all_files

def batch_process_all(base_dir, smiles_file, fixed_length=2520, save_path="spectra_ld50.csv"):
    # Load SMILES–LD50–InChIKey mapping
    smiles_df = pd.read_excel(smiles_file)  # must contain columns: InChIKey, LD50
    ld50_map = dict(zip(smiles_df['InChIKey'], smiles_df['LD50']))

    all_files = find_all_spectra(base_dir)
    print(f"Found {len(all_files)} crest.vibspectrum files.")

    spectra_list = []
    labels_list = []
    inchikey_list = []

    for i, fpath in enumerate(all_files, 1):
        inchikey = Path(fpath).parts[-2]  # assumes folder = InChIKey
        if inchikey not in ld50_map:
            continue
        vec = process_spectrum_file(fpath, fixed_length)
        spectra_list.append(vec)
        labels_list.append(ld50_map[inchikey])
        inchikey_list.append(inchikey)

        if i % 100 == 0 or i == len(all_files):
            print(f"Processed {i}/{len(all_files)}")

    X = np.vstack(spectra_list)
    y = np.array(labels_list)

    # Combine into one CSV (InChIKey + features + label)
    output = np.column_stack([inchikey_list, X, y])
    header = ["InChIKey"] + [f"f{i}" for i in range(X.shape[1])] + ["LD50"]
    np.savetxt(save_path, output, fmt="%s", delimiter=",", header=",".join(header), comments="")

    print(f"Saved {len(y)} spectra+LD50 to {save_path}, shape: {X.shape}")

if __name__ == "__main__":
    base_dir = "/nobackup/slsr63/combined_outputs/"
    smiles_file = "/Users/guilingwei/chem_tools-4/src/chem_tools/SMILES_LD50_Inchikey.xlsx"
    batch_process_all(base_dir, smiles_file, save_path="spectra_ld50.csv")
