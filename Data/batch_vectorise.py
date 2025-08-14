import numpy as np
import os
from glob import glob

def process_spectrum_file(file_path, fixed_length=2520):
    """
    Extract wave numbers (3rd column) from a crest.vibspectrum file,
    skipping headers and 0.00 modes. Pads/truncates to fixed_length.
    Returns a 1D numpy array.
    """
    wave_numbers = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith("#"):
                continue  # Skip header lines
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # Skip incomplete lines
            try:
                wn = float(parts[2])  # 3rd column = wave number
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
    Recursively find all crest.vibspectrum files inside base_dir.
    Includes any under xtb_results subdirectories.
    """
    pattern = os.path.join(base_dir, "**", "crest.vibspectrum")
    all_files = glob(pattern, recursive=True)
    return all_files

def batch_process_all(base_dir, fixed_length=2520, save_path="all_spectra.csv"):
    all_files = find_all_spectra(base_dir)
    print(f"Found {len(all_files)} crest.vibspectrum files.")
    
    spectra_list = []
    for i, fpath in enumerate(all_files, 1):
        vec = process_spectrum_file(fpath, fixed_length)
        spectra_list.append(vec)
        if i % 100 == 0 or i == len(all_files):
            print(f"Processed {i}/{len(all_files)}")

    feature_matrix = np.vstack(spectra_list)
    
    # Save as CSV
    np.savetxt(save_path, feature_matrix, delimiter=",")
    print(f"Saved all spectra to {save_path}, shape: {feature_matrix.shape}")

if __name__ == "__main__":
    base_dir = "/nobackup/slsr63/combined_outputs/"
    batch_process_all(base_dir, save_path="all_spectra.csv")

