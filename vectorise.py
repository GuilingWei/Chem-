import numpy as np

def define(input_path, fixed_length=2520, save_path="processed_spectrum.npy"):
    """
    Process a single crest.vibspectrum file:
    - Extract wave numbers (first column)
    - Remove zeros and invalid rows
    - Pad or truncate to fixed_length
    - Save as a .npy file
    
    Parameters:
        input_path (str): Path to your crest.vibspectrum file
        fixed_length (int): Length to pad/truncate vector to (default 2520)
        save_path (str): Output file to save the processed vector
    """
    wave_numbers = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            try:
                wn = float(parts[0])
            except ValueError:
                continue
            if wn > 0:
                wave_numbers.append(wn)
    
    wave_numbers = np.array(wave_numbers)
    
    if len(wave_numbers) < fixed_length:
        padded = np.pad(wave_numbers, (0, fixed_length - len(wave_numbers)), mode='constant')
    else:
        padded = wave_numbers[:fixed_length]
    
    np.save(save_path, padded)
    print(f"Processed spectrum saved to {save_path}, shape: {padded.shape}")

