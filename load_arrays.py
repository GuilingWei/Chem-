import numpy as np

# Load your spectra data
X = np.loadtxt("all_spectra.csv", delimiter=",")
print("Feature matrix shape:", X.shape)  # e.g., (10000, 2520)

assert not np.isnan(feature_matrix).any()
assert np.isfinite(feature_matrix).all()
