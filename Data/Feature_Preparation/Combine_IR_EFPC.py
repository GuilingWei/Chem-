import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the EFPC data
efpc_df = pd.read_csv('efpc_fingerprints.csv')

# Extract the IR spectra columns
ir_columns = [col for col in df.columns if col.startswith('f') or col.startswith('band_')]

# Merge the IR spectra and EFPC data
combined_df = pd.merge(df[['InChIKey'] + ir_columns], efpc_df, on='InChIKey', how='inner')

# Save the combined dataset with LD50
combined_df['LD50'] = df['LD50']  # Ensure LD50 is included
combined_df.to_csv('combined_features.csv', index=False)
combined_df.to_csv('combined_features.csv', index=False)
