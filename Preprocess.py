#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pathlib import Path
import re

# Load Excel
file = "Concatenated_Toxicity_input_data.xlsx"
df = pd.read_excel(file)

# Drop missing or invalid SMILES
DF = df.dropna(subset=["SMILES"])

# Filter SMILES with .
def has_dot(smiles):
    return re.search(r'\.', smiles) is not None
invalid_smiles = DF[DF["SMILES"].apply(has_dot)]
invalid_smiles.to_excel("SMILES_invalid.xlsx", index=False)

valid_smiles = DF[~DF["SMILES"].apply(has_dot)]
valid_smiles.to_excel("SMILES_ready.xlsx", index=False)

print(f"Saved {len(valid_smiles)} valid and {len(invalid_smiles)} invalid SMILES.")

