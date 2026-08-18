import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToInchiKey

# Load the CSV file
input_file = "SMILES_ready.xlsx"
data = pd.read_excel(input_file)

# Function to generate InChIKey from SMILES
def smiles_to_inchikey(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    return MolToInchiKey(mol)

# Apply function to each row
data["InChIKey"] = data["SMILES"].apply(smiles_to_inchikey)

# Reorder columns
final_data = data[["InChIKey", "LD50", "SMILES"]]

# Save to a new CSV file
output_file = "same_code.csv"
final_data.to_csv(output_file, index=False)
