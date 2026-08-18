import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToInchiKey

input_file = "SMILES_ready.xlsx"
data = pd.read_excel(input_file)

def smiles_to_inchikey(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    return MolToInchiKey(mol)

data["InChIKey"] = data["SMILES"].apply(smiles_to_inchikey)

final_data = data[["InChIKey", "LD50", "SMILES"]]

output_file = "same_code.csv"
final_data.to_csv(output_file, index=False)
