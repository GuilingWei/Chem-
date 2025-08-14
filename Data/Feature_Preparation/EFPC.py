import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the data
df = pd.read_csv('r_smiles_inchikey_ir_ld50.csv')

# Function to generate Morgan fingerprints (equivalent to EFPC in this case)
def generate_efpc(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Generate the Morgan fingerprint with a specific radius and number of bits
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fp)  # Convert bit vector to a list
    return [0] * n_bits  # Return a zero vector if molecule is invalid

# Generate EFPC for each SMILES and store InChIKey
efpc_data = []
for index, row in df.iterrows():
    efpc = generate_efpc(row['SMILES'])
    efpc_data.append([row['InChIKey']] + efpc)

# Save EFPC data to CSV
efpc_df = pd.DataFrame(efpc_data, columns=['InChIKey'] + [f'efpc_{i}' for i in range(len(efpc_data[0]) - 1)])
efpc_df.to_csv('efpc_fingerprints.csv', index=False)
