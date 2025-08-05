import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToInchiKey

def generate_inchikey_map(input_excel: str, output_csv: str) -> None:
    """
    Generates a CSV mapping InChIKeys to SMILES from an Excel input file.

    Parameters
    ----------
    input_excel : str
        Path to the Excel file that contains SMILES strings (must have a 'SMILES' column).
    output_csv : str
        Path to the output CSV file to write InChIKey–SMILES mappings.
    """
    try:
        DF = pd.read_excel(input_excel)
    except Exception as e:
        raise IOError(f"Failed to read input file: {e}")

    if 'SMILES' not in DF.columns:
        raise ValueError("The input file must contain a column named 'SMILES'.")

    DF = DF.dropna(subset=['SMILES'])

    def smiles_to_inchikey(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MolToInchiKey(mol)

    DF['InChIKey'] = DF['SMILES'].apply(smiles_to_inchikey)
    DF = DF.dropna(subset=['InChIKey'])

    try:
        DF[['InChIKey', 'SMILES']].to_csv(output_csv, index=False)
        print(f"InChIKey mapping written to: {output_csv}")
    except Exception as e:
        raise IOError(f"Failed to write output file: {e}")


##generate_inchikey_map("SMILES_ready.xlsx", "inchikey_mapping.csv")
