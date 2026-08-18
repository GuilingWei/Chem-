"""ECFP (Morgan) fingerprint generation."""

from rdkit import Chem
from rdkit.Chem import AllChem

def morgan_bits(smiles, radius=2, n_bits=2048):
    """SMILES -> list of 0/1 bits, length n_bits. All-zero if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * n_bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)