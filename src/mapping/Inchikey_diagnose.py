import os
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToInchiKey
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

ld50_csv = Path("/Users/guilingwei/chem_tools-4/src/chem_tools/InChIKey_LD50_SMILES.csv")
xyz_dir  = Path("/Users/guilingwei/chem_tools-4/src/chem_tools/preparation/xyz_outputs")  

DF = pd.read_csv(ld50_csv)
DF["SMILES"]   = DF["SMILES"].astype(str).str.strip()
DF["InChIKey"] = DF["InChIKey"].astype(str).str.strip()
DF["RootKey"]  = DF["InChIKey"].str[:14]

xyz_files = [p for p in xyz_dir.glob("*.xyz")]
xyz_df = pd.DataFrame({
    "InChIKey_xyz": [p.stem for p in xyz_files],
    "xyz_path": [str(p) for p in xyz_files]
})
xyz_df["RootKey"] = xyz_df["InChIKey_xyz"].str[:14]

root_to_full_xyz = (
    xyz_df.groupby("RootKey")["InChIKey_xyz"]
          .apply(lambda s: set(s.values))
          .to_dict()
)
xyz_full_keys = set(xyz_df["InChIKey_xyz"])

def resolve_to_xyz_key(smiles: str, target_full_keys: set[str]):
    """
    Return (isomeric_smiles, inchikey) that matches one of target_full_keys,
    or (None, None) if no match found.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return (None, None)

    ik0 = MolToInchiKey(m)
    if ik0 in target_full_keys:
        iso0 = Chem.MolToSmiles(m, isomericSmiles=True)
        return (iso0, ik0)

    m1 = Chem.Mol(m)
    Chem.AssignStereochemistry(m1, cleanIt=True, force=True)
    ik1 = MolToInchiKey(m1)
    if ik1 in target_full_keys:
        iso1 = Chem.MolToSmiles(m1, isomericSmiles=True)
        return (iso1, ik1)

    try:
        opts = StereoEnumerationOptions(onlyUnassigned=True, maxIsomers=256)
        isomers = list(EnumerateStereoisomers(m, options=opts))
    except Exception:
        isomers = []

    for iso_m in isomers:
        iso_smi = Chem.MolToSmiles(iso_m, isomericSmiles=True)
        ik = MolToInchiKey(iso_m)
        if ik in target_full_keys:
            return (iso_smi, ik)

    return (None, None)

DF["needs_fix"] = False
DF["status"] = "unchanged"
DF["SMILES_isomeric"] = None
DF["InChIKey_updated"] = None

for i, row in DF.iterrows():
    ik_ld  = row["InChIKey"]
    rk     = row["RootKey"]
    if ik_ld in xyz_full_keys:
        DF.at[i, "status"] = "already_matched"
        continue

    targets = root_to_full_xyz.get(rk, set())
    if targets:
        iso_smi, ik_new = resolve_to_xyz_key(row["SMILES"], targets)
        if ik_new is not None:
            DF.at[i, "SMILES_isomeric"]  = iso_smi
            DF.at[i, "InChIKey_updated"] = ik_new
            DF.at[i, "status"] = "fixed_to_xyz"
            DF.at[i, "needs_fix"] = True
        else:
            DF.at[i, "status"] = "root_match_but_no_full_match"
    else:
        DF.at[i, "status"] = "no_rootkey_overlap_with_xyz"

DF["InChIKey_final"] = DF["InChIKey_updated"].fillna(DF["InChIKey"])

out_all   = ld50_csv.with_name("InChIKey_LD50_SMILES_aligned.csv")
out_fixed = ld50_csv.with_name("InChIKey_LD50_SMILES_fixed_only.csv")
DF.to_csv(out_all, index=False)
DF.loc[DF["status"]=="fixed_to_xyz"].to_csv(out_fixed, index=False)

merged = DF.merge(xyz_df, left_on="InChIKey_final", right_on="InChIKey_xyz", how="inner")
merged_out = ld50_csv.with_name("ld50_xyz_merged.csv")
merged.to_csv(merged_out, index=False)

summary = DF["status"].value_counts().to_dict()
print("Summary:", summary)
print(f"Saved aligned table: {out_all}")
print(f"Saved fixed-only rows: {out_fixed}")
print(f"Merged (ready to use): {merged_out}")
