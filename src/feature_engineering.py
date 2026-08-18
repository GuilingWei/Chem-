"""
Builds the combined feature table (InChIKey + IR + ECFP + LD50).
Fingerprints come from morgan_bits.py. RDKit warnings are suppressed.

    from feature_engineering import build_and_save
    COMBINED = build_and_save("SMILES_Inchikey_IR_LD50.csv", "Broaden_Group.csv")

CLI: python feature_engineering.py --condition broadened/raw
"""

import argparse

import pandas as pd
from rdkit import RDLogger

from ECFP import morgan_bits

RDLogger.DisableLog("rdApp.*")

CONDITIONS = {
    "broadened": dict(input_csv="SMILES_Inchikey_IR_LD50.csv",
                       output_csv="Broaden_Group.csv", ir_prefix="f"),
    "raw": dict(input_csv="SMILES_Inchikey_IR_LD50_RAW.csv",
                output_csv="Broaden_Group_RAW.csv", ir_prefix="f"),
}


def build_and_save(input_csv, output_csv, ir_prefix="f", radius=2, n_bits=2048):
    df = pd.read_csv(input_csv)
    ir_cols = [c for c in df.columns if c.startswith(ir_prefix)]

    ecfp_list = [morgan_bits(smi, radius=radius, n_bits=n_bits) for smi in df["SMILES"]]
    ecfp_cols = [f"ecfp_{i}" for i in range(len(ecfp_list[0]))]
    ecfp = pd.DataFrame(ecfp_list, columns=ecfp_cols)
    ecfp.insert(0, "InChIKey", df["InChIKey"])

    left = df[["InChIKey", "LD50"] + ir_cols]
    combined = left.merge(ecfp, on="InChIKey", how="inner")

    combined.to_csv(output_csv, index=False)
    print(f"Saved {len(combined)} rows -> {output_csv}")
    return combined


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS.keys(), default=None)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--ir-prefix", type=str, default="f")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.condition:
        cfg = CONDITIONS[args.condition]
        input_csv, output_csv, ir_prefix = cfg["input_csv"], cfg["output_csv"], cfg["ir_prefix"]
    else:
        if not args.input or not args.output:
            raise SystemExit("Need --condition or both --input and --output.")
        input_csv, output_csv, ir_prefix = args.input, args.output, args.ir_prefix

    build_and_save(input_csv, output_csv, ir_prefix, args.radius, args.n_bits)


if __name__ == "__main__":
    main()