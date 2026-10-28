# ================================================================
# ================================================================
import numpy as np, pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

SDF_PATH = "data/ChemTastesDB/ChemTastesDB_multilabel_properties.sdf"
SEED_EMBED = 0xF00D
PROPS = ["MolecularWeight", "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount", "Complexity"]
NORM = {"bitter": "bitterness", "bitterness": "bitterness", "sweet": "sweetness", "sweetness": "sweetness",
        "sour": "sourness", "sourness": "sourness", "tasteless": "tastelessness", "tastelessness": "tastelessness",
        "no taste": "tastelessness", "notaste": "tastelessness", "none": "tastelessness"}
MAPS = {"binary": {"bitterness": 0, "sweetness": 1},
        "triclass": {"bitterness": 0, "sweetness": 1, "tastelessness": 2}}

def taste_prop(m):
    if m.HasProp("Class taste"): return "Class taste"
    for p in m.GetPropNames():
        if p.strip().lower() == "class taste": return p
    return None

suppl = Chem.SDMolSupplier(SDF_PATH, sanitize=False, removeHs=False)
recs = []
for mol in suppl:
    r = {"parsed": mol is not None}
    if mol is None: recs.append(r); continue
    ok = True
    try: Chem.SanitizeMol(mol)
    except Exception:
        try: Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception: ok = False
    r["sanitized"] = ok
    if not ok: recs.append(r); continue
    tp = taste_prop(mol)
    r["has_taste_prop"] = tp is not None
    if tp is None: recs.append(r); continue
    raw = mol.GetProp(tp).strip()
    r["raw_label"] = raw
    r["taste_key"] = NORM.get(raw.lower(), raw.lower())
    r["n_missing_props"] = sum(1 for p in PROPS if not mol.HasProp(p) or not mol.GetProp(p).strip() or mol.GetProp(p).strip().lower() == "nan")
    if mol.GetNumConformers() > 0:
        r["conf"] = "has_conformer"; r["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
    else:
        mH = Chem.AddHs(mol, addCoords=True); params = AllChem.ETKDGv3(); params.randomSeed = SEED_EMBED
        if AllChem.EmbedMolecule(mH, params) == 0:
            r["conf"] = "embedded_ok"; r["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mH))
        else:
            r["conf"] = "embed_failed"
    recs.append(r)

df = pd.DataFrame(recs)
print("Total records:", len(df))
print("Parse failures (None):", int((~df.parsed).sum()))
print("Sanitization failures:", int((df.parsed & (df.get("sanitized") == False)).sum()))
print("Missing Class taste property:", int((df.get("has_taste_prop") == False).sum()))
d = df[df.get("has_taste_prop") == True].copy()
print("\n[Raw Class taste label distribution]"); print(d.raw_label.str.lower().value_counts().to_string())

rows = []
for task, mp in MAPS.items():
    in_map = d[d.taste_key.isin(mp)]
    not_in_map = len(d) - len(in_map)
    emb_fail = int((in_map.conf == "embed_failed").sum())
    kept = in_map[in_map.conf != "embed_failed"]
    n_unique = kept.smiles.nunique()
    rows.append({"task": task, "initial_records": len(df),
                 "unparsable_or_unsanitizable": int((~df.parsed).sum() + (df.get("sanitized") == False).sum()),
                 "no_taste_label_prop": int((df.get("has_taste_prop") == False).sum()),
                 "label_not_in_task_map": not_in_map,
                 "conformer_embedding_failed": emb_fail,
                 "FINAL_N": len(kept), "unique_canonical_smiles(groups)": n_unique,
                 "kept_with_any_missing_property": int((kept.n_missing_props > 0).sum()),
                 **{f"final_{k}": int((kept.taste_key == k).sum()) for k in mp}})
out = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print("\n", out.T.to_string())
out.to_csv("exclusion_counts.csv", index=False); print("\nSaved: exclusion_counts.csv")
