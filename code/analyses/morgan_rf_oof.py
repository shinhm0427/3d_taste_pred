# ================================================================================
#
# ================================================================================

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ================================================================
# ================================================================
TASK = "triclass"
SEED = 42
SDF_PATH = "data/ChemTastesDB/ChemTastesDB_multilabel_properties.sdf"
BASE_DIR = "analysis/taste_baseline_cv"
TASK_DIR = f"{BASE_DIR}/{TASK}"
FOLD_JSON_PATH = f"{TASK_DIR}/folds_{TASK}_seed{SEED}.json"
OUT_CSV = f"{TASK_DIR}/morgan_rf_oof_probabilities_{TASK}_seed{SEED}.csv"

LABEL_MAP_BIN = {"bitterness": 0, "sweetness": 1}
LABEL_MAP_TRI = {"bitterness": 0, "sweetness": 1, "tastelessness": 2}
LABEL_MAP = LABEL_MAP_BIN if TASK == "binary" else LABEL_MAP_TRI
NUM_CLASSES = len(LABEL_MAP)

FP_RADIUS = 2
FP_NBITS = 2048

RF_PARAMS = dict(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1)


# ================================================================
# ================================================================
def load_taste_records(sdf_file, label_map):
    norm = {
        "bitter": "bitterness", "bitterness": "bitterness",
        "sweet": "sweetness", "sweetness": "sweetness",
        "sour": "sourness", "sourness": "sourness",
        "tasteless": "tastelessness", "tastelessness": "tastelessness",
        "no taste": "tastelessness", "notaste": "tastelessness",
        "none": "tastelessness", "tastelessness ": "tastelessness",
    }
    label_map = {k.lower().strip(): int(v) for k, v in label_map.items()}
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)

    def normalize_label(raw):
        return norm.get(raw.strip().lower(), raw.strip().lower())

    def get_class_taste_prop(m):
        if m.HasProp("Class taste"):
            return "Class taste"
        for p in m.GetPropNames():
            if p.strip().lower() == "class taste":
                return p
        return None

    smiles_list, labels_list = [], []
    for mol in suppl:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                continue
        taste_prop = get_class_taste_prop(mol)
        if taste_prop is None:
            continue
        taste_key = normalize_label(mol.GetProp(taste_prop))
        if taste_key not in label_map:
            continue
        label = label_map[taste_key]

        if mol.GetNumConformers() == 0:
            mH = Chem.AddHs(mol, addCoords=True)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D
            if AllChem.EmbedMolecule(mH, params) != 0:
                continue
            try:
                AllChem.UFFOptimizeMolecule(mH)
            except Exception:
                pass
            mol = mH

        conf = mol.GetConformer()
        pos = conf.GetPositions()
        if pos is None or len(pos) == 0:
            continue

        can_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        smiles_list.append(can_smiles)
        labels_list.append(label)

    return smiles_list, np.array(labels_list, dtype=int)


def morgan_fp(smiles, radius=FP_RADIUS, n_bits=FP_NBITS):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for b in fp.GetOnBits():
        arr[b] = 1.0
    return arr


def main():
    smiles_all, labels_all = load_taste_records(SDF_PATH, LABEL_MAP)
    n = len(smiles_all)
    print(f"loaded {n} molecules (must match SchNet dataset size exactly)")

    X_all = np.stack([morgan_fp(s) for s in smiles_all], axis=0)

    with open(FOLD_JSON_PATH, "r") as f:
        fold_obj = json.load(f)

    oof_true = np.full(n, -1, dtype=int)
    oof_fold = np.full(n, -1, dtype=int)
    oof_proba = np.full((n, NUM_CLASSES), np.nan, dtype=float)

    for fold_item in fold_obj["folds"]:
        fold = int(fold_item["fold"])
        train_idx = np.asarray(fold_item["train"], dtype=int)
        test_idx = np.asarray(fold_item["test"], dtype=int)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_all[train_idx], labels_all[train_idx])
        proba = clf.predict_proba(X_all[test_idx])

        full_proba = np.zeros((len(test_idx), NUM_CLASSES))
        for j, c in enumerate(clf.classes_):
            full_proba[:, c] = proba[:, j]

        oof_true[test_idx] = labels_all[test_idx]
        oof_fold[test_idx] = fold
        oof_proba[test_idx] = full_proba

        pred = full_proba.argmax(axis=1)
        acc = accuracy_score(labels_all[test_idx], pred)
        f1m = f1_score(labels_all[test_idx], pred, average="macro", zero_division=0)
        print(f"[fold {fold}] Morgan-RF: acc={acc:.4f} macro-F1={f1m:.4f}")

    pred_all = oof_proba.argmax(axis=1)
    print(f"\nOOF overall: acc={accuracy_score(oof_true, pred_all):.4f} "
          f"macro-F1={f1_score(oof_true, pred_all, average='macro', zero_division=0):.4f}")
    print("(Morgan-RF scores should be close to the reference values for this configuration)")

    out_df = pd.DataFrame({"sample_idx": np.arange(n), "fold": oof_fold, "true_label": oof_true})
    for c in range(NUM_CLASSES):
        out_df[f"proba_morgan_rf_class{c}"] = oof_proba[:, c]
    out_df["pred_morgan_rf"] = pred_all
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
