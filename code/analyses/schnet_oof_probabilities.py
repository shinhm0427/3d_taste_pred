"""Extract SchNet out-of-fold class probabilities from saved fold checkpoints (deterministic inference, no retraining).

Outputs are aligned by sample_idx / fold / true_label so that they can be combined with the OOF files of other models (Morgan-RF etc.).
Set SDF_PATH, FOLD_JSON_PATH, OUTPUT_DIR and TASK in the CONFIG block to match the training notebook.
"""
# ================================================================================
#
# ================================================================================

import os
import json
import math
import random
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, GlobalAttention, global_add_pool
# NOTE: torch_scatter import removed — scatter_softmax/scatter_sum were never used in this
# file (pooling uses torch_geometric's GlobalAttention instead). The import was leftover
# copy-paste from a different script and was causing a dlopen/ABI-mismatch crash against
# the locally installed torch build.

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# ================================================================
# ================================================================
TASK = "triclass"
SEED = 42
SDF_PATH = "data/ChemTastesDB/ChemTastesDB_multilabel_properties.sdf"
BASE_DIR = "analysis/taste_baseline_cv"
TASK_DIR = f"{BASE_DIR}/{TASK}"
FOLD_JSON_PATH = f"{TASK_DIR}/folds_{TASK}_seed{SEED}.json"
OUTPUT_DIR = f"{TASK_DIR}/schnet_seed{SEED}"
INNER_VAL_FRACTION = 0.20
BATCH_SIZE = 32

device = torch.device("cpu")

LABEL_MAP_BIN = {"bitterness": 0, "sweetness": 1}
LABEL_MAP_TRI = {"bitterness": 0, "sweetness": 1, "tastelessness": 2}
BASE_PROP_NAMES = [
    "MolecularWeight", "XLogP", "TPSA", "HBondDonorCount",
    "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount", "Complexity",
]

RUN_BINARY = (TASK == "binary")
LABEL_MAP = LABEL_MAP_BIN if RUN_BINARY else LABEL_MAP_TRI
NUM_CLASSES = len(LABEL_MAP)

PROPERTY_DIM = None
PROP_IMPUTER = None
PROP_SCALER = None
KEEP_COLS = None

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ================================================================
# ================================================================
class SchNetInteraction(MessagePassing):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super().__init__(aggr="add")
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, num_filters), nn.Softplus(), nn.Linear(num_filters, hidden_channels)
        )
        self.dense = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.Softplus(), nn.Linear(hidden_channels, hidden_channels)
        )
        self.register_buffer("offset", torch.linspace(0, cutoff, num_gaussians))
        self.register_buffer("coeff", torch.tensor(-0.5 / (cutoff / num_gaussians) ** 2))

    def forward(self, x, edge_index, edge_attr):
        distances = edge_attr.view(-1, 1)
        gauss = torch.exp(self.coeff * (distances - self.offset) ** 2)
        edge_weight = self.filter_network(gauss)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight

    def update(self, aggr_out, x):
        return x + self.dense(aggr_out)


class SchNet(nn.Module):
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6,
                 num_gaussians=100, cutoff=10.0, max_z=100, use_attention_pool=True):
        super().__init__()
        self.embedding = nn.Embedding(max_z + 1, hidden_channels)
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_channels, num_filters, num_gaussians, cutoff)
            for _ in range(num_interactions)
        ])
        if use_attention_pool:
            gate_nn = nn.Sequential(nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(),
                                     nn.Linear(hidden_channels // 2, 1))
            self.pool = GlobalAttention(gate_nn)
        else:
            self.pool = global_add_pool

    def forward(self, batch):
        x = self.embedding(batch.z)
        for interaction in self.interactions:
            x = interaction(x, batch.edge_index, batch.edge_attr)
        return self.pool(x, batch.batch)


class PropertyBranch(nn.Module):
    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class FusionHeadFiLM(nn.Module):
    def __init__(self, h3d_dim=128, prop_dim=32, cls_dim=3, use_ln_after=True):
        super().__init__()
        self.to_gamma = nn.Sequential(nn.Linear(prop_dim, h3d_dim), nn.ReLU(), nn.Linear(h3d_dim, h3d_dim), nn.Tanh())
        self.to_beta = nn.Sequential(nn.Linear(prop_dim, h3d_dim), nn.ReLU(), nn.Linear(h3d_dim, h3d_dim))
        self.ln_after = nn.LayerNorm(h3d_dim) if use_ln_after else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(h3d_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, cls_dim),
        )

    def forward(self, h3d, hprop):
        gamma = self.to_gamma(hprop)
        beta = self.to_beta(hprop)
        fused = self.ln_after(gamma * h3d + beta)
        return self.classifier(fused)


class SchNetTasteClassifier_FiLM(nn.Module):
    def __init__(self, schnet_hidden=128, property_dim=None, num_classes=NUM_CLASSES, use_properties=True):
        super().__init__()
        self.use_properties = use_properties
        self.schnet = SchNet(hidden_channels=schnet_hidden, use_attention_pool=True)
        if self.use_properties:
            self.prop = PropertyBranch(property_dim, out_dim=32)
            self.head = FusionHeadFiLM(h3d_dim=schnet_hidden, prop_dim=32, cls_dim=num_classes, use_ln_after=True)
        else:
            self.prop = None
            self.head = nn.Sequential(
                nn.LayerNorm(schnet_hidden),
                nn.Linear(schnet_hidden, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )

    def forward(self, batch, properties=None):
        h3d = self.schnet(batch)
        if self.use_properties and properties is not None:
            return self.head(h3d, self.prop(properties))
        return self.head(h3d)


class TasteDataset(Dataset):
    def __init__(self, sdf_file, label_map, edge_cutoff=10.0, try_embed_if_no_conf=True, random_seed=0xF00D):
        self.data, self.mols, self.edge_cutoff = [], [], edge_cutoff
        norm = {
            "bitter": "bitterness", "bitterness": "bitterness",
            "sweet": "sweetness", "sweetness": "sweetness",
            "sour": "sourness", "sourness": "sourness",
            "tasteless": "tastelessness", "tastelessness": "tastelessness",
            "no taste": "tastelessness", "notaste": "tastelessness",
            "none": "tastelessness", "tastelessness ": "tastelessness",
        }
        self.label_map = {k.lower().strip(): int(v) for k, v in label_map.items()}
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
            if taste_key not in self.label_map:
                continue
            label = self.label_map[taste_key]

            if mol.GetNumConformers() == 0:
                if not try_embed_if_no_conf:
                    continue
                mH = Chem.AddHs(mol, addCoords=True)
                params = AllChem.ETKDGv3()
                params.randomSeed = random_seed
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

            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            n = len(pos)
            edge_indices, edge_attrs = [], []
            for a in range(n):
                for b in range(a + 1, n):
                    dist = float(np.linalg.norm(pos[a] - pos[b]))
                    if dist < self.edge_cutoff:
                        edge_indices.extend([[a, b], [b, a]])
                        edge_attrs.extend([dist, dist])
            if len(edge_indices) == 0:
                for k in range(n):
                    edge_indices.append([k, k]); edge_attrs.append(0.0)

            vals = []
            for pn in BASE_PROP_NAMES:
                try:
                    v = float(mol.GetProp(pn))
                    vals.append(v if not np.isnan(v) else np.nan)
                except Exception:
                    vals.append(np.nan)

            can_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            self.mols.append(Chem.Mol(mol))
            mol_idx = len(self.mols) - 1

            self.data.append({
                "pos": torch.tensor(pos, dtype=torch.float32),
                "z": torch.tensor(atomic_nums, dtype=torch.long),
                "edge_index": torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
                "edge_attr": torch.tensor(edge_attrs, dtype=torch.float32),
                "properties": torch.tensor(vals, dtype=torch.float32),
                "label": torch.tensor([label], dtype=torch.long),
                "group": can_smiles,
                "mol_idx": mol_idx,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _fit_property_preprocessors_safe(ds, idx_array, iqr_eps=1e-6):
    global PROP_IMPUTER, PROP_SCALER, KEEP_COLS, PROPERTY_DIM
    train_raw = np.stack([ds.data[i]["properties"].numpy() for i in idx_array], axis=0)
    imputer = SimpleImputer(strategy="median").fit(train_raw)
    imp = imputer.transform(train_raw)
    q1, q3 = np.quantile(imp, 0.25, axis=0), np.quantile(imp, 0.75, axis=0)
    keep_cols = np.where((q3 - q1) >= iqr_eps)[0]
    if keep_cols.size == 0:
        keep_cols = np.arange(imp.shape[1])
    scaler = RobustScaler().fit(imp[:, keep_cols])
    PROP_IMPUTER, PROP_SCALER, KEEP_COLS = imputer, scaler, keep_cols
    PROPERTY_DIM = len(keep_cols)


def collate_fn(batch):
    graphs, properties, labels = [], [], []
    for sample in batch:
        graphs.append(Data(x=sample["z"].float(), z=sample["z"], pos=sample["pos"],
                            edge_index=sample["edge_index"], edge_attr=sample["edge_attr"]))
        vals = sample["properties"].numpy()
        vals = PROP_IMPUTER.transform(vals.reshape(1, -1)).reshape(-1)
        vals = vals[KEEP_COLS]
        vals = PROP_SCALER.transform(vals.reshape(1, -1)).reshape(-1)
        properties.append(torch.tensor(vals, dtype=torch.float32))
        labels.append(sample["label"])
    return Batch.from_data_list(graphs), torch.stack(properties), torch.cat(labels)


def load_outer_folds(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj


def split_inner_train_val(ds, outer_train_idx, seed, val_fraction=0.20):
    outer_train_idx = np.asarray(outer_train_idx, dtype=int)
    labels = np.asarray([ds.data[i]["label"].item() for i in outer_train_idx])
    groups = np.asarray([ds.data[i]["group"] for i in outer_train_idx])
    n_splits = max(2, int(round(1.0 / val_fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_train_local, val_local = next(splitter.split(np.zeros(len(outer_train_idx)), labels, groups))
    return outer_train_idx[inner_train_local], outer_train_idx[val_local]


def build_models_dict():
    return {
        "SchNet (3D Only)": SchNetTasteClassifier_FiLM(
            schnet_hidden=128, property_dim=PROPERTY_DIM, num_classes=NUM_CLASSES, use_properties=False),
        "SchNet (3D + Properties)": SchNetTasteClassifier_FiLM(
            schnet_hidden=128, property_dim=PROPERTY_DIM, num_classes=NUM_CLASSES, use_properties=True),
    }


@torch.no_grad()
def evaluate_model_with_probs(model, test_loader):
    model.eval()
    preds, labels, probs = [], [], []
    for batch_graphs, batch_properties, batch_labels in test_loader:
        outputs = model(batch_graphs.to(device), batch_properties.to(device))
        p = torch.softmax(outputs, dim=1)
        preds.extend(outputs.argmax(1).cpu().numpy())
        labels.extend(batch_labels.numpy())
        probs.extend(p.cpu().numpy())
    return np.array(preds), np.array(labels), np.array(probs)


# ================================================================
# ================================================================
def main():
    global NUM_CLASSES
    ds = TasteDataset(SDF_PATH, label_map=LABEL_MAP, edge_cutoff=10.0)
    n = len(ds)
    fold_obj = load_outer_folds(FOLD_JSON_PATH)
    output_dir = Path(OUTPUT_DIR)

    model_names = ["SchNet (3D Only)", "SchNet (3D + Properties)"]
    oof_true = np.full(n, -1, dtype=int)
    oof_fold = np.full(n, -1, dtype=int)
    oof_probs = {name: np.full((n, NUM_CLASSES), np.nan, dtype=float) for name in model_names}

    for fold_item in fold_obj["folds"]:
        fold = int(fold_item["fold"])
        outer_train = np.asarray(fold_item["train"], dtype=int)
        outer_test = np.asarray(fold_item["test"], dtype=int)
        inner_train, inner_val = split_inner_train_val(ds, outer_train, SEED + fold, INNER_VAL_FRACTION)

        y_test = np.asarray([ds.data[i]["label"].item() for i in outer_test])
        oof_true[outer_test] = y_test
        oof_fold[outer_test] = fold

        _fit_property_preprocessors_safe(ds, inner_train)
        test_loader = DataLoader(Subset(ds, outer_test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        for model_name in model_names:
            tag = model_name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
            ckpt_path = output_dir / f"fold_{fold}" / tag / f"{tag}_model.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            model = build_models_dict()[model_name].to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            pred, y_true_loader, probs = evaluate_model_with_probs(model, test_loader)
            if not np.array_equal(y_true_loader, y_test):
                raise RuntimeError(f"fold {fold} / {model_name}: test label order mismatch")
            oof_probs[model_name][outer_test] = probs
            acc = accuracy_score(y_test, pred)
            f1m = f1_score(y_test, pred, average="macro", zero_division=0)
            print(f"[fold {fold}] {model_name}: acc={acc:.4f} macro-F1={f1m:.4f}")

    out_df = pd.DataFrame({"sample_idx": np.arange(n), "fold": oof_fold, "true_label": oof_true})
    for name in model_names:
        tag = "3d_only" if "Only" in name else "3d_property"
        for c in range(NUM_CLASSES):
            out_df[f"proba_{tag}_class{c}"] = oof_probs[name][:, c]
        out_df[f"pred_{tag}"] = oof_probs[name].argmax(axis=1)

    out_path = output_dir / f"schnet_oof_probabilities_{TASK}_seed{SEED}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
