# ================================================================================
#
#
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

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from scipy.stats import binomtest

# ================================================================
# ================================================================
TASK = "binary"
SEED = 42
SDF_PATH = "data/ChemTastesDB/ChemTastesDB_multilabel_properties.sdf"
BASE_DIR = "analysis/taste_baseline_cv"
TASK_DIR = f"{BASE_DIR}/{TASK}"
FOLD_JSON_PATH = f"{TASK_DIR}/folds_{TASK}_seed{SEED}.json"
OUTPUT_DIR = f"{TASK_DIR}/ablation_matrix_seed{SEED}"

SCHNET_OOF_CSV = f"{TASK_DIR}/schnet_seed{SEED}/schnet_oof_probabilities_{TASK}_seed{SEED}.csv"       # schnet_oof_probabilities.py: SchNet-only + SchNet+Property(FiLM)
MORGAN_RF_OOF_CSV = f"{TASK_DIR}/morgan_rf_oof_probabilities_{TASK}_seed{SEED}.csv"                    # morgan_rf_oof.py: Morgan FP-only
SCHNET_FP_PROPERTY_OOF_CSV = f"{TASK_DIR}/schnet_fp_property_seed{SEED}/schnet_fp_property_oof_probabilities_{TASK}_seed{SEED}.csv"  # schnet_fp_property_oof.py: SchNet+Property+FP

NUM_EPOCHS = 300
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
INNER_VAL_FRACTION = 0.20
FP_RADIUS = 2
FP_NBITS = 2048
BOOTSTRAP_B = 5000

device = torch.device("cpu")

LABEL_MAP_BIN = {"bitterness": 0, "sweetness": 1}
LABEL_MAP_TRI = {"bitterness": 0, "sweetness": 1, "tastelessness": 2}
BASE_PROP_NAMES = [
    "MolecularWeight", "XLogP", "TPSA", "HBondDonorCount",
    "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount", "Complexity",
]
LABEL_MAP = LABEL_MAP_BIN if TASK == "binary" else LABEL_MAP_TRI
NUM_CLASSES = len(LABEL_MAP)

RF_PARAMS = dict(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ================================================================
# ================================================================
class TasteDatasetFull(Dataset):
    def __init__(self, sdf_file, label_map, edge_cutoff=10.0, random_seed=0xF00D):
        self.data, self.edge_cutoff = [], edge_cutoff
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
            self.data.append({
                "pos": torch.tensor(pos, dtype=torch.float32),
                "z": torch.tensor(atomic_nums, dtype=torch.long),
                "edge_index": torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
                "edge_attr": torch.tensor(edge_attrs, dtype=torch.float32),
                "properties": torch.tensor(vals, dtype=torch.float32),
                "label": torch.tensor([label], dtype=torch.long),
                "group": can_smiles,
                "smiles": can_smiles,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def morgan_fp(smiles, radius=FP_RADIUS, n_bits=FP_NBITS):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for b in fp.GetOnBits():
        arr[b] = 1.0
    return arr


def metric_dict(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def paired_bootstrap_difference(y_true, pred_a, pred_b, metric="f1_macro", B=BOOTSTRAP_B, seed=SEED):
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def score(y, p):
        if metric == "accuracy":
            return accuracy_score(y, p)
        if metric == "f1_weighted":
            return f1_score(y, p, average="weighted", zero_division=0)
        return f1_score(y, p, average="macro", zero_division=0)

    observed = score(y_true, pred_a) - score(y_true, pred_b)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        diffs[b] = score(y_true[idx], pred_a[idx]) - score(y_true[idx], pred_b[idx])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_two_sided = min(1.0, 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return {"metric": metric, "difference_a_minus_b": observed, "ci95_low": lo, "ci95_high": hi,
            "p_bootstrap_two_sided": p_two_sided, "significant_0.05": bool((lo > 0) or (hi < 0))}


def exact_mcnemar(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    n01 = int(np.sum((~correct_a) & correct_b))
    n10 = int(np.sum(correct_a & (~correct_b)))
    discordant = n01 + n10
    p = 1.0 if discordant == 0 else binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue
    return {"n01_a_wrong_b_right": n01, "n10_a_right_b_wrong": n10, "p_exact": p}


def split_inner_train_val(ds, outer_train_idx, seed, val_fraction=0.20):
    outer_train_idx = np.asarray(outer_train_idx, dtype=int)
    labels = np.asarray([ds.data[i]["label"].item() for i in outer_train_idx])
    groups = np.asarray([ds.data[i]["group"] for i in outer_train_idx])
    n_splits = max(2, int(round(1.0 / val_fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_train_local, val_local = next(splitter.split(np.zeros(len(outer_train_idx)), labels, groups))
    return outer_train_idx[inner_train_local], outer_train_idx[val_local]


# ================================================================
# NEW Variant A: Property-only (MLP, no 3D / no fingerprint)
# ================================================================
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts, beta=0.999, gamma=1.5, num_classes=2):
        super().__init__()
        counts = np.maximum(np.array(class_counts, dtype=np.float64), 1.0)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        t = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
        log_pt = (t * log_probs).sum(dim=1)
        pt = log_pt.exp().clamp(1e-6, 1 - 1e-6)
        focal = (1 - pt).pow(self.gamma) * (-log_pt)
        return (self.weights[targets] * focal).mean()


class PropertyOnlyMLP(nn.Module):
    """Property-only control: same capacity as the SchNet PropertyBranch + classifier, using only the 8 properties (after SAFE preprocessing) without 3D information."""
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, properties):
        return self.net(properties)


def run_property_only(ds, fold_obj, tag="property_only", num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    n = len(ds)
    oof_true = np.full(n, -1, dtype=int)
    oof_fold = np.full(n, -1, dtype=int)
    oof_probs = np.full((n, NUM_CLASSES), np.nan, dtype=float)

    for fold_item in fold_obj["folds"]:
        fold = int(fold_item["fold"])
        outer_train = np.asarray(fold_item["train"], dtype=int)
        outer_test = np.asarray(fold_item["test"], dtype=int)
        inner_train, inner_val = split_inner_train_val(ds, outer_train, SEED + fold, INNER_VAL_FRACTION)

        y_test = np.asarray([ds.data[i]["label"].item() for i in outer_test])
        oof_true[outer_test] = y_test
        oof_fold[outer_test] = fold

        train_raw = np.stack([ds.data[i]["properties"].numpy() for i in inner_train], axis=0)
        imputer = SimpleImputer(strategy="median").fit(train_raw)
        imp = imputer.transform(train_raw)
        q1, q3 = np.quantile(imp, 0.25, axis=0), np.quantile(imp, 0.75, axis=0)
        keep_cols = np.where((q3 - q1) >= 1e-6)[0]
        if keep_cols.size == 0:
            keep_cols = np.arange(imp.shape[1])
        scaler = RobustScaler().fit(imp[:, keep_cols])

        def transform_idx(idx_array):
            raw = np.stack([ds.data[i]["properties"].numpy() for i in idx_array], axis=0)
            x = imputer.transform(raw)[:, keep_cols]
            return scaler.transform(x)

        X_train = torch.tensor(transform_idx(inner_train), dtype=torch.float32)
        X_val = torch.tensor(transform_idx(inner_val), dtype=torch.float32)
        X_test = torch.tensor(transform_idx(outer_test), dtype=torch.float32)
        y_train_t = torch.tensor([ds.data[i]["label"].item() for i in inner_train], dtype=torch.long)
        y_val_t = torch.tensor([ds.data[i]["label"].item() for i in inner_val], dtype=torch.long)

        model = PropertyOnlyMLP(in_dim=X_train.shape[1], num_classes=NUM_CLASSES).to(device)
        class_counts = np.bincount(y_train_t.numpy(), minlength=NUM_CLASSES)
        criterion = ClassBalancedFocalLoss(class_counts, beta=0.999, gamma=1.5, num_classes=NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_f1, patience, counter, delta = -1.0, 20, 0, 1e-4
        best_state = None
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train.to(device)), y_train_t.to(device))
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device)).argmax(1).cpu().numpy()
            val_f1 = f1_score(y_val_t.numpy(), val_pred, average="macro", zero_division=0)
            if val_f1 > best_val_f1 + delta:
                best_val_f1, counter = val_f1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                counter += 1
            if counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            out = model(X_test.to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
        oof_probs[outer_test] = probs
        pred = probs.argmax(axis=1)
        met = metric_dict(y_test, pred)
        print(f"  [{tag}][fold {fold}] acc={met['accuracy']:.4f} macro-F1={met['f1_macro']:.4f} "
              f"(best inner-val macro-F1={best_val_f1:.4f})")

    return _finish_oof(n, oof_fold, oof_true, oof_probs, tag)


# ================================================================
# NEW Variant B: SchNet + Property, simple CONCATENATION control (no FiLM)
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


class ConcatHead(nn.Module):
    """Concatenation control: concatenate [h3d ; hprop] without gamma/beta modulation. Parameter count and layers match the FusionHeadFiLM classifier so that only the conditioning scheme (FiLM vs concatenation) differs."""
    def __init__(self, h3d_dim=128, prop_dim=32, cls_dim=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(h3d_dim + prop_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, cls_dim),
        )

    def forward(self, h3d, hprop):
        return self.classifier(torch.cat([h3d, hprop], dim=1))


class SchNetTasteClassifier_Concat(nn.Module):
    def __init__(self, schnet_hidden=128, property_dim=None, num_classes=2):
        super().__init__()
        self.schnet = SchNet(hidden_channels=schnet_hidden, use_attention_pool=True)
        self.prop = PropertyBranch(property_dim, out_dim=32)
        self.head = ConcatHead(h3d_dim=schnet_hidden, prop_dim=32, cls_dim=num_classes)

    def forward(self, batch, properties):
        h3d = self.schnet(batch)
        return self.head(h3d, self.prop(properties))


def collate_fn_graph_prop(batch, prop_imputer, prop_scaler, keep_cols):
    graphs, properties, labels = [], [], []
    for sample in batch:
        graphs.append(Data(x=sample["z"].float(), z=sample["z"], pos=sample["pos"],
                            edge_index=sample["edge_index"], edge_attr=sample["edge_attr"]))
        vals = sample["properties"].numpy()
        vals = prop_imputer.transform(vals.reshape(1, -1)).reshape(-1)[keep_cols]
        vals = prop_scaler.transform(vals.reshape(1, -1)).reshape(-1)
        properties.append(torch.tensor(vals, dtype=torch.float32))
        labels.append(sample["label"])
    return Batch.from_data_list(graphs), torch.stack(properties), torch.cat(labels)


def build_optimizer_and_scheduler_film(model, base_lr=5e-4, total_epochs=300, prop_lr_mult=2.0,
                                        prop_boost_epochs=30, wd=1e-4):
    prop_params, other_params = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (prop_params if ("prop." in n_ or "to_gamma" in n_ or "to_beta" in n_) else other_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": base_lr, "weight_decay": wd},
        {"params": prop_params, "lr": base_lr * prop_lr_mult, "weight_decay": wd * 0.5},
    ])
    warmup_epochs = max(5, int(0.1 * total_epochs))

    def lr_lambda_other(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def lr_lambda_prop(epoch):
        w = lr_lambda_other(epoch)
        return w * (prop_lr_mult if epoch < prop_boost_epochs else 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda_other, lr_lambda_prop])
    return optimizer, scheduler


def _finish_oof(n, oof_fold, oof_true, oof_probs, tag):
    if np.any(oof_true < 0) or np.any(np.isnan(oof_probs).any(axis=1)):
        raise RuntimeError(f"[{tag}] Some samples have no OOF prediction.")
    pred_all = oof_probs.argmax(axis=1)
    overall = metric_dict(oof_true, pred_all)
    print(f"[{tag}] OOF overall:", overall)

    out_df = pd.DataFrame({"sample_idx": np.arange(n), "fold": oof_fold, "true_label": oof_true})
    for c in range(NUM_CLASSES):
        out_df[f"proba_{tag}_class{c}"] = oof_probs[:, c]
    out_df[f"pred_{tag}"] = pred_all
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / f"{tag}_oof_probabilities_{TASK}_seed{SEED}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return oof_true, pred_all, overall, out_path


def run_schnet_property_concat(ds, fold_obj, tag="schnet_property_concat", num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    n = len(ds)
    oof_true = np.full(n, -1, dtype=int)
    oof_fold = np.full(n, -1, dtype=int)
    oof_probs = np.full((n, NUM_CLASSES), np.nan, dtype=float)

    for fold_item in fold_obj["folds"]:
        fold = int(fold_item["fold"])
        outer_train = np.asarray(fold_item["train"], dtype=int)
        outer_test = np.asarray(fold_item["test"], dtype=int)
        inner_train, inner_val = split_inner_train_val(ds, outer_train, SEED + fold, INNER_VAL_FRACTION)

        y_test = np.asarray([ds.data[i]["label"].item() for i in outer_test])
        oof_true[outer_test] = y_test
        oof_fold[outer_test] = fold

        train_raw = np.stack([ds.data[i]["properties"].numpy() for i in inner_train], axis=0)
        imputer = SimpleImputer(strategy="median").fit(train_raw)
        imp = imputer.transform(train_raw)
        q1, q3 = np.quantile(imp, 0.25, axis=0), np.quantile(imp, 0.75, axis=0)
        keep_cols = np.where((q3 - q1) >= 1e-6)[0]
        if keep_cols.size == 0:
            keep_cols = np.arange(imp.shape[1])
        scaler = RobustScaler().fit(imp[:, keep_cols])
        property_dim = len(keep_cols)

        def cfn(batch):
            return collate_fn_graph_prop(batch, imputer, scaler, keep_cols)

        train_labels = np.asarray([ds.data[i]["label"].item() for i in inner_train])
        counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        weights_per_class = 1.0 / np.maximum(counts, 1)
        sample_weights = torch.DoubleTensor([weights_per_class[y] for y in train_labels])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(Subset(ds, inner_train), batch_size=BATCH_SIZE, sampler=sampler, collate_fn=cfn)
        val_loader = DataLoader(Subset(ds, inner_val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=cfn)
        test_loader = DataLoader(Subset(ds, outer_test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=cfn)

        model = SchNetTasteClassifier_Concat(schnet_hidden=128, property_dim=property_dim, num_classes=NUM_CLASSES).to(device)
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        criterion = ClassBalancedFocalLoss(class_counts, beta=0.999, gamma=1.5, num_classes=NUM_CLASSES).to(device)
        optimizer, scheduler = build_optimizer_and_scheduler_film(model, base_lr=lr, total_epochs=num_epochs)

        best_val_f1, patience, counter, delta = -1.0, 20, 0, 1e-4
        save_path = Path(OUTPUT_DIR) / tag / f"fold_{fold}" / "model.pth"
        for epoch in range(num_epochs):
            model.train()
            for g, p, y in train_loader:
                g, p, y = g.to(device), p.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(g, p), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            val_preds, val_gts = [], []
            with torch.no_grad():
                for g, p, y in val_loader:
                    out = model(g.to(device), p.to(device))
                    val_preds.extend(out.argmax(1).cpu().numpy())
                    val_gts.extend(y.numpy())
            val_f1 = f1_score(val_gts, val_preds, average="macro", zero_division=0)

            if val_f1 > best_val_f1 + delta:
                best_val_f1, counter = val_f1, 0
                os.makedirs(save_path.parent, exist_ok=True)
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
            if counter >= patience:
                break

        if save_path.exists():
            model.load_state_dict(torch.load(save_path, map_location=device))

        model.eval()
        preds, probs_list = [], []
        with torch.no_grad():
            for g, p, y in test_loader:
                out = model(g.to(device), p.to(device))
                probs_list.append(torch.softmax(out, dim=1).cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)
        oof_probs[outer_test] = probs
        pred = probs.argmax(axis=1)
        met = metric_dict(y_test, pred)
        print(f"  [{tag}][fold {fold}] acc={met['accuracy']:.4f} macro-F1={met['f1_macro']:.4f} "
              f"(property_dim={property_dim}, best inner-val macro-F1={best_val_f1:.4f})")

    return _finish_oof(n, oof_fold, oof_true, oof_probs, tag)


# ================================================================
# NEW Variant C: FP + Property (RF, no SchNet)
# ================================================================
def run_fp_plus_property(ds, fold_obj, tag="fp_plus_property"):
    n = len(ds)
    smiles_all = [ds.data[i]["smiles"] for i in range(n)]
    labels_all = np.array([ds.data[i]["label"].item() for i in range(n)])
    props_all = np.stack([ds.data[i]["properties"].numpy() for i in range(n)], axis=0)
    fp_all = np.stack([morgan_fp(s) for s in smiles_all], axis=0)

    oof_true = np.full(n, -1, dtype=int)
    oof_fold = np.full(n, -1, dtype=int)
    oof_probs = np.full((n, NUM_CLASSES), np.nan, dtype=float)

    for fold_item in fold_obj["folds"]:
        fold = int(fold_item["fold"])
        train_idx = np.asarray(fold_item["train"], dtype=int)
        test_idx = np.asarray(fold_item["test"], dtype=int)

        imputer = SimpleImputer(strategy="median").fit(props_all[train_idx])
        prop_train = imputer.transform(props_all[train_idx])
        prop_test = imputer.transform(props_all[test_idx])
        q1, q3 = np.quantile(prop_train, 0.25, axis=0), np.quantile(prop_train, 0.75, axis=0)
        keep_cols = np.where((q3 - q1) >= 1e-6)[0]
        if keep_cols.size == 0:
            keep_cols = np.arange(prop_train.shape[1])
        scaler = RobustScaler().fit(prop_train[:, keep_cols])
        prop_train = scaler.transform(prop_train[:, keep_cols])
        prop_test = scaler.transform(prop_test[:, keep_cols])

        X_train = np.concatenate([fp_all[train_idx], prop_train], axis=1)
        X_test = np.concatenate([fp_all[test_idx], prop_test], axis=1)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, labels_all[train_idx])
        proba = clf.predict_proba(X_test)
        full_proba = np.zeros((len(test_idx), NUM_CLASSES))
        for j, c in enumerate(clf.classes_):
            full_proba[:, c] = proba[:, j]

        oof_true[test_idx] = labels_all[test_idx]
        oof_fold[test_idx] = fold
        oof_probs[test_idx] = full_proba

        pred = full_proba.argmax(axis=1)
        met = metric_dict(labels_all[test_idx], pred)
        print(f"  [{tag}][fold {fold}] acc={met['accuracy']:.4f} macro-F1={met['f1_macro']:.4f}")

    return _finish_oof(n, oof_fold, oof_true, oof_probs, tag)


# ================================================================
# ================================================================
def aggregate_ablation_table():
    rows = []
    pred_lookup = {}
    y_true_ref = None

    def _load(path, pred_col, label, true_col="true_label"):
        nonlocal y_true_ref
        if not os.path.exists(path):
            print(f"[Note] {path} not found; variant '{label}' is excluded from the table.")
            return
        df = pd.read_csv(path).sort_values("sample_idx")
        y_true = df[true_col].to_numpy()
        pred = df[pred_col].to_numpy()
        if y_true_ref is None:
            y_true_ref = y_true
        elif not np.array_equal(y_true, y_true_ref):
            print(f"[Warning] {label}: true_label differs from the other variants; check sample_idx alignment and filtering.")
        m = metric_dict(y_true, pred)
        rows.append({"variant": label, **m})
        pred_lookup[label] = pred

    _load(SCHNET_OOF_CSV, "pred_3d_only", "SchNet-only (3D)")
    _load(SCHNET_OOF_CSV, "pred_3d_property", "SchNet + Property (FiLM)")
    _load(MORGAN_RF_OOF_CSV, "pred_morgan_rf", "Morgan FP-only (RF)")
    _load(SCHNET_FP_PROPERTY_OOF_CSV, "pred_fp_property", "SchNet + Property + FP")
    _load(Path(OUTPUT_DIR) / f"property_only_oof_probabilities_{TASK}_seed{SEED}.csv", "pred_property_only", "Property-only (MLP)")
    _load(Path(OUTPUT_DIR) / f"schnet_property_concat_oof_probabilities_{TASK}_seed{SEED}.csv", "pred_schnet_property_concat", "SchNet + Property (concat.)")
    _load(Path(OUTPUT_DIR) / f"fp_plus_property_oof_probabilities_{TASK}_seed{SEED}.csv", "pred_fp_plus_property", "FP + Property (RF)")

    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    print("\n=== 6-way ablation + concatenation control (task=%s) ===" % TASK)
    print(summary.to_string(index=False))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = Path(OUTPUT_DIR) / f"ablation_matrix_summary_{TASK}_seed{SEED}.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if "SchNet + Property (FiLM)" in pred_lookup and "SchNet + Property (concat.)" in pred_lookup:
        print("\n=== FiLM vs concatenation: significance test ===")
        for m in ["accuracy", "f1_weighted", "f1_macro"]:
            print(paired_bootstrap_difference(y_true_ref, pred_lookup["SchNet + Property (FiLM)"],
                                               pred_lookup["SchNet + Property (concat.)"], metric=m))
        print(exact_mcnemar(y_true_ref, pred_lookup["SchNet + Property (concat.)"], pred_lookup["SchNet + Property (FiLM)"]))

    return summary


def main():
    ds = TasteDatasetFull(SDF_PATH, label_map=LABEL_MAP, edge_cutoff=10.0)
    with open(FOLD_JSON_PATH, "r") as f:
        fold_obj = json.load(f)
    print(f"loaded {len(ds)} molecules for task={TASK}")

    print("\n--- Variant: Property-only (MLP) ---")
    run_property_only(ds, fold_obj)

    print("\n--- Variant: SchNet + Property (concatenation control) ---")
    run_schnet_property_concat(ds, fold_obj)

    print("\n--- Variant: FP + Property (RF) ---")
    run_fp_plus_property(ds, fold_obj)

    print("\n--- Merging with existing results into the final comparison table ---")
    aggregate_ablation_table()


if __name__ == "__main__":
    main()
