# ================================================================
# ================================================================
import random
from pathlib import Path
import numpy as np
import torch
import schnet_fp_property_oof as s4

SEEDS = [1, 2, 3, 4]

for s in SEEDS:
    out_dir = Path(f"{s4.TASK_DIR}/schnet_fp_property_trainseed{s}")
    csv = out_dir / f"schnet_fp_property_oof_probabilities_{s4.TASK}_seed{s}.csv"
    if csv.exists():
        print(f"[skip] seed {s}: {csv} already exists"); continue
    print(f"\n######## TASK={s4.TASK}  train seed={s} ########")
    s4.SEED = s
    s4.OUTPUT_DIR = str(out_dir)
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    s4.main()
