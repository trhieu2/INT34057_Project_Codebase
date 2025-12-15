from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .model import MLP

class TestDataset(Dataset):
    def __init__(self, pids, emb_map):
        self.pids = pids
        self.emb_map = emb_map
    def __len__(self): return len(self.pids)
    def __getitem__(self, i):
        pid = self.pids[i]
        x = torch.tensor(self.emb_map[pid], dtype=torch.float32)
        return pid, x

def predict_aspect(cfg: dict, aspect: str, test_pids: list[str], emb_map: dict, ckpt_path: str, mlb_path: str) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]

    mcfg = cfg["model"]
    model = MLP(int(mcfg["input_dim"]), int(mcfg["hidden1"]), int(mcfg["hidden2"]), float(mcfg["dropout"]), len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    thr = float(cfg["predict"]["thresholds"][aspect])
    min_preds = int(cfg["predict"]["min_preds_per_protein"])

    loader = DataLoader(TestDataset(test_pids, emb_map), batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False, num_workers=3)

    rows = []
    with torch.no_grad():
        for pids, xb in tqdm(loader, desc=f"Predict {aspect}"):
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            for i, pid in enumerate(pids):
                s = probs[i]
                idx = np.where(s >= thr)[0]
                if len(idx) < min_preds:
                    idx = np.argsort(s)[-min_preds:][::-1]
                for j in idx:
                    rows.append((pid, str(classes[j]), float(s[j])))

    df = pd.DataFrame(rows, columns=["pid","term","p"])
    df = df.groupby(["pid","term"])["p"].max().reset_index()
    return df
