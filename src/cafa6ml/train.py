from __future__ import annotations
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from .model import MLP

class TrainDataset(Dataset):
    def __init__(self, pids, emb_map, Y):
        self.pids = pids
        self.emb_map = emb_map
        self.Y = Y

    def __len__(self): return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        x = torch.tensor(self.emb_map[pid], dtype=torch.float32)
        y = torch.tensor(self.Y[i], dtype=torch.float32)
        return pid, x, y

def train_one(aspect: str, cfg: dict, train_pids: list[str], emb_map: dict, Y: np.ndarray, ckpt_path: str, mlb_path: str, mlb):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TrainDataset(train_pids, emb_map, Y)

    n_train = int(len(ds) * float(cfg["train"]["train_val_split"]))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(int(cfg["seed"])))

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]))

    mcfg = cfg["model"]
    model = MLP(int(mcfg["input_dim"]), int(mcfg["hidden1"]), int(mcfg["hidden2"]), float(mcfg["dropout"]), Y.shape[1]).to(device)

    crit = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)

    best_val = 1e18
    best_state = None

    for ep in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        tl = 0.0
        for _, xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tl += float(loss.item())

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for _, xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += float(crit(model(xb), yb).item())

        sch.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{aspect}] epoch={ep:02d} train_loss={tl:.4f} val_loss={vl:.4f} best={best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"state_dict": model.state_dict(), "classes": mlb.classes_}, ckpt_path)
    with open(mlb_path, "wb") as f:
        pickle.dump(mlb, f)
