from __future__ import annotations
import numpy as np
import pandas as pd
from .fasta import to_acc

def load_embeddings_map(ids_csv: str, emb_npy: str) -> dict[str, np.ndarray]:
    ids = pd.read_csv(ids_csv)
    col = "protein_id" if "protein_id" in ids.columns else ids.columns[0]
    raw = ids[col].astype(str).tolist()
    pids = [to_acc(x) for x in raw]

    X = np.load(emb_npy, mmap_mode="r")
    assert X.shape[0] == len(pids), (X.shape, len(pids))

    # if duplicates after normalization, keep first (rare but safe)
    out = {}
    for i, pid in enumerate(pids):
        if pid not in out:
            out[pid] = X[i]
    return out
