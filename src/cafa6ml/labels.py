from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def build_mlb_Y_and_train_pids(train_terms_tsv: str, aspect: str, all_train_pids: list[str]):
    terms = pd.read_csv(train_terms_tsv, sep="\t")
    aspect_df = terms[terms["aspect"].astype(str) == aspect].copy()
    if len(aspect_df) == 0:
        raise ValueError(f"No annotations for aspect '{aspect}'")

    aspect_df["EntryID"] = aspect_df["EntryID"].astype(str)
    aspect_df["term"] = aspect_df["term"].astype(str)

    protein_to_terms = aspect_df.groupby("EntryID")["term"].apply(list).to_dict()

    #keep proteins that have BOTH embeddings and terms
    train_pids = [pid for pid in all_train_pids if pid in protein_to_terms]

    labels = [protein_to_terms[pid] for pid in train_pids]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels).astype(np.float32)
    return train_pids, mlb, Y
