from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from tqdm import tqdm
from .ontology import get_all_descendants

def load_goa_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()
    for col in ["protein_id","go_term","qualifier"]:
        if col not in df.columns:
            raise ValueError(f"GOA CSV missing column: {col}")
    df["protein_id"] = df["protein_id"].astype(str)
    df["go_term"] = df["go_term"].astype(str)
    df["qualifier"] = df["qualifier"].astype(str)
    return df

def build_negative_keys(goa_df: pd.DataFrame, go_children: Dict[str, List[str]], neg_token: str = "NOT") -> Set[str]:
    neg = goa_df[goa_df["qualifier"].str.contains(neg_token, na=False)]
    if len(neg) == 0:
        return set()
    neg = neg[["protein_id","go_term"]].drop_duplicates()
    neg_by_protein = neg.groupby("protein_id")["go_term"].apply(list).to_dict()

    cache = {}
    negative_keys = set()
    for pid, terms in tqdm(neg_by_protein.items(), desc="GOA negative propagation"):
        all_neg = set(terms)
        for t in terms:
            all_neg |= get_all_descendants(t, go_children, cache)
        for t in all_neg:
            negative_keys.add(f"{pid}_{t}")
    return negative_keys

def build_positive_df(goa_df: pd.DataFrame, score: float = 1.0, neg_token: str = "NOT") -> pd.DataFrame:
    pos = goa_df[~goa_df["qualifier"].str.contains(neg_token, na=False)]
    pos = pos[["protein_id","go_term"]].drop_duplicates()
    pos = pos.rename(columns={"protein_id":"pid","go_term":"term"})
    pos["p"] = float(score)
    return pos[["pid","term","p"]]

def apply_negative_filter(pred_df: pd.DataFrame, negative_keys: Set[str]) -> pd.DataFrame:
    if not negative_keys:
        return pred_df
    df = pred_df.copy()
    df["k"] = df["pid"].astype(str) + "_" + df["term"].astype(str)
    before = len(df)
    df = df[~df["k"].isin(negative_keys)].drop(columns=["k"])
    print(f"GOA negative removed {before-len(df)} rows ({before}->{len(df)})")
    return df

def add_goa_ground_truth(pred_df: pd.DataFrame, goa_pos_df: pd.DataFrame) -> pd.DataFrame:
    test_pids = set(pred_df["pid"].unique())
    add = goa_pos_df[goa_pos_df["pid"].isin(test_pids)].copy()
    combined = pd.concat([pred_df, add], ignore_index=True)
    combined = combined.groupby(["pid","term"])["p"].max().reset_index()
    print(f"GOA positives merged: {len(pred_df)} -> {len(combined)}")
    return combined
