from __future__ import annotations
import os
import yaml
import pandas as pd

from .utils import seed_everything, ensure_dir
from .fasta import load_fasta_ids
from .embeddings import load_embeddings_map
from .labels import build_mlb_Y_and_train_pids
from .train import train_one
from .predict import predict_aspect
from .ontology import parse_obo, propagate_predictions
from .goa import (
    load_goa_csv,
    build_negative_keys,
    build_positive_df,
    apply_negative_filter,
    add_goa_ground_truth,
)

def load_cfg(cfg_path: str) -> dict:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    base = cfg["paths"]["base_dir"]
    cfg["paths"]["train_terms"] = f"{base}/Train/train_terms.tsv"
    cfg["paths"]["train_fasta"] = f"{base}/Train/train_sequences.fasta"
    cfg["paths"]["test_fasta"]  = f"{base}/Test/testsuperset.fasta"
    cfg["paths"]["obo"]         = f"{base}/Train/go-basic.obo"
    cfg["paths"]["goa_csv"]     = f"{base}/goa_uniprot_all.csv"
    cfg["paths"]["esm_dir"]     = f"{base}/esm-650m-embeds"
    cfg["paths"]["esm_embeddings"] = f"{cfg['paths']['esm_dir']}/protein_embeddings.npy"
    cfg["paths"]["esm_ids"]        = f"{cfg['paths']['esm_dir']}/protein_ids.csv"
    return cfg

def cmd_train(cfg_path: str):
    cfg = load_cfg(cfg_path)
    seed_everything(int(cfg["seed"]))
    p = cfg["paths"]
    ensure_dir("outputs/checkpoints")

    # all train ids from FASTA, normalized to accession by fasta.to_acc()
    all_train_ids = load_fasta_ids(p["train_fasta"])
    emb_map = load_embeddings_map(p["esm_ids"], p["esm_embeddings"])

    # keep only those with embeddings
    all_train_pids = [pid for pid in all_train_ids if pid in emb_map]
    print("all_train_pids with embeddings:", len(all_train_pids))

    for asp in cfg["train"]["aspects"]:
        #train only proteins that have labels for this aspect
        train_pids_asp, mlb, Y = build_mlb_Y_and_train_pids(p["train_terms"], asp, all_train_pids)
        print(f"[{asp}] train_pids={len(train_pids_asp)} Y={Y.shape} classes={len(mlb.classes_)}")

        ckpt = f"outputs/checkpoints/mlp_{asp}.pt"
        mlb_path = f"outputs/checkpoints/mlb_{asp}.pkl"
        train_one(asp, cfg, train_pids_asp, emb_map, Y, ckpt, mlb_path, mlb)

def cmd_predict(cfg_path: str):
    cfg = load_cfg(cfg_path)
    seed_everything(int(cfg["seed"]))
    p = cfg["paths"]
    ensure_dir("outputs/submissions")

    test_ids = load_fasta_ids(p["test_fasta"])
    emb_map = load_embeddings_map(p["esm_ids"], p["esm_embeddings"])
    test_pids = [pid for pid in test_ids if pid in emb_map]
    print("test_pids with embeddings:", len(test_pids))

    dfs = []
    for asp in cfg["train"]["aspects"]:
        ckpt = f"outputs/checkpoints/mlp_{asp}.pt"
        mlb_path = f"outputs/checkpoints/mlb_{asp}.pkl"
        if not os.path.exists(ckpt) or not os.path.exists(mlb_path):
            raise FileNotFoundError(f"Missing checkpoint for {asp}: {ckpt} or {mlb_path}")
        dfs.append(predict_aspect(cfg, asp, test_pids, emb_map, ckpt, mlb_path))

    pred_df = pd.concat(dfs, ignore_index=True)
    pred_df = pred_df.groupby(["pid","term"])["p"].max().reset_index()

    go_parents, go_children = parse_obo(p["obo"])
    prop_df = propagate_predictions(pred_df, go_parents)

    if cfg["goa"]["enable"]:
        goa_df = load_goa_csv(p["goa_csv"])
        neg_keys = build_negative_keys(goa_df, go_children, neg_token=cfg["goa"]["negative_contains"])
        prop_df = apply_negative_filter(prop_df, neg_keys)
        pos_df = build_positive_df(goa_df, score=float(cfg["goa"]["add_score"]), neg_token=cfg["goa"]["negative_contains"])
        prop_df = add_goa_ground_truth(prop_df, pos_df)

    out = "submission.tsv" 
    prop_df = prop_df.sort_values("p", ascending=False)
    prop_df.to_csv(out, sep="\t", index=False, header=False)
    print("Saved:", out)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("command", choices=["train", "predict"])
    args = ap.parse_args()

    if args.command == "train":
        cmd_train(args.config)
    else:
        cmd_predict(args.config)

if __name__ == "__main__":
    main()
