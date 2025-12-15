import os, yaml, numpy as np, pandas as pd
from scipy import sparse
from cafa6ml.fasta import load_fasta_ids
from cafa6ml.embeddings import load_embeddings_map
from cafa6ml.ontology import parse_obo
from cafa6ml.goa import load_goa_csv, build_negative_keys, build_positive_df

cfg=yaml.safe_load(open("configs/base.yaml"))
base=cfg["paths"]["base_dir"]

paths={
  "train_terms": f"{base}/Train/train_terms.tsv",
  "train_fasta": f"{base}/Train/train_sequences.fasta",
  "test_fasta":  f"{base}/Test/testsuperset.fasta",
  "obo":         f"{base}/Train/go-basic.obo",
  "goa":         f"{base}/goa_uniprot_all.csv",
  "ids":         f"{base}/esm-650m-embeds/protein_ids.csv",
  "emb":         f"{base}/esm-650m-embeds/protein_embeddings.npy",
}

print("== DATA CHECK ==")
for k,v in paths.items():
    print(k, "exists=", os.path.exists(v), "|", v)

print("\n== EMBEDDINGS CHECK ==")
emb_map = load_embeddings_map(paths["ids"], paths["emb"])
# print one emb dim
one = next(iter(emb_map.values()))
print("embeddings proteins:", len(emb_map), "dim:", len(one))

print("\n== FASTA IDS CHECK ==")
train_ids = load_fasta_ids(paths["train_fasta"])
test_ids  = load_fasta_ids(paths["test_fasta"])
train_hit = sum(1 for x in train_ids if x in emb_map)
test_hit  = sum(1 for x in test_ids  if x in emb_map)
print("train fasta:", len(train_ids), "hit:", train_hit)
print("test  fasta:", len(test_ids),  "hit:", test_hit)

print("\n== ONTOLOGY CHECK ==")
go_parents, go_children = parse_obo(paths["obo"])
print("go_parents terms:", len(go_parents), "go_children terms:", len(go_children))

print("\n== GOA CHECK ==")
if os.path.exists(paths["goa"]):
    goa = load_goa_csv(paths["goa"])
    neg = goa[goa["qualifier"].str.contains(cfg["goa"]["negative_contains"], na=False)]
    pos = goa[~goa["qualifier"].str.contains(cfg["goa"]["negative_contains"], na=False)]
    print("goa rows:", len(goa), "neg rows:", len(neg), "pos rows:", len(pos))

    neg_keys = build_negative_keys(goa, go_children, neg_token=cfg["goa"]["negative_contains"])
    pos_df = build_positive_df(goa, score=float(cfg["goa"]["add_score"]), neg_token=cfg["goa"]["negative_contains"])
    print("neg_keys:", len(neg_keys), "pos_df:", len(pos_df))
else:
    print("no goa file")

print("\n== OUTPUT CHECK ==")
sub_path="outputs/submissions/submission.tsv"
if os.path.exists(sub_path):
    df=pd.read_csv(sub_path, sep="\t", header=None, names=["pid","term","p"])
    c=df.groupby("pid").size()
    print("submission proteins:", c.shape[0], "rows:", len(df))
    print("preds/protein min/mean/p50/p90:", int(c.min()), float(c.mean()), float(c.median()), float(c.quantile(0.9)))
    print("p min/max:", float(df.p.min()), float(df.p.max()))
else:
    print("no submission yet:", sub_path)
