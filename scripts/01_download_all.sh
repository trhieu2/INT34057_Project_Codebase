#!/usr/bin/env bash
set -e

BASE="data/cafa-6-protein-function-prediction"
mkdir -p "$BASE"

echo "== Download competition data =="
kaggle competitions download -c cafa-6-protein-function-prediction -p "$BASE"
unzip -o "$BASE"/*.zip -d "$BASE"

echo "== Download ESM-650M embeddings dataset =="
mkdir -p data/_tmp_esm650m
kaggle datasets download -d xunphongtrn/esm-650m-embeds -p data/_tmp_esm650m --unzip
mkdir -p "$BASE/esm-650m-embeds"
find data/_tmp_esm650m -name "protein_embeddings.npy" -exec cp -f {} "$BASE/esm-650m-embeds/" \;
find data/_tmp_esm650m -name "protein_ids.csv"        -exec cp -f {} "$BASE/esm-650m-embeds/" \;

echo "== Download GOA CSV dataset =="
mkdir -p data/_tmp_goa
kaggle datasets download -d seddiktrk/protein-go-annotations -p data/_tmp_goa --unzip
cp -f data/_tmp_goa/goa_uniprot_all.csv "$BASE/goa_uniprot_all.csv"

echo "== Sanity check =="
ls -lah "$BASE/Train/train_terms.tsv"
ls -lah "$BASE/Train/go-basic.obo"
ls -lah "$BASE/esm-650m-embeds/protein_embeddings.npy"
ls -lah "$BASE/esm-650m-embeds/protein_ids.csv"
ls -lah "$BASE/goa_uniprot_all.csv"
