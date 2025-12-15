#!/usr/bin/env bash
set -e

COMP="cafa-6-protein-function-prediction"
FILE="${1:-submission.tsv}"
MSG="${2:-Project replication submit}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install: pip install kaggle"
  exit 1
fi

if [ ! -f "$FILE" ]; then
  echo "submission file not found: $FILE"
  exit 1
fi

# sanity check: 3 columns tab-separated
python - <<PY
import pandas as pd, sys
f="$FILE"
df=pd.read_csv(f, sep="\\t", header=None, names=["pid","term","p"])
assert df.shape[1]==3
assert df["p"].min()>0 and df["p"].max()<=1.0
print("submission shape:", df.shape, "| proteins:", df["pid"].nunique())
PY

kaggle competitions submit -c "$COMP" -f "$FILE" -m "$MSG"
echo "submitted: $FILE"
