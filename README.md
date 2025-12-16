# CAFA6 Protein Function Prediction

**Bài toán:** dự đoán các **GO terms** (multi-label) cho protein ở 3 nhánh GO (**C/F/P**) từ **precomputed ESM-650M embeddings**.  
**Input:** embeddings + train labels (`train_terms.tsv`) + GO ontology (`go-basic.obo`) + (tuỳ chọn) GOA (`goa_uniprot_all.csv`)  
**Output:** `submission.tsv` (3 cột tab-separated: `protein_id`, `go_term`, `score`).
## Pipeline (high level)

```text
  ├─ Download data/embeddings
  ├─ Build label space & targets per aspect (C/F/P)
  ├─ Train 3-head MLP (thực tế là 3 model: C, F, P)
  ├─ Predict test scores (C/F/P) → merge
  ├─ Ontology propagation (child → parent, max score)
  ├─ (Optional) GOA negative removal + add GOA positives
  └─ Export submission.tsv
## TL;DR / Quickstart (3–5 phút)

### 1) Create environment + install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
