# CAFA6 Protein Function Prediction

**Bài toán:** dự đoán các **GO terms** (multi-label) cho protein ở 3 nhánh GO (**C/F/P**) từ **precomputed ESM-650M embeddings**.  
**Input:** embeddings + train labels (`train_terms.tsv`) + GO ontology (`go-basic.obo`) + (tuỳ chọn) GOA (`goa_uniprot_all.csv`)  
**Output:** `submission.tsv` (3 cột tab-separated: `protein_id`, `go_term`, `score`).
