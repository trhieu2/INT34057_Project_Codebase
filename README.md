## INT34057 Project — CAFA-6 Protein Function Prediction

Repo này là codebase cho cuộc thi Kaggle **CAFA-6 Protein Function Prediction**: dự đoán **GO terms** cho **protein** dựa trên **chuỗi acid amin**

- **Input:** CAFA6 dataset (FASTA + train_terms + go-basic.obo) + embeddings (protein_ids + protein_embeddings) + (optional) GOA annotations.
- **Output:** `submission.tsv` gồm 3 cột tab-separated: `protein_id`, `go_term`, `score`.

## What this project does (pipeline overview)

1. Download CAFA6 competition data + embeddings (+ optional GOA).
2. Build label space & training targets theo 3 aspects của GO: **C / F / P**.
3. Train 3 model MLP (mỗi aspect một model).
4. Predict trên test set và merge các dự đoán lại thành bảng `(protein_id, term, score)`.
5. **Ontology propagation:** lan truyền score từ child terms lên ancestor terms bằng max.
6. (Optional) **GOA postprocess:** loại các term “NOT” và/hoặc thêm GOA positives.
7. Export `submission.tsv`.

## TL;DR / Quickstart

> Các lệnh dưới đây chạy được trên local / Vast.ai / Colab / server bất kỳ, miễn là có Python và (khuyến nghị) GPU.

### 0) Clone repo
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```
### 1) Create environment + install
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -e .
```
### 2) (Required) Kaggle API token for data and submit
Tạo token kaggle.json trên Kaggle: Account → API → Create New Token.
Locate token đúng vị trí:
```bash
mkdir -p ~/.config/kaggle
cp /path/to/kaggle.json ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json
```
### 3) Download data + embeddings (+ optional GOA)
```bash
scripts/01_download_all.sh
```
### 4) Train
```bash
scripts/02_train.sh
```
### 5) Predict + create submission
```bash
scripts/03_predict.sh
```




