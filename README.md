# Music Genre CNN (CQT / HPSS)

研究で使用したCNNベースの音楽ジャンル分類コードです。

## 特徴
- CQT/HPSS特徴に対応
- 学習・評価スクリプト（`scripts/`）
- Streamlitデモ（`src/apps/recco_streamlit_fixed.py`）

## セットアップ
```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -r requirements.txt
