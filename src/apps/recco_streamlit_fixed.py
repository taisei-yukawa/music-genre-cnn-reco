# -*- coding: utf-8 -*-
"""
Streamlit UI demo (fixed-config version) + 可視化＆除外機能（Streamlit標準連携版）
 - 推薦Top-Kの平均ベクトルを棒グラフ & レーダーチャート（matplotlib）で表示
 - PCA(2D)で「推薦地図」を Altair で表示（クエリを原点へ平行移動）
 - 楽曲を直接アップロード時、同一ファイル名のトラックを推薦から除外
 - さらに .mp3 ファイルのアップロードに対応し、librosa による CQT 変換機能を追加
Run:
  streamlit run recco_streamlit_fixed.py
"""

import io
import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

# 追加ライブラリ: mp3 から CQT を計算するために librosa を読み込む
import librosa

# 標準連携の可視化ライブラリ
import matplotlib.pyplot as plt
import altair as alt
from sklearn.decomposition import PCA

# ======================
# CONFIG
# ======================
CKPT_PATH = r"C:\Yukawa\Lab\CNN\singlelabel_recco_2_result_1\Models\filters_32_64_128_256_512_best_model.pth"
FILTERS   = [32, 64, 128, 256, 512]
INDEX_DIR = r"C:\Yukawa\Lab\CNN\singlelabel_recco_2_result_1\Reco_Index"
DEFAULT_NORM = "softmax"
DEFAULT_METRIC = "cosine"  # "cosine" | "l2" | "l1diff" | "jsd"
DEFAULT_TOPK = 10
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CLIP_SEC = 10

# CQT 計算用パラメータ
# モデル学習時の前処理に合わせる必要がある。ここでは仮の設定として
# 7 オクターブ * 12 ビン/オクターブ = 84 ビンとする。
CQT_SR = 22050
CQT_HOP_LENGTH = 512
CQT_N_BINS = 84
CQT_BINS_PER_OCTAVE = 12
CQT_FMIN = librosa.note_to_hz('C1')

# ======================
# Constants
# ======================
GENRES = ['blues','classical','country','disco','hiphop','jazz','metal','rock','pop']
GENRE2IDX = {g:i for i,g in enumerate(GENRES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Model (same as training)
# ======================
def _same_pad(ks, dil=(1,1)):
    return ((ks[0]-1)*dil[0]//2, (ks[1]-1)*dil[1]//2)

class ImprovedMusicGenreCNN(nn.Module):
    def __init__(self, filters, dropout=0.2, n_classes=9):
        super().__init__()
        defs = [
            {"ks":(7,3),"st":(2,1)}, {"ks":(5,3),"st":(2,1)},
            {"ks":(3,3),"st":(2,1)}, {"ks":(3,3),"st":(1,2)},
            {"ks":(3,3),"st":(1,1)},
        ]
        def block(in_ch, out_ch, ks, st):
            pad = _same_pad(ks)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=st, padding=pad, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )
        self.conv1 = block(1,          filters[0], **defs[0])
        self.conv2 = block(filters[0],  filters[1], **defs[1])
        self.conv3 = block(filters[1],  filters[2], **defs[2])
        self.conv4 = block(filters[2],  filters[3], **defs[3])
        self.conv5 = block(filters[3],  filters[4], **defs[4])
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(filters[4], 256)
        self.fc2   = nn.Linear(256, 64)
        self.fc3   = nn.Linear(64, n_classes)

    def forward(self, x):  # (B,1,F,T)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x); x = self.conv5(x)
        x = self.pool(x).flatten(1)
        x = self.drop(torch.nn.GELU()(self.fc1(x)))
        x = torch.nn.GELU()(self.fc2(x))
        return self.fc3(x)  # logits

# ======================
# Vector helpers
# ======================
def l2_unit(x, eps=1e-9):
    n = np.linalg.norm(x); return x/(n+eps)

def l1_prob(x, eps=1e-12):
    s = x.sum(); return x/(s+eps) if s>0 else x

def softmax_from_logits(z, temperature=1.0):
    z = z/float(max(1e-6, temperature)); z = z - z.max()
    p = np.exp(z); p /= (p.sum()+1e-12)
    return p

def js_distance(p, q, eps=1e-12):
    p = l1_prob(p.copy(), eps); q = l1_prob(q.copy(), eps)
    m = 0.5*(p+q)
    def kl(a,b): return np.sum(a*(np.log(a+eps)-np.log(b+eps)))
    return math.sqrt(max(0.0, 0.5*kl(p,m)+0.5*kl(q,m)))

def search(V: np.ndarray, q: np.ndarray, metric="cosine", topk=10):
    if metric == "cosine":
        Vn = np.stack([l2_unit(p) for p in V]); qn = l2_unit(q)
        sims = Vn @ qn
        idx = np.argsort(-sims)[:topk]; return idx, sims[idx]
    elif metric == "l2":
        d = np.linalg.norm(V - q[None,:], axis=1)
        idx = np.argsort(d)[:topk]; return idx, -d[idx]
    elif metric == "l1diff":
        d = np.sum(np.abs(V - q[None,:]), axis=1)
        idx = np.argsort(d)[:topk]; return idx, -d[idx]
    elif metric == "jsd":
        d = np.array([js_distance(p,q) for p in V])
        idx = np.argsort(d)[:topk]; return idx, -d[idx]
    else:
        raise ValueError(metric)

def extract_clip(cqt: np.ndarray, sr=22050, hop_length=512, clip_sec=10, start_frame=0) -> np.ndarray:
    assert cqt.ndim == 2, "Expected cqt shape (F, T)"
    fps = sr / hop_length; need = int(fps*clip_sec)
    F,T = cqt.shape
    if T <= need: return cqt
    start = max(0, min(T-need, int(start_frame)))
    return cqt[:, start:start+need]

def logits_to_space(vec: np.ndarray, norm: str, temperature: float) -> np.ndarray:
    if norm == "softmax": return softmax_from_logits(vec, temperature=temperature)
    if norm == "l2":      return l2_unit(vec)
    return vec  # "none"

# ======================
# Cached loaders 
# ======================
@st.cache_resource(show_spinner=False)
def load_model_fixed(ckpt_path: str, filters: list):
    model = ImprovedMusicGenreCNN(filters).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_index_fixed(index_dir: str, norm: str):
    idx_path = os.path.join(index_dir, f"index_{norm}.npy")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Index not found: {idx_path}")
    V = np.load(idx_path)
    meta_csv = os.path.join(index_dir, f"index_{norm}_meta.csv")
    meta = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else None
    return V, meta

# ======================
# Utils: 除外・可視化
# ======================
def get_exclude_indices(meta: pd.DataFrame, uploaded_names: list) -> list:
    """meta の 'path' または 'filename' とアップロード名（basename）を比較し、除外 index を返す"""
    if meta is None or len(uploaded_names) == 0:
        return []
    low_names = {n.lower() for n in uploaded_names}
    if 'path' in meta.columns:
        series = meta['path'].astype(str).fillna("").apply(lambda p: os.path.basename(p).lower())
    elif 'filename' in meta.columns:
        series = meta['filename'].astype(str).fillna("").str.lower()
    else:
        return []
    mask = series.isin(low_names)
    return np.where(mask.values)[0].tolist()

def plot_radar_mpl(genres, values, title="Mean of Top-K (Radar)", size=2.8, dpi=120):
    """
    matplotlib でレーダーチャート Figure を作成（小さめ＆余白少なめ）
    """
    angles = np.linspace(0, 2*np.pi, len(genres), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    vals   = np.concatenate([values, values[:1]])

    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi, subplot_kw={'projection': 'polar'})
    # 描画
    ax.plot(angles, vals, linewidth=1.2)
    ax.fill(angles, vals, alpha=0.25)

    # 文字サイズやパディングを厳しめに
    ax.set_thetagrids(np.degrees(angles[:-1]), genres)
    ax.set_title(title, fontsize=8, pad=6)
    ax.tick_params(labelsize=7, pad=2)
    # 図のスパインを細めにして見た目を締める
    ax.spines['polar'].set_linewidth(1.0)
    ax.grid(alpha=0.35)

    # 余白を手動で詰める（左, 下, 幅, 高さ）
    ax.set_position([0.07, 0.09, 0.86, 0.82])

    # tight_layout は極座標だと効きづらいので使わない
    return fig

def show_fig_tight(fig, width_px=420, dpi=200, pad_inches=0.02):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    buf.seek(0)
    st.image(buf, width=width_px)   # 必要なら use_container_width=True 試す
    plt.close(fig)

def plot_pca_map_altair(q_vec: np.ndarray, rec_vecs: np.ndarray, labels: list = None, title="Recommendation Map (PCA 2D)"):
    """Altair で PCA(2D) 散布図（クエリを原点へ平行移動）を返す"""
    X = np.vstack([q_vec[None, :], rec_vecs])  # (K+1, D)
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)
    origin = X2[0].copy()
    X2 = X2 - origin  # 平行移動でクエリを原点へ

    data = []
    # クエリ点
    data.append({"x": 0.0, "y": 0.0, "kind": "query", "label": "query"})
    # 推薦点
    for i in range(1, X2.shape[0]):
        data.append({
            "x": float(X2[i, 0]),
            "y": float(X2[i, 1]),
            "kind": "recommendation",
            "label": labels[i-1] if labels and i-1 < len(labels) else f"idx={i-1}"
        })
    df = pd.DataFrame(data)

    base = alt.Chart(df, title=title).encode(
        x=alt.X('x:Q', axis=alt.Axis(title='PC 1', grid=True)),
        y=alt.Y('y:Q', axis=alt.Axis(title='PC 2', grid=True)),
        tooltip=['kind:N', 'label:N', 'x:Q', 'y:Q']
    ).properties(width='container', height=400)

    rec = base.transform_filter(alt.datum.kind == "recommendation").mark_point(filled=True).encode(
        shape=alt.value('circle'),
        size=alt.value(80),
        color=alt.value('#1f77b4')
    )

    qry = base.transform_filter(alt.datum.kind == "query").mark_point(filled=True).encode(
        shape=alt.value('star'),
        size=alt.value(200),
        color=alt.value('#d62728')
    )

    return (rec + qry).interactive()

# ======================
# UI
# ======================
st.set_page_config(page_title="Music Recommender (Fixed Config)", layout="wide")
st.title("🎧 Music Recommender – UI Demo")
st.caption("※ モデルとインデックスの場所はコード先頭の CONFIG で固定してます。")

# サイドバー（必要最低限の操作のみ）
with st.sidebar:
    st.header("🔧 検索パラメータ")
    norm = st.selectbox("インデックスの正規化", ["softmax","l2","none"], index=["softmax","l2","none"].index(DEFAULT_NORM))
    metric = st.selectbox("類似度/距離", ["cosine","l2","l1diff","jsd"], index=["cosine","l2","l1diff","jsd"].index(DEFAULT_METRIC))
    topk = st.number_input("Top-K", min_value=1, max_value=100, value=DEFAULT_TOPK, step=1)
    temperature = st.number_input("Softmax 温度", min_value=0.05, max_value=5.0, value=DEFAULT_TEMPERATURE, step=0.05)
    clip_sec = st.number_input("クリップ秒数", min_value=1, max_value=30, value=DEFAULT_CLIP_SEC, step=1)

# 自動ロード & ステータス表示
model, V, meta = None, None, None
colA, colB = st.columns(2)
with colA:
    try:
        model = load_model_fixed(CKPT_PATH, FILTERS)
        st.success(f"✅ モデル読み込みOK: {os.path.basename(CKPT_PATH)} / FILTERS={FILTERS}")
    except Exception as e:
        st.error(f"モデル読み込み失敗: {e}")
with colB:
    try:
        V, meta = load_index_fixed(INDEX_DIR, norm)
        st.success(f"✅ インデックス読み込みOK: index_{norm}.npy (shape={None if V is None else V.shape})")
    except Exception as e:
        st.error(f"インデックス読み込み失敗: {e}")

# タブ
tab_song, tab_impr, tab_slider = st.tabs(["🎵 楽曲を直接入力", "🧠 印象ベクトル（整数）", "🎚 スライダー入力"])

# --- Tab 1: 楽曲を直接入力 ---
with tab_song:
    # .npy だけでなく mp3 も受け付ける
    up_files = st.file_uploader("cqt.npy を選択（複数可）", type=["npy", "mp3"], accept_multiple_files=True)
    uploaded_names = [up.name for up in up_files] if up_files else []

    if up_files and (model is not None) and (V is not None):
        try:
            vecs = []
            per_track_rows = []

            for up in up_files:
                # ファイル拡張子で分岐
                if up.name.lower().endswith('.npy'):
                    # 既存のロジック: npy から CQT を読み込む
                    cqt = np.load(io.BytesIO(up.getvalue()))
                elif up.name.lower().endswith('.mp3'):
                    # mp3 を librosa で読み込み、CQT を計算
                    y, sr_native = librosa.load(io.BytesIO(up.getvalue()), sr=CQT_SR, mono=True)
                    cqt_complex = librosa.cqt(
                        y,
                        sr=CQT_SR,
                        hop_length=CQT_HOP_LENGTH,
                        n_bins=CQT_N_BINS,
                        bins_per_octave=CQT_BINS_PER_OCTAVE,
                        fmin=CQT_FMIN
                    )
                    cqt = np.abs(cqt_complex)
                else:
                    # 想定外の拡張子
                    st.warning(f"未対応のファイル形式です: {up.name}")
                    continue

                # 共通処理: clip を抽出してモデルへ投げる
                clip = extract_clip(cqt, sr=CQT_SR, hop_length=CQT_HOP_LENGTH, clip_sec=clip_sec, start_frame=0)
                x = torch.tensor(clip[None, None, ...], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(x).squeeze(0).cpu().numpy()

                v = logits_to_space(logits, norm, temperature)
                if norm == "l2":
                    v = l2_unit(v)
                vecs.append(v)
                per_track_rows.append({"file": up.name, **{g: v[i] for i, g in enumerate(GENRES)}})

            if len(vecs) == 0:
                st.warning("有効なファイルがありませんでした。")
            else:
                Q = np.stack(vecs, axis=0)  # (M, 9)
                q_vec_song = Q.mean(axis=0)
                if norm == "softmax":
                    q_vec_song = l1_prob(q_vec_song)
                elif norm == "l2":
                    q_vec_song = l2_unit(q_vec_song)

                st.success(f"{len(vecs)} クエリを作成しました。")
                df_tracks = pd.DataFrame(per_track_rows)
                st.markdown("**各曲のベクトル**")
                st.dataframe(df_tracks, use_container_width=True)
                st.markdown("**集約クエリ（代表）**")
                st.write(pd.DataFrame({"genre": GENRES, "value": q_vec_song}))

        except Exception as e:
            st.error(f"処理に失敗しました: {e}")

# --- Tab 2: 印象ベクトル（整数） ---
with tab_impr:
    st.subheader("🧠 印象ベクトル（整数入力→確率に正規化）")
    st.caption("各ジャンルの整数（例: 0〜10）を入れて L1 正規化します。")
    cols = st.columns(3)
    impr_raw = {g:0 for g in GENRES}
    for i,g in enumerate(GENRES):
        with cols[i%3]:
            impr_raw[g] = st.number_input(g, min_value=0, max_value=100, value=0, step=1, key=f"impr_{g}")
    q_vec_impr = None
    if st.button("正規化してクエリにする", key="btn_vec_impr"):
        v = np.array([float(impr_raw[g]) for g in GENRES], dtype=np.float32)
        if v.sum() == 0:
            st.warning("すべて 0 です。1つ以上に正の値を入れてください。")
        else:
            q_vec_impr = l1_prob(v)
            if norm == "l2": q_vec_impr = l2_unit(q_vec_impr)
            st.success("クエリベクトルを生成しました。")
            st.write(pd.DataFrame({"genre": GENRES, "value": q_vec_impr}))

# --- Tab 3: スライダー入力 ---
with tab_slider:
    st.subheader("🎚 スライダー入力（整数→確率に正規化）")
    st.caption("UIスライダーで各ジャンルの重みを設定します。")
    cols2 = st.columns(3)
    slide_vals = {}
    for i,g in enumerate(GENRES):
        with cols2[i%3]:
            slide_vals[g] = st.slider(g, min_value=0, max_value=10, value=0, step=1, key=f"slide_{g}")
    q_vec_slide = None
    if st.button("正規化してクエリにする", key="btn_vec_slide"):
        v = np.array([float(slide_vals[g]) for g in GENRES], dtype=np.float32)
        if v.sum() == 0:
            st.warning("すべて 0 です。1つ以上に正の値を入れてください。")
        else:
            q_vec_slide = l1_prob(v)
            if norm == "l2": q_vec_slide = l2_unit(q_vec_slide)
            st.success("クエリベクトルを生成しました。")
            st.write(pd.DataFrame({"genre": GENRES, "value": q_vec_slide}))

# ---------------------
# Consolidate query vector
# ---------------------
q = None
for cand in [locals().get("q_vec_song"), locals().get("q_vec_impr"), locals().get("q_vec_slide")]:
    if isinstance(cand, np.ndarray):
        q = cand

st.markdown("---")
st.subheader("🔎 検索")
if V is None:
    st.info("インデックスが読み込めていません。CONFIG の INDEX_DIR / norm を確認してください。")
elif q is None:
    st.info("上のタブでクエリベクトルを作成してください。")
else:
    if st.button("検索を実行", type="primary"):
        try:
            # 除外対象（アップロードされたファイル名と一致）
            exclude_indices = get_exclude_indices(meta, uploaded_names)
            req_topk = int(topk)
            fetch_k = req_topk + len(exclude_indices)

            # 一旦多めに取得
            idx_raw, scores_raw = search(V, q, metric=metric, topk=fetch_k)

            # 除外適用
            idx_filtered = [i for i in idx_raw if i not in exclude_indices]
            scores_filtered = [s for i, s in zip(idx_raw, scores_raw) if i not in exclude_indices]

            # 最終Top-K
            idx = np.array(idx_filtered[:req_topk], dtype=int)
            scores = np.array(scores_filtered[:req_topk], dtype=float)

            approx_genres = [GENRES[int(np.argmax(V[i]))] for i in idx]
            df = pd.DataFrame({
                "rank": np.arange(1, len(idx)+1),
                "index": idx,
                "approx_genre": approx_genres,
                "score": np.round(scores, 6),
            })

            # メタ付与（あれば）
            if meta is not None:
                for c in ["true_genre", "path", "filename", "track_id", "title", "artist"]:
                    if c in meta.columns:
                        vals = []
                        for i in idx:
                            try:
                                vals.append(meta.iloc[i][c])
                            except Exception:
                                vals.append("")
                        df[c] = vals

            st.success("検索完了")
            st.dataframe(df, use_container_width=True)

            # === 推薦結果の平均ベクトル（棒 + レーダー） ===
            if len(idx) > 0:
                mean_vec = V[idx].mean(axis=0)  # (9,)
                st.markdown("###### 推薦結果の平均ベクトル")
                st.bar_chart(pd.DataFrame({"value": mean_vec}, index=GENRES))

                st.markdown("###### 推薦結果の平均ベクトル")
                fig_radar = plot_radar_mpl(GENRES, mean_vec, title="Mean of Top-K (Radar)", size=2.8, dpi=120)
                show_fig_tight(fig_radar, width_px=360)  # ← 幅を小さめに固定

            # === 推薦地図（PCA 2D, Altair） ===
            if len(idx) > 0:
                # hover 用ラベル
                labels = []
                if meta is not None and len(meta) > 0:
                    for i in idx:
                        label = None
                        title = (str(meta.iloc[i]["title"]) if "title" in meta.columns else "")
                        artist = (str(meta.iloc[i]["artist"]) if "artist" in meta.columns else "")
                        if title and title != "nan":
                            label = title
                            if artist and artist != "nan":
                                label = f"{title} - {artist}"
                        if not label:
                            if "filename" in meta.columns:
                                fn = str(meta.iloc[i]["filename"])
                                if fn and fn != "nan":
                                    label = fn
                        if not label and "path" in meta.columns:
                            p = str(meta.iloc[i]["path"])
                            if p and p != "nan":
                                label = os.path.basename(p)
                        labels.append(label if label else f"idx={i}")
                else:
                    labels = [f"idx={i}" for i in idx]

                st.markdown("###### 推薦地図（PCA）")
                chart = plot_pca_map_altair(q_vec=q, rec_vecs=V[idx], labels=labels, title="Recommendation Map (PCA)")
                st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"検索に失敗しました: {e}")

st.markdown("---")
with st.expander("ℹ️運用メモ"):
    st.write(f"""
- モデルとインデックスは**固定パス**（コード先頭の CONFIG を編集）。
- 現在の設定:
  - CKPT_PATH = `{CKPT_PATH}`
  - FILTERS   = {FILTERS}
  - INDEX_DIR = `{INDEX_DIR}`
- `norm` は UI で切替可。対応する `index_{{norm}}.npy` が `{INDEX_DIR}` に存在する必要あり。
- **アップロード曲の推薦除外**: meta の `path` または `filename` とアップロードファイル名（basename）を一致照合して除外。
- 推薦Top-Kの**平均ベクトル**を棒グラフ＆レーダーチャート（matplotlib）で表示。
- クエリと推薦Top-Kを PCA(2D) で可視化（Altair）。クエリは原点（0,0）に配置。
""")