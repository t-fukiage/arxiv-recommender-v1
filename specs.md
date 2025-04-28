# arXiv Recommender - 技術仕様書

本仕様書は Python 3.10+ で動くコマンドラインツール `arxiv-recommender` の技術的な詳細をまとめたものです。自身の BibTeX ライブラリに基づいて arXiv の新着論文から推薦を生成します。Google Gemini API を活用（埋め込み生成、再ランク、クラスタラベリング）して動作します。BibTeX エントリをクラスタリングし、トピックごとに推薦を生成する機能も備えています。

## 1. アーキテクチャ全体

```
├── config.yaml                  # モデル・パラメータ等を一元管理
├── requirements.txt             # Python 依存パッケージ (pip)
├── environment.yml              # Conda 環境定義 (faiss-cpu を含む場合)
├── src/
│   └── arxiv_recommender/
│       ├── __init__.py
│       └── core/
│           ├── __init__.py
│           ├── main.py          # CLI エントリーポイント、メイン処理フロー
│           ├── ingest.py        # BibTeX & arXiv データ取り込み
│           ├── embed_gemini.py  # Gemini API 埋め込み
│           ├── index.py         # Faiss ANN インデックスラッパー
│           ├── rerank.py        # Gemini API 再ランク
│           ├── cluster.py       # HDBSCAN クラスタリング、重心計算、ラベル生成
│           └── utils.py         # 設定読込、ロギング設定
├── tests/                       # pytest テストコード (現状 basic)
├── cache/                       # Query Embedding, Clustering 結果キャッシュ (デフォルト)
└── output/                      # JSON/HTML 結果出力先 (デフォルト)
```

**データフロー:**

1.  **Ingest (`ingest.py`)**:
    *   `load_bibtex`: 指定された `.bib` ファイルを `bibtexparser` で読み込み、Title/Abstract を含む辞書のリストを生成。
    *   `fetch_arxiv`: `arxiv` ライブラリを使用し、指定日 (`--date`) または当日 (`None`) の新着論文を指定カテゴリ (`config.fetch.arxiv_categories` またはデフォルト `CS_CATS`) から取得。
2.  **Embed Query (`embed_gemini.py`)**:
    *   BibTeX エントリの Title/Abstract からベクトルを生成。
    *   `task_type="clustering"` が指定される。
    *   結果は `cache/` にモデル名を含むファイル名 (`.npz`) でキャッシュ。`--refresh-embeddings` で再生成。
3.  **Cluster (`cluster.py`, オプション `--cluster`)**:
    *   キャッシュされた BibTeX ベクトルを使用 (なければ Embed Query から)。
    *   `hdbscan` ライブラリを用いてベクトルをクラスタリング (`min_cluster_size` をパラメータとして使用)。ノイズ点は `-1` に割り当て。
    *   各クラスタの重心ベクトルを計算。
    *   クラスタリング結果 (ラベル、重心) を `cache/` にキャッシュ。`--refresh-embeddings` で再生成。
    *   `label_clusters_gemini`: 各クラスタからサンプルテキストを抽出し、`config.cluster.label_model` (Gemini) を使ってクラスタラベルを生成。ラベルもキャッシュ。
4.  **Embed Corpus (`embed_gemini.py`)**:
    *   `fetch_arxiv` で取得した arXiv 新着論文の Title/Abstract からベクトルを生成 (キャッシュなし)。
    *   `task_type="clustering"` が指定される。
5.  **Index (`index.py`)**:
    *   `ANNIndex(dim, factory, nprobe)`: Faiss インデックスのラッパークラス。
    *   `factory`: `config.index.factory` 文字列でインデックスタイプを指定。
    *   `nprobe`: `config.index.nprobe` 値 (IVF 系インデックス用)。
    *   `add(vectors)`: ベクトルをインデックスに追加。
    *   `search(query_vectors, k)`: k-NN 検索を実行。
6.  **Search & Rerank (`main.py`, `rerank.py`)**:
    *   **Clustering Mode (`--cluster`)**:
        *   各クラスタの重心ベクトルと arXiv 論文の所属クラスタ (最近傍重心) に基づき、クラスタごとの候補を検索。
        *   `config.rerank.enable = true` の場合、`rerank_gemini` で候補を再ランク。
    *   **Non-Clustering Mode**:
        *   BibTeX 全体の重心ベクトルで Faiss インデックスを検索 (`top_k * oversample`)。
        *   `config.rerank.enable = true` の場合、`rerank_gemini` で候補を再ランク。
    *   再ランク無効時は ANN の距離スコア（負値）を使用。
7.  **Output (`main.py`)**:
    *   最終的な推薦結果（上位 `top_k`）を `output/` に `arxiv_reco_YYYYMMDD.json` および `.html` として保存。HTML はクラスタリング時にグループ化される。

## 2. 設定ファイル (`config.yaml` 例)

```yaml
# config.yaml

cache_dir: cache            # Query Embedding, Clustering 結果キャッシュ

# --- Gemini API Settings --- 
gemini:
  model_name: models/text-embedding-004 # Gemini Embedding モデル名
  api_key: null             # APIキー (null or 省略時は環境変数 GEMINI_API_KEY を使用)
  batch_size: 10            # 埋め込みバッチサイズ

# --- ANN Index Settings (Faiss) ---
index:
  factory: HNSW32           # Faiss Index Factory 文字列 (例: "Flat", "HNSW32", "IVF1024,Flat")
  nprobe: 32                # IVF系インデックスの検索時 nprobe 値

# --- Reranking Settings ---
rerank:
  enable: false             # true で再ランク有効化
  oversample: 4             # ANN検索 -> 再ランク時の候補数オーバーサンプル係数 (top_k * oversample)
  gemini_model: gemini-2.0-flash-lite # 再ランク用 Gemini モデル (enable=true 時)

# --- Clustering Settings (オプション --cluster 使用時) ---
cluster:
  algorithm: hdbscan        # 現在 hdbscan のみサポート (ライブラリ依存)
  min_cluster_size: 40      # hdbscan の min_cluster_size パラメータ
  label_model: gemini-2.0-flash-lite # クラスタラベル生成用 Gemini モデル

# --- Output Settings ---
output:
  top_k: 100                # 最終的な推薦数 (全体またはクラスタごと)
  dir: output               # 結果出力ディレクトリ

# --- ArXiv Fetch Settings ---
fetch:
  arxiv_categories:         # 取得する arXiv カテゴリのリスト (オプション)
    - cs.AI                 # 例: ['cs.LG', 'stat.ML']
    - cs.LG                 # 空リストまたはキーが存在しない場合、ingest.py 内の CS_CATS デフォルトを使用
    - cs.CV
    - cs.CL
    - cs.GR
    - cs.HC
    - stat.ML
    - q-bio.NC
    - q-bio.QM
```

## 3. 主要モジュール仕様

### 3.1 `ingest.py`

-   `load_bibtex(path)`: `bibtexparser` を使用。BibTeX ファイルをパースし、`title`, `abstract`, `id` (DOI または arXiv ID) を含む辞書のリストを返す。Abstract がない場合は Title を使用。
-   `fetch_arxiv(date, cats)`: `arxiv` ライブラリを使用。指定 `date` (YYYY-MM-DD) の論文を取得。`cats` リストが提供されれば、それらのカテゴリに合致する論文のみを取得 (`OR` 検索)。`cats` が空または `None` の場合、内部定義の `CS_CATS` リストを使用。取得した論文情報を `title`, `abstract`, `id` (arXiv ID) を含む辞書のリストとして返す。

### 3.2 `embed_gemini.py`

-   `embed_gemini(texts, api_key, model_name, batch_size, task_type)`: `google.generativeai` ライブラリを使用。
    -   `api_key`: 環境変数 `GEMINI_API_KEY` または config から取得。
    -   `model_name`: `config.gemini.model_name` を使用。
    -   指定 `task_type` (e.g., "clustering") で API にリクエスト。
    -   レート制限 (429) 時に指数バックオフ付きリトライを実装。

### 3.3 `index.py`

-   `ANNIndex(dim, factory, nprobe)`: Faiss インデックスのラッパークラス。
    -   `factory`: `config.index.factory` 文字列でインデックスタイプを指定。
    -   `nprobe`: `config.index.nprobe` 値 (IVF 系インデックス用)。
    -   `add(vectors)`: ベクトルをインデックスに追加。
    -   `search(query_vectors, k)`: k-NN 検索を実行。

### 3.4 `rerank.py`

-   `rerank_gemini(pairs, api_key, model_name)`: `google.generativeai.GenerativeModel` を使用。
    -   `api_key`: 環境変数 `GEMINI_API_KEY` または config から取得。
    -   `model_name`: `config.rerank.gemini_model` を使用。
    -   各 `(query, passage)` ペアについて、関連度スコア (0.0-1.0) を返すようプロンプトで指示 (`temperature=0.0`)。
    -   レート制限 (429) 時に指数バックオフ付きリトライを実装。
    -   APIエラーやスコア抽出失敗時は `0.0` を返すフォールバックあり。

### 3.5 `cluster.py`

-   `cluster_embeddings(embeddings, min_cluster_size)`: `hdbscan` ライブラリを使用。
    -   入力された `embeddings` をクラスタリング。
    -   `min_cluster_size` をパラメータとして渡す。
    -   結果としてクラスタラベル配列と、各クラスタIDに対応する重心ベクトル辞書を返す。
-   `sample_texts_per_cluster(cluster_labels, texts)`: 各クラスタから代表的なテキスト（Title+Abstract）をサンプリングする（ラベル生成用）。
-   `label_clusters_gemini(samples, api_key, model_name, cache_file)`:
    -   `samples` (クラスタID -> テキストリスト) を受け取る。
    -   `google.generativeai` を使用し、指定 `model_name` (`config.cluster.label_model`) で各クラスタのトピックを表す短いラベルを生成するようプロンプトで指示。
    -   結果は `cache_file` (JSON) にキャッシュされ、次回実行時に再利用可能。

### 3.6 `utils.py`

-   `load_config(path)`: `PyYAML` を使用して設定ファイルを読み込む。
-   `setup_logger()`: 標準の `logging` モジュールを設定。ログレベルは環境変数 `LOGLEVEL` で制御可能（デフォルト `INFO`）。

## 4. 依存関係 (`requirements.txt`)

```txt
# 主要ライブラリ
arxiv                 # arXiv API アクセス
bibtexparser          # BibTeX ファイルのパース
PyYAML                # 設定ファイル (config.yaml) の読み込み
numpy                 # 数値計算、ベクトル操作
tqdm                  # プログレスバー表示

# ANN 検索用 (Faiss)
faiss-cpu             # (Conda でのインストール推奨、下記参照)

# クラスタリング用 (オプション --cluster)
scikit-learn          # HDBSCAN の依存関係など
hdbscan               # HDBSCAN クラスタリングアルゴリズム実装

# Google Gemini API 用
google-generativeai   # Gemini API クライアント

# テスト用
pytest                # テストフレームワーク
```
**注意:** `faiss-cpu` は `pip` でのインストールが複雑な場合があるため、特に macOS では `conda` でのインストールが推奨されます。`hdbscan` ライブラリも別途インストールが必要です (`pip install hdbscan`)。`requirements.txt` がこれらの全てをリストしているか確認が必要です。

## 5. コマンドライン引数

`python src/arxiv_recommender/core/main.py [OPTIONS]`

-   `--bib <path>`: **(必須)** 入力 BibTeX ファイルのパス。
-   `--config <path>`: 設定ファイルのパス (デフォルト: `config.yaml`)。
-   `--date <YYYY-MM-DD>`: arXiv 論文を取得する日付 (デフォルト: 実行日の UTC)。`today` も可。
-   `--topk <N>`: 出力する推薦数 (全体またはクラスタごと) (config の `output.top_k` を上書き)。
-   `--cluster`: このフラグを立てると BibTeX ライブラリのクラスタリングを実行し、結果をクラスタごとに表示。
-   `--refresh-embeddings`: このフラグを立てると、キャッシュされた BibTeX 埋め込みとクラスタリング結果を無視して再計算・再キャッシュする。BibTeX ファイル更新時やモデル変更時に使用。
-   `--debug`: このフラグを立てると、BibTeX の先頭 100 件のみを使用してデバッグ実行する。
-   `--n_clusters <N>`: *(現在 HDBSCAN 使用のため実質的に未使用)*

## 6. 環境変数

-   `GEMINI_API_KEY`: Google AI API キー。`config.yaml` の `gemini.api_key` より優先される。設定されていない場合はエラー。
-   `LOGLEVEL`: ログレベル (`DEBUG`, `INFO`, `WARNING`, `ERROR`)。デフォルト `INFO`。

## 7. 環境構築例 (Conda + pip ハイブリッド)

`faiss-cpu` のインストールを考慮した推奨手順。

```bash
# 1. リポジトリをクローン (任意)
# git clone <repository_url>
# cd <repository_directory>

# 2. Conda 環境を作成・有効化 (environment.yml を使用する場合)
# environment.yml 例:
# name: arxiv-reco
# channels:
#   - conda-forge
#   - defaults
# dependencies:
#   - python=3.10
#   - faiss-cpu # Faiss を Conda で管理
# conda env create -f environment.yml
# conda activate arxiv-reco

# (environment.yml を使わない場合)
# conda create -n arxiv-reco python=3.10 -y
# conda activate arxiv-reco
# conda install -c conda-forge faiss-cpu -y

# 3. pip をアップグレードし、requirements.txt で残りをインストール
pip install --upgrade pip
pip install -r requirements.txt
# hdbscan が requirements.txt になければ追加インストール
# pip install hdbscan

# 4. 動作確認
export GEMINI_API_KEY='YOUR_API_KEY' # Gemini API キーを設定
python src/arxiv_recommender/core/main.py --bib path/to/your.bib --date today --debug
```

## 8. 拡張アイデア

-   Faiss PQ (Product Quantization) の明示的なサポートとパラメータ調整。
-   `usearch` や他の ANN ライブラリのサポート。
-   ベクトル DB (Chroma, LanceDB) 連携オプション。
-   HTML 出力の改善、静的サイトホスティング連携。
-   通知機能 (Slack, Email)。
-   他のクラスタリングアルゴリズム (KMeansなど) のサポートと `--n_clusters` 引数の活用。

