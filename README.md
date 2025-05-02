# arXiv Recommender

BibTeX ライブラリに基づいてパーソナライズされた arXiv 論文推薦ツールです。Google Gemini を利用しています。

このコマンドラインツールは、手持ちの BibTeX ライブラリ内の論文アブストラクトやタイトルに基づいて、arXiv の新着論文から推薦を生成します。Google Gemini API を活用し、高品質なテキスト埋め込み、オプションでの再ランク付け、クラスタラベリングを行います。

## クイックスタート (日本語)

1.  **環境設定:**
    (以下のスクリプト実行前に、事前にAnaconda (Conda) のインストールが必要です。)

    ```bash
    ./setup_arxiv_reco_env.sh
    ```

2.  **`run_all.sh` の設定:**
    - `run_all.sh` ファイルを開きます。
    - `BIB_FILE` 変数をあなたの BibTeX ファイルのパスに更新します。
    - `GAI_API_KEY` 変数をあなたの実際の Google AI API キーに更新します。

3.  **アプリケーションの実行:**
    ```bash
    ./run_all.sh
    ```
    このスクリプトは以下を実行します:
    - conda 環境を有効化します。
    - 推薦リストと HTML 出力を生成します。
    - プロキシサーバーと HTTP サーバーをバックグラウンドで起動します。

4.  **結果の確認:**
    - Web ブラウザを開き、`http://localhost:8001` にアクセスします。

サーバーを停止するには、`run_all.sh` 完了時にターミナルに表示される `kill` コマンドを使用してください。

## 主な機能

-   Google Gemini Embeddings (`text-embedding-004` など) を使用して推薦を生成します。
-   オプションで Gemini 生成モデルを使用して初期候補を再ランク付けし、関連性を向上させます。（**注意:** この機能は現在テスト段階であり、デフォルトでは無効 (`config.yaml` の `rerank.enable: false`) にしておくことを推奨します。）
-   オプションで HDBSCAN を使用して BibTeX ライブラリをクラスタリングし、トピック固有の推薦を提供します。
-   その際、Gemini 生成モデルを使用してクラスタに人間が読めるラベルを生成します。
    -   クラスタ数が多い場合、free tierのGemini APIのrate limit（RPM）を一時的に超過し、一部のクラスタでラベル付けができず`nan`になることがありますが、再度スクリプトを実行すると自動的に残りのラベル付けが行われるため、心配は不要です。
-   オプションで、推薦された論文の要約（Explanation）を Gemini を使って生成し、HTML 出力上でインタラクティブに表示します。
-   BibTeX 埋め込みとクラスタ結果をキャッシュし、次回以降の実行を高速化します。
-   推薦結果を JSON とユーザーフレンドリーな HTML 形式の両方で出力します。
-   arXiv 論文をカテゴリでフィルタリングできます。

## 手動セットアップ

1.  **環境:**
    *   Python 3.10+ が必要です。
    *   依存関係、特に `faiss-cpu` の管理には Conda の使用を推奨します。
    *   詳細な Conda + pip による手順は `specs.md` の「環境構築例」を参照してください。
    *   依存関係のインストール:
        ```bash
        # Conda 環境 (arxiv-reco-v1) を新規作成 (Python 3.10)
        conda create -y -n arxiv-reco-v1 python=3.10

        # 環境をアクティベート
        conda activate arxiv-reco-v1

        # faiss-cpu を conda-forge からインストール
        conda install -y -c conda-forge faiss-cpu

        # pip をアップグレード
        pip install --upgrade pip
        pip install -r requirements.txt
        ```

2.  **`config.yaml`:**
    *   `config.yaml` がなければコピーするか、既存のファイルを編集します。
    *   Gemini モデル名、インデックスパラメータ、出力オプションなどの設定を調整します。
    *   **(新規)** `fetch.arxiv_categories` を設定して、特定の arXiv カテゴリで論文をフィルタリングします (例: `cs.LG`, `stat.ML`)。空リストの場合はデフォルトの CS カテゴリ (`src/arxiv_recommender/data/ingest.py` 内で定義) を使用します。複数のカテゴリを指定する場合は、リスト形式で記述します。
        ```yaml
        fetch:
          arxiv_categories:
            - cs.AI
            - cs.LG
            - stat.ML
        ```
    *   **論文要約 (Explanation) プロンプト:** `explanation.enable: true` の場合、要約生成に使用されるプロンプトをカスタマイズできます。
        *   `explanation.files_api_prompt_template`: PDF 全体を処理する Files API を使用する場合のプロンプトテンプレートです。`{target_language}` プレースホルダを使用できます。
        *   `explanation.prompt_template`: Files API を使用しない場合やフォールバック時のプロンプトテンプレートです。`{title}`, `{abstract}`, `{target_language}` プレースホルダを使用できます。
        *   `explanation.target_language`: 生成する要約の言語（例: "Japanese", "English"）を指定します。プロンプトテンプレート内の `{target_language}` に反映されます。
    *   **クラスタラベリングプロンプト:** `cluster.algorithm: hdbscan` の場合、`cluster.label_model` で指定されたモデルがクラスタのラベル付けを行います。プロンプトは現在コード内にハードコードされていますが、将来的に設定ファイルでカスタマイズ可能になる可能性があります。
    *   **注意:** 再ランク機能 (`rerank.enable`) は現在実験的なものです。この機能を特にテストする場合を除き、`false` に設定しておくことを推奨します。
    *   すべての設定オプションの詳細については `specs.md` を参照してください。

3.  **API キー:**
    *   `GAI_API_KEY` 環境変数を設定します:
        ```bash
        export GAI_API_KEY='YOUR_API_KEY'
        ```
    *   または、`config.yaml` の `gemini` セクションで `api_key` を指定することもできますが、通常は環境変数の使用が推奨されます。

## 使い方

以下の手順は、run_all.shで一気に実行することもできます。

1.  **環境のアクティベート:** Conda 環境をアクティベートします。
    ```bash
    conda activate arxiv-reco-v1
    ```
2.  **API キーの設定:** `GAI_API_KEY` 環境変数が設定されていることを確認します (セットアップ参照)。
3.  **推薦の実行:** メインスクリプトをお好みのオプションで実行します。例えば、今日の論文をクラスタリングモードで推薦する場合:
    ```bash
    python src/arxiv_recommender/core/main.py \
      --bib path/to/your/library.bib \
      --date today \
      --cluster
    ```
    > **Note:**  
    > BibTeXファイル内の論文数が数百件程度と少ない場合、`--cluster` オプションによるクラスタリングがうまく機能しないことがあります。  
    > その場合は `--cluster` オプションを外して実行することを推奨します。

    *(主要な引数は下記参照)*
4.  **プロキシサーバーの実行 (Explanation 機能用):** Explanation 機能 (`config.yaml` の `explanation.enable: true`) を有効にした場合は、**別のターミナル**でプロキシサーバーを起動します:
    ```bash
    python src/arxiv_recommender/server/proxy.py
    ```
    *(このサーバーが HTML 内の Explain ボタンからのリクエストを処理します)*
5.  **HTTP サーバーの実行 (HTML 閲覧用):** 出力ディレクトリ (デフォルト: `output/`) に移動し、**さらに別のターミナル**でシンプルな HTTP サーバーを起動します:
    ```bash
    cd output
    python -m http.server 8001
    ```
    *(例としてポート 8001 を使用していますが、利用可能なポートであれば何でも構いません)*
6.  **結果の表示:** Web ブラウザを開き、`http://localhost:8001` (または使用したポート) にアクセスします。
7.  **サーバーの停止:** 終了したら、各サーバーのプロセス ID (PID) を見つけて `kill` コマンドを使用するか、それぞれのターミナルで `Ctrl+C` を押してサーバーを停止します。
    ```bash
    # 例 (PID は実行ごとに異なります):
    # kill <proxy_server_pid>
    # kill <http_server_pid>
    ```

**`main.py` の主要な引数:**

-   `--bib` (必須): BibTeX ファイルへのパス。
-   `--config`: 設定ファイルへのパス (デフォルト: `config.yaml`)。
-   `--date`: arXiv 取得対象日 (デフォルト: 今日の UTC)。
-   `--topk`: 推薦数 (クラスタごと/全体) (デフォルト: config の `output.top_k`)。
-   `--cluster`: BibTeX ライブラリのクラスタリングを有効にし、トピック固有の推薦を行います。
-   `--refresh-embeddings`: クエリ埋め込みとクラスタ結果のキャッシュを再計算します。BibTeX や設定を大幅に変更した場合に使用します。
-   `--debug`: BibTeX ライブラリの小さなサブセットで実行し、簡単なテストを行います。

すべての引数の詳細については `specs.md` を参照してください。

## 出力

結果は `config.yaml` の `output.dir` で指定されたディレクトリ (デフォルト: `output/`) に保存されます。

-   **JSON:** `arxiv_reco_<YYYYMMDD>.json`
    -   `--cluster` 使用時: キーがクラスタラベル、値が `[論文情報, スコア]` のリストである辞書。
    -   それ以外: `[論文情報, スコア]` の単一リスト。
-   **HTML:** `arxiv_reco_<YYYYMMDD>.html`
    -   人間が読める形式の推薦概要。クラスタリング有効時はクラスタごとにグループ化されます。

`スコア` は関連性を示します (高いほど良い)。再ランク付けが有効な場合はリランカーから、無効な場合は ANN 距離の負の値に基づきます。

## Explanation (論文要約) 機能

`config.yaml` の `explanation.enable: true` を設定すると、HTML出力に各論文の「Explain」ボタンが表示されます。

このボタンをクリックすると、論文の要約が生成・表示されます。要約生成には Gemini API を利用するため、API キーの設定が必要です。

**プロキシサーバーの実行:**

要約生成は、バックグラウンドで動作するローカルプロキシサーバー (`src/arxiv_recommender/server/proxy.py`) を介して行われます。HTMLファイルを開く前に、別のターミナルで以下のコマンドを実行してプロキシサーバーを起動してください:

```bash
python src/arxiv_recommender/server/proxy.py
```

サーバーはデフォルトで `http://localhost:5001` でリクエストを待ち受けます。

**注意:**

-   プロキシサーバーは、メインの推薦スクリプトと同じ `config.yaml` と API キー環境変数 (`GAI_API_KEY`) を参照します。
-   要約は生成後にブラウザの `localStorage` にキャッシュされるため、同じ論文のExplainボタンを再度クリックしてもAPIコールは発生しません（ブラウザのキャッシュが有効な限り）。
-   `config.yaml` の `explanation` セクションで、要約に使用するモデルやプロンプト、キャッシュ設定などを調整できます。

## 技術詳細

設定オプション、コマンドライン引数、環境変数、クラスタリングプロセスに関する詳細情報は `specs.md` を参照してください。

## ライセンス

MIT ライセンス

---

# arXiv Recommender

Personalized arXiv recommendation tool based on your BibTeX library, powered by Google Gemini.

This command-line tool generates recommendations for new arXiv papers based on the abstracts and titles in your existing BibTeX library. It leverages the Google Gemini API for high-quality text embeddings and optional re-ranking and cluster labeling.

## Quick Start

1.  **Set up the environment:**
    (Note: Anaconda (Conda) must be installed before running the script below.)

    ```bash
    ./setup_arxiv_reco_env.sh
    ```

2.  **Configure `run_all.sh`:**
    - Open the `run_all.sh` file.
    - Update the `BIB_FILE` variable with the path to your BibTeX file.
    - Update the `GAI_API_KEY` variable with your actual Google AI API key.

3.  **Run the application:**
    ```bash
    ./run_all.sh
    ```
    This script will:
    - Activate the conda environment.
    - Generate recommendations and the HTML output.
    - Start a proxy server and an HTTP server in the background.

4.  **Access the results:**
    - Open your web browser and navigate to: `http://localhost:8001`

To stop the servers, use the `kill` commands displayed in the terminal when `run_all.sh` finishes.

## Features

-   Generates recommendations using Google Gemini embeddings (`text-embedding-004` or similar).
-   Optionally re-ranks initial candidates using a Gemini generative model for improved relevance. (**Note:** This feature is experimental and currently recommended to be kept disabled (`rerank.enable: false` in `config.yaml`).)
-   Optionally clusters your BibTeX library using HDBSCAN to provide topic-specific recommendations.
-   When clustering, optionally generates human-readable labels for clusters using a Gemini generative model.
    -   If there are many clusters, you may temporarily exceed the free tier Gemini API rate limit (RPM), causing some clusters to be labeled as `nan`. However, if you rerun the script, the remaining clusters will be automatically labeled, so there is no need to worry.
-   Optionally generates summaries (explanations) for recommended papers using Gemini and displays them interactively in the HTML output.
-   Caches BibTeX embeddings and cluster results for faster subsequent runs.
-   Outputs recommendations in both JSON and user-friendly HTML formats.
-   Allows filtering arXiv papers by category.

## Manual Setup

1.  **Environment:**
    *   Requires Python 3.10+.
    *   Using Conda is recommended for managing dependencies, especially `faiss-cpu`.
    *   See the "Environment Setup Example" in `specs.md` for detailed Conda + pip instructions.
    *   Install dependencies:
        ```bash
        # Create a new Conda environment (arxiv-reco-v1) with Python 3.10
        conda create -y -n arxiv-reco-v1 python=3.10

        # Activate the environment
        conda activate arxiv-reco-v1

        # Install faiss-cpu from conda-forge
        conda install -y -c conda-forge faiss-cpu

        # Upgrade pip
        pip install --upgrade pip
        pip install -r requirements.txt
        ```

2.  **`config.yaml`:**
    *   Copy `config.yaml` (if not present) or edit the existing one.
    *   Adjust settings like Gemini model names, index parameters, and output options.
    *   **(New)** Configure `fetch.arxiv_categories` to filter papers by specific arXiv categories (e.g., `cs.LG`, `stat.ML`). An empty list uses the default CS categories (defined in `src/arxiv_recommender/data/ingest.py`). Specify multiple categories as a list:
        ```yaml
        fetch:
          arxiv_categories:
            - cs.AI
            - cs.LG
            - stat.ML
        ```
    *   **Paper Explanation Prompts:** If `explanation.enable: true`, you can customize the prompts used for generating summaries:
        *   `explanation.files_api_prompt_template`: The prompt template used when processing the full PDF with the Files API. You can use the `{target_language}` placeholder.
        *   `explanation.prompt_template`: The prompt template used as a fallback or when not using the Files API. You can use `{title}`, `{abstract}`, and `{target_language}` placeholders.
        *   `explanation.target_language`: Specifies the target language for the generated summaries (e.g., "Japanese", "English"). This replaces `{target_language}` in the templates.
    *   **Cluster Labeling Prompt:** If `cluster.algorithm: hdbscan`, the model specified in `cluster.label_model` labels the clusters. The prompt is currently hardcoded but might become configurable in the future.
    *   **Note:** The rerank feature (`rerank.enable`) is currently experimental. It is recommended to keep it set to `false` unless specifically testing this feature.
    *   See `specs.md` for a detailed explanation of all configuration options.

3.  **API Key:**
    *   Set the `GAI_API_KEY` environment variable:
        ```bash
        export GAI_API_KEY='YOUR_API_KEY'
        ```
    *   Alternatively, you can specify the `api_key` under the `gemini` section in `config.yaml`, but using the environment variable is generally preferred.

## Usage

Note: The following steps can also be executed at once using `run_all.sh`.

1.  **Activate Environment:** Make sure your Conda environment is activated.
    ```bash
    conda activate arxiv-reco-v1
    ```
2.  **Set API Key:** Ensure the `GAI_API_KEY` environment variable is set (see Setup).
3.  **Run Recommendation:** Execute the main script with your desired options. For example, to run with clustering for today's papers:
    ```bash
    python src/arxiv_recommender/core/main.py \
      --bib path/to/your/library.bib \
      --date today \
      --cluster
    ```
    > **Note:**
    > If your BibTeX file contains only a few hundred papers, clustering with the `--cluster` option might not work well.
    > In that case, it's recommended to run without the `--cluster` option.

    *(See below for Key Arguments)*
4.  **Run Proxy Server (for Explanations):** If you enabled the explanation feature (`explanation.enable: true` in `config.yaml`), start the proxy server in a **separate terminal**:
    ```bash
    python src/arxiv_recommender/server/proxy.py
    ```
    *(This server handles requests from the Explain buttons in the HTML)*
5.  **Run HTTP Server (to view HTML):** Navigate to the output directory (default: `output/`) and start a simple HTTP server in **another separate terminal**:
    ```bash
    cd output
    python -m http.server 8001
    ```
    *(Using port 8001 as an example, you can use any available port)*
6.  **View Results:** Open your web browser and go to `http://localhost:8001` (or the port you used).
7.  **Stop Servers:** When finished, stop the servers by finding their Process IDs (PIDs) and using the `kill` command, or by pressing `Ctrl+C` in their respective terminals.
    ```bash
    # Example (PIDs will vary):
    # kill <proxy_server_pid>
    # kill <http_server_pid>
    ```

**Key Arguments for `main.py`:**

-   `--bib` (required): Path to your BibTeX file.
-   `--config`: Path to the configuration file (default: `config.yaml`).
-   `--date`: Target date for arXiv fetch (default: today UTC).
-   `--topk`: Number of recommendations per cluster/overall (default: `output.top_k` in config).
-   `--cluster`: Enable clustering of your BibTeX library for topic-specific recommendations.
-   `--refresh-embeddings`: Recompute and cache query embeddings and cluster results. Use if your BibTeX or config changes significantly.
-   `--debug`: Run with a small subset of your BibTeX library for quick testing.

See `specs.md` for details on all arguments.

## Output

Results are saved in the directory specified by `output.dir` in `config.yaml` (default: `output/`).

-   **JSON:** `arxiv_reco_<YYYYMMDD>.json`
    -   If `--cluster` is used: A dictionary where keys are cluster labels and values are lists of `[paper_info, score]`.
    -   Otherwise: A single list of `[paper_info, score]`.
-   **HTML:** `arxiv_reco_<YYYYMMDD>.html`
    -   A human-readable summary of the recommendations, grouped by cluster if enabled.

The `score` represents relevance (higher is better), either from the re-ranker or based on negative ANN distance if re-ranking is disabled.

## Explanation Feature

If you set `explanation.enable: true` in `config.yaml`, an "Explain" button will appear for each paper in the HTML output.

Clicking this button generates and displays a summary of the paper. This requires a Gemini API key as it uses the API for generation.

**Running the Proxy Server:**

Summary generation happens via a local proxy server (`src/arxiv_recommender/server/proxy.py`) running in the background. Before opening the HTML file, start the proxy server in a separate terminal:

```bash
python src/arxiv_recommender/server/proxy.py
```

The server listens on `http://localhost:5001` by default.

**Notes:**

-   The proxy server uses the same `config.yaml` and `GAI_API_KEY` environment variable as the main recommendation script.
-   Summaries are cached in the browser's `localStorage` after generation. Clicking the Explain button again for the same paper will not trigger another API call (as long as the browser cache is valid).
-   You can configure the explanation model, prompt, caching behavior, etc., in the `explanation` section of `config.yaml`.

## Technical Details

See `specs.md` for detailed information on configuration options, command-line arguments, environment variables, and the clustering process.

## License

MIT License.