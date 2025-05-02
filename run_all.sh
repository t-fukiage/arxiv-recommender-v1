#!/bin/bash

# このスクリプトの実行方法：
# ./run_all.sh

# 実行時に "zsh: permission denied: ./run_all.sh" などのエラーが出た場合は、
# 以下のコマンドで実行権限を付与してください：
#   chmod +x run_all.sh


# エラーが発生したらスクリプトを停止する
set -e

# --- 設定項目 (必要に応じて編集) ---
# 警告: APIキーをスクリプトに直接記述するのはセキュリティ上推奨されません。
# 可能であれば、この行を削除し、スクリプト実行前に環境変数を設定してください。
# 例: export GAI_API_KEY='...'
export GAI_API_KEY='YOUR_API_KEY_HERE' # <<< 必ず実際のAPIキーに置き換えてください！

# BibTeXファイルのパス (実際のパスに置き換えてください)
BIB_FILE="my.bib" # <<< 修正してください

# HTTPサーバーのポート番号 (もし8001が使用中なら変更)
HTTP_PORT=8001
# プロキシサーバーのポート番号 (proxy.pyのデフォルトに合わせる)
PROXY_PORT=5001
# 出力ディレクトリ
OUTPUT_DIR="output"
# ログファイル
PROXY_LOG="proxy.log"
HTTP_LOG="http_server.log"

# --- 実行 ---

# 既存のログファイルを削除 (任意)
rm -f "$PROXY_LOG" "$HTTP_LOG"

echo "ステップ 1: Conda環境のアクティベート (arxiv-reco-v1)..."
# 事前に `conda init bash` (または `conda init zsh`) が実行されている必要があります
source $(conda info --base)/etc/profile.d/conda.sh # condaコマンドをシェルスクリプト内で使えるようにする
conda activate arxiv-reco-v1
echo "Conda環境 OK."

echo "ステップ 2: APIキーの確認..."
if [ -z "$GAI_API_KEY" ] || [ "$GAI_API_KEY" == "YOUR_API_KEY_HERE" ]; then
  echo "エラー: GAI_API_KEYが設定されていないか、プレースホルダーのままです。"
  echo "スクリプト内の 'YOUR_API_KEY_HERE' を実際のキーに置き換えるか、"
  echo "スクリプト実行前に export GAI_API_KEY='...' を実行してください。"
  exit 1
fi
echo "APIキー OK."

echo "ステップ 3: 今日のarXiv論文の推薦リストとHTMLを生成 (クラスターモード)..."
# 出力ディレクトリを作成 (存在しない場合)
mkdir -p "$OUTPUT_DIR"
# 推薦スクリプト実行 (設定ファイル config.yaml を使用)
python src/arxiv_recommender/core/main.py --bib "$BIB_FILE" --date today --cluster --config config.yaml
# python src/arxiv_recommender/core/main.py --bib "$BIB_FILE\" --mode gemini --date today --cluster
echo "HTML生成完了。"

echo "ステップ 4: プロキシサーバーをバックグラウンドで起動 (localhost:$PROXY_PORT)..."
# プロキシサーバーをバックグラウンドで起動し、ログをリダイレクト
python src/arxiv_recommender/server/proxy.py --port "$PROXY_PORT" > "$PROXY_LOG" 2>&1 &
proxy_pid=$!
echo "プロキシサーバー起動完了 (PID: $proxy_pid)。ログ: $PROXY_LOG"

# サーバーが起動するまで少し待つ (任意)
sleep 2

echo "ステップ 5: HTTPサーバーをバックグラウンドで起動 (localhost:$HTTP_PORT)..."
# HTTPサーバーをバックグラウンドで起動し、ログをリダイレクト
cd "$OUTPUT_DIR" # outputディレクトリに移動
python -m http.server "$HTTP_PORT" > "../$HTTP_LOG" 2>&1 & # ログは一つ上のディレクトリに保存
http_server_pid=$!
cd .. # 元のディレクトリに戻る
echo "HTTPサーバー起動完了 (PID: $http_server_pid)。ログ: $HTTP_LOG"

echo ""
echo "--- 準備完了 ---"
echo "ブラウザで以下のURLを開いてください:"
echo "http://localhost:$HTTP_PORT"
echo ""

# macOSなら自動でブラウザを開く (Linuxは xdg-open http://localhost:$HTTP_PORT)
# open "http://localhost:$HTTP_PORT"

# ユーザーがEnterを押すまで待機
read -p "推薦結果の確認が終わったら、このターミナルでEnterキーを押してサーバーを停止してください..."

echo ""
echo "サーバーを停止しています..."

# killコマンドが失敗してもスクリプトが止まらないようにする
kill $proxy_pid 2>/dev/null || echo "プロキシサーバー ($proxy_pid) の停止に失敗しました (すでに停止している可能性があります)。"
kill $http_server_pid 2>/dev/null || echo "HTTPサーバー ($http_server_pid) の停止に失敗しました (すでに停止している可能性があります)。"

echo "サーバーは停止しました。"
echo ""

# スクリプトが終了しないように待機する (必要であれば)
# この部分をコメントアウトすれば、URL表示後にスクリプトは終了します
# echo "スクリプトはこのまま待機します。Ctrl+Cで終了できますが、サーバーは別途killする必要があります。"
# wait $proxy_pid $http_server_pid