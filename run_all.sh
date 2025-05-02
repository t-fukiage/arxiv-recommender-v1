#!/bin/bash

# このスクリプトの実行方法：
# ./run_all.sh

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

# --- 実行 ---

echo "ステップ 1: Conda環境のアクティベート (arxiv-reco)..."
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
python src/arxiv_recommender/core/main.py --bib "$BIB_FILE" --date today --cluster
# python src/arxiv_recommender/core/main.py --bib "$BIB_FILE" --mode gemini --date today --cluster
echo "HTML生成完了。"

{
  echo "ステップ 4: プロキシサーバーをバックグラウンドで起動 (localhost:5001)..."
  python src/arxiv_recommender/server/proxy.py &
  proxy_pid=$!
  echo "プロキシサーバー起動完了 (PID: $proxy_pid)。"
  echo "Server PID: $proxy_pid"
} >> proxy.log 2>&1
# サーバーが起動するまで少し待つ (任意)
sleep 2

{
  echo "ステップ 5: HTTPサーバーをバックグラウンドで起動 (localhost:$HTTP_PORT)..."
  python -m http.server -d output $HTTP_PORT &
  http_server_pid=$!
  echo "HTTPサーバー起動完了 (PID: $http_server_pid)。"
  echo "Server PID: $http_server_pid"
} >> http_server.log 2>&1

echo ""
echo "--- 準備完了 ---"
echo "ブラウザで以下のURLを開いてください:"
echo "http://localhost:$HTTP_PORT"
echo ""
echo "サーバーを停止するには、以下のコマンドを実行してください:"
echo "kill $proxy_pid"
echo "kill $http_server_pid"
echo ""

# スクリプトが終了しないように待機する (必要であれば)
# この部分をコメントアウトすれば、URL表示後にスクリプトは終了します
# echo "スクリプトはこのまま待機します。Ctrl+Cで終了できますが、サーバーは別途killする必要があります。"
# wait $proxy_pid $http_server_pid