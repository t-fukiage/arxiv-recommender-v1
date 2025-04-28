#!/bin/bash

# arXiv Recommender 環境セットアップスクリプト
# Conda環境名: arxiv-reco-v1

# 1. Conda環境作成 (Python 3.10)
conda create -y -n arxiv-reco-v1 python=3.10

# 2. 環境アクティベート
conda activate arxiv-reco-v1 || source activate arxiv-reco-v1

# 3. faiss-cpuをconda-forgeからインストール
conda install -y -c conda-forge faiss-cpu

# 4. pipアップグレード
pip install --upgrade pip

# 5. requirements.txtからインストール
pip install -r requirements.txt

# 7. 完了メッセージ
echo "\n[INFO] arxiv-reco-v1 環境セットアップ完了！"