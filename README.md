---
title: 競馬予想デモアプリ
emoji: 🏇
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.55.0"
app_file: app.py
pinned: false
license: mit
---

# 競馬予想デモアプリ

過去の走破データに基づく統計分析で、今週のレースを分析するデモアプリです。

## 機能

- JRA公式サイトから出馬表を取得して分析
- LightGBM + SHAP による予測と根拠の可視化
- 騎手成績（通算・コース別・競馬場別）の参照
- 馬の比較機能

## 技術スタック

- **モデル**: LightGBM（勾配ブースティング）
- **説明可能性**: SHAP（TreeExplainer）
- **UI**: Streamlit
- **データ**: JRA公式サイト（公開情報）/ Kaggle JRAレース結果

## 免責事項

本アプリは統計分析による参考情報の提供を目的としています。馬券購入は自己責任でお願いします。

---

Built by [KaleidoFuture](https://kaleidofuture.com)
