"""
競馬予想AI - LightGBM モデル訓練・予測・SHAP分析
「事前予測用」: 出馬表の時点で確定している特徴量のみ使用
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score

MODEL_PATH = Path(__file__).parent / "data" / "model.pkl"
EXPLAINER_PATH = Path(__file__).parent / "data" / "explainer.pkl"


# ─── 特徴量定義 ──────────────────────────────────────────────
# 出馬表の時点で利用可能な特徴量のみ
FEATURE_COLS = [
    # 基本情報
    "馬齢", "斤量", "距離(m)",
    # 前走データ
    "前走_着順", "前走_人気", "前走_頭数", "前走_距離",
    "前走_タイム", "前走_馬体重", "前走_上がり3F",
    # 2走前データ
    "2走前_着順", "2走前_人気",
    # 戦績
    "出走数", "勝率", "複勝率",
    # 騎手成績
    "騎手_勝率", "騎手_複勝率", "騎手_騎乗数",
    # 騎手×条件
    "騎手_コース別勝率", "騎手_場別勝率",
]

JOCKEY_STATS_PATH = Path(__file__).parent / "data" / "jockey_stats.csv"

TARGET_COL = "is_win"


# ─── 特徴量エンジニアリング ────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """データフレームから特徴量を準備する"""
    df = df.copy()

    # 1着フラグ
    rank_col = next((c for c in ["着順", "rank"] if c in df.columns), None)
    if rank_col:
        df["is_win"] = (pd.to_numeric(df[rank_col], errors="coerce") == 1).astype(int)

    # 利用可能な特徴量だけ使う
    available = [c for c in FEATURE_COLS if c in df.columns]

    # 数値変換
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, available


def build_past_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle過去データから前走特徴量を構築する。

    同じ馬の過去レース結果を参照して、前走着順・タイム等を付与する。
    未来のデータを参照しないよう、日付順にソートして処理する。
    """
    df = df.copy()
    df["レース日付"] = pd.to_datetime(df["レース日付"])
    df = df.sort_values(["レース日付", "レース番号"]).reset_index(drop=True)

    # 数値化
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["人気"] = pd.to_numeric(df["人気"], errors="coerce")
    df["単勝"] = pd.to_numeric(df["単勝"], errors="coerce")
    df["馬体重"] = pd.to_numeric(df["馬体重"], errors="coerce")
    df["距離(m)"] = pd.to_numeric(df.get("距離(m)", pd.Series(dtype=float)), errors="coerce")

    # タイムを秒数に変換
    if "タイム" in df.columns:
        df["タイム秒"] = df["タイム"].apply(_parse_time)

    # 上がり3Fを数値化
    if "上り3F" in df.columns:
        df["上がり3F"] = pd.to_numeric(df["上り3F"], errors="coerce")
    elif "上がり3F" not in df.columns:
        df["上がり3F"] = np.nan

    # 出走頭数（レースごと）
    df["頭数"] = df.groupby(["レース日付", "競馬場名", "レース番号"])["馬名"].transform("count")

    # 馬ごとにグループ化して前走データを構築
    past_cols = {
        "前走_着順": [], "前走_人気": [], "前走_頭数": [],
        "前走_距離": [], "前走_タイム": [], "前走_馬体重": [],
        "前走_上がり3F": [],
        "2走前_着順": [], "2走前_人気": [],
        "出走数": [], "勝率": [], "複勝率": [],
    }

    # 馬名でグループ化し、各行に前走情報を付与
    for col in past_cols:
        past_cols[col] = [np.nan] * len(df)

    horse_history = {}  # {馬名: [(着順, 人気, 頭数, 距離, タイム, 馬体重, 上がり3F), ...]}

    for idx, row in df.iterrows():
        horse = row["馬名"]
        if pd.isna(horse):
            continue

        history = horse_history.get(horse, [])

        # 前走データを設定
        if len(history) >= 1:
            prev = history[-1]
            past_cols["前走_着順"][idx] = prev[0]
            past_cols["前走_人気"][idx] = prev[1]
            past_cols["前走_頭数"][idx] = prev[2]
            past_cols["前走_距離"][idx] = prev[3]
            past_cols["前走_タイム"][idx] = prev[4]
            past_cols["前走_馬体重"][idx] = prev[5]
            past_cols["前走_上がり3F"][idx] = prev[6]

        if len(history) >= 2:
            prev2 = history[-2]
            past_cols["2走前_着順"][idx] = prev2[0]
            past_cols["2走前_人気"][idx] = prev2[1]

        # 戦績
        past_cols["出走数"][idx] = len(history)
        if len(history) > 0:
            wins = sum(1 for h in history if h[0] == 1)
            top3 = sum(1 for h in history if h[0] <= 3)
            past_cols["勝率"][idx] = wins / len(history)
            past_cols["複勝率"][idx] = top3 / len(history)
        else:
            past_cols["勝率"][idx] = 0.0
            past_cols["複勝率"][idx] = 0.0

        # 現在のレースを履歴に追加
        history.append((
            row["着順"],
            row["人気"],
            row.get("頭数", np.nan),
            row.get("距離(m)", np.nan),
            row.get("タイム秒", np.nan),
            row["馬体重"],
            row.get("上がり3F", np.nan),
        ))
        horse_history[horse] = history

    for col, vals in past_cols.items():
        df[col] = vals

    # 騎手成績をマージ
    df = _merge_jockey_stats(df)

    return df


def _merge_jockey_stats(df: pd.DataFrame) -> pd.DataFrame:
    """騎手の通算成績をマージする。jockey_stats.csv を参照。"""
    if not JOCKEY_STATS_PATH.exists():
        return df

    jstats = pd.read_csv(JOCKEY_STATS_PATH, encoding="utf-8-sig")

    # スペース除去した騎手名でマッチング
    if "騎手_key" not in jstats.columns:
        jstats["騎手_key"] = jstats["騎手"].str.replace(" ", "").str.replace("\u3000", "")

    jockey_col = "騎手" if "騎手" in df.columns else None
    if jockey_col is None:
        return df

    import re as _re

    def _normalize_jockey(name):
        s = str(name).replace(" ", "").replace("\u3000", "")
        s = _re.sub(r"^[A-Za-z]\.", "", s)
        return s

    df = df.copy()
    df["_jockey_key"] = df[jockey_col].apply(_normalize_jockey)

    jstats["_jkey_norm"] = jstats["騎手_key"].apply(_normalize_jockey)
    lookup = jstats.set_index("_jkey_norm")[["騎手_勝率", "騎手_複勝率", "騎乗数"]].to_dict("index")

    avg_win = jstats["騎手_勝率"].mean()
    avg_top3 = jstats["騎手_複勝率"].mean()
    avg_rides = jstats["騎乗数"].median()

    df["騎手_勝率"] = df["_jockey_key"].map(lambda k: lookup.get(k, {}).get("騎手_勝率", avg_win))
    df["騎手_複勝率"] = df["_jockey_key"].map(lambda k: lookup.get(k, {}).get("騎手_複勝率", avg_top3))
    df["騎手_騎乗数"] = df["_jockey_key"].map(lambda k: lookup.get(k, {}).get("騎乗数", avg_rides))

    # 騎手×コース別勝率（芝/ダート）
    avg_course_win = avg_win
    course_col = "芝・ダート区分" if "芝・ダート区分" in df.columns else None
    if course_col and "騎手_芝_勝率" in jstats.columns:
        turf_lookup = jstats.set_index("_jkey_norm")["騎手_芝_勝率"].to_dict()
        dirt_lookup = jstats.set_index("_jkey_norm")["騎手_ダート_勝率"].to_dict()
        df["騎手_コース別勝率"] = [
            turf_lookup.get(k, avg_course_win) if c == "芝" else dirt_lookup.get(k, avg_course_win)
            for k, c in zip(df["_jockey_key"], df[course_col])
        ]
    else:
        df["騎手_コース別勝率"] = avg_win

    # 騎手×競馬場別勝率（サンプル少はブレンド）
    venue_csv = Path(__file__).parent / "data" / "jockey_venue_stats.csv"
    if venue_csv.exists() and "競馬場名" in df.columns:
        vstats = pd.read_csv(venue_csv, encoding="utf-8-sig")
        vstats["騎手_key"] = vstats["騎手_key"].apply(_normalize_jockey)
        venue_lookup = {}
        for _, vrow in vstats.iterrows():
            venue_lookup[(vrow["騎手_key"], vrow["競馬場名"])] = (vrow["勝率"], vrow["騎乗数"])

        overall = {k: v.get("騎手_勝率", avg_win)
                  for k, v in jstats.set_index("_jkey_norm").to_dict("index").items()}

        def _venue_rate(jkey, venue):
            entry = venue_lookup.get((jkey, venue))
            if entry is None:
                return overall.get(jkey, avg_win)
            rate, rides = entry
            if rides < 30:
                weight = rides / 30
                return rate * weight + overall.get(jkey, avg_win) * (1 - weight)
            return rate

        df["騎手_場別勝率"] = [
            _venue_rate(k, v)
            for k, v in zip(df["_jockey_key"], df["競馬場名"])
        ]
    else:
        df["騎手_場別勝率"] = avg_win

    df.drop(columns=["_jockey_key"], inplace=True)
    return df


def _parse_time(t) -> float:
    """'1:34.3' → 94.3 に変換"""
    try:
        parts = str(t).split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(t)
    except Exception:
        return np.nan


# ─── モデル訓練 ───────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """LightGBMモデルを訓練して保存する"""
    print("前走特徴量を構築中...")
    df = build_past_race_features(df)

    df, features = prepare_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"ターゲット列 '{TARGET_COL}' がデータに存在しません")

    # 前走データがある行のみ使用（新馬戦は除外）
    mask = df["出走数"] > 0
    df_train = df[mask]
    print(f"訓練対象: {len(df_train):,}行（前走あり） / 全{len(df):,}行")

    X = df_train[features].fillna(-1)
    y = df_train[TARGET_COL]

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, eval_set=[(X, y)])

    # 保存
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, features), f)

    # SHAP explainer を作成・保存
    explainer = shap.TreeExplainer(model)
    with open(EXPLAINER_PATH, "wb") as f:
        pickle.dump(explainer, f)

    print(f"モデル保存完了: {MODEL_PATH}")
    print(f"特徴量: {features}")
    return model, features


def load_model():
    """保存済みモデルを読み込む"""
    with open(MODEL_PATH, "rb") as f:
        model, features = pickle.load(f)
    with open(EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)
    return model, features, explainer


# ─── 予測 ────────────────────────────────────────────────────

def predict_race(df: pd.DataFrame, model, features: list) -> pd.DataFrame:
    """
    レースデータから各馬の勝利確率を予測する
    レース内でソフトマックス正規化して相対スコアに変換
    """
    df, _ = prepare_features(df)

    # モデルが要求する特徴量で不足分を-1埋め
    for col in features:
        if col not in df.columns:
            df[col] = -1

    X = df[features].fillna(-1)
    proba = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["win_prob"] = proba

    # レース内相対スコア（合計100%になるよう正規化）
    df["win_score"] = proba / proba.sum() * 100

    return df.sort_values("win_score", ascending=False)


# ─── SHAP分析 ────────────────────────────────────────────────

def get_shap_explanation(row: pd.Series, features: list,
                          explainer) -> list[dict]:
    """1頭分のSHAP値を計算して、説明用のリストを返す"""
    # モデルが要求する特徴量で不足分を-1埋め
    vals = []
    for f in features:
        vals.append(row.get(f, -1))
    X = np.array(vals, dtype=float).reshape(1, -1)
    X = np.nan_to_num(X, nan=-1)

    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        shap_vals = raw[1][0]
    else:
        shap_vals = raw[0]

    result = []
    for feat, val, sv in zip(features, X[0], shap_vals):
        result.append({
            "feature": feat,
            "value": float(val),
            "shap": float(sv),
            "direction": "positive" if sv > 0 else "negative",
        })

    result.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return result[:8]


# ─── 評価 ────────────────────────────────────────────────────

def evaluate_model(model, df: pd.DataFrame, features: list) -> dict:
    """モデルの精度・回収率を評価する"""
    df = build_past_race_features(df)
    df, _ = prepare_features(df)

    for col in features:
        if col not in df.columns:
            df[col] = -1

    X = df[features].fillna(-1)

    if TARGET_COL not in df.columns:
        return {}

    y = df[TARGET_COL]
    proba = model.predict_proba(X)[:, 1]

    metrics = {}
    try:
        metrics["auc"] = roc_auc_score(y, proba)
    except Exception:
        metrics["auc"] = 0.0

    # 回収率計算（オッズがある場合）
    if "単勝" in df.columns:
        df = df.copy()
        df["pred_prob"] = proba
        df["単勝"] = pd.to_numeric(df["単勝"], errors="coerce")
        bets = df[df["pred_prob"] * df["単勝"] > 0.8].dropna(subset=["単勝"])
        if len(bets) > 0:
            wins = bets[bets[TARGET_COL] == 1]
            total_bet = len(bets)
            total_return = (wins["単勝"] * 100).sum() if len(wins) > 0 else 0
            metrics["roi"] = (total_return / (total_bet * 100) - 1) * 100
            metrics["bet_count"] = total_bet
            metrics["win_count"] = len(wins)
        else:
            metrics["roi"] = 0.0
            metrics["bet_count"] = 0
            metrics["win_count"] = 0

    return metrics
