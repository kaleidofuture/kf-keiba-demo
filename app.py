"""
競馬予想デモアプリ - Streamlit
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).parent
sys.path.append(str(APP_DIR))
from model import predict_race, load_model, get_shap_explanation
from scraper import get_shutsuba_races, shutsuba_to_predict_df

DATA_DIR = APP_DIR / "data"
MODEL_PATH = DATA_DIR / "model.pkl"
LOGO_PATH = APP_DIR / "assets" / "logo.png"

# ─── カラーパレット ──────────────────────────────────────────
# 濃緑（芝）+ ゴールド（格式）+ ダークグレー（文字）
C_PRIMARY = "#1B4332"
C_ACCENT = "#B8860B"
C_BG = "#FAFAFA"
C_CARD = "#FFFFFF"
C_TEXT = "#1a1a1a"
C_MUTED = "#6c757d"
C_POS = "#1B7A3D"
C_NEG = "#C0392B"


# ─── カスタムCSS ─────────────────────────────────────────────
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Noto Sans JP', sans-serif;
    }}

    /* Streamlitデフォルト要素を非表示 */
    header[data-testid="stHeader"] {{ display: none !important; }}
    footer {{ display: none !important; }}
    .stDeployButton {{ display: none !important; }}
    #MainMenu {{ display: none !important; }}

    /* 上部余白を詰める */
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }}

    /* タブのスタイル */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 2px solid #e0e0e0;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 12px 24px;
        font-weight: 500;
        color: {C_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        color: {C_PRIMARY};
        border-bottom: 3px solid {C_PRIMARY};
        font-weight: 700;
    }}

    /* ボタン */
    .stButton > button {{
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        border-color: {C_PRIMARY};
        color: {C_PRIMARY};
        box-shadow: 0 2px 8px rgba(27, 67, 50, 0.15);
    }}

    /* プライマリボタン */
    button[kind="primary"] {{
        background-color: {C_PRIMARY} !important;
        color: white !important;
        border: none !important;
    }}
    button[kind="primary"]:hover {{
        background-color: #143728 !important;
    }}

    /* メトリクス */
    [data-testid="stMetric"] {{
        background: {C_CARD};
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 3px solid {C_PRIMARY};
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}

    /* セレクトボックス */
    .stSelectbox label {{
        font-weight: 500;
        color: {C_TEXT};
    }}

    /* キャプション */
    .stCaption {{
        color: {C_MUTED} !important;
    }}

    /* カスタムヘッダー */
    .app-header {{
        background: linear-gradient(135deg, {C_PRIMARY} 0%, #2D6A4F 100%);
        padding: 1.2rem 2rem;
        border-radius: 0 0 12px 12px;
        margin: -1.5rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .app-header h1 {{
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.02em;
    }}
    .app-header .subtitle {{
        color: rgba(255,255,255,0.7);
        font-size: 0.8rem;
        margin: 0;
    }}
    .app-header .badge {{
        background: {C_ACCENT};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .app-header .logo-img {{
        height: 36px;
        margin-right: 12px;
        vertical-align: middle;
    }}
    .app-header .title-area {{
        display: flex;
        align-items: center;
    }}

    /* カスタムフッター */
    .app-footer {{
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
        color: {C_MUTED};
        font-size: 0.8rem;
    }}
    .app-footer a {{
        color: {C_PRIMARY};
        text-decoration: none;
        font-weight: 500;
    }}
    .app-footer a:hover {{
        text-decoration: underline;
    }}
</style>
"""


@st.dialog("騎手名鑑", width="large")
def _show_jockey_directory(df_raw):
    """開催に出場する全騎手の成績をポップアップ表示する"""
    import re as _re
    jockey_csv = DATA_DIR / "jockey_stats.csv"
    if not jockey_csv.exists():
        st.warning("騎手データがありません")
        return

    jstats = pd.read_csv(jockey_csv, encoding="utf-8-sig")

    def _norm(name):
        s = str(name).replace(" ", "").replace("\u3000", "")
        s = _re.sub(r"^[A-Za-z]\.", "", s)
        return s

    jstats["_norm"] = jstats["騎手"].apply(_norm)
    lookup = jstats.set_index("_norm").to_dict("index")

    # 出場騎手を集計
    jockeys = df_raw["騎手名"].unique().tolist()
    rows = []
    for j in jockeys:
        norm = _norm(j)
        info = lookup.get(norm)
        if info:
            rows.append({
                "騎手": j,
                "勝率": f"{info['騎手_勝率']:.1%}",
                "複勝率": f"{info['騎手_複勝率']:.1%}",
                "騎乗数": f"{int(info['騎乗数']):,}",
                "勝率_raw": info["騎手_勝率"],
            })
        else:
            rows.append({
                "騎手": j,
                "勝率": "-",
                "複勝率": "-",
                "騎乗数": "-",
                "勝率_raw": -1,
            })

    df_j = pd.DataFrame(rows).sort_values("勝率_raw", ascending=False).reset_index(drop=True)
    df_j.index = range(1, len(df_j) + 1)
    df_j.index.name = "#"

    st.caption(f"出場騎手 {len(df_j)}名（成績は学習データ 〜2021年7月 に基づく）")
    st.dataframe(df_j[["騎手", "勝率", "複勝率", "騎乗数"]], use_container_width=True)

    # データなしの騎手
    no_data = [r["騎手"] for r in rows if r["勝率_raw"] < 0]
    if no_data:
        st.caption(f"※ データなし（2021年以降デビュー等）: {', '.join(no_data)}")


@st.dialog("馬の詳細情報", width="large")
def _show_horse_detail(row, features, explainer):
    """馬の詳細情報をポップアップ表示する"""
    name = row.get("馬名", "不明")
    score = row.get("win_score", 0)

    st.subheader(f"{name}")
    st.metric("予測スコア", f"{score:.1f}%")

    col1, col2, col3 = st.columns(3)
    col1.metric("馬齢", f"{int(row.get('馬齢', 0))}歳" if pd.notna(row.get('馬齢')) else "-")
    col2.metric("斤量", f"{row.get('斤量', '-')}kg" if pd.notna(row.get('斤量')) else "-")
    col3.metric("騎手", row.get("騎手", "-"))

    st.divider()

    st.markdown("**戦績（直近4戦ベース）**")
    col1, col2, col3 = st.columns(3)
    races_count = row.get("出走数", 0)
    col1.metric("出走数", f"{int(races_count)}戦" if pd.notna(races_count) and races_count >= 0 else "-")
    win_rate = row.get("勝率", 0)
    col2.metric("勝率", f"{win_rate:.0%}" if pd.notna(win_rate) else "-")
    top3_rate = row.get("複勝率", 0)
    col3.metric("複勝率", f"{top3_rate:.0%}" if pd.notna(top3_rate) else "-")

    st.divider()
    st.markdown("**近走成績**")
    for prefix, label in [("前走", "前走"), ("2走前", "2走前")]:
        rank = row.get(f"{prefix}_着順", -1)
        if pd.notna(rank) and rank > 0:
            pop = row.get(f"{prefix}_人気", -1)
            heads = row.get(f"{prefix}_頭数", -1)
            dist = row.get(f"{prefix}_距離", -1)
            time_val = row.get(f"{prefix}_タイム", -1)
            weight = row.get(f"{prefix}_馬体重", -1)
            f3 = row.get(f"{prefix}_上がり3F", -1)

            parts = [f"**{label}**: {int(rank)}着"]
            if pd.notna(pop) and pop > 0: parts.append(f"{int(pop)}番人気")
            if pd.notna(heads) and heads > 0: parts.append(f"({int(heads)}頭立)")
            if pd.notna(dist) and dist > 0: parts.append(f"/ {int(dist)}m")
            if pd.notna(time_val) and time_val > 0:
                m, s = divmod(time_val, 60)
                parts.append(f"/ {int(m)}:{s:04.1f}")
            if pd.notna(weight) and weight > 0: parts.append(f"/ {int(weight)}kg")
            if pd.notna(f3) and f3 > 0: parts.append(f"/ 上がり3F {f3:.1f}")
            st.markdown(" ".join(parts))
        else:
            if prefix == "前走":
                st.markdown(f"**{label}**: データなし（初出走）")

    # 騎手成績
    st.divider()
    jockey_name = row.get("騎手", "")
    st.markdown(f"**騎手: {jockey_name}**")
    j_total = row.get("騎手_勝率", -1)
    j_top3 = row.get("騎手_複勝率", -1)
    j_rides = row.get("騎手_騎乗数", -1)
    j_course = row.get("騎手_コース別勝率", -1)
    j_venue = row.get("騎手_場別勝率", -1)

    if pd.notna(j_total) and j_total >= 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("通算勝率", f"{j_total:.1%}")
        col2.metric("通算複勝率", f"{j_top3:.1%}" if pd.notna(j_top3) else "-")
        col3.metric("騎乗数", f"{int(j_rides):,}" if pd.notna(j_rides) else "-")

        col4, col5 = st.columns(2)
        col4.metric("コース別勝率", f"{j_course:.1%}" if pd.notna(j_course) and j_course >= 0 else "-",
                    help="今回のコース（芝/ダート）での勝率")
        col5.metric("競馬場別勝率", f"{j_venue:.1%}" if pd.notna(j_venue) and j_venue >= 0 else "-",
                    help="今回の競馬場での勝率")
    else:
        st.caption("騎手成績データなし（2021年以降デビュー等）")

    st.divider()
    st.markdown("**特徴量の影響度**")
    shap_items = get_shap_explanation(row, features, explainer)
    shap_df = pd.DataFrame(shap_items[:6])
    fig = px.bar(
        shap_df, x="shap", y="feature", orientation="h",
        color="direction",
        color_discrete_map={"positive": C_POS, "negative": C_NEG},
    )
    fig.update_layout(
        height=250, showlegend=False,
        xaxis_title="影響度（右=有利 / 左=不利）",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _assign_marks(scores) -> list[str]:
    """勝率予測スコア(降順)から印を付与する。"""
    n = len(scores)
    if n == 0:
        return []

    avg = 100 / n
    top3_range = scores[0] - scores[min(2, n - 1)]

    marks = [""] * n

    if top3_range < avg * 0.2:
        for i in range(min(round(n * 0.5), n)):
            marks[i] = "△"
    else:
        if n >= 1: marks[0] = "◎"
        if n >= 2: marks[1] = "○"
        if n >= 3: marks[2] = "▲"

        delta_slots = max(0, round(n * 0.5) - 3)
        cutoff = scores[min(2, n - 1)] * 0.3
        delta_count = 0
        for i in range(3, n):
            if delta_count < delta_slots and scores[i] >= cutoff:
                marks[i] = "△"
                delta_count += 1

    return marks


# ─── ページ設定 ──────────────────────────────────────────────
st.set_page_config(
    page_title="競馬予想デモ",
    page_icon="🏇",
    layout="wide",
)

# CSS注入
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── カスタムヘッダー ────────────────────────────────────────
import base64

def _get_logo_b64():
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""

_logo_b64 = _get_logo_b64()
_logo_html = f'<img src="data:image/png;base64,{_logo_b64}" class="logo-img">' if _logo_b64 else ""

st.markdown(f"""
<div class="app-header">
    <div class="title-area">
        {_logo_html}
        <div>
            <h1>競馬予想デモアプリ</h1>
            <p class="subtitle">過去の走破データに基づく統計分析</p>
        </div>
    </div>
    <span class="badge">DEMO</span>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("モデルが見つかりません（data/model.pkl）")

# ─── メインエリア ────────────────────────────────────────────
tab1, tab2 = st.tabs(["出馬表・予測", "使い方"])

# ── Tab1: 今週の出馬表 ───────────────────────────────────────
with tab1:
    tab1_container = st.empty()

    with tab1_container.container():
        if st.session_state.get("shutsuba_loading"):
            kaisai = st.session_state.get("shutsuba_loading_label", "")
            st.info(f"{kaisai}の出馬表を取得中...（12レース分）")
            from scraper import get_shutsuba_table
            races = st.session_state.get("shutsuba_races", [])
            race = next(r for r in races if r["label"] == kaisai)
            try:
                df_raw = get_shutsuba_table(race["cname"], race["url"])
                if df_raw is not None and len(df_raw) > 0:
                    st.session_state["shutsuba_df"] = df_raw
                    st.session_state["shutsuba_kaisai"] = kaisai
            except Exception as e:
                st.error(f"取得エラー: {e}")
            st.session_state.pop("shutsuba_loading", None)
            st.session_state.pop("shutsuba_loading_label", None)
            st.rerun()

        elif "shutsuba_df" in st.session_state:
            df_raw = st.session_state["shutsuba_df"]
            kaisai = st.session_state.get("shutsuba_kaisai", "")

            col_info, col_back = st.columns([3, 1])
            with col_info:
                st.success(f"{kaisai} — {len(df_raw)}頭（{df_raw['レース番号'].nunique()}レース）")
            with col_back:
                if st.button("← 開催選択に戻る"):
                    for key in ["shutsuba_df", "shutsuba_kaisai", "shutsuba_result",
                                "shutsuba_features", "shutsuba_explainer", "shutsuba_race_label"]:
                        st.session_state.pop(key, None)
                    st.rerun()

            col_race_label, col_jockey_btn = st.columns([3, 1])
            with col_race_label:
                st.markdown("**レースを選ぶ**")
            with col_jockey_btn:
                if st.button("騎手名鑑", use_container_width=True):
                    _show_jockey_directory(df_raw)
            race_nums = sorted(df_raw["レース番号"].unique())
            n_cols = 4
            for row_start in range(0, len(race_nums), n_cols):
                btn_cols = st.columns(n_cols)
                for col_idx, rn in enumerate(race_nums[row_start:row_start + n_cols]):
                    rd = df_raw[df_raw["レース番号"] == rn].iloc[0]
                    full_name = rd['レース名']
                    short_name = full_name[:6] + "…" if len(full_name) > 6 else full_name
                    label = f"R{rn} {short_name}"
                    with btn_cols[col_idx]:
                        if st.button(label, key=f"rn_{rn}", use_container_width=True, help=full_name):
                            full_label = f"R{rn} {full_name}"
                            df_race = df_raw[df_raw["レース番号"] == rn].copy()
                            with st.spinner("分析中..."):
                                try:
                                    df_pred = shutsuba_to_predict_df(df_race)
                                    model, features, explainer = load_model()
                                    df_result = predict_race(df_pred.copy(), model, features)
                                    st.session_state["shutsuba_result"] = df_result
                                    st.session_state["shutsuba_features"] = features
                                    st.session_state["shutsuba_explainer"] = explainer
                                    st.session_state["shutsuba_race_label"] = full_label
                                except Exception as e:
                                    st.error(f"分析エラー: {e}")

            if "shutsuba_result" in st.session_state:
                df_result = st.session_state["shutsuba_result"]
                features = st.session_state["shutsuba_features"]
                explainer = st.session_state["shutsuba_explainer"]

                race_label = st.session_state.get("shutsuba_race_label", "")
                st.subheader(f"{race_label} — 分析結果")
                st.caption("◎本命 ○対抗 ▲単穴 △連下候補 無印=見送り｜上位拮抗時は◎なし（△のみ）｜※出走数・勝率は直近4戦ベース")
                name_col = "馬名" if "馬名" in df_result.columns else None

                df_display = df_result.copy().reset_index(drop=True)
                df_display["印"] = _assign_marks(df_display["win_score"].values)
                df_display["勝率予測"] = df_display["win_score"].apply(lambda x: f"{x:.1f}%")
                # 順位列を予測スコア順で付与
                df_display["順位"] = range(1, len(df_display) + 1)
                sort_mode = st.radio("並び順", ["馬番順", "予測順"], horizontal=True, label_visibility="collapsed")
                if sort_mode == "馬番順" and "馬番" in df_display.columns:
                    df_display = df_display.sort_values("馬番").reset_index(drop=True)
                else:
                    df_display = df_display.sort_values("win_score", ascending=False).reset_index(drop=True)

                show_cols = []
                if "枠" in df_display.columns:
                    show_cols.append("枠")
                if "馬番" in df_display.columns:
                    show_cols.append("馬番")
                show_cols.append("順位")
                show_cols.append("印")
                if name_col:
                    show_cols.append(name_col)
                show_cols.append("勝率予測")
                for c in ["騎手", "馬齢", "出走数", "勝率", "前走_着順", "前走_人気", "前走_頭数"]:
                    if c in df_display.columns:
                        show_cols.append(c)

                df_show = df_display[show_cols].copy()
                if "勝率" in df_show.columns:
                    df_show["勝率"] = df_show["勝率"].apply(lambda x: f"{x:.0%}" if pd.notna(x) and x >= 0 else "-")
                if "前走_着順" in df_show.columns:
                    df_show["前走_着順"] = df_show["前走_着順"].apply(lambda x: f"{int(x)}着" if pd.notna(x) and x > 0 else "-")
                if "前走_人気" in df_show.columns:
                    df_show["前走_人気"] = df_show["前走_人気"].apply(lambda x: f"{int(x)}番人気" if pd.notna(x) and x > 0 else "-")
                if "出走数" in df_show.columns:
                    df_show["出走数"] = df_show["出走数"].apply(lambda x: f"{int(x)}戦" if pd.notna(x) and x >= 0 else "-")
                if "前走_頭数" in df_show.columns:
                    df_show["前走_頭数"] = df_show["前走_頭数"].apply(lambda x: f"{int(x)}頭立" if pd.notna(x) and x > 0 else "-")

                df_show.index = range(1, len(df_show) + 1)

                # 枠番カラー定義
                WAKU_STYLE = {
                    1: ("background:#fff;color:#333;border:1px solid #ccc", "白"),
                    2: ("background:#333;color:#fff", "黒"),
                    3: ("background:#e74c3c;color:#fff", "赤"),
                    4: ("background:#3498db;color:#fff", "青"),
                    5: ("background:#f1c40f;color:#333", "黄"),
                    6: ("background:#27ae60;color:#fff", "緑"),
                    7: ("background:#e67e22;color:#fff", "橙"),
                    8: ("background:#e91e9b;color:#fff", "桃"),
                }

                # === プロフェッショナルHTMLテーブル ===
                top_score = df_show["勝率予測"].str.rstrip("%").astype(float).max() if "勝率予測" in df_show.columns else 100

                html = """<style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap');
                .rt{width:100%;border-collapse:separate;border-spacing:0;font-family:'Inter','Noto Sans JP',system-ui,sans-serif;font-size:13px;border-radius:8px;overflow:hidden;box-shadow:0 0 0 1px rgba(0,0,0,.06),0 2px 8px rgba(0,0,0,.04)}
                .rt thead th{padding:0;margin:0;height:0;border:none;overflow:hidden;line-height:0;font-size:0}
                .rt .rth{display:flex;align-items:center;background:linear-gradient(135deg,#1a1f36 0%,#252b48 100%);padding:0 4px}
                .rt .rth span{display:inline-block;padding:11px 4px;color:rgba(255,255,255,.45);font-size:10px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;text-align:center;white-space:nowrap}
                .rt tbody tr{transition:all .15s ease}
                .rt tbody tr:hover{background:rgba(37,136,116,.04)}
                .rt tbody tr:nth-child(even){background:rgba(0,0,0,.015)}
                .rt tbody tr:nth-child(even):hover{background:rgba(37,136,116,.04)}
                .rt td{padding:10px 6px;border-bottom:1px solid rgba(0,0,0,.04);vertical-align:middle;font-variant-numeric:tabular-nums}

                .wk{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:4px;font-weight:700;font-size:11px;line-height:1;box-shadow:0 1px 2px rgba(0,0,0,.1)}
                .wk1{background:#f5f5f5;color:#444;box-shadow:0 1px 2px rgba(0,0,0,.08),inset 0 0 0 1px rgba(0,0,0,.12)}
                .wk2{background:#2c2c2c;color:#fff}
                .wk3{background:#c0392b;color:#fff}
                .wk4{background:#2471a3;color:#fff}
                .wk5{background:#f0c929;color:#1a1a1a}
                .wk6{background:#1e8449;color:#fff}
                .wk7{background:#d35400;color:#fff}
                .wk8{background:#c2185b;color:#fff}

                .rk{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;padding:0 4px;border-radius:10px;font-weight:700;font-size:10px}
                .rk1{background:linear-gradient(135deg,#1a8870,#21a88a);color:#fff}
                .rk2{background:linear-gradient(135deg,#3a9e8e,#4db8a4);color:#fff}
                .rk3{background:linear-gradient(135deg,#6bb5a8,#82c9bd);color:#fff}
                .rkn{background:none;color:#bbb;font-weight:500}

                .mk{text-align:center;font-weight:800;font-size:16px;line-height:1}
                .mkh{color:#c0392b}.mkt{color:#d35400}.mks{color:#2471a3}.mkr{color:#999}

                .hn{font-weight:600;color:#1a1f36;font-size:13px;white-space:nowrap;letter-spacing:-.2px}
                .jk{color:#888;font-size:11.5px;white-space:nowrap}

                .sc-wrap{display:flex;align-items:center;gap:6px}
                .sc-bar{flex:1;height:4px;background:rgba(0,0,0,.04);border-radius:2px;overflow:hidden;min-width:40px}
                .sc-fill{height:100%;border-radius:2px;transition:width .3s ease}
                .sc-val{font-weight:700;font-size:13px;white-space:nowrap;min-width:38px;text-align:right}
                .sc-hi{color:#1a8870}.sc-hi .sc-fill{background:linear-gradient(90deg,#1a8870,#21a88a)}
                .sc-md{color:#1a1f36}.sc-md .sc-fill{background:#a0aec0}
                .sc-lo{color:#ccc}.sc-lo .sc-fill{background:#e2e8f0}

                .sub{color:#999;font-size:11.5px;text-align:center}
                .num{text-align:center;color:#666;font-weight:500}
                </style>"""

                # ヘッダー行（flexで幅制御）
                col_labels = {"枠":"枠","馬番":"番","順位":"#","印":"印","馬名":"馬名","勝率予測":"勝率予測",
                              "騎手":"騎手","馬齢":"齢","出走数":"戦","勝率":"通算","前走_着順":"前走","前走_人気":"人気","前走_頭数":"頭数"}
                col_widths = {"枠":"34px","馬番":"30px","順位":"30px","印":"26px","馬名":"2fr","勝率予測":"3fr",
                              "騎手":"1.5fr","馬齢":"26px","出走数":"34px","勝率":"44px","前走_着順":"38px","前走_人気":"56px","前走_頭数":"48px"}

                grid_cols = " ".join(col_widths.get(c, "1fr") for c in df_show.columns)
                header_html = '<div class="rth" style="display:grid;grid-template-columns:' + grid_cols + ';gap:2px">'
                for col in df_show.columns:
                    header_html += f'<span>{col_labels.get(col, col)}</span>'
                header_html += '</div>'

                html += '<table class="rt"><thead><tr><th>' + header_html + '</th></tr></thead><tbody>'

                MARK_CLS = {"◎":"mkh","○":"mkt","▲":"mks","△":"mkr"}

                for _, row in df_show.iterrows():
                    row_html = '<tr><td><div style="display:grid;grid-template-columns:' + grid_cols + ';gap:2px;align-items:center">'
                    for col in df_show.columns:
                        val = row[col]
                        if col == "枠":
                            w = int(val) if pd.notna(val) and val != 0 else 0
                            row_html += f'<div style="text-align:center"><span class="wk wk{w}">{w}</span></div>' if w > 0 else '<div style="text-align:center">-</div>'
                        elif col == "馬番":
                            row_html += f'<div class="num">{int(val)}</div>'
                        elif col == "順位":
                            r = int(val)
                            cls = f"rk{r}" if r <= 3 else "rkn"
                            row_html += f'<div style="text-align:center"><span class="rk {cls}">{r}</span></div>'
                        elif col == "印":
                            cls = MARK_CLS.get(str(val), "")
                            row_html += f'<div class="mk {cls}">{val}</div>'
                        elif col == "馬名":
                            row_html += f'<div><span class="hn">{val}</span></div>'
                        elif col == "騎手":
                            row_html += f'<div><span class="jk">{val}</span></div>'
                        elif col == "勝率予測":
                            sv = float(str(val).rstrip("%"))
                            pct = min(sv / top_score * 100, 100)
                            cls = "sc-hi" if sv >= top_score * 0.7 else ("sc-md" if sv >= top_score * 0.3 else "sc-lo")
                            row_html += f'<div class="sc-wrap {cls}"><span class="sc-val">{val}</span><div class="sc-bar"><div class="sc-fill" style="width:{pct:.0f}%"></div></div></div>'
                        elif col == "馬齢":
                            row_html += f'<div class="num">{int(val) if pd.notna(val) else "-"}</div>'
                        else:
                            row_html += f'<div class="sub">{val}</div>'
                    row_html += '</div></td></tr>'
                    html += row_html
                html += '</tbody></table>'
                st.markdown(html, unsafe_allow_html=True)

                # 馬の詳細をボタンで表示
                st.caption("馬名をクリックすると詳細を表示")
                horse_names = df_display[name_col].tolist() if name_col else []
                btn_row_size = 6
                for row_start in range(0, len(horse_names), btn_row_size):
                    cols = st.columns(btn_row_size)
                    for j, hname in enumerate(horse_names[row_start:row_start + btn_row_size]):
                        with cols[j]:
                            if st.button(hname, key=f"detail_{row_start+j}", use_container_width=True):
                                sel_row = df_display[df_display[name_col] == hname].iloc[0]
                                _show_horse_detail(sel_row, features, explainer)

                if name_col:
                    fig = px.bar(
                        df_result.head(10), x=name_col, y="win_score",
                        title="予測スコア（上位10頭）",
                        labels={"win_score": "予測スコア (%)"},
                        color="win_score",
                        color_continuous_scale=[[0, "#C0392B"], [0.5, C_ACCENT], [1, C_POS]],
                    )
                    fig.update_layout(
                        showlegend=False, height=320,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Noto Sans JP"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # 馬の比較
                st.subheader("馬を比較する")

                compare_features = [
                    ("前走_着順", "前走着順", True),      # (列名, 表示名, 低い方が良いか)
                    ("前走_人気", "前走人気", True),
                    ("勝率", "通算勝率", False),
                    ("騎手_勝率", "騎手勝率", False),
                    ("騎手_場別勝率", "騎手×場別勝率", False),
                    ("前走_上がり3F", "前走上がり3F", True),
                ]

                horse_options = df_result[name_col].tolist() if name_col else list(range(len(df_result)))

                col_a, col_b = st.columns(2)
                with col_a:
                    horse_a = st.selectbox("馬A", horse_options, index=0, key="shap_horse_a")
                with col_b:
                    horse_b_options = ["（未選択）"] + [h for h in horse_options if h != horse_a]
                    horse_b = st.selectbox("馬B（比較）", horse_b_options, index=0, key="shap_horse_b")

                if horse_a:
                    row_a = df_result[df_result[name_col] == horse_a].iloc[0]

                    def _fmt_val(feat, val):
                        if pd.isna(val) or val == -1:
                            return "-"
                        if "勝率" in feat:
                            return f"{val:.1%}"
                        if "着順" in feat:
                            return f"{int(val)}着"
                        if "人気" in feat:
                            return f"{int(val)}番人気"
                        if "3F" in feat:
                            return f"{val:.1f}秒"
                        return f"{val}"

                    def _raw_val(row, feat):
                        v = row.get(feat, -1)
                        return float(v) if pd.notna(v) and v != -1 else 0

                    if horse_b != "（未選択）":
                        row_b = df_result[df_result[name_col] == horse_b].iloc[0]

                        # 比較テーブル（勝っている方をハイライト）
                        compare_rows = []
                        for feat, disp, lower_better in compare_features:
                            va = _raw_val(row_a, feat)
                            vb = _raw_val(row_b, feat)
                            fa = _fmt_val(feat, row_a.get(feat, -1))
                            fb = _fmt_val(feat, row_b.get(feat, -1))

                            if va != 0 and vb != 0:
                                if lower_better:
                                    winner = "A" if va < vb else ("B" if vb < va else "-")
                                else:
                                    winner = "A" if va > vb else ("B" if vb > va else "-")
                            else:
                                winner = "-"

                            compare_rows.append({
                                "項目": disp,
                                horse_a: fa,
                                horse_b: fb,
                                "優勢": horse_a if winner == "A" else (horse_b if winner == "B" else "-"),
                            })

                        df_compare = pd.DataFrame(compare_rows)

                        # 勝敗カウント
                        a_wins = sum(1 for r in compare_rows if r["優勢"] == horse_a)
                        b_wins = sum(1 for r in compare_rows if r["優勢"] == horse_b)

                        col1, col2, col3 = st.columns([2, 1, 2])
                        with col1:
                            st.metric(horse_a, f"{a_wins}項目で優勢")
                        with col2:
                            st.markdown("<div style='text-align:center;padding-top:1rem;font-size:1.5rem;font-weight:bold;'>vs</div>", unsafe_allow_html=True)
                        with col3:
                            st.metric(horse_b, f"{b_wins}項目で優勢")

                        st.dataframe(
                            df_compare.set_index("項目"),
                            use_container_width=True,
                        )
                    else:
                        # 単体表示
                        st.markdown(f"**{horse_a}の分析データ**")
                        single_rows = []
                        for feat, disp, _ in compare_features:
                            single_rows.append({
                                "項目": disp,
                                "値": _fmt_val(feat, row_a.get(feat, -1)),
                            })
                        st.dataframe(
                            pd.DataFrame(single_rows).set_index("項目"),
                            use_container_width=True,
                        )

        else:
            # 初期画面
            st.subheader("今週の出馬表から分析")
            st.caption("JRA公式サイトの公開情報（出馬表）を取得し、過去データに基づいて分析します。木曜16時以降に利用可能。")

            if not MODEL_PATH.exists():
                st.warning("モデルが見つかりません（data/model.pkl）")
            else:
                if st.button("出馬表を取得", type="primary"):
                    with st.spinner("JRAサイトから開催情報を取得中..."):
                        try:
                            races = get_shutsuba_races()
                            if races:
                                st.session_state["shutsuba_races"] = races
                                st.rerun()
                            else:
                                st.warning("出馬表がまだ公開されていません（木曜16時以降に再試行してください）")
                        except Exception as e:
                            st.error(f"取得エラー: {e}")

                if "shutsuba_races" in st.session_state:
                    races = st.session_state["shutsuba_races"]
                    st.success(f"{len(races)}開催を取得しました")
                    st.markdown("**開催を選ぶ**")
                    race_cols = st.columns(min(len(races), 3))
                    for i, race in enumerate(races):
                        with race_cols[i % 3]:
                            if st.button(race["label"], key=f"race_{i}", use_container_width=True):
                                st.session_state["shutsuba_loading"] = True
                                st.session_state["shutsuba_loading_label"] = race["label"]
                                st.rerun()

# ── Tab2: 使い方 ──────────────────────────────────────────────
with tab2:
    st.subheader("使い方ガイド")
    st.markdown("""
### 基本の操作

1. **「出馬表・予測」タブ** で「出馬表を取得」を押す（木曜16時以降）
2. **開催を選択**（例: 3/21 2回中山7日）
3. **レースを選択**（例: R11）→ 分析結果が表示されます
4. **馬名ボタン** をクリック → 馬の詳細（戦績・騎手成績・影響度分析）を表示
5. **「馬を比較する」** で2頭の実値を項目ごとに比較
6. **「騎手名鑑」** ボタンで出場騎手の成績一覧を確認

### 分析の仕組み

過去のレース結果データ（約24万件）を学習した統計モデルが、出馬表の情報から各馬の勝率を推定します。

**主な分析要素（20項目）：**
- 馬の基本情報（馬齢・斤量・距離）
- 前走・2走前の着順・人気・上がり3F
- 通算勝率・複勝率
- 騎手の通算勝率・コース別勝率・競馬場別勝率

### 印の付け方

- ◎○▲は上位3頭に各1頭ずつ（上位が拮抗している場合は◎なし）
- △は出走頭数の半分まで
- 無印は見送り
- ※出走数・勝率は直近4戦ベース

### 技術スタック

| 項目 | 技術 |
|------|------|
| 予測モデル | LightGBM（勾配ブースティング） |
| 根拠の可視化 | SHAP（TreeExplainer） |
| データ取得 | JRA公式サイト（公開情報） |
| 学習データ | Kaggle JRAレース結果（2016〜2021年） |

### 免責事項

> 本アプリは統計分析による参考情報の提供を目的としています。
> 馬券購入は自己責任でお願いします。
> 分析結果は将来の的中を保証するものではありません。
""")

# ─── フッター ────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built by <a href="https://kaleidofuture.com" target="_blank">KaleidoFuture</a>
</div>
""", unsafe_allow_html=True)
