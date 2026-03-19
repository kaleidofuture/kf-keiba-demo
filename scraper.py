"""
競馬予想AI - JRA 出馬表スクレイパー
公開情報（出馬表）を JRA 公式サイトから取得する
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from pathlib import Path

BASE_URL = "https://www.jra.go.jp"
ACCESS_D = f"{BASE_URL}/JRADB/accessD.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": BASE_URL,
}


def _post(session: requests.Session, url: str, cname: str) -> BeautifulSoup:
    resp = session.post(url, data={"cname": cname}, headers=HEADERS, timeout=15)
    text = resp.content.decode("cp932", errors="replace")
    return BeautifulSoup(text, "html.parser")


def _get(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, headers=HEADERS, timeout=15)
    text = resp.content.decode("cp932", errors="replace")
    return BeautifulSoup(text, "html.parser")


def get_shutsuba_races() -> list[dict]:
    """
    今週の出馬表から開催一覧を取得する。
    戻り値: [{"label": "2回中山7日", "cname": "pw01drl...", "url": "..."}]
    """
    session = requests.Session()
    soup = _post(session, ACCESS_D, "pw01dli00/F3")

    races = []
    for el in soup.find_all(onclick=True):
        onclick = el.get("onclick", "")
        m = re.search(r"doAction\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", onclick)
        if not m:
            continue
        url_part, cname = m.group(1), m.group(2)
        if not cname.startswith("pw01drl"):
            continue
        label = el.get_text(strip=True)
        if label and len(label) > 1:
            # cnameから日付を抽出 (pw01drl...20260321/xx → 3/21)
            date_str = cname.split("/")[0][-8:]
            try:
                m = int(date_str[4:6])
                d = int(date_str[6:8])
                label_with_date = f"{m}/{d} {label}"
            except (ValueError, IndexError):
                label_with_date = label
            races.append({"label": label_with_date, "cname": cname, "url": BASE_URL + url_part})

    return races


def get_shutsuba_table(cname: str, url: str) -> pd.DataFrame | None:
    """
    特定の開催のレース選択ページから各レースの出馬表を取得する。
    1. POST でレース選択ページ（12レース一覧）を開く
    2. 各レースの「出馬表」ボタン href を取得
    3. 各レースの出馬表ページ（GET）をパースして馬データを取得
    """
    session = requests.Session()

    # 出馬表トップ → 開催選択（セッション確立）
    _post(session, ACCESS_D, "pw01dli00/F3")
    soup = _post(session, url, cname)

    # cnameから競馬場コードを特定
    # pw01drl0006... → 0006=中山, 0009=阪神, 0007=中京 etc.
    venue_code_map = {
        "0001": "札幌", "0002": "函館", "0003": "福島", "0004": "新潟",
        "0005": "東京", "0006": "中山", "0007": "中京", "0008": "京都",
        "0009": "阪神", "0010": "小倉",
    }
    venue_name = ""
    m = re.search(r"pw01drl(\d{4})", cname)
    if m:
        venue_name = venue_code_map.get(m.group(1), "")

    # フォールバック: ページタイトルから抽出
    if not venue_name:
        venue_names = ["中山", "阪神", "中京", "東京", "京都", "福島", "新潟", "小倉", "札幌", "函館"]
        page_text = soup.get_text()
        for v in venue_names:
            if v in page_text:
                venue_name = v
                break

    # レース選択テーブルから「出馬表」ボタンの href を取得
    race_links = []
    table = soup.find("table")
    if not table:
        return None

    rows = table.find_all("tr")
    for i, tr in enumerate(rows[1:], 1):  # skip header row
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue

        race_num = i
        race_name = cells[1].get_text(strip=True)
        course_info = cells[3].get_text(strip=True)  # e.g. "ダート1,200m16頭"

        # 「出馬表」ボタンの href を探す
        shutsuba_link = None
        for a in tr.find_all("a"):
            href = a.get("href", "")
            if "CNAME=" in href and "出馬表" in a.get_text(strip=True):
                shutsuba_link = BASE_URL + href
                break

        if shutsuba_link:
            race_links.append({
                "race_num": race_num,
                "race_name": race_name,
                "course_info": course_info,
                "url": shutsuba_link,
            })

    if not race_links:
        return None

    # 各レースの出馬表をGETで取得してパース
    all_rows = []
    for race in race_links:
        r_soup = _get(session, race["url"])
        horses = _parse_race_shutsuba(r_soup)
        for h in horses:
            h["レース番号"] = race["race_num"]
            h["レース名"] = race["race_name"]
            h["コース距離"] = race["course_info"]
            h["競馬場名"] = venue_name
        all_rows.extend(horses)

    if not all_rows:
        return None

    return pd.DataFrame(all_rows)


def _parse_race_shutsuba(soup: BeautifulSoup) -> list[dict]:
    """個別レースの出馬表ページをパースして馬データを返す。

    JRA出馬表のセル構造:
    - [0] 枠 (金曜確定前は空)
    - [1] 馬番 (金曜確定前は空)
    - [2] 馬名+調教師名+血統 (複合セル)
    - [3] 性齢/毛色+負担重量+騎手名 (複合セル)
    - [4-7] 前走〜4走前
    """
    horses = []

    table = soup.find("table")
    if not table:
        return horses

    for tr in table.find_all("tr")[1:]:  # skip header
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        # セル[2]: 馬名+調教師 → 子要素から分離
        cell2 = tds[2]
        parts2 = [c.get_text(strip=True) for c in cell2.children
                   if hasattr(c, "get_text") and c.get_text(strip=True)]
        horse_name = parts2[0] if parts2 else ""
        trainer = parts2[1] if len(parts2) > 1 else ""

        if not horse_name:
            continue

        # セル[3]: 性齢/毛色+負担重量+騎手名
        cell3 = tds[3]
        parts3 = [c.get_text(strip=True) for c in cell3.children
                   if hasattr(c, "get_text") and c.get_text(strip=True)]
        sex_age = parts3[0] if parts3 else ""
        weight_kg = parts3[1] if len(parts3) > 1 else ""
        jockey = parts3[2] if len(parts3) > 2 else ""

        # 枠・馬番（金曜確定前は空）
        waku = tds[0].get_text(strip=True)
        umaban = tds[1].get_text(strip=True)

        row = {
            "馬名": horse_name,
            "枠": waku,
            "馬番": umaban,
            "性齢": sex_age,
            "負担重量": weight_kg,
            "騎手名": jockey.lstrip("▲△☆◇"),
            "調教師名": trainer,
        }

        # セル[4-7]: 前走〜4走前のデータを抽出
        past_races = _parse_past_races(tds[4:8])
        row.update(past_races)

        horses.append(row)

    return horses


def _parse_past_races(cells) -> dict:
    """前走〜4走前のセルから特徴量を抽出する。

    各セルの parts 構造:
    [0] "2026年1月17日中山"
    [1] "菜の花賞牝1勝ク"
    [2] "1着8頭1番2番人気"
    [3] "横山 武史55.0kg"
    [4] "1600芝1:34.4良462kg"
    [5] "3333F 34.2ファンクション(0.0)"
    """
    result = {}
    win_count = 0
    top3_count = 0
    race_count = 0

    for i, cell in enumerate(cells):
        text = cell.get_text(strip=True) if hasattr(cell, "get_text") else ""
        if not text:
            continue

        race_count += 1
        prefix = f"前走" if i == 0 else f"{i+1}走前"

        parts = [c.get_text(strip=True) for c in cell.children
                 if hasattr(c, "get_text") and c.get_text(strip=True)]

        # 着順・頭数・人気を抽出 (parts[2]: "1着8頭1番2番人気")
        if len(parts) > 2:
            rank_text = parts[2]
            m_rank = re.search(r"(\d+)着", rank_text)
            m_heads = re.search(r"(\d+)頭", rank_text)
            m_pop = re.search(r"(\d+)番人気", rank_text)

            if m_rank:
                rank = int(m_rank.group(1))
                result[f"{prefix}_着順"] = rank
                if rank == 1:
                    win_count += 1
                if rank <= 3:
                    top3_count += 1
            if m_heads:
                result[f"{prefix}_頭数"] = int(m_heads.group(1))
            if m_pop:
                result[f"{prefix}_人気"] = int(m_pop.group(1))

        # 距離・タイム・馬場・馬体重 (parts[4]: "1600芝1:34.4良462kg")
        if len(parts) > 4:
            detail = parts[4]
            m_dist = re.match(r"(\d{3,5})", detail)
            if m_dist:
                result[f"{prefix}_距離"] = int(m_dist.group(1))

            m_time = re.search(r"(\d):(\d{2}\.\d)", detail)
            if m_time:
                time_sec = float(m_time.group(1)) * 60 + float(m_time.group(2))
                result[f"{prefix}_タイム"] = time_sec

            m_weight = re.search(r"(\d{3,4})kg", detail)
            if m_weight:
                result[f"{prefix}_馬体重"] = int(m_weight.group(1))

            # 芝/ダート
            if "芝" in detail:
                result[f"{prefix}_コース"] = 1  # 芝
            elif "ダ" in detail:
                result[f"{prefix}_コース"] = 0  # ダート

        # 上がり3F (parts[5]: "3333F 34.2ファンクション(0.0)")
        if len(parts) > 5:
            m_3f = re.search(r"3F\s*(\d{2}\.\d)", parts[5])
            if m_3f:
                result[f"{prefix}_上がり3F"] = float(m_3f.group(1))

    result["出走数"] = race_count
    result["勝利数"] = win_count
    result["複勝数"] = top3_count
    if race_count > 0:
        result["勝率"] = win_count / race_count
        result["複勝率"] = top3_count / race_count
    else:
        result["勝率"] = 0.0
        result["複勝率"] = 0.0

    return result


def shutsuba_to_predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    出馬表 DataFrame をモデルの predict_race() に渡せる形式に変換する。
    JRA サイトの列名とモデルの列名のマッピングを行う。
    """
    out = pd.DataFrame()

    # そのままコピーする列
    direct_cols = [
        "馬名", "レース番号", "レース名", "コース距離", "競馬場名",
        "前走_着順", "前走_人気", "前走_頭数", "前走_距離",
        "前走_タイム", "前走_馬体重", "前走_上がり3F",
        "2走前_着順", "2走前_人気",
        "出走数", "勝率", "複勝率",
    ]
    for col in direct_cols:
        if col in df.columns:
            out[col] = df[col]

    # リネームが必要な列
    if "騎手名" in df.columns:
        out["騎手"] = df["騎手名"]
    if "負担重量" in df.columns:
        out["斤量"] = df["負担重量"].astype(str).str.replace("kg", "", regex=False)
        out["斤量"] = pd.to_numeric(out["斤量"], errors="coerce")
    if "性齢" in df.columns:
        out["馬齢"] = df["性齢"].astype(str).str.extract(r"(\d+)")[0]
        out["馬齢"] = pd.to_numeric(out["馬齢"], errors="coerce")

    # コース距離からメートル数を抽出（「ダート1,200m16頭」→ 1200）
    if "コース距離" in out.columns:
        out["距離(m)"] = out["コース距離"].astype(str).str.replace(",", "").str.extract(r"(\d{3,5})m")[0]
        out["距離(m)"] = pd.to_numeric(out["距離(m)"], errors="coerce")

    # 数値変換
    numeric_cols = [
        "前走_着順", "前走_人気", "前走_頭数", "前走_距離",
        "前走_タイム", "前走_馬体重", "前走_上がり3F",
        "2走前_着順", "2走前_人気",
        "出走数", "勝率", "複勝率",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # 騎手成績をルックアップ
    jockey_csv = Path(__file__).parent / "data" / "jockey_stats.csv"
    if jockey_csv.exists() and "騎手" in out.columns:
        jstats = pd.read_csv(jockey_csv, encoding="utf-8-sig")
        if "騎手_key" not in jstats.columns:
            jstats["騎手_key"] = jstats["騎手"].str.replace(" ", "").str.replace("\u3000", "")

        def _normalize_jockey(name):
            """騎手名を正規化: スペース除去 + 外国人の'X.'プレフィックス除去"""
            s = str(name).replace(" ", "").replace("\u3000", "")
            # "C.ルメール" → "ルメール", "M.デムーロ" → "デムーロ"
            s = re.sub(r"^[A-Za-z]\.", "", s)
            return s

        out["_jkey"] = out["騎手"].apply(_normalize_jockey)
        jstats["_jkey_norm"] = jstats["騎手_key"].apply(_normalize_jockey)
        lookup = jstats.set_index("_jkey_norm")[["騎手_勝率", "騎手_複勝率", "騎乗数"]].to_dict("index")

        # データにない騎手は全騎手の平均値で埋める
        avg_win = jstats["騎手_勝率"].mean()
        avg_top3 = jstats["騎手_複勝率"].mean()
        avg_rides = jstats["騎乗数"].median()

        out["騎手_勝率"] = out["_jkey"].map(lambda k: lookup.get(k, {}).get("騎手_勝率", avg_win))
        out["騎手_複勝率"] = out["_jkey"].map(lambda k: lookup.get(k, {}).get("騎手_複勝率", avg_top3))
        out["騎手_騎乗数"] = out["_jkey"].map(lambda k: lookup.get(k, {}).get("騎乗数", avg_rides))

        # 騎手×コース別勝率（芝/ダート）
        if "コース距離" in out.columns:
            turf_lookup = {k: v.get("騎手_芝_勝率", avg_win)
                          for k, v in jstats.set_index("_jkey_norm").to_dict("index").items()}
            dirt_lookup = {k: v.get("騎手_ダート_勝率", avg_win)
                          for k, v in jstats.set_index("_jkey_norm").to_dict("index").items()}
            out["騎手_コース別勝率"] = [
                turf_lookup.get(k, avg_win) if "芝" in str(c) else dirt_lookup.get(k, avg_win)
                for k, c in zip(out["_jkey"], out["コース距離"])
            ]

        # 騎手×競馬場別勝率
        venue_csv = Path(__file__).parent / "data" / "jockey_venue_stats.csv"
        if venue_csv.exists() and "競馬場名" in out.columns:
            vstats = pd.read_csv(venue_csv, encoding="utf-8-sig")
            vstats["_vkey"] = vstats["騎手_key"].apply(_normalize_jockey)
            # (騎手, 競馬場) → (勝率, 騎乗数) のルックアップ
            vlookup = {}
            for _, vrow in vstats.iterrows():
                vlookup[(vrow["_vkey"], vrow["競馬場名"])] = (vrow["勝率"], vrow["騎乗数"])

            # 騎手の全体勝率ルックアップ
            overall = {k: v.get("騎手_勝率", avg_win)
                      for k, v in jstats.set_index("_jkey_norm").to_dict("index").items()}

            def _venue_rate(jkey, venue):
                entry = vlookup.get((jkey, venue))
                if entry is None:
                    return overall.get(jkey, avg_win)
                rate, rides = entry
                # 30騎乗未満はサンプル不足 → 全体勝率とブレンド
                if rides < 30:
                    weight = rides / 30
                    return rate * weight + overall.get(jkey, avg_win) * (1 - weight)
                return rate

            out["騎手_場別勝率"] = [
                _venue_rate(k, v)
                for k, v in zip(out["_jkey"], out["競馬場名"])
            ]

        out.drop(columns=["_jkey"], inplace=True)

    return out
