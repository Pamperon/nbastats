# app.py — NBA Stats + Bet365 (versione veloce)
# - Batch: Giocatore, Linea, Over 5, Over 10, Over intera stagione
# - Singolo: nessun Push
# - Bet365: solo mercato "Più di", output Giocatore/Linea, deduplicato, solo Excel
# - get_player_gamelog: singolo tentativo (più rapido)

import math
import datetime as dt
import time
import unicodedata
import io
import re
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog

# -------------------- CONFIG UI --------------------
st.set_page_config(page_title="NBA Stats + Bet365", layout="centered")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.75rem;
        max-width: 900px;
        margin: auto;
    }
    label[data-testid="stWidgetLabel"] > div:empty {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 NBA Stats — Props-style Analyzer")
st.caption(
    "Analisi batch, ricerca singolo giocatore e estrazione HTML Bet365 (solo mercato 'Più di')."
)

# -------------------- UTILITIES --------------------
def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode("utf-8").lower().strip()

def season_string_for_today(today: Optional[dt.date] = None) -> str:
    d = today or dt.date.today()
    if d.month >= 10:
        start = d.year
    else:
        start = d.year - 1
    end = (start + 1) % 100
    return f"{start}-{end:02d}"

def prev_season(season: str) -> str:
    y1 = int(season[:4]) - 1
    y2 = (y1 + 1) % 100
    return f"{y1}-{y2:02d}"

CURRENT_SEASON = season_string_for_today()

def force_half(value: float, min_v: float = 0.0, max_v: float = 120.0) -> float:
    v = math.floor(float(value)) + 0.5
    if v < min_v + 0.5:
        v = min_v + 0.5
    if v > max_v - 0.5:
        v = max_v - 0.5
    return v

# -------------------- DATA LAYERS --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_all_players() -> List[Dict]:
    return players.get_players()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_teams() -> List[Dict]:
    return teams.get_teams()

def find_player_id_by_name(player_name: str) -> Optional[int]:
    custom_map = {
        "pj washington": "p.j. washington",
        "ron holland ii": "ronald holland ii",
    }
    norm_in = normalize_name(player_name)
    candidate = custom_map.get(norm_in, player_name)

    all_p = cached_all_players()
    norm_candidate = normalize_name(candidate)

    for p in all_p:
        if normalize_name(p["full_name"]) == norm_candidate:
            return p["id"]
    for p in all_p:
        if norm_candidate in normalize_name(p["full_name"]):
            return p["id"]
    return None

# ✅ versione veloce: un solo tentativo
@st.cache_data(ttl=1800, show_spinner=False)
def get_player_gamelog(player_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for c in ("PTS", "AST", "REB"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["PAR"] = df["PTS"] + df["AST"] + df["REB"]
    return df.sort_values("GAME_DATE", ascending=False)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_player_full_history(player_id: int, start_year: int = 2015) -> pd.DataFrame:
    frames = []
    current_year = dt.date.today().year
    for year in range(start_year, current_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        try:
            df = playergamelog.PlayerGameLog(
                player_id=player_id, season=season, season_type_all_star="Regular Season"
            ).get_data_frames()[0]
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
            for c in ("PTS", "AST", "REB"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["PAR"] = df["PTS"] + df["AST"] + df["REB"]
            frames.append(df)
            time.sleep(0.1)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("GAME_DATE", ascending=False)

# -------------------- HELPERS STATS --------------------
def percent_over(series: pd.Series, line: float) -> Tuple[float, int, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    total = len(s)
    if total == 0:
        return 0.0, 0, 0
    over = int((s > line).sum())
    return round(100 * over / total, 1), over, total

def calculate_over_under(series: pd.Series, line: float):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 0.0, 0, 0
    over_c = int((s > line).sum())
    under_c = int((s < line).sum())
    total = len(s)
    return (round(100 * over_c / total, 1),
            round(100 * under_c / total, 1),
            over_c, under_c)

def filter_game_type(df: pd.DataFrame, game_type: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if game_type == "Casa":
        if "MATCHUP" not in df.columns:
            return df.head(0)
        return df[df["MATCHUP"].str.contains("vs", na=False)]
    if game_type == "Ospite":
        if "MATCHUP" not in df.columns:
            return df.head(0)
        return df[df["MATCHUP"].str.contains("@", na=False)]
    return df

# -------------------- PLOTTING --------------------
def plot_bar(df: pd.DataFrame, col: str, line: float, title: str, rotate: int = 45):
    if df.empty:
        st.warning("⚠️ Nessun dato disponibile per il grafico.")
        return
    dd = df.sort_values("GAME_DATE")
    labels = dd["GAME_DATE"].dt.strftime("%m/%d")
    values = dd[col].astype(float)

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#10B981" if v > line else "#EF4444" for v in values]
    bars = ax.bar(range(len(values)), values, width=0.6, color=colors)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{values.iloc[i]:.0f}", ha="center", va="bottom", fontsize=8, color="#e5e7eb")
    ax.axhline(line, color="#9CA3AF", linestyle="--", linewidth=1.2, label=f"Linea {line:g}")
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#121821")
    ax.tick_params(colors="#e5e7eb", labelsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=8)
    ax.legend(facecolor="#121821", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=9)
    ax.set_title(title, fontsize=13, color="#e5e7eb")
    st.pyplot(fig)

# -------------------- BET365 PARSER --------------------
def _norm_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(s.split()).strip()

def _contains_over(label: str) -> bool:
    lab = unicodedata.normalize("NFKD", label.lower()).encode("ascii", "ignore").decode("ascii")
    return ("piu di" in lab) or ("over" in lab)

def extract_bet365(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    pods = soup.select(".gl-MarketGroupPod.src-FixtureSubGroup")
    for pod in pods:
        fix_el = pod.select_one(".src-FixtureSubGroupButton_Text")
        players_list = [_norm_text(e.get_text()) for e in pod.select(".srb-ParticipantLabelWithTeam_Name")]
        for market in pod.select(".gl-Market.gl-Market_General-columnheader"):
            header_el = market.select_one(".gl-MarketColumnHeader")
            market_name = _norm_text(header_el.get_text() if header_el else "")
            if not _contains_over(market_name):
                continue
            parts = market.select(".gl-ParticipantCenteredStacked.gl-Participant_General")
            entries = []
            for p in parts:
                line_el = p.select_one(".gl-ParticipantCenteredStacked_Handicap")
                line = _norm_text(line_el.get_text()) if line_el else ""
                entries.append(line)
            for i, name in enumerate(players_list):
                if i < len(entries):
                    rows.append({"Giocatore": name, "Linea": entries[i]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["Giocatore", "Linea"]).reset_index(drop=True)
    return df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    bio.seek(0)
    return bio.read()

# -------------------- UI --------------------
tab_batch, tab_single, tab_bet365 = st.tabs(["📥 Batch", "🔍 Singolo", "🧩 Bet365"])

# ========== BATCH ==========
with tab_batch:
    st.subheader("📥 Analisi batch da Excel")
    st.caption(f"File con colonne Giocatore e Linea — stagione {CURRENT_SEASON}")
    metric = st.radio("📊 Metrica", ["Punti", "Assist", "Rimbalzi", "P+A+R"], horizontal=True)
    metric_map = {"Punti": "PTS", "Assist": "AST", "Rimbalzi": "REB", "P+A+R": "PAR"}
    upl = st.file_uploader("Carica file (.xlsx)", type=["xlsx"])
    if upl:
        df_in = pd.read_excel(upl)
        if not {"Giocatore", "Linea"}.issubset(df_in.columns):
            st.error("Mancano colonne Giocatore o Linea.")
            st.stop()
        results = []
        progress = st.progress(0)
        for i, row in df_in.iterrows():
            player = str(row["Giocatore"])
            try:
                line = float(str(row["Linea"]).replace(",", "."))
            except:
                line = 0.0
            pid = find_player_id_by_name(player)
            if pid is None:
                results.append({"Giocatore": player, "Linea": line, "Over 5": "N/D", "Over 10": "N/D", "Over stagione": "N/D"})
                progress.progress((i+1)/len(df_in))
                continue
            try:
                glog = get_player_gamelog(pid)
            except:
                results.append({"Giocatore": player, "Linea": line, "Over 5": "ERR", "Over 10": "ERR", "Over stagione": "ERR"})
                progress.progress((i+1)/len(df_in))
                continue
            col = metric_map[metric]
            last5 = glog.head(5)
            last10 = glog.head(10)
            p5, _, _ = percent_over(last5[col], line)
            p10, _, _ = percent_over(last10[col], line)
            p_all, _, oc, uc = calculate_over_under(glog[col], line)
            results.append({"Giocatore": player, "Linea": line, "Over 5": f"{p5}%", "Over 10": f"{p10}%", "Over stagione": f"{p_all}%"})
            progress.progress((i+1)/len(df_in))
        df_out = pd.DataFrame(results)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("⬇️ Scarica Excel", data=to_excel_bytes(df_out), file_name="risultati_batch.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ========== SINGOLO ==========
with tab_single:
    st.subheader("🔍 Analisi giocatore singolo")
    q = st.text_input("Nome giocatore:")
    if q.strip():
        pid = find_player_id_by_name(q)
        if pid:
            m = st.radio("📌 Metrica", ["Punti", "Assist", "Rimbalzi", "P+A+R"], horizontal=True)
            col = {"Punti": "PTS", "Assist": "AST", "Rimbalzi": "REB", "P+A+R": "PAR"}[m]
            try:
                df = get_player_gamelog(pid)
            except Exception as e:
                st.error(f"Errore NBA: {e}")
                st.stop()
            line_raw = st.number_input(f"🎯 Linea {m.lower()}", 0.0, 120.0, 20.5, 1.0, format="%.1f")
            line = force_half(line_raw)
            st.subheader("📈 Grafico")
            plot_bar(df.head(10), col, line, f"{q} — {m}")
            st.subheader("📊 Statistiche")
            p5, o5, t5 = percent_over(df.head(5)[col], line)
            p10, o10, t10 = percent_over(df.head(10)[col], line)
            pall, _, oc, uc = calculate_over_under(df[col], line)
            st.write(f"Ultime 5: {p5}% ({o5}/{t5})")
            st.write(f"Ultime 10: {p10}% ({o10}/{t10})")
            st.write(f"Intera stagione: {pall}% over ({oc}/{len(df)})")

# ========== BET365 ==========
with tab_bet365:
    st.subheader("🧩 Estrazione Bet365 (Più di)")
    html_file = st.file_uploader("Carica file .html / .txt", type=["html", "htm", "txt"])
    html_text = st.text_area("Oppure incolla qui il codice HTML", height=200)
    html_content = ""
    if html_file:
        html_content = html_file.read().decode("utf-8", errors="ignore")
    elif html_text.strip():
        html_content = html_text
    if st.button("🔎 Estrai"):
        if not html_content.strip():
            st.warning("Incolla o carica l'HTML prima di procedere.")
            st.stop()
        df = extract_bet365(html_content)
        if df.empty:
            st.error("Nessun dato trovato.")
        else:
            st.success(f"{len(df)} righe trovate")
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Scarica Excel", data=to_excel_bytes(df),
                               file_name="bet365_linee.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
