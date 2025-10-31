# app.py — NBA Stats + Bet365 Extractor (centrato/compatto)
# - Batch: rimosso teams_roster_map(); sigla squadra dal TEAM_ABBREVIATION del gamelog
# - Valori sulle barre SEMPRE visibili
# - "Ultime 5/10" cross-stagione; "Intera stagione" = solo stagione corrente
# - Vs avversario: stagione corrente, precedente, carriera
# - Nuova scheda: 🧩 Estrazione Bet365 (HTML → Excel/CSV)

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

# Limita la larghezza massima del contenitore centrale
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
st.caption("Ricerca giocatore, percentuali Over/Under, grafico a barre, filtri casa/trasferta, storico vs avversario. + 🧩 Estrazione Bet365 HTML.")

# -------------------- UTILITIES (NBA) --------------------
def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8").lower().strip()

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

def with_retry(fn, *args, attempts: int = 3, wait_secs: float = 0.8, **kwargs):
    last_exc = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(wait_secs)
    raise last_exc

# forza qualunque numero al formato N + 0.5 (ed applica i limiti)
def force_half(value: float, min_v: float = 0.0, max_v: float = 120.0) -> float:
    v = math.floor(value) + 0.5
    if v < min_v + 0.5:
        v = min_v + 0.5
    if v > max_v - 0.5:
        v = max_v - 0.5
    return v

# -------------------- DATA LAYERS (NBA, CACHED) --------------------
@st.cache_data(ttl=3600)
def cached_all_players() -> List[Dict]:
    return players.get_players()

@st.cache_data(ttl=3600)
def cached_teams() -> List[Dict]:
    return teams.get_teams()

def find_player_id_by_name(player_name: str) -> Optional[int]:
    custom_map = {"pj washington": "p.j. washington", "ron holland ii": "ronald holland ii"}
    norm_in = normalize_name(player_name)
    candidate = custom_map.get(norm_in, player_name)

    all_players = cached_all_players()
    norm_candidate = normalize_name(candidate)

    for p in all_players:
        if normalize_name(p["full_name"]) == norm_candidate:
            return p["id"]
    for p in all_players:
        if norm_candidate in normalize_name(p["full_name"]):
            return p["id"]
    return None

@st.cache_data(ttl=1800)
def get_player_gamelog(player_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    def _call():
        return playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    df = with_retry(_call)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for c in ("PTS", "AST", "REB"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["PAR"] = df["PTS"] + df["AST"] + df["REB"]
    return df.sort_values("GAME_DATE", ascending=False)

@st.cache_data(ttl=6 * 3600)
def get_player_full_history(player_id: int, start_year: int = 2000) -> pd.DataFrame:
    frames = []
    for year in range(start_year, dt.date.today().year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        try:
            def _call():
                return playergamelog.PlayerGameLog(
                    player_id=player_id, season=season, season_type_all_star="Regular Season"
                ).get_data_frames()[0]
            df = with_retry(_call)
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
            for c in ("PTS", "AST", "REB"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["PAR"] = df["PTS"] + df["AST"] + df["REB"]
            frames.append(df)
            time.sleep(0.2)
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

def calculate_over_under_push(series: pd.Series, line: float):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 0.0, 0.0, 0, 0, 0
    over_c = int((s > line).sum())
    under_c = int((s < line).sum())
    push_c = int((s == line).sum())
    total = len(s)
    return (round(100 * over_c / total, 1), round(100 * under_c / total, 1),
            round(100 * push_c / total, 1), over_c, under_c, push_c)

def filter_game_type(df: pd.DataFrame, game_type: str) -> pd.DataFrame:
    if game_type == "Casa":
        return df[df["MATCHUP"].str.contains("vs", na=False)]
    if game_type == "Ospite":
        return df[df["MATCHUP"].str.contains("@", na=False)]
    return df

def get_last_n_games_cross_seasons(player_id: int, n: int, game_type: str) -> pd.DataFrame:
    cur = get_player_gamelog(player_id, season=CURRENT_SEASON)
    cur = filter_game_type(cur, game_type)
    if len(cur) >= n:
        return cur.head(n)
    prev = prev_season(CURRENT_SEASON)
    try:
        prev_df = get_player_gamelog(player_id, season=prev)
        prev_df = filter_game_type(prev_df, game_type)
    except Exception:
        prev_df = pd.DataFrame()
    combo = pd.concat([cur, prev_df], ignore_index=True).sort_values("GAME_DATE", ascending=False)
    return combo.head(n)

# -------------------- PLOTTING (sempre con valori/etichette) --------------------
def plot_bar(df: pd.DataFrame, col: str, line: float, title: str,
             rotate: int = 45, compact: bool = False):
    if df.empty:
        st.warning("⚠️ Nessun dato disponibile per il grafico.")
        return

    dd = df.sort_values("GAME_DATE")
    labels = dd["GAME_DATE"].dt.strftime("%m/%d")
    values = dd[col].astype(float)

    fig, ax = plt.subplots(figsize=(8, 3))  # compatto
    colors = ["#10B981" if v > line else "#EF4444" for v in values]
    bars = ax.bar(range(len(values)), values, width=0.6, color=colors)

    # Valori SEMPRE visibili
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{values.iloc[i]:.0f}", ha="center", va="bottom", fontsize=8, color="#e5e7eb")

    ax.axhline(line, color="#9CA3AF", linestyle="--", linewidth=1.2, label=f"Linea {line:g}")

    # stile dark
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#121821")
    ax.tick_params(colors="#e5e7eb", labelsize=9)
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.title.set_color("#e5e7eb")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(col, fontsize=10)
    ax.set_xlabel("")

    # Date SEMPRE visibili sotto (anche intera stagione)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=8)

    ax.legend(facecolor="#121821", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=9)
    st.pyplot(fig)

# -------------------- HELPERS (Bet365 extractor) --------------------
def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = " ".join(s.split())
    return s.strip()

def _contains_over(label: str) -> bool:
    lab = label.lower()
    lab_noaccent = unicodedata.normalize("NFKD", lab).encode("ascii", "ignore").decode("ascii")
    return ("piu di" in lab_noaccent) or ("over" in lab_noaccent)

def _contains_under(label: str) -> bool:
    lab = label.lower()
    lab_noaccent = unicodedata.normalize("NFKD", lab).encode("ascii", "ignore").decode("ascii")
    return ("meno di" in lab_noaccent) or ("under" in lab_noaccent)

def _to_float_odds(x: str):
    if x is None:
        return None
    x = x.strip().replace(",", ".")
    try:
        return float(x)
    except:
        m = re.search(r"\d+[.,]\d+", x)
        if m:
            try:
                return float(m.group(0).replace(",", "."))
            except:
                return None
        return None

def parse_over_under_layout(soup: BeautifulSoup, market_filter: str):
    rows = []
    pods = soup.select(".gl-MarketGroupPod.src-FixtureSubGroup")
    if not pods:
        return rows

    for pod in pods:
        fix_el = pod.select_one(".src-FixtureSubGroupButton_Text")
        fixture = _norm_text(fix_el.get_text()) if fix_el else ""
        players_list = [_norm_text(e.get_text()) for e in pod.select(".srb-ParticipantLabelWithTeam_Name")]

        for market in pod.select(".gl-Market.gl-Market_General-columnheader"):
            header_el = market.select_one(".gl-MarketColumnHeader")
            market_name = _norm_text(header_el.get_text() if header_el else "")

            if market_filter == "over" and not _contains_over(market_name):
                continue
            if market_filter == "under" and not _contains_under(market_name):
                continue

            parts = market.select(".gl-ParticipantCenteredStacked.gl-Participant_General")
            entries = []
            for p in parts:
                line_el = p.select_one(".gl-ParticipantCenteredStacked_Handicap")
                odds_el = p.select_one(".gl-ParticipantCenteredStacked_Odds")
                line = _norm_text(line_el.get_text()) if line_el else ""
                odds = _norm_text(odds_el.get_text()) if odds_el else ""
                entries.append((line, _to_float_odds(odds)))

            n = min(len(players_list), len(entries))
            for i in range(n):
                line, odds = entries[i]
                rows.append({
                    "Fixture": fixture,
                    "Player": players_list[i],
                    "Market": market_name,
                    "Line": line,
                    "Odds": odds
                })
    return rows

def parse_columns_layout(soup: BeautifulSoup):
    rows = []
    fixture_el = soup.select_one(".src-FixtureSubGroupButton_Text")
    fixture = _norm_text(fixture_el.get_text()) if fixture_el else ""
    players_list = [_norm_text(e.get_text()) for e in soup.select(".srb-ParticipantLabelWithTeam_Name")]
    if not players_list:
        return rows
    columns = soup.select(".srb-HScrollPlaceColumnMarket")
    if not columns:
        return rows

    for col in columns:
        header_el = col.select_one(".srb-HScrollPlaceHeader")
        header = _norm_text(header_el.get_text()) if header_el else ""
        odds_spans = col.select(".gl-ParticipantOddsOnly_Odds")
        odds = [_to_float_odds(_norm_text(sp.get_text())) for sp in odds_spans]

        n = min(len(players_list), len(odds))
        for i in range(n):
            rows.append({
                "Fixture": fixture,
                "Player": players_list[i],
                "Market": header,
                "Line": header,
                "Odds": odds[i],
            })
    return rows

def extract_bet365(html: str, market_filter: str = "over") -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = parse_over_under_layout(soup, market_filter=market_filter)
    if not rows:
        rows = parse_columns_layout(soup)
    df = pd.DataFrame(rows, columns=["Fixture", "Player", "Market", "Line", "Odds"])
    return df

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="estratto")
    bio.seek(0)
    return bio.read()

# -------------------- UI: TABS --------------------
tab_batch, tab_single, tab_bet365 = st.tabs([
    "📥 Analisi batch da Excel",
    "🔍 Ricerca giocatore singolo",
    "🧩 Estrazione Bet365 (HTML → Excel)"
])

# ==================== TAB: BATCH ====================
with tab_batch:
    st.subheader("📥 Carica file Excel per analisi batch")
    st.caption(f"Il file deve contenere le colonne **Giocatore** e **Linea**. Stagione corrente: **{CURRENT_SEASON}** (Regular Season).")

    metric_choice = st.radio("📊 Scegli la metrica da analizzare",
                             ["Punti", "Assist", "Rimbalzi", "P+A+R"], horizontal=True, key="batch_metric")
    metric_map = {"Punti": "PTS", "Assist": "AST", "Rimbalzi": "REB", "P+A+R": "PAR"}

    uploaded = st.file_uploader("📁 Carica Excel (.xlsx)", type=["xlsx"], key="batch_upload")

    if uploaded:
        try:
            df_in = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"❌ Impossibile leggere il file: {e}")
            st.stop()

        if not {"Giocatore", "Linea"}.issubset(df_in.columns):
            st.error("❌ Il file deve contenere le colonne 'Giocatore' e 'Linea'.")
            st.stop()

        st.success("File caricato correttamente. Avvio analisi…")
        results = []
        progress = st.progress(0)

        for i, row in df_in.iterrows():
            player_name = str(row["Giocatore"])
            line = float(row["Linea"])
            pid = find_player_id_by_name(player_name)

            if pid is None:
                results.append({
                    "Giocatore": player_name, "Squadra": "N/D", "Linea": line,
                    "% Over 5G": "N/D", "% Over 10G": "N/D",
                    "% Over Stagione": "N/D", "% Under Stagione": "N/D", "% Push Stagione": "N/D",
                })
                progress.progress((i + 1) / len(df_in))
                continue

            try:
                glog = get_player_gamelog(pid)  # stagione corrente (per % stagione e TEAM_ABBREVIATION)
            except Exception:
                results.append({
                    "Giocatore": player_name, "Squadra": "N/D", "Linea": line,
                    "% Over 5G": "ERR", "% Over 10G": "ERR",
                    "% Over Stagione": "ERR", "% Under Stagione": "ERR", "% Push Stagione": "ERR",
                })
                progress.progress((i + 1) / len(df_in))
                continue

            # 👇 squadra direttamente dal gamelog
            team = "N/D"
            if "TEAM_ABBREVIATION" in glog.columns and not glog.empty:
                try:
                    team = str(glog["TEAM_ABBREVIATION"].iloc[0])
                except Exception:
                    team = "N/D"

            col = metric_map[metric_choice]

            # Ultime 5/10 cross-stagione
            last5 = get_last_n_games_cross_seasons(pid, 5, "Totale")
            last10 = get_last_n_games_cross_seasons(pid, 10, "Totale")

            p5, _, _ = percent_over(last5[col], line)
            p10, _, _ = percent_over(last10[col], line)
            over_all, under_all, push_all, oc, uc, pc = calculate_over_under_push(glog[col], line)

            results.append({
                "Giocatore": player_name,
                "Squadra": team,
                "Linea": line,
                "% Over 5G": f"{p5}%",
                "% Over 10G": f"{p10}%",
                "% Over Stagione": f"{over_all}%",
                "% Under Stagione": f"{under_all}%",
                "% Push Stagione": f"{push_all}%",
            })
            progress.progress((i + 1) / len(df_in))

        df_out = pd.DataFrame(results)
        st.dataframe(df_out, use_container_width=True, hide_index=True)

        bio = io.BytesIO()
        df_out.to_excel(bio, index=False, engine="openpyxl")
        bio.seek(0)
        st.download_button("⬇️ Scarica risultati (Excel)", data=bio.read(),
                           file_name="risultati_over_under.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

# ==================== TAB: SINGLE ====================
with tab_single:
    st.subheader("🔍 Ricerca giocatore singolo")
    q = st.text_input("Inserisci il nome del giocatore (es. LeBron James)")

    if q.strip():
        norm_q = normalize_name(q)
        active = players.get_active_players()
        matches = [p for p in active if norm_q in normalize_name(p["full_name"])]
        if not matches:
            st.error("❌ Nessun giocatore trovato tra gli attivi. Prova il nome completo.")
        else:
            sel = st.selectbox("Seleziona il giocatore", matches, format_func=lambda p: p["full_name"])
            pid = sel["id"]

            m = st.radio("📌 Scegli la metrica", ["Punti", "Assist", "Rimbalzi", "P+A+R"],
                         horizontal=True, key="single_metric")
            game_type = st.radio("🎯 Tipo di partita", ["Totale", "Casa", "Ospite"],
                                 horizontal=True, key="single_gtype")

            col_map = {"Punti": "PTS", "Assist": "AST", "Rimbalzi": "REB", "P+A+R": "PAR"}
            col = col_map[m]

            try:
                df_cur = get_player_gamelog(pid)  # stagione corrente
            except Exception as e:
                st.error(f"Errore nel recupero dati: {e}")
                st.stop()

            df_cur = filter_game_type(df_cur, game_type)

            # --- Linea: solo .5, scatti di 1 ---
            defaults = {"PTS": 20.5, "AST": 5.5, "REB": 6.5, "PAR": 30.5}
            default_line = defaults[col]
            line_raw = st.number_input(
                f"🎯 Inserisci la linea {m.lower()}",
                min_value=0.0, max_value=120.0, value=default_line, step=1.0, format="%.1f",
                help="Si muove di 1 alla volta ed è sempre .5 (es. 20.5 → 21.5 → 22.5)."
            )
            line = force_half(line_raw)

            # --- Grafico (valori e date sempre visibili) ---
            st.subheader("📈 Grafico")
            chart_range = st.selectbox("Intervallo", ["Ultime 5", "Ultime 10", "Intera stagione"])
            if chart_range == "Ultime 5":
                dplot = get_last_n_games_cross_seasons(pid, 5, game_type)
                title = f"{sel['full_name']} | Ultime 5 — {m}"
                plot_bar(dplot, col, line, title, rotate=45, compact=False)
            elif chart_range == "Ultime 10":
                dplot = get_last_n_games_cross_seasons(pid, 10, game_type)
                title = f"{sel['full_name']} | Ultime 10 — {m}"
                plot_bar(dplot, col, line, title, rotate=45, compact=False)
            else:
                title = f"{sel['full_name']} | Intera stagione — {m}"
                plot_bar(df_cur, col, line, title, rotate=45, compact=False)

            # --- Statistiche ---
            st.subheader(f"📊 Statistiche {m.lower()}")
            last5 = get_last_n_games_cross_seasons(pid, 5, game_type)
            last10 = get_last_n_games_cross_seasons(pid, 10, game_type)

            p5, o5, t5 = percent_over(last5[col], line)
            p10, o10, t10 = percent_over(last10[col], line)
            pall_over, pall_under, pall_push, oc, uc, pc = calculate_over_under_push(df_cur[col], line)  # corrente

            st.write(f"**Ultime 5 (cross-stagione)**: {p5}% over ({o5}/{t5})")
            st.write(f"**Ultime 10 (cross-stagione)**: {p10}% over ({o10}/{t10})")
            st.write(
                f"**Intera stagione (corrente)**: Over {pall_over}% ({oc}/{len(df_cur)}), "
                f"Under {pall_under}% ({uc}/{len(df_cur)}), Push {pall_push}% ({pc}/{len(df_cur)})"
            )

            # --- Vs Avversario: stagione corrente, stagione precedente, carriera ---
            st.subheader("🆚 Storico vs avversario")
            team_abbrs = sorted({t["abbreviation"] for t in cached_teams()})
            opp = st.selectbox("Seleziona squadra avversaria", ["—"] + team_abbrs)

            if opp != "—":
                # Corrente
                df_vs_cur = df_cur[df_cur["MATCHUP"].str.contains(opp, na=False)]
                pov_cur, ovc_cur, totc_cur = percent_over(df_vs_cur[col], line)

                # Precedente
                prev = prev_season(CURRENT_SEASON)
                try:
                    df_prev = get_player_gamelog(pid, season=prev)
                    df_prev = filter_game_type(df_prev, game_type)
                except Exception:
                    df_prev = pd.DataFrame()
                df_vs_prev = df_prev[df_prev["MATCHUP"].str.contains(opp, na=False)]
                pov_prev, ovc_prev, totc_prev = percent_over(df_vs_prev[col], line)

                # Carriera
                df_hist = get_player_full_history(pid)
                df_hist = filter_game_type(df_hist, game_type)
                df_vs_all = df_hist[df_hist["MATCHUP"].str.contains(opp, na=False)]
                pov_all, ovc_all, totc_all = percent_over(df_vs_all[col], line)

                st.write(f"**Stagione corrente vs {opp}**: {pov_cur}% over ({ovc_cur}/{totc_cur})")
                st.write(f"**Stagione precedente vs {opp}**: {pov_prev}% over ({ovc_prev}/{totc_prev})")
                st.write(f"**Carriera vs {opp}**: {pov_all}% over ({ovc_all}/{totc_all})")

# ==================== TAB: BET365 EXTRACTOR ====================
with tab_bet365:
    st.subheader("🧩 Estrazione Bet365 (HTML → Excel/CSV)")
    st.caption("Incolla o carica l’HTML Bet365. Estrae **Giocatore, Linea, Quota**. Supporto per Over/Under e layout a colonne (0, 5, 10, ...).")

    market_opt = st.selectbox("Mercato da estrarre", ["Più di", "Meno di", "Entrambi"], index=0)
    market_val = {"Più di": "over", "Meno di": "under", "Entrambi": "both"}[market_opt]
    deduplicate = st.checkbox("Rimuovi duplicati (per Fixture+Player+Line+Odds)", value=(market_val != "both"))

    tab_file, tab_paste = st.tabs(["📁 Carica file HTML/TXT", "📋 Incolla HTML"])
    html_content = ""

    with tab_file:
        up = st.file_uploader("Carica un file .html / .txt esportato da Bet365", type=["html", "htm", "txt"])
        if up is not None:
            html_content = up.read().decode("utf-8", errors="ignore")

    with tab_paste:
        txt = st.text_area("Incolla qui il codice HTML", height=260, placeholder="<!doctype html> ...")
        if txt.strip():
            html_content = txt

    if st.button("🔎 Estrai dati", type="primary", use_container_width=True):
        if not html_content.strip():
            st.warning("Inserisci o carica l'HTML prima di procedere.")
            st.stop()

        with st.spinner("Estrazione in corso..."):
            df_ext = extract_bet365(html_content, market_filter=market_val)
            if df_ext.empty and market_val != "both":
                # se non trova nel filtro, prova fallback both (magari header localizzati)
                df_ext = extract_bet365(html_content, market_filter="both")

            if deduplicate and not df_ext.empty:
                df_ext = df_ext.drop_duplicates(subset=["Fixture", "Player", "Line", "Odds"]).reset_index(drop=True)

        if df_ext.empty:
            st.error("Nessun dato riconosciuto. Verifica di aver incollato l’HTML corretto.")
        else:
            st.success(f"✅ Righe estratte: {len(df_ext)}")
            st.dataframe(df_ext, use_container_width=True, hide_index=True)

            # Download
            csv_bytes = df_ext.to_csv(index=False, encoding="utf-8").encode("utf-8")
            bio_xlsx = io.BytesIO()
            with pd.ExcelWriter(bio_xlsx, engine="openpyxl") as writer:
                df_ext.to_excel(writer, index=False, sheet_name="estratto")
            bio_xlsx.seek(0)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Scarica CSV", data=csv_bytes, file_name="bet365_estratto.csv",
                                   mime="text/csv", use_container_width=True)
            with c2:
                st.download_button("⬇️ Scarica Excel", data=bio_xlsx.read(), file_name="bet365_estratto.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
