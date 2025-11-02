# app.py — Safe-boot (nessun import pesante finché non clicchi)
# Mantiene: grafici compatti con valori/date; 5/10 cross-stagione; intera stagione solo corrente;
# vs avversario (corrente/precedente/carriera); estrazione Bet365.
# Tutto lazy: niente import di nba_api/bs4 fino al click.

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
import importlib.util as ilu

st.set_page_config(page_title="NBA Stats + Bet365", layout="centered")
st.markdown(
    """
    <style>
    .block-container { padding-top: 0.75rem; max-width: 900px; margin: auto; }
    label[data-testid="stWidgetLabel"] > div:empty {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 NBA Stats — Props-style Analyzer")
st.caption("Ricerca giocatore, percentuali Over/Under, grafico a barre, filtri casa/trasferta, storico vs avversario. + 🧩 Estrazione Bet365 HTML.")

# --------- Utils base (non fanno I/O) ---------
def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8").lower().strip()

def season_string_for_today(today: Optional[dt.date] = None) -> str:
    d = today or dt.date.today()
    start = d.year if d.month >= 10 else d.year - 1
    end = (start + 1) % 100
    return f"{start}-{end:02d}"

def prev_season(season: str) -> str:
    y1 = int(season[:4]) - 1
    y2 = (y1 + 1) % 100
    return f"{y1}-{y2:02d}"

CURRENT_SEASON = season_string_for_today()

def with_retry(fn, *args, attempts: int = 3, wait_secs: float = 0.7, **kwargs):
    last_exc = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(wait_secs * (2 ** i))
    raise last_exc

def force_half(value: float, min_v: float = 0.0, max_v: float = 120.0) -> float:
    v = math.floor(float(value)) + 0.5
    v = max(min_v + 0.5, min(v, max_v - 0.5))
    return v

def module_exists(modname: str) -> bool:
    return ilu.find_spec(modname) is not None

# --------- PLOT ---------
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

    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{values.iloc[i]:.0f}",
                ha="center", va="bottom", fontsize=8, color="#e5e7eb")

    ax.axhline(line, color="#9CA3AF", linestyle="--", linewidth=1.2, label=f"Linea {line:g}")

    fig.patch.set_facecolor("#0b0f14"); ax.set_facecolor("#121821")
    ax.tick_params(colors="#e5e7eb", labelsize=9); ax.title.set_color("#e5e7eb")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(col, fontsize=10); ax.set_xlabel("")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=8)
    ax.legend(facecolor="#121821", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=9)
    st.pyplot(fig)

# --------- STAT helpers ---------
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
    over_c = int((s > line).sum()); under_c = int((s < line).sum()); push_c = int((s == line).sum())
    total = len(s)
    return round(100*over_c/total,1), round(100*under_c/total,1), round(100*push_c/total,1), over_c, under_c, push_c

def filter_game_type(df: pd.DataFrame, game_type: str) -> pd.DataFrame:
    if game_type == "Casa":
        return df[df["MATCHUP"].str.contains("vs", na=False)]
    if game_type == "Ospite":
        return df[df["MATCHUP"].str.contains("@", na=False)]
    return df

# --------- Cache (senza spinner) — tutte lazy nei body ---------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_all_players():
    import importlib
    players_mod = importlib.import_module("nba_api.stats.static.players")
    return players_mod.get_players()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_teams():
    import importlib
    teams_mod = importlib.import_module("nba_api.stats.static.teams")
    return teams_mod.get_teams()

@st.cache_data(ttl=1800, show_spinner=False)
def get_player_gamelog(player_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    import importlib
    PlayerGameLog = importlib.import_module("nba_api.stats.endpoints.playergamelog").PlayerGameLog
    def _call():
        return PlayerGameLog(player_id=player_id, season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    df = with_retry(_call)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for c in ("PTS", "AST", "REB"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["PAR"] = df["PTS"] + df["AST"] + df["REB"]
    return df.sort_values("GAME_DATE", ascending=False)

@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_player_full_history(player_id: int, start_year: int = 2000) -> pd.DataFrame:
    import importlib
    PlayerGameLog = importlib.import_module("nba_api.stats.endpoints.playergamelog").PlayerGameLog
    frames = []
    for year in range(start_year, dt.date.today().year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        try:
            def _call():
                return PlayerGameLog(player_id=player_id, season=season, season_type_all_star="Regular Season").get_data_frames()[0]
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
    return pd.concat(frames, ignore_index=True).sort_values("GAME_DATE", ascending=False)

@st.cache_data(ttl=900, show_spinner=False)
def get_dual_gamelogs(player_id: int, game_type: str):
    cur = get_player_gamelog(player_id, season=CURRENT_SEASON)
    prev = get_player_gamelog(player_id, season=prev_season(CURRENT_SEASON))
    cur = filter_game_type(cur, game_type)
    prev = filter_game_type(prev, game_type)
    return cur, prev

# --------- Bet365 helpers (lazy bs4) ---------
def _make_soup(html: str):
    from bs4 import BeautifulSoup  # import solo qui
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

def _norm_text(s: str) -> str:
    if s is None: return ""
    return " ".join(s.split()).strip()

def _contains_over(label: str) -> bool:
    lab_noaccent = unicodedata.normalize("NFKD", label.lower()).encode("ascii","ignore").decode("ascii")
    return ("piu di" in lab_noaccent) or ("over" in lab_noaccent)

def _contains_under(label: str) -> bool:
    lab_noaccent = unicodedata.normalize("NFKD", label.lower()).encode("ascii","ignore").decode("ascii")
    return ("meno di" in lab_noaccent) or ("under" in lab_noaccent)

def _to_float_odds(x: str):
    if x is None: return None
    x = x.strip().replace(",", ".")
    try:
        return float(x)
    except:
        m = re.search(r"\d+[.,]\d+", x)
        if m:
            try: return float(m.group(0).replace(",", "."))
            except: return None
        return None

def parse_over_under_layout(soup, market_filter: str):
    rows = []
    pods = soup.select(".gl-MarketGroupPod.src-FixtureSubGroup")
    if not pods: return rows
    for pod in pods:
        fix_el = pod.select_one(".src-FixtureSubGroupButton_Text")
        fixture = _norm_text(fix_el.get_text()) if fix_el else ""
        players_list = [_norm_text(e.get_text()) for e in pod.select(".srb-ParticipantLabelWithTeam_Name")]
        for market in pod.select(".gl-Market.gl-Market_General-columnheader"):
            header_el = market.select_one(".gl-MarketColumnHeader")
            market_name = _norm_text(header_el.get_text() if header_el else "")
            if market_filter == "over" and not _contains_over(market_name): continue
            if market_filter == "under" and not _contains_under(market_name): continue
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
                rows.append({"Fixture": fixture, "Player": players_list[i], "Market": market_name, "Line": line, "Odds": odds})
    return rows

def parse_columns_layout(soup):
    rows = []
    fixture_el = soup.select_one(".src-FixtureSubGroupButton_Text")
    fixture = _norm_text(fixture_el.get_text()) if fixture_el else ""
    players_list = [_norm_text(e.get_text()) for e in soup.select(".srb-ParticipantLabelWithTeam_Name")]
    if not players_list: return rows
    columns = soup.select(".srb-HScrollPlaceColumnMarket")
    if not columns: return rows
    for col in columns:
        header_el = col.select_one(".srb-HScrollPlaceHeader")
        header = _norm_text(header_el.get_text()) if header_el else ""
        odds_spans = col.select(".gl-ParticipantOddsOnly_Odds")
        odds = [_to_float_odds(_norm_text(sp.get_text())) for sp in odds_spans]
        n = min(len(players_list), len(odds))
        for i in range(n):
            rows.append({"Fixture": fixture, "Player": players_list[i], "Market": header, "Line": header, "Odds": odds[i]})
    return rows

def extract_bet365(html: str, market_filter: str = "over") -> pd.DataFrame:
    soup = _make_soup(html)
    rows = parse_over_under_layout(soup, market_filter=market_filter)
    if not rows:
        rows = parse_columns_layout(soup)
    return pd.DataFrame(rows, columns=["Fixture","Player","Market","Line","Odds"])

# --------- TABS ---------
tab_batch, tab_single, tab_bet365 = st.tabs([
    "📥 Analisi batch da Excel",
    "🔍 Ricerca giocatore singolo",
    "🧩 Estrazione Bet365 (HTML → Excel)"
])

# ================= TAB: BATCH =================
with tab_batch:
    st.subheader("📥 Carica file Excel per analisi batch")
    st.caption(f"Il file deve contenere le colonne **Giocatore** e **Linea**. Stagione corrente: **{CURRENT_SEASON}** (Regular Season).")

    if not module_exists("nba_api"):
        st.error("Manca il pacchetto **nba_api**. Aggiungilo al requirements.txt e ridistribuisci.")
        st.code("nba_api==1.5.2\nrequests==2.32.3\nurllib3==2.2.3", language="text")
    else:
        metric_choice = st.radio("📊 Scegli la metrica", ["Punti","Assist","Rimbalzi","P+A+R"], horizontal=True, key="batch_metric")
        metric_map = {"Punti":"PTS","Assist":"AST","Rimbalzi":"REB","P+A+R":"PAR"}
        uploaded = st.file_uploader("📁 Carica Excel (.xlsx)", type=["xlsx"], key="batch_upload")

        if uploaded:
            try:
                df_in = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"❌ Impossibile leggere il file: {e}")
                st.stop()

            if not {"Giocatore","Linea"}.issubset(df_in.columns):
                st.error("❌ Il file deve contenere le colonne 'Giocatore' e 'Linea'.")
                st.stop()

            if st.button("🔎 Analizza file", type="primary", use_container_width=True):
                # Cache locale per questo run
                local_cache: Dict[Tuple[int, str], pd.DataFrame] = {}
                def get_gl(pid: int, season: str) -> pd.DataFrame:
                    key = (pid, season)
                    if key not in local_cache:
                        local_cache[key] = get_player_gamelog(pid, season=season)
                    return local_cache[key]

                import importlib
                results = []
                progress = st.progress(0.0)

                # trova ID by name (usa cached_all_players)
                def find_player_id_by_name(player_name: str) -> Optional[int]:
                    all_players = cached_all_players()
                    norm_candidate = normalize_name(player_name)
                    # match esatto
                    for p in all_players:
                        if normalize_name(p.get("full_name","")) == norm_candidate:
                            return p["id"]
                    # match parziale
                    for p in all_players:
                        if norm_candidate in normalize_name(p.get("full_name","")):
                            return p["id"]
                    return None

                for i, row in df_in.iterrows():
                    player_name = str(row["Giocatore"])
                    try:
                        line = float(row["Linea"])
                    except Exception:
                        try:
                            line = float(str(row["Linea"]).replace(",", "."))
                        except Exception:
                            line = 0.0

                    pid = find_player_id_by_name(player_name)

                    if pid is None:
                        results.append({"Giocatore": player_name, "Squadra": "N/D", "Linea": line,
                                        "% Over 5G": "N/D", "% Over 10G": "N/D",
                                        "% Over Stagione": "N/D", "% Under Stagione": "N/D", "% Push Stagione": "N/D"})
                        progress.progress((i+1)/len(df_in)); continue

                    try:
                        glog = get_gl(pid, CURRENT_SEASON)  # stagione corrente
                    except Exception:
                        results.append({"Giocatore": player_name, "Squadra": "N/D", "Linea": line,
                                        "% Over 5G": "ERR", "% Over 10G": "ERR",
                                        "% Over Stagione": "ERR", "% Under Stagione": "ERR", "% Push Stagione": "ERR"})
                        progress.progress((i+1)/len(df_in)); continue

                    team = "N/D"
                    if "TEAM_ABBREVIATION" in glog.columns and not glog.empty:
                        team = str(glog["TEAM_ABBREVIATION"].iloc[0])

                    col = metric_map[metric_choice]
                    prev_gl = get_gl(pid, prev_season(CURRENT_SEASON))
                    last5  = pd.concat([filter_game_type(glog, "Totale"),
                                        filter_game_type(prev_gl, "Totale")], ignore_index=True)\
                                .sort_values("GAME_DATE", ascending=False).head(5)
                    last10 = pd.concat([filter_game_type(glog, "Totale"),
                                        filter_game_type(prev_gl, "Totale")], ignore_index=True)\
                                .sort_values("GAME_DATE", ascending=False).head(10)

                    p5, _, _ = percent_over(last5[col], line)
                    p10, _, _ = percent_over(last10[col], line)
                    over_all, under_all, push_all, oc, uc, pc = calculate_over_under_push(glog[col], line)

                    results.append({"Giocatore": player_name, "Squadra": team, "Linea": line,
                                    "% Over 5G": f"{p5}%", "% Over 10G": f"{p10}%",
                                    "% Over Stagione": f"{over_all}%", "% Under Stagione": f"{under_all}%",
                                    "% Push Stagione": f"{push_all}%"})
                    progress.progress((i+1)/len(df_in))

                df_out = pd.DataFrame(results)
                st.dataframe(df_out, use_container_width=True, hide_index=True)

                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as w:
                    df_out.to_excel(w, index=False, sheet_name="risultati")
                bio.seek(0)
                st.download_button("⬇️ Scarica risultati (Excel)", data=bio.read(),
                                   file_name="risultati_over_under.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

# ================= TAB: SINGOLO =================
with tab_single:
    st.subheader("🔍 Ricerca giocatore singolo")

    if not module_exists("nba_api"):
        st.error("Manca il pacchetto **nba_api**. Aggiungilo al requirements.txt e ridistribuisci.")
    else:
        q = st.text_input("Inserisci il nome del giocatore (es. LeBron James)")
        if q.strip():
            # defer import finché serve
            if st.button("🔁 Calcola / Aggiorna", type="primary", use_container_width=True):
                try:
                    active = cached_all_players()
                except Exception as e:
                    st.error(f"Errore nel caricamento giocatori: {e}")
                    st.stop()

                norm_q = normalize_name(q)
                matches = [p for p in active if norm_q in normalize_name(p.get("full_name",""))]
                if not matches:
                    st.error("❌ Nessun giocatore trovato tra gli attivi. Prova il nome completo.")
                    st.stop()

                sel = st.selectbox("Seleziona il giocatore", matches, format_func=lambda p: p["full_name"])
                pid = sel["id"]

                m = st.radio("📌 Scegli la metrica", ["Punti","Assist","Rimbalzi","P+A+R"], horizontal=True)
                game_type = st.radio("🎯 Tipo di partita", ["Totale","Casa","Ospite"], horizontal=True)

                col_map = {"Punti":"PTS","Assist":"AST","Rimbalzi":"REB","P+A+R":"PAR"}; col = col_map[m]

                defaults = {"PTS":20.5,"AST":5.5,"REB":6.5,"PAR":30.5}
                line_raw = st.number_input(f"🎯 Inserisci la linea {m.lower()}",
                                           min_value=0.0, max_value=120.0, value=defaults[col],
                                           step=1.0, format="%.1f",
                                           help="Si muove di 1 alla volta ed è sempre .5 (20.5 → 21.5 → 22.5).")
                line = force_half(line_raw)

                try:
                    df_cur, df_prev = get_dual_gamelogs(pid, game_type)
                except Exception as e:
                    st.error(f"Errore nel recupero dati: {e}")
                    st.stop()

                st.subheader("📈 Grafico")
                chart_range = st.selectbox("Intervallo", ["Ultime 5","Ultime 10","Intera stagione"])
                if chart_range == "Ultime 5":
                    dplot = pd.concat([df_cur, df_prev], ignore_index=True).sort_values("GAME_DATE", ascending=False).head(5)
                    title = f"{sel['full_name']} | Ultime 5 — {m}"
                elif chart_range == "Ultime 10":
                    dplot = pd.concat([df_cur, df_prev], ignore_index=True).sort_values("GAME_DATE", ascending=False).head(10)
                    title = f"{sel['full_name']} | Ultime 10 — {m}"
                else:
                    dplot = df_cur
                    title = f"{sel['full_name']} | Intera stagione — {m}"

                plot_bar(dplot, col, line, title, rotate=45)

                st.subheader(f"📊 Statistiche {m.lower()}")
                last5  = pd.concat([df_cur, df_prev], ignore_index=True).sort_values("GAME_DATE", ascending=False).head(5)
                last10 = pd.concat([df_cur, df_prev], ignore_index=True).sort_values("GAME_DATE", ascending=False).head(10)
                p5,  o5,  t5  = percent_over(last5[col], line)
                p10, o10, t10 = percent_over(last10[col], line)
                pall_over, pall_under, pall_push, oc, uc, pc = calculate_over_under_push(df_cur[col], line)

                st.write(f"**Ultime 5 (cross-stagione)**: {p5}% over ({o5}/{t5})")
                st.write(f"**Ultime 10 (cross-stagione)**: {p10}% over ({o10}/{t10})")
                st.write(f"**Intera stagione (corrente)**: Over {pall_over}% ({oc}/{len(df_cur)}), Under {pall_under}% ({uc}/{len(df_cur)}), Push {pall_push}% ({pc}/{len(df_cur)})")

                st.subheader("🆚 Storico vs avversario")
                try:
                    team_abbrs = sorted([t["abbreviation"] for t in cached_teams()])
                except Exception:
                    team_abbrs = []
                opp = st.selectbox("Seleziona squadra avversaria", ["—"] + team_abbrs)
                if opp != "—":
                    df_vs_cur  = df_cur[df_cur["MATCHUP"].str.contains(opp, na=False)]
                    df_vs_prev = df_prev[df_prev["MATCHUP"].str.contains(opp, na=False)]
                    try:
                        df_hist = get_player_full_history(pid)
                        df_hist = filter_game_type(df_hist, game_type)
                        df_vs_all = df_hist[df_hist["MATCHUP"].str.contains(opp, na=False)]
                    except Exception:
                        df_vs_all = pd.DataFrame()

                    pov_cur,  ovc_cur,  totc_cur  = percent_over(df_vs_cur[col],  line)
                    pov_prev, ovc_prev, totc_prev = percent_over(df_vs_prev[col], line)
                    pov_all,  ovc_all,  totc_all  = percent_over(df_vs_all[col],  line)

                    st.write(f"**Stagione corrente vs {opp}**: {pov_cur}% over ({ovc_cur}/{totc_cur})")
                    st.write(f"**Stagione precedente vs {opp}**: {pov_prev}% over ({ovc_prev}/{totc_prev})")
                    st.write(f"**Carriera vs {opp}**: {pov_all}% over ({ovc_all}/{totc_all})")
        else:
            st.info("Inserisci un nome e premi **Calcola / Aggiorna**.")

# ================= TAB: BET365 =================
with tab_bet365:
    st.subheader("🧩 Estrazione Bet365 (HTML → Excel/CSV)")
    if not module_exists("beautifulsoup4"):
        st.error("Manca **beautifulsoup4**. Aggiungila al requirements.txt e ridistribuisci.")
        st.code("beautifulsoup4==4.12.3\nlxml==4.9.4", language="text")
    else:
        st.caption("Incolla o carica l’HTML Bet365. Estrae **Giocatore, Linea, Quota**. Supporta Over/Under e layout a colonne.")
        market_opt = st.selectbox("Mercato da estrarre", ["Più di","Meno di","Entrambi"], index=0)
        market_val = {"Più di":"over","Meno di":"under","Entrambi":"both"}[market_opt]
        deduplicate = st.checkbox("Rimuovi duplicati (Fixture+Player+Line+Odds)", value=(market_val != "both"))

        tab_file, tab_paste = st.tabs(["📁 Carica file HTML/TXT", "📋 Incolla HTML"])
        html_content = ""
        with tab_file:
            up = st.file_uploader("Carica un file .html / .txt esportato da Bet365", type=["html","htm","txt"])
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
            df_ext = extract_bet365(html_content, market_filter=market_val)
            if df_ext.empty and market_val != "both":
                df_ext = extract_bet365(html_content, market_filter="both")
            if deduplicate and not df_ext.empty:
                df_ext = df_ext.drop_duplicates(subset=["Fixture","Player","Line","Odds"]).reset_index(drop=True)

            if df_ext.empty:
                st.error("Nessun dato riconosciuto. Verifica l’HTML.")
            else:
                st.success(f"✅ Righe estratte: {len(df_ext)}")
                st.dataframe(df_ext, use_container_width=True, hide_index=True)
                csv_bytes = df_ext.to_csv(index=False).encode("utf-8")
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
