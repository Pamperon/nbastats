# app.py — NBA Stats (build stabile "classica")
# - Solo 2 tab: Batch + Giocatore singolo (niente Bet365)
# - Ultime 5/10 cross-stagione; Intera stagione = solo stagione corrente
# - Grafico compatto con valori e date sempre visibili
# - Linea .5 con step 1
# - Vs avversario: stagione corrente, precedente, carriera
# - Cache dati con show_spinner=False (niente "Running ...")
# - Nessun hardening HTTP, nessuna cache su disco

import io
import math
import time
import datetime as dt
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog

# ---------------- UI ----------------
st.set_page_config(page_title="NBA Stats", layout="centered")
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
st.caption("Ricerca giocatore, percentuali Over/Under, grafico a barre, filtri casa/trasferta, storico vs avversario.")

# ---------------- Utils ----------------
def normalize_name(name: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode("utf-8").lower().strip()

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

def force_half(value: float, min_v: float = 0.0, max_v: float = 120.0) -> float:
    """Rende la linea sempre N + 0.5 e rispetta i limiti."""
    v = math.floor(float(value)) + 0.5
    v = max(min_v + 0.5, min(v, max_v - 0.5))
    return v

# ---------------- Cache data ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_all_players() -> List[Dict]:
    return players.get_players()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_teams() -> List[Dict]:
    return teams.get_teams()

def find_player_id_by_name(player_name: str) -> Optional[int]:
    custom_map = {"pj washington": "p.j. washington", "ron holland ii": "ronald holland ii"}
    norm_in = normalize_name(player_name)
    candidate = custom_map.get(norm_in, player_name)

    all_players = cached_all_players()
    norm_candidate = normalize_name(candidate)

    # match esatto
    for p in all_players:
        if normalize_name(p["full_name"]) == norm_candidate:
            return p["id"]
    # match parziale
    for p in all_players:
        if norm_candidate in normalize_name(p["full_name"]):
            return p["id"]
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_player_gamelog(player_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    df = playergamelog.PlayerGameLog(
        player_id=player_id, season=season, season_type_all_star="Regular Season"
    ).get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for c in ("PTS", "AST", "REB"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["PAR"] = df["PTS"] + df["AST"] + df["REB"]
    return df.sort_values("GAME_DATE", ascending=False)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_player_full_history(player_id: int, start_year: int = 2000) -> pd.DataFrame:
    frames = []
    for year in range(start_year, dt.date.today().year + 1):
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
            time.sleep(0.2)  # respira un attimo tra le stagioni
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("GAME_DATE", ascending=False)

# ---------------- Stat helpers ----------------
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

# ---------------- Plot ----------------
def plot_bar(df: pd.DataFrame, col: str, line: float, title: str, rotate: int = 45):
    if df.empty:
        st.warning("⚠️ Nessun dato disponibile per il grafico.")
        return

    dd = df.sort_values("GAME_DATE")
    labels = dd["GAME_DATE"].dt.strftime("%m/%d")
    values = dd[col].astype(float)

    fig, ax = plt.subplots(figsize=(8, 3))  # compatto
    colors = ["#10B981" if v > line else "#EF4444" for v in values]
    bars = ax.bar(range(len(values)), values, width=0.6, color=colors)

    # Etichette sopra le barre
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{values.iloc[i]:.0f}",
            ha="center", va="bottom", fontsize=8, color="#e5e7eb"
        )

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

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=8)

    ax.legend(facecolor="#121821", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=9)
    st.pyplot(fig)

# ---------------- UI: TABS ----------------
tab_batch, tab_single = st.tabs([
    "📥 Analisi batch da Excel",
    "🔍 Ricerca giocatore singolo"
])

# ========== TAB: BATCH ==========
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

        # cache locale per il run
        local_cache: Dict[Tuple[int, str], pd.DataFrame] = {}
        def get_gl(pid: int, season: str) -> pd.DataFrame:
            key = (pid, season)
            if key not in local_cache:
                local_cache[key] = get_player_gamelog(pid, season=season)
            return local_cache[key]

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
                results.append({
                    "Giocatore": player_name, "Squadra": "N/D", "Linea": line,
                    "% Over 5G": "N/D", "% Over 10G": "N/D",
                    "% Over Stagione": "N/D", "% Under Stagione": "N/D", "% Push Stagione": "N/D",
                })
                progress.progress((i + 1) / len(df_in))
                continue

            try:
                glog = get_gl(pid, CURRENT_SEASON)  # stagione corrente
            except Exception:
                results.append({
                    "Giocatore": player_name, "Squadra": "N/D", "Linea": line,
                    "% Over 5G": "ERR", "% Over 10G": "ERR",
                    "% Over Stagione": "ERR", "% Under Stagione": "ERR", "% Push Stagione": "ERR",
                })
                progress.progress((i + 1) / len(df_in))
                continue

            # squadra dal gamelog
            team = "N/D"
            if "TEAM_ABBREVIATION" in glog.columns and not glog.empty:
                team = str(glog["TEAM_ABBREVIATION"].iloc[0])

            col = metric_map[metric_choice]

            # Ultime 5/10 cross-stagione (Totale)
            prev_gl = get_gl(pid, prev_season(CURRENT_SEASON))
            last5 = pd.concat(
                [filter_game_type(glog, "Totale"), filter_game_type(prev_gl, "Totale")],
                ignore_index=True
            ).sort_values("GAME_DATE", ascending=False).head(5)
            last10 = pd.concat(
                [filter_game_type(glog, "Totale"), filter_game_type(prev_gl, "Totale")],
                ignore_index=True
            ).sort_values("GAME_DATE", ascending=False).head(10)

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
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            df_out.to_excel(w, index=False, sheet_name="risultati")
        bio.seek(0)
        st.download_button("⬇️ Scarica risultati (Excel)", data=bio.read(),
                           file_name="risultati_over_under.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

# ========== TAB: SINGOLO ==========
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

            # Linea .5 con step 1
            defaults = {"PTS": 20.5, "AST": 5.5, "REB": 6.5, "PAR": 30.5}
            line_raw = st.number_input(
                f"🎯 Inserisci la linea {m.lower()}",
                min_value=0.0, max_value=120.0, value=defaults[col], step=1.0, format="%.1f",
                help="Si muove di 1 alla volta ed è sempre .5 (es. 20.5 → 21.5 → 22.5)."
            )
            line = force_half(line_raw)

            try:
                df_cur = get_player_gamelog(pid)  # stagione corrente
            except Exception as e:
                st.error(f"Errore nel recupero dati: {e}")
                st.stop()

            df_cur = filter_game_type(df_cur, game_type)

            # Grafico
            st.subheader("📈 Grafico")
            chart_range = st.selectbox("Intervallo", ["Ultime 5", "Ultime 10", "Intera stagione"])
            if chart_range == "Ultime 5":
                dplot = get_last_n_games_cross_seasons(pid, 5, game_type)
                plot_bar(dplot, col, line, f"{sel['full_name']} | Ultime 5 — {m}", rotate=45)
            elif chart_range == "Ultime 10":
                dplot = get_last_n_games_cross_seasons(pid, 10, game_type)
                plot_bar(dplot, col, line, f"{sel['full_name']} | Ultime 10 — {m}", rotate=45)
            else:
                plot_bar(df_cur, col, line, f"{sel['full_name']} | Intera stagione — {m}", rotate=45)

            # Statistiche
            st.subheader(f"📊 Statistiche {m.lower()}")
            last5 = get_last_n_games_cross_seasons(pid, 5, game_type)
            last10 = get_last_n_games_cross_seasons(pid, 10, game_type)

            p5, o5, t5 = percent_over(last5[col], line)
            p10, o10, t10 = percent_over(last10[col], line)
            pall_over, pall_under, pall_push, oc, uc, pc = calculate_over_under_push(df_cur[col], line)

            st.write(f"**Ultime 5 (cross-stagione)**: {p5}% over ({o5}/{t5})")
            st.write(f"**Ultime 10 (cross-stagione)**: {p10}% over ({o10}/{t10})")
            st.write(
                f"**Intera stagione (corrente)**: Over {pall_over}% ({oc}/{len(df_cur)}), "
                f"Under {pall_under}% ({uc}/{len(df_cur)}), Push {pall_push}% ({pc}/{len(df_cur)})"
            )

            # Vs avversario
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

                # Carriera (tutte le stagioni)
                df_hist = get_player_full_history(pid)
                df_hist = filter_game_type(df_hist, game_type)
                df_vs_all = df_hist[df_hist["MATCHUP"].str.contains(opp, na=False)]
                pov_all, ovc_all, totc_all = percent_over(df_vs_all[col], line)

                st.write(f"**Stagione corrente vs {opp}**: {pov_cur}% over ({ovc_cur}/{totc_cur})")
                st.write(f"**Stagione precedente vs {opp}**: {pov_prev}% over ({ovc_prev}/{totc_prev})")
                st.write(f"**Carriera vs {opp}**: {pov_all}% over ({ovc_all}/{totc_all})")
