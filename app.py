# app.py — NBA Stats (revamped)
# -----------------------------------------------------------
# Migliorie principali:
# - Cache aggressiva con @st.cache_data (TTL configurabile)
# - Retry semplice integrato (senza dipendenze esterne)
# - UI a tab: "Analisi batch da Excel" / "Ricerca giocatore singolo"
# - Stagione NBA calcolata dinamicamente
# - Grafico potenziato (media mobile 5 gare, stile dark)
# - Statistiche Over / Under / Push
# - Filtri Casa / Ospite anche per storico vs avversario
# - Codice più modulare e robusto
# -----------------------------------------------------------

import datetime as dt
import time
import unicodedata
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster

# -------------------- CONFIG UI --------------------
st.set_page_config(page_title="NBA Stats", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem;}
    label[data-testid="stWidgetLabel"] > div:empty {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 NBA Stats — Props-style Analyzer")
st.caption("Ricerca giocatore, percentuali Over/Under, grafico a barre, filtri casa/trasferta, storico vs avversario.")

# -------------------- UTILITIES --------------------
def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8").lower().strip()

def season_string_for_today(today: Optional[dt.date] = None) -> str:
    """Calcola automaticamente la stagione NBA corrente (formato 'YYYY-YY').
    Regola empirica: stagione inizia ~ottobre. Se mese >=10, stagione es. 2025-26, altrimenti 2024-25."""
    d = today or dt.date.today()
    if d.month >= 10:
        start = d.year
    else:
        start = d.year - 1
    end = (start + 1) % 100
    return f"{start}-{end:02d}"

CURRENT_SEASON = season_string_for_today()

def with_retry(fn, *args, attempts: int = 3, wait_secs: float = 0.8, **kwargs):
    """Retry semplice per gestire momentanei errori delle API NBA."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(wait_secs)
    # rilancia l'ultima eccezione se fallisce comunque
    raise last_exc

# -------------------- DATA LAYERS (CACHED) --------------------
@st.cache_data(ttl=3600)
def cached_active_players() -> List[Dict]:
    return players.get_players()  # include attivi + inattivi

@st.cache_data(ttl=3600)
def cached_teams() -> List[Dict]:
    return teams.get_teams()

def find_player_id_by_name(player_name: str) -> Optional[int]:
    # Correzioni manuali utili
    custom_map = {
        "pj washington": "p.j. washington",
        "ron holland ii": "ronald holland ii",
    }
    norm_in = normalize_name(player_name)
    candidate = custom_map.get(norm_in, player_name)

    all_players = cached_active_players()
    norm_candidate = normalize_name(candidate)

    # match esatto
    for p in all_players:
        if normalize_name(p["full_name"]) == norm_candidate:
            return p["id"]
    # match parziale (contiene)
    for p in all_players:
        if norm_candidate in normalize_name(p["full_name"]):
            return p["id"]
    return None

@st.cache_data(ttl=1800)
def get_player_gamelog(player_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Gamelog Regular Season per stagione specifica."""
    def _call():
        return playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="Regular Season"
        ).get_data_frames()[0]

    df = with_retry(_call)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    # ensure numeric
    for c in ("PTS", "AST", "REB"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["PAR"] = df["PTS"] + df["AST"] + df["REB"]
    return df.sort_values("GAME_DATE", ascending=False)

@st.cache_data(ttl=6 * 3600)
def get_player_full_history(player_id: int, start_year: int = 2000) -> pd.DataFrame:
    """Storico multistagione consolidato (Regular Season)."""
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
            # mini pausa per cortesia API
            time.sleep(0.2)
        except Exception:
            # salta stagioni non disponibili
            continue

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("GAME_DATE", ascending=False)

@st.cache_data(ttl=6 * 3600)
def teams_roster_map(season: str = CURRENT_SEASON) -> Dict[int, str]:
    """Mappa PLAYER_ID -> TEAM_ABBR per la stagione indicata."""
    mapping: Dict[int, str] = {}
    for t in cached_teams():
        try:
            def _call():
                return commonteamroster.CommonTeamRoster(team_id=t["id"], season=season).get_data_frames()[0]
            roster = with_retry(_call)
            for _, row in roster.iterrows():
                mapping[int(row["PLAYER_ID"])] = t["abbreviation"]
            time.sleep(0.2)
        except Exception:
            continue
    return mapping

# -------------------- STATS HELPERS --------------------
def calculate_over_under_push(series: pd.Series, line: float) -> Tuple[float, float, float, int, int, int]:
    """Restituisce (over%, under%, push%, over_count, under_count, push_count)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 0.0, 0.0, 0, 0, 0
    over_c = int((s > line).sum())
    under_c = int((s < line).sum())
    push_c = int((s == line).sum())
    total = len(s)
    return (
        round(100 * over_c / total, 1),
        round(100 * under_c / total, 1),
        round(100 * push_c / total, 1),
        over_c,
        under_c,
        push_c,
    )

def percent_over(series: pd.Series, line: float) -> Tuple[float, int, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    total = len(s)
    if total == 0:
        return 0.0, 0, 0
    over = int((s > line).sum())
    return round(100 * over / total, 1), over, total

# -------------------- PLOTTING --------------------
def plot_bar_line(df: pd.DataFrame, col: str, line: float, title: str,
                  show_vals: bool = False, rolling_n: int = 5,
                  rotate: int = 45, compact: bool = False):
    if df.empty:
        st.warning("⚠️ Nessun dato disponibile per il grafico.")
        return

    dd = df.sort_values("GAME_DATE")
    labels = dd["GAME_DATE"].dt.strftime("%m/%d")
    values = dd[col].astype(float)

    fig, ax = plt.subplots(figsize=(12, 4))
    # colori: verde sopra linea, rosso sotto/uguale
    colors = ["#10B981" if v > line else "#EF4444" for v in values]
    bars = ax.bar(range(len(values)), values, width=0.6, color=colors)

    if show_vals and not compact:
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{values.iloc[i]:.0f}", ha="center", va="bottom", fontsize=9, color="#e5e7eb")

    # linea soglia
    ax.axhline(line, color="#9CA3AF", linestyle="--", linewidth=1.5, label=f"Linea {line:g}")

    # media mobile
    if rolling_n and rolling_n > 1 and len(values) >= rolling_n:
        roll = values.rolling(rolling_n).mean()
        ax.plot(range(len(values)), roll, linewidth=2, label=f"Media mobile ({rolling_n})", color="#F59E0B")

    # stile dark
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#121821")
    ax.tick_params(colors="#e5e7eb")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.title.set_color("#e5e7eb")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(col)
    ax.set_xlabel("Data" if not compact else "")

    if not compact:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=9)
    else:
        ax.set_xticks([])

    ax.legend(facecolor="#121821", edgecolor="#374151", labelcolor="#e5e7eb")
    st.pyplot(fig)

# -------------------- UI: TABS --------------------
tab_batch, tab_single = st.tabs(["📥 Analisi batch da Excel", "🔍 Ricerca giocatore singolo"])

# ==================== TAB: BATCH ====================
with tab_batch:
    st.subheader("📥 Carica file Excel per analisi batch")
    st.caption("Il file deve contenere le colonne **Giocatore** e **Linea**. La stagione è calcolata automaticamente: "
               f"**{CURRENT_SEASON}** (Regular Season).")

    metric_choice = st.radio(
        "📊 Scegli la metrica da analizzare",
        ["Punti", "Assist", "Rimbalzi", "P+A+R"],
        horizontal=True,
        key="batch_metric",
    )
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
        roster_map = teams_roster_map()

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
                glog = get_player_gamelog(pid)
            except Exception as e:
                results.append({
                    "Giocatore": player_name, "Squadra": roster_map.get(pid, "N/D"), "Linea": line,
                    "% Over 5G": "ERR", "% Over 10G": "ERR",
                    "% Over Stagione": "ERR", "% Under Stagione": "ERR", "% Push Stagione": "ERR",
                })
                progress.progress((i + 1) / len(df_in))
                continue

            col = metric_map[metric_choice]
            vals = glog[col].astype(float)

            # percentuali
            p5, _, _ = percent_over(vals.head(5), line)
            p10, _, _ = percent_over(vals.head(10), line)
            over_all, under_all, push_all, oc, uc, pc = calculate_over_under_push(vals, line)

            results.append({
                "Giocatore": player_name,
                "Squadra": roster_map.get(pid, "N/D"),
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

        # Download
        import io
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
        # propone solo attivi (UX migliore), ma fallback su tutti in find_player_id_by_name
        active = players.get_active_players()
        matches = [p for p in active if norm_q in normalize_name(p["full_name"])]
        if not matches:
            st.error("❌ Nessun giocatore trovato tra gli attivi. Prova il nome completo.")
        else:
            sel = st.selectbox("Seleziona il giocatore", matches, format_func=lambda p: p["full_name"])
            pid = sel["id"]

            m = st.radio("📌 Scegli la metrica", ["Punti", "Assist", "Rimbalzi", "P+A+R"], horizontal=True, key="single_metric")
            game_type = st.radio("🎯 Tipo di partita", ["Totale", "Casa", "Ospite"], horizontal=True, key="single_gtype")

            col_map = {"Punti": "PTS", "Assist": "AST", "Rimbalzi": "REB", "P+A+R": "PAR"}
            col = col_map[m]

            # scarica gamelog stagione corrente
            try:
                df = get_player_gamelog(pid)
            except Exception as e:
                st.error(f"Errore nel recupero dati: {e}")
                st.stop()

            if game_type == "Casa":
                df = df[df["MATCHUP"].str.contains("vs", na=False)]
            elif game_type == "Ospite":
                df = df[df["MATCHUP"].str.contains("@", na=False)]

            # linea default sensata per metrica
            defaults = {"PTS": 20.5, "AST": 5.5, "REB": 6.5, "PAR": 30.5}
            default_line = defaults[col]
            line = st.number_input(f"🎯 Inserisci la linea {m.lower()}",
                                   min_value=0.0, max_value=120.0, value=default_line, step=0.5)
            # evita interi .0: se l’utente mette 20.0 suggeriamo 20.5 (UX props)
            if line % 1 == 0:
                line += 0.5

            # --- Grafici ---
            st.subheader("📈 Grafico")
            c1, c2, c3 = st.columns(3)
            with c1:
                chart_range = st.selectbox("Intervallo", ["Ultime 5", "Ultime 10", "Intera stagione"])
            with c2:
                show_vals = st.checkbox("Mostra valori sulle barre", value=False)
            with c3:
                rolling_n = st.number_input("Media mobile (N)", min_value=0, max_value=20, value=5, step=1)

            if chart_range == "Ultime 5":
                dplot = df.head(5)
                title = f"Ultime 5 — {m}"
            elif chart_range == "Ultime 10":
                dplot = df.head(10)
                title = f"Ultime 10 — {m}"
            else:
                dplot = df
                title = f"Intera stagione — {m}"

            plot_bar_line(dplot, col, line, f"{sel['full_name']} | {title}",
                          show_vals=show_vals, rolling_n=rolling_n,
                          rotate=45 if chart_range != "Intera stagione" else 0,
                          compact=(chart_range == "Intera stagione"))

            # --- Statistiche ---
            st.subheader(f"📊 Statistiche {m.lower()}")
            p5, o5, t5 = percent_over(df[col].head(5), line)
            p10, o10, t10 = percent_over(df[col].head(10), line)
            pall_over, pall_under, pall_push, oc, uc, pc = calculate_over_under_push(df[col], line)

            st.write(f"**Ultime 5**: {p5}% over ({o5}/{t5})")
            st.write(f"**Ultime 10**: {p10}% over ({o10}/{t10})")
            st.write(
                f"**Intera stagione**: Over {pall_over}% ({oc}/{len(df)}), "
                f"Under {pall_under}% ({uc}/{len(df)}), Push {pall_push}% ({pc}/{len(df)})"
            )

            # --- Vs Avversario ---
            st.subheader("🆚 Storico vs avversario")
            team_abbrs = sorted({t["abbreviation"] for t in cached_teams()})
            opp = st.selectbox("Seleziona squadra avversaria", ["—"] + team_abbrs)

            if opp != "—":
                hist = get_player_full_history(pid)
                # ricalcola PAR se serve
                # (già presente ma in caso)
                hist["PAR"] = hist["PTS"] + hist["AST"] + hist["REB"]

                if game_type == "Casa":
                    hist = hist[hist["MATCHUP"].str.contains("vs", na=False)]
                elif game_type == "Ospite":
                    hist = hist[hist["MATCHUP"].str.contains("@", na=False)]

                df_vs = hist[hist["MATCHUP"].str.contains(opp, na=False)]
                pov, ovc, totc = percent_over(df_vs[col], line)
                st.write(f"**Carriera vs {opp}**: {pov}% over ({ovc}/{totc})")
