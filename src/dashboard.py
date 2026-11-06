"""Dashboard Deportivo Pro v3: Radar EN VIVO (Momentum), Monte Carlo 10k (Hándicaps), Fatiga/Descanso, Ensamble ML (Gradient Boosting), Jugadores, Árbitro y Kelly."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import date, datetime, timedelta, timezone
import concurrent.futures
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.config import MAX_PARALLEL_WORKERS
from src.analytics.auto_learner import get_auto_learner
from src.analytics.entity_resolver import EntityResolver
from src.analytics.predictor import (
    MatchPrediction,
    TeamFormMetrics,
    _poisson_prob_over,
    calculate_kelly_stake,
    compute_team_form,
    predict_live_in_play,
    predict_match_poisson,
)
from src.models.database import (
    get_bankroll_ledger_history,
    get_current_bankroll,
    get_db_session,
    get_or_create_initial_bankroll,
    get_portfolio_bets,
    get_portfolio_kpis,
    get_precomputed_matches_for_date,
    init_db,
    record_bankroll_transaction,
    save_portfolio_bets,
    save_team_alias,
    upsert_match,
)
from src.analytics.portfolio_allocator import (
    AllocatedBet,
    BetOpportunity,
    DutchingOpportunity,
    SurebetOpportunity,
    find_surebets,
    optimize_value_bets,
    solve_dutching,
)
from src.analytics.settlement_engine import (
    SettlementReport,
    audit_and_settle_open_bets,
    evaluate_bet,
)
from src.models.schemas import (
    LiveMomentumSummarySchema,
    MatchDetailSchema,
    PlayerPropSchema,
    TeamSchema,
    TournamentSchema,
)
from src.scrapers.flashscore import FlashscoreScraper
from src.scrapers.sofascore import SofascoreScraper

COLOMBIA_TZ = timezone(timedelta(hours=-5))


def format_colombia_time(dt_or_ts: Any, fmt: str = "%H:%M") -> str:
    """Convierte un timestamp o datetime UTC a la hora oficial de Colombia (UTC-5)."""
    if dt_or_ts is None:
        return "--:--"
    try:
        if isinstance(dt_or_ts, (int, float)):
            dt = datetime.fromtimestamp(dt_or_ts, tz=COLOMBIA_TZ)
            return dt.strftime(fmt)
        elif isinstance(dt_or_ts, datetime):
            if dt_or_ts.tzinfo is None:
                dt_or_ts = dt_or_ts.replace(tzinfo=timezone.utc)
            return dt_or_ts.astimezone(COLOMBIA_TZ).strftime(fmt)
    except Exception:
        pass
    return "--:--"


st.set_page_config(
    page_title="ProSports AI v3 — Live Momentum, Monte Carlo, ML & Jugadores",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner=False)
def _bootstrap_database():
    """Inicializa la base de datos una única vez por ciclo del servidor (acelera las recargas)."""
    init_db()
    return True

_bootstrap_database()

# Bloquear traducción automática del navegador que rompe nodos del Virtual DOM de React
components.html(
    """
    <script>
    try {
        const doc = window.parent.document;
        doc.documentElement.setAttribute('translate', 'no');
        doc.documentElement.classList.add('notranslate');
        if (!doc.querySelector('meta[name="google"][content="notranslate"]')) {
            const meta = doc.createElement('meta');
            meta.name = 'google';
            meta.content = 'notranslate';
            doc.head.appendChild(meta);
        }
    } catch (e) {}
    </script>
    """,
    height=0,
)

# Estilos CSS personalizados (Dark Sports Analytics Pro v3)
st.markdown(
    """
    <style>
    .scoreboard-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 22px 26px;
        color: #f8fafc;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
    }
    .league-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .ref-pill {
        display: inline-block;
        background: rgba(245, 158, 11, 0.16);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .team-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        margin: 6px 0 4px 0;
    }
    .elo-tag {
        font-size: 0.83rem;
        color: #94a3b8;
        font-weight: 600;
    }
    .badge-w {
        display: inline-block; background: #16a34a; color: white;
        font-weight: 700; font-size: 0.74rem; padding: 2px 7px;
        border-radius: 5px; margin: 0 2px;
    }
    .badge-d {
        display: inline-block; background: #ca8a04; color: white;
        font-weight: 700; font-size: 0.74rem; padding: 2px 7px;
        border-radius: 5px; margin: 0 2px;
    }
    .badge-l {
        display: inline-block; background: #dc2626; color: white;
        font-weight: 700; font-size: 0.74rem; padding: 2px 7px;
        border-radius: 5px; margin: 0 2px;
    }
    .market-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 15px 17px;
        height: 100%;
        margin-bottom: 14px;
    }
    .market-header {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .market-pick {
        font-size: 1.10rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .conf-high {
        background: rgba(34, 197, 94, 0.18); color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 2px 9px; border-radius: 999px; font-size: 0.73rem; font-weight: 700;
    }
    .conf-med {
        background: rgba(250, 204, 21, 0.18); color: #facc15;
        border: 1px solid rgba(250, 204, 21, 0.4);
        padding: 2px 9px; border-radius: 999px; font-size: 0.73rem; font-weight: 700;
    }
    .conf-low {
        background: rgba(148, 163, 184, 0.18); color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.4);
        padding: 2px 9px; border-radius: 999px; font-size: 0.73rem; font-weight: 700;
    }
    .hit-yes {
        background: rgba(22, 163, 74, 0.25); color: #4ade80;
        border: 1px solid #22c55e;
        padding: 2px 9px; border-radius: 6px; font-size: 0.74rem; font-weight: 800;
    }
    .hit-no {
        background: rgba(220, 38, 38, 0.25); color: #f87171;
        border: 1px solid #ef4444;
        padding: 2px 9px; border-radius: 6px; font-size: 0.74rem; font-weight: 800;
    }
    .prob-bar-bg {
        width: 100%; background: #0f172a; height: 8px;
        border-radius: 99px; overflow: hidden; margin: 9px 0 6px 0;
    }
    .prob-bar-fill {
        height: 100%; border-radius: 99px;
        background: linear-gradient(90deg, #38bdf8, #22c55e);
    }
    .stat-compare-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_scrapers() -> tuple[SofascoreScraper, FlashscoreScraper]:
    return (
        SofascoreScraper(rate_limit_seconds=0.30, use_cache=True),
        FlashscoreScraper(rate_limit_seconds=0.45, use_cache=True),
    )


def render_form_badges(form_str: str) -> str:
    if not form_str or form_str == "N/A":
        return "<span style='color:#64748b'>Sin historial</span>"
    html_parts = []
    for token in form_str.split("-"):
        if token == "W":
            html_parts.append("<span class='badge-w'>V</span>")
        elif token == "D":
            html_parts.append("<span class='badge-d'>E</span>")
        elif token == "L":
            html_parts.append("<span class='badge-l'>D</span>")
    return "".join(html_parts)


def plot_momentum_chart(
    momentum: LiveMomentumSummarySchema, home_name: str, away_name: str
) -> go.Figure:
    """Construye el gráfico interactivo de Presión Minuto a Minuto (Attack Momentum)."""
    mins = [pt.minute for pt in momentum.points]
    vals = [pt.value for pt in momentum.points]
    colors = ["#38bdf8" if v >= 0 else "#f43f5e" for v in vals]

    fig = go.Figure(
        data=[
            go.Bar(
                x=mins,
                y=vals,
                marker_color=colors,
                name="Presión",
                hovertemplate="Min %{x}: %{y:.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"⚡ Gráfico de Presión Minuto a Minuto (Arriba Azul: {home_name} | Abajo Rojo: {away_name})",
        xaxis_title="Minuto de Partido",
        yaxis_title="Índice de Presión Ofensiva (Momentum)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        bargap=0.05,
    )
    return fig


def _normalize_market_category(cat: str) -> str:
    """Normaliza los nombres de los 10 mercados para agregación uniforme."""
    c = cat.lower()
    if "resultado" in c or "1x2" in c or "doble oportunidad" in c:
        return "Resultado (Ensamble Opta + H2H + ML)"
    if "tiempo completo" in c or "mercado de goles" in c or "over 2.5" in c:
        return "Mercado de Goles (Tiempo Completo)"
    if "ambos" in c or "btts" in c:
        return "Ambos Equipos Anotan (BTTS)"
    if "1er tiempo" in c or "1h" in c or "descanso" in c:
        return "Goles en el 1er Tiempo (Opta 1ST)"
    if "remates al arco" in c or "shots on target" in c or "sot" in c:
        return "Remates al Arco (Shots on Target)"
    if "atajadas" in c or "portero" in c or "saves" in c:
        return "Atajadas del Portero (GK Saves)"
    if "remates totales" in c or "total shots" in c:
        return "Remates Totales del Partido"
    if "córners" in c or "corners" in c or "saques de esquina" in c:
        return "Córners (Saques de Esquina)"
    if "tarjetas" in c or "árbitro" in c or "arbitro" in c or "cards" in c:
        return "Tarjetas Amarillas & Árbitro"
    if "hándicap" in c or "handicap" in c or "dnb" in c or "10k" in c:
        return "Hándicap Asiático & Equipos (10K Sims)"
    return cat


def _extract_specific_bet_line(pick_text: str, category: str) -> str:
    """
    Normaliza cada pronóstico exacto en su línea de apuesta específica y comprensible.
    Ej: '+1.5 Goles FT', '+2.5 Goles FT', '3+ Atajadas Portero', '+8.5 Córners', '+9.5 Córners', etc.
    """
    p = pick_text.lower()

    # 1. Goles Totales FT
    if "más de 1.5 goles" in p and "1er tiempo" not in p and "equ." not in p:
        return "⚽ Over 1.5 Goles FT (+1.5)"
    if "más de 2.5 goles" in p or "over 2.5" in p:
        return "⚽ Over 2.5 Goles FT (+2.5)"
    if "menos de 2.5 goles" in p or "under 2.5" in p:
        return "⚽ Under 2.5 Goles FT (-2.5)"
    if "ambos anotan: sí" in p or "ambos anotan: s" in p:
        return "🔥 Ambos Equipos Anotan: SÍ (BTTS)"
    if "ambos anotan: no" in p:
        return "⛔ Ambos Equipos Anotan: NO"

    # 2. 1er Tiempo (1H)
    if "+0.5 1h" in p or ("más de 0.5 goles" in p and "1er tiempo" in p):
        return "⏱️ Over 0.5 Goles 1H (+0.5 1er Tiempo)"
    if "-1.5 1h" in p or ("menos de 1.5 goles" in p and "1er tiempo" in p):
        return "⏱️ Under 1.5 Goles 1H (-1.5 1er Tiempo)"

    # 3. Porteros (GK Saves)
    if "3+ atajadas" in p or ("+2.5" in p and "atajadas" in p):
        return "🧤 Portero: 3+ Atajadas (+2.5 Saves)"
    if "2+ atajadas" in p or ("+1.5" in p and "atajadas" in p):
        return "🧤 Portero: 2+ Atajadas (+1.5 Saves)"
    if "5.5 atajadas" in p:
        return "🧤 Más de 5.5 Atajadas Totales"

    # 4. Córners (Tiros de Esquina)
    if "córner" in p or "corner" in p or "crner" in p or "saque" in p:
        if "9.5" in p and ("más" in p or "over" in p or "+" in p):
            return "🚩 Over 9.5 Córners (+9.5 Tiros de Esquina)"
        if "8.5" in p and ("más" in p or "over" in p or "+" in p):
            return "🚩 Over 8.5 Córners (+8.5 Tiros de Esquina)"
        if "7.5" in p and ("más" in p or "over" in p or "+" in p):
            return "🚩 Over 7.5 Córners (+7.5 Tiros de Esquina)"
        if "6.5" in p and ("más" in p or "over" in p or "+" in p):
            return "🚩 Over 6.5 Córners (+6.5 Tiros de Esquina)"
        if "menos" in p or "under" in p:
            return "🚩 Under Córners (-9.5 Tiros de Esquina)"
        return "🚩 Córners / Saques de Esquina"

    # 5. Remates al Arco (Shots on Target)
    if "remates al arco" in p or "al arco" in p or "shots on target" in p or "sot" in p:
        if "8.5" in p and "más" in p:
            return "🎯 Over 8.5 Remates al Arco (+8.5 SoT)"
        if "7.5" in p and "más" in p:
            return "🎯 Over 7.5 Remates al Arco (+7.5 SoT)"
        if "menos" in p:
            return "🎯 Under 8.5 Remates al Arco"
        return "🎯 Remates al Arco (SoT)"

    # 6. Remates Totales
    if "remates totales" in p or "total shots" in p:
        if "24.5" in p and "más" in p:
            return "💥 Over 24.5 Remates Totales"
        if "22.5" in p and "más" in p:
            return "💥 Over 22.5 Remates Totales"
        if "menos" in p:
            return "💥 Under Remates Totales (-24.5)"
        return "💥 Remates Totales del Partido"

    # 7. Tarjetas Amarillas
    if "tarjeta" in p or "card" in p:
        if "4.5" in p and "más" in p:
            return "🟨 Over 4.5 Tarjetas Amarillas"
        if "3.5" in p and "más" in p:
            return "🟨 Over 3.5 Tarjetas Amarillas"
        if "menos" in p:
            return "🟨 Under Tarjetas Amarillas (-4.5)"
        return "🟨 Tarjetas Amarillas"

    # 8. Resultado / Doble Oportunidad / DNB / Equipos
    if "doble oportunidad" in p:
        return "🛡️ Doble Oportunidad (1X / X2)"
    if "apuesta sin empate" in p or "dnb" in p:
        return "⚖️ Apuesta Sin Empate (DNB)"
    if "equ." in p or "equipo más de 1.5" in p:
        return "⚽ Equipo Individual: Over 1.5 Goles"
    if "hándicap" in p or "handicap" in p:
        return "📈 Hándicap Asiático (+1.5)"
    if "victoria" in p:
        return "🏆 Victoria Directa 1X2"

    return pick_text


def evaluate_all_systematic_lines_for_match(
    match: MatchDetailSchema, pred: MatchPrediction
) -> List[Dict[str, Any]]:
    """
    Evalúa sistemáticamente TODAS las líneas estándar de apuestas para un partido finalizado:
    +1.5, +2.5, -2.5, -3.5 Goles, BTTS Sí/No, +0.5 1H, -1.5 1H,
    Córners (+6.5, +7.5, +8.5, +9.5, +10.5, -10.5), Remates al arco (+6.5, +7.5, +8.5, -8.5),
    Remates totales (+20.5, +22.5, +24.5), Tarjetas (+3.5, +4.5, -4.5), Atajadas (2+, 3+),
    Doble Oportunidad (1X, X2) y Goles por equipo.
    Permite auditar el comportamiento global de cada línea en el 100% de los partidos.
    """
    hs = match.home_score
    as_ = match.away_score
    tot_g = (hs + as_) if (hs is not None and as_ is not None) else None
    ht_hs = match.home_score_ht
    ht_as = match.away_score_ht
    ht_g = (ht_hs + ht_as) if (ht_hs is not None and ht_as is not None) else None

    st = match.stats
    crn = (
        (st.corner_kicks_home + st.corner_kicks_away)
        if (st and st.corner_kicks_home is not None and st.corner_kicks_away is not None)
        else None
    )
    sot = (
        (st.shots_on_target_home + st.shots_on_target_away)
        if (st and st.shots_on_target_home is not None and st.shots_on_target_away is not None)
        else None
    )
    sh = (
        (st.total_shots_home + st.total_shots_away)
        if (st and st.total_shots_home is not None and st.total_shots_away is not None)
        else None
    )
    crd = (
        (st.yellow_cards_home + st.yellow_cards_away)
        if (st and st.yellow_cards_home is not None and st.yellow_cards_away is not None)
        else None
    )
    sv_h = st.saves_home if (st and st.saves_home is not None) else None
    sv_a = st.saves_away if (st and st.saves_away is not None) else None
    max_sv = max(sv_h, sv_a) if (sv_h is not None and sv_a is not None) else None
    tot_sv = (sv_h + sv_a) if (sv_h is not None and sv_a is not None) else None

    match_label = f"{match.home_team.name} vs {match.away_team.name}"
    lines: List[Dict[str, Any]] = []

    # 1. Goles Totales FT
    if tot_g is not None:
        lines.append({
            "line_name": "⚽ Over 1.5 Goles FT (+1.5)",
            "category": "Mercado de Goles (Tiempo Completo)",
            "prob": pred.prob_over_1_5,
            "hit": tot_g >= 2,
            "actual": f"Total: {tot_g} goles ({hs}-{as_})",
            "match": match_label,
        })
        lines.append({
            "line_name": "⚽ Over 2.5 Goles FT (+2.5)",
            "category": "Mercado de Goles (Tiempo Completo)",
            "prob": pred.prob_over_2_5,
            "hit": tot_g >= 3,
            "actual": f"Total: {tot_g} goles ({hs}-{as_})",
            "match": match_label,
        })
        lines.append({
            "line_name": "⚽ Under 2.5 Goles FT (-2.5)",
            "category": "Mercado de Goles (Tiempo Completo)",
            "prob": pred.prob_under_2_5,
            "hit": tot_g <= 2,
            "actual": f"Total: {tot_g} goles ({hs}-{as_})",
            "match": match_label,
        })
        lines.append({
            "line_name": "⚽ Under 3.5 Goles FT (-3.5)",
            "category": "Mercado de Goles (Tiempo Completo)",
            "prob": round(max(1.0, 100.0 - pred.prob_over_3_5), 1),
            "hit": tot_g <= 3,
            "actual": f"Total: {tot_g} goles ({hs}-{as_})",
            "match": match_label,
        })
        lines.append({
            "line_name": "🔥 Ambos Equipos Anotan: SÍ (BTTS)",
            "category": "Ambos Equipos Anotan (BTTS)",
            "prob": pred.prob_btts,
            "hit": (hs >= 1 and as_ >= 1),
            "actual": f"Marcador: {hs}-{as_}",
            "match": match_label,
        })
        lines.append({
            "line_name": "⛔ Ambos Equipos Anotan: NO",
            "category": "Ambos Equipos Anotan (BTTS)",
            "prob": pred.prob_btts_no,
            "hit": (hs == 0 or as_ == 0),
            "actual": f"Marcador: {hs}-{as_}",
            "match": match_label,
        })

    # 2. Goles 1er Tiempo (1H)
    if ht_g is not None:
        lines.append({
            "line_name": "⏱️ Over 0.5 Goles 1H (+0.5 1er Tiempo)",
            "category": "Goles en el 1er Tiempo (Opta 1ST)",
            "prob": pred.prob_1h_over_0_5,
            "hit": ht_g >= 1,
            "actual": f"Descanso: {ht_hs}-{ht_as} ({ht_g} goles)",
            "match": match_label,
        })
        lines.append({
            "line_name": "⏱️ Under 1.5 Goles 1H (-1.5 1er Tiempo)",
            "category": "Goles en el 1er Tiempo (Opta 1ST)",
            "prob": round(max(1.0, 100.0 - pred.prob_1h_over_1_5), 1),
            "hit": ht_g <= 1,
            "actual": f"Descanso: {ht_hs}-{ht_as} ({ht_g} goles)",
            "match": match_label,
        })

    # 3. Córners (Tiros de Esquina)
    if crn is not None:
        exp_crn = pred.expected_corners_total or 9.5
        lines.append({
            "line_name": "🚩 Over 6.5 Córners (+6.5 Tiros de Esquina)",
            "category": "Córners (Saques de Esquina)",
            "prob": round(_poisson_prob_over(6.5, exp_crn) * 100.0, 1),
            "hit": crn >= 7,
            "actual": f"{crn} córners totales",
            "match": match_label,
        })
        lines.append({
            "line_name": "🚩 Over 7.5 Córners (+7.5 Tiros de Esquina)",
            "category": "Córners (Saques de Esquina)",
            "prob": round(_poisson_prob_over(7.5, exp_crn) * 100.0, 1),
            "hit": crn >= 8,
            "actual": f"{crn} córners totales",
            "match": match_label,
        })
        lines.append({
            "line_name": "🚩 Over 8.5 Córners (+8.5 Tiros de Esquina)",
            "category": "Córners (Saques de Esquina)",
            "prob": pred.prob_over_8_5_corners,
            "hit": crn >= 9,
            "actual": f"{crn} córners totales",
            "match": match_label,
        })
        lines.append({
            "line_name": "🚩 Over 9.5 Córners (+9.5 Tiros de Esquina)",
            "category": "Córners (Saques de Esquina)",
            "prob": pred.prob_over_9_5_corners,
            "hit": crn >= 10,
            "actual": f"{crn} córners totales",
            "match": match_label,
        })
        lines.append({
            "line_name": "🚩 Under 10.5 Córners (-10.5 Tiros de Esquina)",
            "category": "Córners (Saques de Esquina)",
            "prob": round(max(1.0, 100.0 - pred.prob_over_10_5_corners), 1),
            "hit": crn <= 10,
            "actual": f"{crn} córners totales",
            "match": match_label,
        })

    # 4. Remates al Arco (Shots on Target)
    if sot is not None:
        exp_sot = pred.expected_sot_total or 8.5
        lines.append({
            "line_name": "🎯 Over 6.5 Remates al Arco (+6.5 SoT)",
            "category": "Remates al Arco (Shots on Target)",
            "prob": round(_poisson_prob_over(6.5, exp_sot) * 100.0, 1),
            "hit": sot >= 7,
            "actual": f"{sot} remates al arco",
            "match": match_label,
        })
        lines.append({
            "line_name": "🎯 Over 7.5 Remates al Arco (+7.5 SoT)",
            "category": "Remates al Arco (Shots on Target)",
            "prob": pred.prob_over_7_5_sot,
            "hit": sot >= 8,
            "actual": f"{sot} remates al arco",
            "match": match_label,
        })
        lines.append({
            "line_name": "🎯 Over 8.5 Remates al Arco (+8.5 SoT)",
            "category": "Remates al Arco (Shots on Target)",
            "prob": pred.prob_over_8_5_sot,
            "hit": sot >= 9,
            "actual": f"{sot} remates al arco",
            "match": match_label,
        })
        lines.append({
            "line_name": "🎯 Under 8.5 Remates al Arco",
            "category": "Remates al Arco (Shots on Target)",
            "prob": round(max(1.0, 100.0 - pred.prob_over_8_5_sot), 1),
            "hit": sot <= 8,
            "actual": f"{sot} remates al arco",
            "match": match_label,
        })

    # 5. Remates Totales
    if sh is not None:
        exp_sh = pred.expected_shots_total or 23.0
        lines.append({
            "line_name": "💥 Over 20.5 Remates Totales",
            "category": "Remates Totales del Partido",
            "prob": round(_poisson_prob_over(20.5, exp_sh) * 100.0, 1),
            "hit": sh >= 21,
            "actual": f"{sh} remates totales",
            "match": match_label,
        })
        lines.append({
            "line_name": "💥 Over 22.5 Remates Totales",
            "category": "Remates Totales del Partido",
            "prob": pred.prob_over_22_5_shots,
            "hit": sh >= 23,
            "actual": f"{sh} remates totales",
            "match": match_label,
        })
        lines.append({
            "line_name": "💥 Over 24.5 Remates Totales",
            "category": "Remates Totales del Partido",
            "prob": pred.prob_over_24_5_shots,
            "hit": sh >= 25,
            "actual": f"{sh} remates totales",
            "match": match_label,
        })

    # 6. Tarjetas Amarillas
    if crd is not None:
        lines.append({
            "line_name": "🟨 Over 3.5 Tarjetas Amarillas",
            "category": "Tarjetas Amarillas & Árbitro",
            "prob": pred.prob_over_3_5_cards,
            "hit": crd >= 4,
            "actual": f"{crd} tarjetas amarillas",
            "match": match_label,
        })
        lines.append({
            "line_name": "🟨 Over 4.5 Tarjetas Amarillas",
            "category": "Tarjetas Amarillas & Árbitro",
            "prob": pred.prob_over_4_5_cards,
            "hit": crd >= 5,
            "actual": f"{crd} tarjetas amarillas",
            "match": match_label,
        })
        lines.append({
            "line_name": "🟨 Under 4.5 Tarjetas Amarillas",
            "category": "Tarjetas Amarillas & Árbitro",
            "prob": round(max(1.0, 100.0 - pred.prob_over_4_5_cards), 1),
            "hit": crd <= 4,
            "actual": f"{crd} tarjetas amarillas",
            "match": match_label,
        })

    # 7. Atajadas de Portero
    if max_sv is not None:
        lines.append({
            "line_name": "🧤 Portero: 2+ Atajadas (+1.5 Saves)",
            "category": "Atajadas del Portero (GK Saves)",
            "prob": max(pred.prob_home_gk_2plus_saves, pred.prob_away_gk_2plus_saves),
            "hit": max_sv >= 2,
            "actual": f"Máx portero: {max_sv} atajadas",
            "match": match_label,
        })
        lines.append({
            "line_name": "🧤 Portero: 3+ Atajadas (+2.5 Saves)",
            "category": "Atajadas del Portero (GK Saves)",
            "prob": max(pred.prob_home_gk_3plus_saves, pred.prob_away_gk_3plus_saves),
            "hit": max_sv >= 3,
            "actual": f"Máx portero: {max_sv} atajadas",
            "match": match_label,
        })

    # 8. Doble Oportunidad & Resultado
    if hs is not None and as_ is not None:
        lines.append({
            "line_name": "🛡️ Doble Oportunidad 1X (Local o Empate)",
            "category": "Resultado (Ensamble Opta + H2H + ML)",
            "prob": pred.prob_1x,
            "hit": hs >= as_,
            "actual": f"Marcador: {hs}-{as_}",
            "match": match_label,
        })
        lines.append({
            "line_name": "🛡️ Doble Oportunidad X2 (Visitante o Empate)",
            "category": "Resultado (Ensamble Opta + H2H + ML)",
            "prob": pred.prob_x2,
            "hit": as_ >= hs,
            "actual": f"Marcador: {hs}-{as_}",
            "match": match_label,
        })
        lines.append({
            "line_name": "⚽ Equipo Individual: Over 1.5 Goles",
            "category": "Hándicap Asiático & Equipos (10K Sims)",
            "prob": max(pred.prob_home_over_1_5, pred.prob_away_over_1_5),
            "hit": (hs >= 2 or as_ >= 2),
            "actual": f"Marcador: {hs}-{as_}",
            "match": match_label,
        })

    return lines


def compute_market_accuracy_breakdown(
    predictions: List[MatchPrediction],
    matches_map: Optional[Dict[str, MatchDetailSchema]] = None,
) -> Dict[str, Any]:
    """
    Analiza el rendimiento y porcentaje de aciertos:
    1. Por los 10 mercados principales (categorías estándar).
    2. Por LÍNEAS ESPECÍFICAS DE APUESTA seleccionadas por el modelo (+1.5, +2.5, +8.5 Córners, etc.).
    3. AUDITORÍA EXHAUSTIVA DE TODAS LAS LÍNEAS en CADA partido (+1.5 Goles, +2.5 Goles, +6.5 Córners, etc. con muestra N completa).
    """
    canonical_categories = [
        "Resultado (Ensamble Opta + H2H + ML)",
        "Mercado de Goles (Tiempo Completo)",
        "Ambos Equipos Anotan (BTTS)",
        "Goles en el 1er Tiempo (Opta 1ST)",
        "Remates al Arco (Shots on Target)",
        "Atajadas del Portero (GK Saves)",
        "Remates Totales del Partido",
        "Córners (Saques de Esquina)",
        "Tarjetas Amarillas & Árbitro",
        "Hándicap Asiático & Equipos (10K Sims)",
    ]

    category_map: Dict[str, Dict[str, Any]] = {
        cat: {
            "category": cat,
            "total": 0,
            "hits": 0,
            "misses": 0,
            "win_rate": 0.0,
            "probs": [],
            "picks": [],
        }
        for cat in canonical_categories
    }

    specific_lines_map: Dict[str, Dict[str, Any]] = {}
    exhaustive_lines_map: Dict[str, Dict[str, Any]] = {}

    overall_total = 0
    overall_hits = 0
    matches_audited = 0

    ex_overall_total = 0
    ex_overall_hits = 0

    for p in predictions:
        if not p.all_market_picks:
            continue
        has_finished = any(pk.get("hit") is not None for pk in p.all_market_picks)
        if has_finished:
            matches_audited += 1

        # 1. EVALUACIÓN DE LAS 10 TARJETAS SELECCIONADAS POR EL MODELO
        for pk in p.all_market_picks:
            raw_cat = pk.get("category", "")
            pick_label = pk.get("pick", "")
            hit = pk.get("hit")
            if hit is None:
                continue

            # Agregación por Categoría General
            norm_cat = _normalize_market_category(raw_cat)
            if norm_cat not in category_map:
                category_map[norm_cat] = {
                    "category": norm_cat,
                    "total": 0,
                    "hits": 0,
                    "misses": 0,
                    "win_rate": 0.0,
                    "probs": [],
                    "picks": [],
                }

            category_map[norm_cat]["total"] += 1
            overall_total += 1
            if hit is True:
                category_map[norm_cat]["hits"] += 1
                overall_hits += 1
            else:
                category_map[norm_cat]["misses"] += 1

            category_map[norm_cat]["probs"].append(pk.get("prob", 50.0))
            category_map[norm_cat]["picks"].append({
                "match": f"{p.home_team} vs {p.away_team}",
                "pick": pick_label,
                "prob": pk.get("prob"),
                "hit": hit,
                "actual": pk.get("actual"),
            })

            # Agregación por Línea Específica Recomendada
            spec_line = _extract_specific_bet_line(pick_label, raw_cat)
            if spec_line not in specific_lines_map:
                specific_lines_map[spec_line] = {
                    "line_name": spec_line,
                    "category": norm_cat,
                    "total": 0,
                    "hits": 0,
                    "misses": 0,
                    "win_rate": 0.0,
                    "probs": [],
                    "picks": [],
                }

            specific_lines_map[spec_line]["total"] += 1
            if hit is True:
                specific_lines_map[spec_line]["hits"] += 1
            else:
                specific_lines_map[spec_line]["misses"] += 1

            specific_lines_map[spec_line]["probs"].append(pk.get("prob", 50.0))
            specific_lines_map[spec_line]["picks"].append({
                "match": f"{p.home_team} vs {p.away_team}",
                "pick": pick_label,
                "prob": pk.get("prob"),
                "hit": hit,
                "actual": pk.get("actual"),
            })

        # 2. AUDITORÍA EXHAUSTIVA DE TODAS LAS LÍNEAS SISTEMÁTICAS EN ESTE PARTIDO
        if matches_map and p.event_id in matches_map:
            m_obj = matches_map[p.event_id]
            sys_lines = evaluate_all_systematic_lines_for_match(m_obj, p)
            for s_line in sys_lines:
                l_name = s_line["line_name"]
                l_cat = s_line["category"]
                l_hit = s_line["hit"]
                l_prob = s_line["prob"]

                if l_name not in exhaustive_lines_map:
                    exhaustive_lines_map[l_name] = {
                        "line_name": l_name,
                        "category": l_cat,
                        "total": 0,
                        "hits": 0,
                        "misses": 0,
                        "win_rate": 0.0,
                        "probs": [],
                        "picks": [],
                    }

                exhaustive_lines_map[l_name]["total"] += 1
                ex_overall_total += 1
                if l_hit is True:
                    exhaustive_lines_map[l_name]["hits"] += 1
                    ex_overall_hits += 1
                else:
                    exhaustive_lines_map[l_name]["misses"] += 1

                exhaustive_lines_map[l_name]["probs"].append(l_prob)
                exhaustive_lines_map[l_name]["picks"].append({
                    "match": s_line["match"],
                    "pick": l_name,
                    "prob": l_prob,
                    "hit": l_hit,
                    "actual": s_line["actual"],
                })

    # Procesar Categorías
    stats_list = []
    for cat, data in category_map.items():
        tot = data["total"]
        h = data["hits"]
        m = data["misses"]
        wr = round((h / tot) * 100.0, 1) if tot > 0 else 0.0
        data["win_rate"] = wr
        data["avg_prob"] = round(sum(data["probs"]) / len(data["probs"]), 1) if data["probs"] else 0.0

        if tot == 0:
            status_tag = "Sin datos en partidos analizados"
            badge = "⚪ Pendiente"
        elif wr >= 72.0:
            status_tag = "Alta Efectividad (Top Acierto)"
            badge = "🟢 Muy Confiable"
        elif wr >= 55.0:
            status_tag = "Rendimiento Estable"
            badge = "🔵 Estable"
        elif wr >= 45.0:
            status_tag = "Rendimiento Moderado (Cautela)"
            badge = "🟡 Cautela"
        else:
            status_tag = "Alta Volatilidad (Fallo Frecuente)"
            badge = "🔴 Riesgoso"

        data["status_tag"] = status_tag
        data["badge"] = badge
        stats_list.append(data)

    stats_list.sort(key=lambda x: (x["total"] > 0, x["win_rate"], x["total"]), reverse=True)

    # Procesar Líneas Específicas Recomendadas
    lines_list = []
    for line_name, data in specific_lines_map.items():
        tot = data["total"]
        h = data["hits"]
        m = data["misses"]
        wr = round((h / tot) * 100.0, 1) if tot > 0 else 0.0
        data["win_rate"] = wr
        data["avg_prob"] = round(sum(data["probs"]) / len(data["probs"]), 1) if data["probs"] else 0.0

        if tot == 0:
            status_tag = "Sin datos"
            badge = "⚪ Pendiente"
        elif wr >= 72.0:
            status_tag = "Línea Estrella (Muy Alta Efectividad)"
            badge = "🟢 Estrella"
        elif wr >= 55.0:
            status_tag = "Línea Sólida"
            badge = "🔵 Sólida"
        elif wr >= 45.0:
            status_tag = "Línea Moderada"
            badge = "🟡 Cautela"
        else:
            status_tag = "Línea de Alto Fallo (Evitar / Revisar)"
            badge = "🔴 Evitar"

        data["status_tag"] = status_tag
        data["badge"] = badge
        lines_list.append(data)

    lines_list.sort(key=lambda x: (x["total"] > 0, x["win_rate"], x["total"]), reverse=True)

    # Procesar Líneas Sistemáticas Exhaustivas
    exhaustive_list = []
    for line_name, data in exhaustive_lines_map.items():
        tot = data["total"]
        h = data["hits"]
        m = data["misses"]
        wr = round((h / tot) * 100.0, 1) if tot > 0 else 0.0
        data["win_rate"] = wr
        data["avg_prob"] = round(sum(data["probs"]) / len(data["probs"]), 1) if data["probs"] else 0.0

        if tot == 0:
            status_tag = "Sin datos"
            badge = "⚪ Pendiente"
        elif wr >= 75.0:
            status_tag = "Línea Estrella (Muy Alta Frecuencia de Acierto)"
            badge = "🟢 Estrella"
        elif wr >= 55.0:
            status_tag = "Línea Sólida / Frecuente"
            badge = "🔵 Sólida"
        elif wr >= 45.0:
            status_tag = "Línea Moderada (50/50)"
            badge = "🟡 Cautela"
        else:
            status_tag = "Línea de Alto Fallo / Riesgo Elevado"
            badge = "🔴 Riesgo Alto"

        data["status_tag"] = status_tag
        data["badge"] = badge
        exhaustive_list.append(data)

    exhaustive_list.sort(key=lambda x: (x["total"] > 0, x["win_rate"], x["total"]), reverse=True)

    active_cats = [s for s in stats_list if s["total"] > 0]
    best_cat = active_cats[0] if active_cats else None
    worst_cat = active_cats[-1] if active_cats else None

    active_lines = [s for s in lines_list if s["total"] > 0]
    best_line = active_lines[0] if active_lines else None
    worst_line = active_lines[-1] if active_lines else None

    active_ex = [s for s in exhaustive_list if s["total"] > 0]
    best_ex_line = active_ex[0] if active_ex else None
    worst_ex_line = active_ex[-1] if active_ex else None

    overall_win_rate = round((overall_hits / overall_total) * 100.0, 1) if overall_total > 0 else 0.0
    ex_win_rate = round((ex_overall_hits / ex_overall_total) * 100.0, 1) if ex_overall_total > 0 else 0.0

    return {
        "matches_audited": matches_audited,
        "overall_total": overall_total,
        "overall_hits": overall_hits,
        "overall_misses": overall_total - overall_hits,
        "overall_win_rate": overall_win_rate,
        "category_stats": stats_list,
        "specific_lines_stats": lines_list,
        "exhaustive_lines_stats": exhaustive_list,
        "exhaustive_total": ex_overall_total,
        "exhaustive_hits": ex_overall_hits,
        "exhaustive_win_rate": ex_win_rate,
        "best_category": best_cat,
        "worst_category": worst_cat,
        "best_line": best_line,
        "worst_line": worst_line,
        "best_exhaustive_line": best_ex_line,
        "worst_exhaustive_line": worst_ex_line,
    }


def plot_specific_lines_chart(lines_list: List[Dict[str, Any]], overall_win_rate: float, title: str) -> go.Figure:
    """Construye un gráfico horizontal de barras de todas las líneas específicas ordenadas por % de acierto."""
    active = [s for s in lines_list if s["total"] > 0]
    if not active:
        active = lines_list

    active_rev = list(reversed(active))
    labels = [s["line_name"] for s in active_rev]
    win_rates = [s["win_rate"] for s in active_rev]

    colors = []
    text_labels = []
    for s in active_rev:
        wr = s["win_rate"]
        tot = s["total"]
        h = s["hits"]
        if tot == 0:
            colors.append("#64748b")
            text_labels.append("Sin datos")
        elif wr >= 72.0:
            colors.append("#10b981")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos) 🏆")
        elif wr >= 55.0:
            colors.append("#38bdf8")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos)")
        elif wr >= 45.0:
            colors.append("#f59e0b")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos)")
        else:
            colors.append("#f43f5e")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos) ⚠️")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=win_rates,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.25)", width=1),
            ),
            text=text_labels,
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="#ffffff", size=12, family="sans-serif"),
            hovertemplate="<b>%{y}</b><br>Efectividad Real: %{x:.1f}%<extra></extra>",
        )
    )

    fig.add_vline(
        x=50.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="50% Umbral Neutro",
        annotation_position="top right",
        annotation_font=dict(color="#94a3b8", size=11),
    )

    if overall_win_rate > 0:
        fig.add_vline(
            x=overall_win_rate,
            line_width=2,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"Promedio: {overall_win_rate:.1f}%",
            annotation_position="bottom right",
            annotation_font=dict(color="#38bdf8", size=11),
        )

    chart_height = max(420, len(active) * 36)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color="#f8fafc"),
        ),
        xaxis=dict(
            title="% de Acierto Real",
            range=[0, 105],
            ticksuffix="%",
            gridcolor="#334155",
            color="#94a3b8",
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            color="#f8fafc",
            tickfont=dict(size=12, weight="bold"),
        ),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        height=chart_height,
        margin=dict(l=20, r=30, t=50, b=30),
    )
    return fig


def plot_market_accuracy_chart(breakdown: Dict[str, Any]) -> go.Figure:
    """Construye un gráfico horizontal de barras de los 10 mercados coloreado por efectividad."""
    stats = [s for s in breakdown["category_stats"] if s["total"] > 0]
    if not stats:
        stats = breakdown["category_stats"]

    stats_rev = list(reversed(stats))
    categories = [s["category"] for s in stats_rev]
    win_rates = [s["win_rate"] for s in stats_rev]

    colors = []
    text_labels = []
    for s in stats_rev:
        wr = s["win_rate"]
        tot = s["total"]
        h = s["hits"]
        if tot == 0:
            colors.append("#64748b")
            text_labels.append("Sin datos evaluados")
        elif wr >= 72.0:
            colors.append("#10b981")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos) 🏆")
        elif wr >= 55.0:
            colors.append("#38bdf8")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos)")
        elif wr >= 45.0:
            colors.append("#f59e0b")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos)")
        else:
            colors.append("#f43f5e")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos) ⚠️")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=categories,
            x=win_rates,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.25)", width=1),
            ),
            text=text_labels,
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="#ffffff", size=13, family="sans-serif"),
            hovertemplate="<b>%{y}</b><br>Efectividad Real: %{x:.1f}%<extra></extra>",
        )
    )

    fig.add_vline(
        x=50.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="50% Umbral Neutro",
        annotation_position="top right",
        annotation_font=dict(color="#94a3b8", size=11),
    )

    if breakdown["overall_total"] > 0:
        fig.add_vline(
            x=breakdown["overall_win_rate"],
            line_width=2,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"Promedio Global: {breakdown['overall_win_rate']:.1f}%",
            annotation_position="bottom right",
            annotation_font=dict(color="#38bdf8", size=12),
        )

    fig.update_layout(
        title=dict(
            text=f"📊 Efectividad y Tasa de Acierto por Mercado ({breakdown['matches_audited']} Partidos Auditados • {breakdown['overall_hits']}/{breakdown['overall_total']} Aciertos Totales)",
            font=dict(size=15, color="#f8fafc"),
        ),
        xaxis=dict(
            title="% de Acierto Real",
            range=[0, 105],
            ticksuffix="%",
            gridcolor="#334155",
            color="#94a3b8",
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            color="#f8fafc",
            tickfont=dict(size=12, weight="bold"),
        ),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        height=450,
        margin=dict(l=20, r=30, t=50, b=30),
    )
    return fig


@st.cache_data(ttl=1200, show_spinner=False)
def load_day_catalog(date_str: str, include_all_leagues: bool) -> Dict[str, Any]:
    """Carga rápida de todos los partidos de la fecha con prioridad de caché Supabase (0.2s) y fallback automático."""
    # 0. Si no se solicitan todas las ligas del mundo, verificar si ya existen datos precalculados en Supabase
    if not include_all_leagues:
        try:
            precomputed_records = get_precomputed_matches_for_date(date_str)
            if precomputed_records:
                matches: List[MatchDetailSchema] = []
                precomputed_deep_cache: Dict[int, Dict[str, Any]] = {}
                precomputed_preds: Dict[int, MatchPrediction] = {}
                fs_lookup: Dict[tuple[int, int], str] = {}

                for r in precomputed_records:
                    try:
                        m_obj = MatchDetailSchema.model_validate(r["match_detail"])
                        matches.append(m_obj)

                        deep_data = r.get("deep_analysis") or {}
                        pred_obj = None
                        if r.get("prediction"):
                            try:
                                pred_obj = MatchPrediction.model_validate(r["prediction"])
                                precomputed_preds[m_obj.event_id] = pred_obj
                            except Exception as pred_err:
                                logger.warning(f"Error parseando prediccion para {m_obj.event_id}: {pred_err}")

                        fs_id = deep_data.get("flashscore_id", "N/A")
                        fs_lookup[(m_obj.home_team.id, m_obj.away_team.id)] = fs_id

                        if "home_form" in deep_data and "away_form" in deep_data:
                            try:
                                precomputed_deep_cache[m_obj.event_id] = {
                                    "match": m_obj,
                                    "home_form": TeamFormMetrics.model_validate(deep_data["home_form"]),
                                    "away_form": TeamFormMetrics.model_validate(deep_data["away_form"]),
                                    "home_recent": [MatchDetailSchema.model_validate(x) for x in deep_data.get("home_recent", [])],
                                    "away_recent": [MatchDetailSchema.model_validate(x) for x in deep_data.get("away_recent", [])],
                                    "home_players": [PlayerPropSchema.model_validate(x) for x in deep_data.get("home_players", [])],
                                    "away_players": [PlayerPropSchema.model_validate(x) for x in deep_data.get("away_players", [])],
                                    "prediction": pred_obj,
                                    "standings_map": deep_data.get("standings_map", {}),
                                    "flashscore_id": fs_id,
                                }
                            except Exception as dec_err:
                                logger.debug(f"Error decodificando deep data para {m_obj.event_id}: {dec_err}")
                    except Exception as err:
                        logger.warning(f"Error deserializando partido precalculado {r.get('event_id')}: {err}")

                if matches:
                    return {
                        "matches": matches,
                        "fs_total": len(matches),
                        "fs_lookup": fs_lookup,
                        "entity_links": [],
                        "source": "⚡ Supabase Cloud Cache (Ultra Rápido)",
                        "precomputed_deep_cache": precomputed_deep_cache,
                        "precomputed_predictions": precomputed_preds,
                    }
        except Exception as e:
            logger.warning(f"Error consultando caché de Supabase: {e}")

    sf_scraper, fs_scraper = get_scrapers()
    resolver = EntityResolver(confidence_threshold=78.0)

    # 1. Extraer partidos de Flashscore (CDN global de alta disponibilidad)
    try:
        t_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        day_offset = max(-7, min(7, (t_dt - date.today()).days))
        fs_matches = fs_scraper.get_daily_matches(day_offset=day_offset)
    except Exception as e:
        logger.warning(f"Error consultando Flashscore: {e}")
        fs_matches = []

    # 2. Consultar partidos de Sofascore
    try:
        matches = sf_scraper.get_scheduled_events(
            date_str=date_str, only_top_tournaments=not include_all_leagues
        )
        if not matches and not include_all_leagues:
            matches = sf_scraper.get_scheduled_events(
                date_str=date_str, only_top_tournaments=False
            )
    except Exception as e:
        logger.warning(f"Error consultando Sofascore: {e}")
        matches = []

    source_label = "Sofascore Opta"

    # 3. Fallback inteligente a Flashscore si Sofascore está bloqueado o vacío en la nube
    if not matches and fs_matches:
        source_label = "Flashscore Cloud"
        curated_countries = {
            "france", "francia", "england", "inglaterra", "spain", "españa",
            "italy", "italia", "germany", "alemania", "colombia", "usa",
            "united states", "estados unidos", "argentina", "saudi arabia",
            "arabia saudita", "turkey", "turquía", "turquia", "brazil",
            "brasil", "portugal", "mexico", "méxico", "europe", "europa",
            "south america", "sudamérica", "sudamerica", "world", "mundo",
            "international", "internacional", "concacaf", "conmebol", "uefa"
        }

        for fsm in fs_matches:
            try:
                parts = fsm.tournament_title.split(":")
                raw_country = parts[0].strip() if len(parts) > 1 else fsm.country
                country_clean = raw_country.lower()

                # Si no está marcado "Cargar todas las ligas", filtrar solo por las 12 ligas, copas y torneos internacionales solicitados
                if not include_all_leagues and not any(c in country_clean for c in curated_countries):
                    continue

                # Filtrar que el partido de Flashscore realmente corresponda a la fecha solicitada
                tz_local = timezone(timedelta(hours=-5))
                fsm_date_local = datetime.fromtimestamp(fsm.start_timestamp, tz=tz_local).strftime("%Y-%m-%d")
                if fsm_date_local != date_str:
                    continue

                numeric_id = abs(hash(fsm.flashscore_id)) % (10**9)
                status_map = {"1": "notstarted", "2": "inprogress", "3": "finished"}
                if fsm.status_code in status_map.values():
                    status_type = fsm.status_code
                else:
                    status_type = status_map.get(str(fsm.status_code), "notstarted")
                t_name = parts[1].strip() if len(parts) > 1 else fsm.tournament_title
                home_id = abs(hash(fsm.home_team_name)) % (10**7)
                away_id = abs(hash(fsm.away_team_name)) % (10**7)

                m_obj = MatchDetailSchema(
                    event_id=numeric_id,
                    custom_id=fsm.flashscore_id,
                    slug=f"{fsm.home_team_name.lower().replace(' ', '-')}-vs-{fsm.away_team_name.lower().replace(' ', '-')}",
                    start_timestamp=fsm.start_timestamp,
                    start_time_utc=fsm.start_time_utc,
                    status_type=status_type,
                    tournament=TournamentSchema(
                        id=abs(hash(t_name)) % (10**6),
                        name=t_name,
                        country=raw_country,
                    ),
                    home_team=TeamSchema(id=home_id, name=fsm.home_team_name, country=raw_country),
                    away_team=TeamSchema(id=away_id, name=fsm.away_team_name, country=raw_country),
                    home_score=fsm.home_score,
                    away_score=fsm.away_score,
                )
                matches.append(m_obj)
            except Exception as e_fsm:
                logger.debug(f"Error parseando partido fallback de Flashscore: {e_fsm}")

    fs_lookup: Dict[tuple[int, int], str] = {}
    entity_links: List[Dict[str, Any]] = []

    if source_label == "Flashscore Cloud":
        # Optimización Ultra-Rápida: Mapeo directo 1:1 sin sobrecarga de EntityResolver
        for m in matches:
            if m.custom_id:
                fs_lookup[(m.home_team.id, m.away_team.id)] = m.custom_id
    else:
        # Cruce fuzzy Sofascore <-> Flashscore con registro canónico
        canonical_teams = []
        for m in matches:
            canonical_teams.append((m.home_team.id, m.home_team.name))
            canonical_teams.append((m.away_team.id, m.away_team.name))
        resolver.register_canonical_teams(canonical_teams)

        for fsm in fs_matches:
            rh = resolver.resolve(fsm.home_team_name)
            ra = resolver.resolve(fsm.away_team_name)
            if rh and rh.is_confident and ra and ra.is_confident:
                fs_lookup[(rh.canonical_id, ra.canonical_id)] = fsm.flashscore_id
                save_team_alias(
                    "flashscore",
                    fsm.home_team_id,
                    fsm.home_team_name,
                    rh.canonical_id,
                    rh.canonical_name,
                    rh.similarity_score,
                )
                save_team_alias(
                    "flashscore",
                    fsm.away_team_id,
                    fsm.away_team_name,
                    ra.canonical_id,
                    ra.canonical_name,
                    ra.similarity_score,
                )
                entity_links.append(
                    {
                        "Torneo Flashscore": fsm.tournament_title,
                        "Flashscore ID": fsm.flashscore_id,
                        "Local (Flashscore)": fsm.home_team_name,
                        "Local (Sofascore)": rh.canonical_name,
                        "Similitud Local (%)": rh.similarity_score,
                        "Visita (Flashscore)": fsm.away_team_name,
                        "Visita (Sofascore)": ra.canonical_name,
                        "Similitud Visita (%)": ra.similarity_score,
                    }
                )

    return {
        "matches": matches,
        "fs_total": len(fs_matches),
        "fs_lookup": fs_lookup,
        "entity_links": entity_links,
        "source": source_label,
    }


@st.cache_data(ttl=900, show_spinner=False)
def analyze_single_match_deep(match_dict: Dict[str, Any], fs_id: str) -> Dict[str, Any]:
    """
    Extrae estadísticas profundas bajo demanda:
    - Árbitro, cuotas de apertura vs actuales, gráfico de presión (Momentum)
    - Historial separando Local vs Visitante + cálculo de Días de Descanso / Fatiga
    - Jugadores líderes en remates al arco, totales, goles y bajas
    - Predicción híbrida (Dixon-Coles + Monte Carlo 10k + Gradient Boosting ML)
    """
    sf_scraper, _ = get_scrapers()
    match = MatchDetailSchema.model_validate(match_dict)

    sf_scraper.enrich_match(
        match,
        include_stats=True,
        include_odds=True,
        include_h2h=True,
        include_lineups=True,
        include_streaks=True,
        include_referee=True,
    )

    ut_id = match.tournament.unique_id or match.tournament.id
    s_id = match.season_id or 0
    standings_map = (
        sf_scraper.get_tournament_standings_map(ut_id, s_id)
        if (ut_id and s_id)
        else {}
    )

    # Peticiones concurrentes del mismo partido (Local y Visitante en paralelo)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as inner_exec:
        fut_home_rec = inner_exec.submit(
            sf_scraper.get_team_recent_matches, match.home_team.id, 6, True, 5
        )
        fut_away_rec = inner_exec.submit(
            sf_scraper.get_team_recent_matches, match.away_team.id, 6, True, 5
        )
        home_recent = fut_home_rec.result()
        away_recent = fut_away_rec.result()

    home_form = compute_team_form(
        match.home_team.id,
        match.home_team.name,
        home_recent,
        standing_info=standings_map.get(match.home_team.id),
        target_venue="home",
        reference_timestamp=match.start_timestamp,
        standings_map=standings_map,
    )
    away_form = compute_team_form(
        match.away_team.id,
        match.away_team.name,
        away_recent,
        standing_info=standings_map.get(match.away_team.id),
        target_venue="away",
        reference_timestamp=match.start_timestamp,
        standings_map=standings_map,
    )

    home_missing = match.home_lineup.missing_players if match.home_lineup else []
    away_missing = match.away_lineup.missing_players if match.away_lineup else []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as inner_exec:
        fut_home_pl = inner_exec.submit(
            sf_scraper.get_team_top_players,
            match.home_team.id,
            match.home_team.name,
            ut_id,
            s_id,
            home_missing,
            match.home_lineup,
            away_form.shots_on_target_against_avg,
        )
        fut_away_pl = inner_exec.submit(
            sf_scraper.get_team_top_players,
            match.away_team.id,
            match.away_team.name,
            ut_id,
            s_id,
            away_missing,
            match.away_lineup,
            home_form.shots_on_target_against_avg,
        )
        home_top_players = fut_home_pl.result()
        away_top_players = fut_away_pl.result()

    prediction = predict_match_poisson(
        match,
        home_form,
        away_form,
        home_players=home_top_players,
        away_players=away_top_players,
    )

    upsert_match(match)

    try:
        if "audited_predictions_cache" in st.session_state and match.status_type == "finished" and match.home_score is not None:
            st.session_state["audited_predictions_cache"][match.event_id] = prediction
            if "audited_matches_cache" not in st.session_state:
                st.session_state["audited_matches_cache"] = {}
            st.session_state["audited_matches_cache"][match.event_id] = match
    except Exception:
        pass

    return {
        "match": match,
        "home_form": home_form,
        "away_form": away_form,
        "home_recent": home_recent,
        "away_recent": away_recent,
        "prediction": prediction,
        "home_players": home_top_players,
        "away_players": away_top_players,
        "standings_map": standings_map,
        "flashscore_id": fs_id,
    }


def analyze_matches_batch_parallel(
    match_pairs: List[Any],
    progress_bar=None,
    status_text_prefix: str = "Procesando",
    max_workers: int = MAX_PARALLEL_WORKERS,
) -> List[Dict[str, Any]]:
    """
    Ejecuta el análisis profundo (Opta + H2H + Jugadores + ML) de múltiples partidos
    en paralelo utilizando un pool de hilos de alto rendimiento.
    """
    if not match_pairs:
        return []

    results_map: Dict[int, Dict[str, Any]] = {}
    total = len(match_pairs)
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, item in enumerate(match_pairs):
            if isinstance(item, (tuple, list)):
                m_obj, fs_code = item[0], item[1]
            else:
                m_obj, fs_code = item, "N/A"
            m_dict = m_obj.model_dump() if hasattr(m_obj, "model_dump") else m_obj
            future_to_idx[executor.submit(analyze_single_match_deep, m_dict, fs_code)] = idx

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed_count += 1
            item = match_pairs[idx]
            m_item = item[0] if isinstance(item, (tuple, list)) else item
            
            h_name = getattr(getattr(m_item, 'home_team', None), 'name', 'Local')
            a_name = getattr(getattr(m_item, 'away_team', None), 'name', 'Visitante')
            team_label = f"{h_name} vs {a_name}"
            
            if progress_bar:
                try:
                    progress_bar.progress(
                        completed_count / total,
                        text=f"{status_text_prefix} ({completed_count}/{total}): {team_label}",
                    )
                except Exception:
                    pass

            try:
                res = future.result()
                results_map[idx] = res
            except Exception as e:
                logger.error(f"Error procesando partido {team_label} en paralelo: {e}")

    return [results_map[i] for i in range(total) if i in results_map]


def render_stat_comparison_bar(label: str, val_home: float, val_away: float, unit: str = "") -> None:
    total = max(0.01, val_home + val_away)
    pct_home = min(92.0, max(8.0, (val_home / total) * 100.0))
    html = (
        f"<div class='stat-compare-box notranslate' translate='no'>"
        f"<div style='display:flex; justify-content:space-between; font-size:0.88rem; font-weight:700; margin-bottom:5px;'>"
        f"<span style='color:#38bdf8;'>{val_home}{unit}</span>"
        f"<span style='color:#cbd5e1;'>{label}</span>"
        f"<span style='color:#f43f5e;'>{val_away}{unit}</span>"
        f"</div>"
        f"<div style='width:100%; height:7px; background:#334155; border-radius:99px; display:flex; overflow:hidden;'>"
        f"<div style='width:{pct_home:.1f}%; background:#38bdf8; height:100%;'></div>"
        f"<div style='width:{100.0-pct_home:.1f}%; background:#f43f5e; height:100%;'></div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    if "bet_slip" not in st.session_state:
        st.session_state["bet_slip"] = []
    if "audited_predictions_cache" not in st.session_state:
        st.session_state["audited_predictions_cache"] = {}
    if "audited_matches_cache" not in st.session_state:
        st.session_state["audited_matches_cache"] = {}

    with st.sidebar:
        st.markdown("## ⚽ ProSports AI v3")
        st.caption("Live Momentum • Monte Carlo 10k • Hándicaps • Fatiga • Ensamble ML • Kelly")
        st.divider()

        selected_date = st.date_input("📅 1. Selecciona la Fecha", value=date.today())
        include_all = st.toggle(
            "🌍 Cargar ligas globales adicionales",
            value=False,
            help="Por defecto carga el catálogo Élite: 12 Ligas Top (Francia, Inglaterra, España, Italia, Alemania, Colombia, MLS, Argentina, Arabia Saudita, Turquía, Brasil, Portugal), sus Copas Nacionales, Torneos Continentales y Selecciones. Si lo activas, incluye ligas menores de todo el mundo.",
        )

        date_str = selected_date.strftime("%Y-%m-%d")
        catalog = load_day_catalog(date_str, include_all)
        all_matches: List[MatchDetailSchema] = catalog["matches"]

        # Población automática de caché si proviene de Supabase precalculado (Ultra Rápido)
        if "precomputed_predictions" in catalog:
            st.session_state["audited_predictions_cache"].update(catalog["precomputed_predictions"])
        if "precomputed_deep_cache" in catalog:
            if "deep_analysis_cache" not in st.session_state:
                st.session_state["deep_analysis_cache"] = {}
            st.session_state["deep_analysis_cache"].update(catalog["precomputed_deep_cache"])
        for m in all_matches:
            st.session_state["audited_matches_cache"][m.event_id] = m

        tournaments_available = sorted(
            list({f"{m.tournament.name} ({m.tournament.country or 'Int'})" for m in all_matches})
        )
        selected_league = st.selectbox(
            "🏆 Filtrar por Torneo / Liga",
            options=["Todas las Ligas (" + str(len(all_matches)) + " partidos)"] + tournaments_available,
        )
        search_team = st.text_input("🔎 Buscar equipo", placeholder="Ej. Barcelona, Bayern, River...")

        filtered_matches = []
        for m in all_matches:
            t_label = f"{m.tournament.name} ({m.tournament.country or 'Int'})"
            if not selected_league.startswith("Todas las Ligas") and t_label != selected_league:
                continue
            if search_team:
                q = search_team.lower().strip()
                if q not in m.home_team.name.lower() and q not in m.away_team.name.lower():
                    continue
            filtered_matches.append(m)

        filter_finished = st.checkbox("🏁 Ver solo partidos finalizados (con aciertos/fallos auditados)", value=False)
        if filter_finished:
            filtered_matches = [m for m in filtered_matches if m.status_type == "finished"]

        st.markdown(f"**Partidos filtrados:** `{len(filtered_matches)}`")

        if filtered_matches:
            match_options = []
            for m in filtered_matches:
                if m.status_type == "finished":
                    st_tag = f"[{m.home_score}-{m.away_score} FIN]"
                elif m.status_type == "inprogress":
                    st_tag = f"[{m.home_score}-{m.away_score} VIVO]"
                else:
                    col_time = format_colombia_time(m.start_timestamp or m.start_time_utc, "%H:%M")
                    st_tag = f"[{col_time} COL]"
                match_options.append(
                    f"{st_tag} | {m.home_team.name} vs {m.away_team.name} ({m.tournament.name})"
                )

            selected_match_idx = st.selectbox(
                "🎯 2. Selecciona un Partido para Analizar",
                options=range(len(match_options)),
                format_func=lambda idx: match_options[idx],
            )
        else:
            selected_match_idx = None

        st.divider()
        st.markdown(f"🎟️ **Selecciones en tu Boleto:** `{len(st.session_state['bet_slip'])}`")
        if st.session_state["bet_slip"]:
            if st.button("🗑️ Vaciar Boleto de Combinada", use_container_width=True):
                st.session_state["bet_slip"] = []
                st.rerun()

        st.divider()
        from src.config import DATABASE_URL
        db_badge = "🟢 Supabase (PostgreSQL)" if DATABASE_URL else "🟡 SQLite Local"
        st.markdown(f"💾 **Base de Datos:** `{db_badge}`")
        src_badge = catalog.get("source", "Sofascore Opta")
        st.markdown(f"📡 **Fuente de Datos:** `{src_badge}`")
        st.markdown(f"📊 **Partidos Disponibles:** `{len(all_matches)}`")

        if st.button("🔄 Forzar Recarga (Limpiar Caché)", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # SELECTOR DE MODO SUPERIOR (6 Modos con Auditoría Destacada)
    top_col1, top_col2 = st.columns([1.1, 2.9])
    with top_col1:
        st.markdown(f"### 📅 Jornada `{date_str}` ({len(filtered_matches)} P)")
    with top_col2:
        view_mode = st.radio(
            "Modo de visualización:",
            options=[
                "🎯 Análisis del Partido",
                "📊 Auditoría de Pronósticos (Aciertos/Fallos)",
                "🧠 Auto-Aprendizaje e IA Adaptativa",
                "💼 Asignador Maestro de Capital & Portafolio",
                "🔴 En Vivo & Momentum",
                "🔥 Radar Top Picks",
                "📋 Catálogo Completo",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )

    # MODO DEDICADO: AUDITORÍA DE PRONÓSTICOS (PROMEDIOS, GRÁFICA Y RANKING DE LOS 10 MERCADOS)
    if view_mode == "📊 Auditoría de Pronósticos (Aciertos/Fallos)":
        st.subheader("📊 Auditoría de Pronósticos: Promedios de Acierto, Gráfica y Ranking de Mercados y Líneas")
        st.caption(
            "Este módulo audita automáticamente los pronósticos generados por el modelo contra los marcadores y estadísticas oficiales "
            "de partidos finalizados para responder exactamente **cuántos aciertos tuvo, cuál es el promedio general y cuáles son las líneas (+1.5, +2.5 goles, córners, atajadas) que más acierta y las que más falla**."
        )

        st.markdown("### 🎛️ Alcance y Selección de Partidos a Auditar")
        scope_col1, scope_col2 = st.columns([1.6, 2.4])
        with scope_col1:
            scope_mode = st.radio(
                "📅 Selecciona el rango temporal de partidos finalizados:",
                [
                    "📅 Partidos Finalizados de Hoy",
                    "📅 Últimos 3 Días (Hoy, Ayer y Antier)",
                    "📅 Última Semana Completa (7 Días)",
                ],
                horizontal=False,
            )

        days_to_check = 1
        if "3 Días" in scope_mode:
            days_to_check = 3
        elif "Semana" in scope_mode:
            days_to_check = 7

        base_date = selected_date
        scope_finished_matches: List[tuple[MatchDetailSchema, str]] = []
        seen_events = set()

        for d_offset in range(days_to_check):
            d_target = base_date - timedelta(days=d_offset)
            d_target_str = d_target.strftime("%Y-%m-%d")
            day_cat = catalog if (d_target_str == date_str) else load_day_catalog(d_target_str, False)
            for m_item in day_cat["matches"]:
                if (
                    m_item.status_type == "finished"
                    and m_item.home_score is not None
                    and m_item.event_id not in seen_events
                ):
                    seen_events.add(m_item.event_id)
                    fs_code = day_cat["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A")
                    scope_finished_matches.append((m_item, fs_code))

        if not scope_finished_matches:
            st.warning("No se encontraron partidos finalizados en el período seleccionado. Cambia la fecha o el rango temporal.")
            return

        with scope_col2:
            st.markdown(f"**Partidos finalizados detectados en este período:** `{len(scope_finished_matches)}`")
            audit_slider_val = st.slider(
                "O selecciona una cantidad específica a auditar:",
                min_value=1,
                max_value=max(1, len(scope_finished_matches)),
                value=min(20, len(scope_finished_matches)),
            )

        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1.2, 0.9, 1.6])
        with btn_col1:
            btn_audit_all = st.button(
                f"🚀 Auditar TODOS los Partidos ({len(scope_finished_matches)} partidos)",
                type="primary",
                use_container_width=True,
            )
        with btn_col2:
            btn_audit_slider = st.button(
                f"⚡ Auditar Selección ({audit_slider_val} partidos)",
                use_container_width=True,
            )
        with btn_col3:
            btn_clear_audit = st.button(
                "🗑️ Limpiar",
                use_container_width=True,
                help="Reinicia la memoria de auditoría",
            )
        with btn_col4:
            btn_feed_brain = st.button(
                "🧠 Entrenar IA con esto",
                use_container_width=True,
                help="Asimila estos partidos auditados en el cerebro de la IA para actualizar pesos y calibración",
            )

        if btn_feed_brain:
            audited_pairs_to_feed = []
            for m_id, pred_item in audited_cache.items():
                if m_id in matches_cache:
                    audited_pairs_to_feed.append((matches_cache[m_id], pred_item))
            if audited_pairs_to_feed:
                learner_inst = get_auto_learner()
                rep = learner_inst.learn_from_audited_matches(
                    audited_pairs_to_feed, evaluate_all_systematic_lines_for_match
                )
                st.success(
                    f"🧠 ¡Cerebro de la IA actualizado! Asimiló {len(audited_pairs_to_feed)} partidos. "
                    f"Nuevos pesos: Dixon-Coles {rep['new_weights']['dixon']*100:.1f}% | ML {rep['new_weights']['ml']*100:.1f}%."
                )
                st.rerun()
            else:
                st.warning("Primero audita partidos finalizados para poder alimentar a la IA.")

        if btn_clear_audit:
            st.session_state["audited_predictions_cache"] = {}
            st.session_state["audited_matches_cache"] = {}
            st.rerun()

        audited_cache = st.session_state.get("audited_predictions_cache", {})
        matches_cache = st.session_state.get("audited_matches_cache", {})

        # Auto-auditar los primeros 8-10 partidos para dar visualización inmediata
        if len(audited_cache) == 0 and scope_finished_matches:
            auto_sample = scope_finished_matches[: min(10, len(scope_finished_matches))]
            with st.spinner(f"⚡ Procesando auditoría inicial de {len(auto_sample)} partidos en paralelo..."):
                batch_res = analyze_matches_batch_parallel(auto_sample, max_workers=MAX_PARALLEL_WORKERS)
                for res_auto in batch_res:
                    m_val = res_auto["match"]
                    st.session_state["audited_predictions_cache"][m_val.event_id] = res_auto["prediction"]
                    st.session_state["audited_matches_cache"][m_val.event_id] = m_val
            audited_cache = st.session_state["audited_predictions_cache"]
            matches_cache = st.session_state["audited_matches_cache"]

        if btn_audit_all or btn_audit_slider:
            target_list = scope_finished_matches if btn_audit_all else scope_finished_matches[:audit_slider_val]
            prog = st.progress(0, text="Iniciando auditoría paralela de alta velocidad...")
            batch_res = analyze_matches_batch_parallel(
                target_list,
                progress_bar=prog,
                status_text_prefix="Auditando en paralelo",
                max_workers=MAX_PARALLEL_WORKERS,
            )
            for res_deep in batch_res:
                m_val = res_deep["match"]
                st.session_state["audited_predictions_cache"][m_val.event_id] = res_deep["prediction"]
                st.session_state["audited_matches_cache"][m_val.event_id] = m_val
            prog.empty()
            st.success(f"✅ Se auditaron exitosamente {len(batch_res)} partidos finalizados a máxima velocidad.")
            st.rerun()

        all_audited_preds = list(st.session_state.get("audited_predictions_cache", {}).values())
        all_audited_matches = st.session_state.get("audited_matches_cache", {})

        if not all_audited_preds:
            st.warning("No hay partidos auditados aún. Pulsa 'Auditar TODOS' para comenzar.")
            return

        breakdown = compute_market_accuracy_breakdown(all_audited_preds, matches_map=all_audited_matches)

        # Selector de Vista: Líneas Específicas vs 10 Categorías
        tab_spec_lines, tab_general_cats = st.tabs(
            [
                "🎯 Líneas Específicas (+1.5, +2.5 Goles, Córners, Atajadas, etc.)",
                "📁 Resumen por 10 Categorías Generales",
            ]
        )

        with tab_spec_lines:
            st.markdown("#### 🎯 Tasa de Acierto y Rendimiento por Línea Específica")
            st.caption(
                "Consolidación global de todas las líneas concretas: evalúa cuántas veces acertó en **+1.5 goles**, **+2.5 goles**, "
                "**+6.5, +7.5, +8.5 córners**, **atajadas**, etc."
            )

            audit_mode_view = st.radio(
                "📐 Modalidad de Auditoría de Líneas:",
                [
                    "🌐 Auditoría Exhaustiva de TODAS las Líneas en CADA Partido (Recomendada: Muestra Completa de +1.5, +2.5, Córners, SoT, etc.)",
                    "🎯 Solo Pronósticos Recomendados por el Modelo (Las 10 Tarjetas Seleccionadas)",
                ],
                horizontal=False,
            )

            is_exhaustive = audit_mode_view.startswith("🌐")
            if is_exhaustive and breakdown.get("exhaustive_lines_stats"):
                active_lines_dataset = breakdown["exhaustive_lines_stats"]
                active_win_rate = breakdown["exhaustive_win_rate"]
                active_title = f"🌐 Efectividad Exhaustiva de TODAS las Líneas ({breakdown['matches_audited']} Partidos Auditados • {breakdown['exhaustive_hits']}/{breakdown['exhaustive_total']} Aciertos Totales)"
                top_line_disp = breakdown.get("best_exhaustive_line")
                worst_line_disp = breakdown.get("worst_exhaustive_line")
                disp_total_evals = breakdown["exhaustive_total"]
            else:
                active_lines_dataset = breakdown["specific_lines_stats"]
                active_win_rate = breakdown["overall_win_rate"]
                active_title = f"🎯 Efectividad de Líneas Recomendadas por el Modelo ({breakdown['matches_audited']} Partidos Auditados • {breakdown['overall_hits']}/{breakdown['overall_total']} Aciertos Totales)"
                top_line_disp = breakdown.get("best_line")
                worst_line_disp = breakdown.get("worst_line")
                disp_total_evals = breakdown["overall_total"]

            # TARJETAS KPI RESUMEN DE LA MODALIDAD SELECCIONADA
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "🎯 % Acierto Promedio",
                f"{active_win_rate:.1f}%",
                f"{breakdown['exhaustive_hits'] if is_exhaustive else breakdown['overall_hits']} aciertos de {disp_total_evals}",
            )
            top_wr = f"{top_line_disp['win_rate']:.1f}%" if top_line_disp else "-"
            top_name = top_line_disp["line_name"] if top_line_disp else "N/A"
            k2.metric("🏆 Línea Estrella Más Certera", top_wr, top_name)

            worst_wr = f"{worst_line_disp['win_rate']:.1f}%" if worst_line_disp else "-"
            worst_name = worst_line_disp["line_name"] if worst_line_disp else "N/A"
            k3.metric("⚠️ Línea de Mayor Riesgo", worst_wr, worst_name)

            k4.metric(
                "🏟️ Partidos Auditados",
                str(breakdown["matches_audited"]),
                f"{disp_total_evals} líneas evaluadas",
            )

            st.markdown("---")

            # Gráfica Plotly de Líneas
            st.plotly_chart(
                plot_specific_lines_chart(
                    active_lines_dataset,
                    active_win_rate,
                    active_title,
                ),
                use_container_width=True,
            )

            # Cuadros de Diagnóstico: Líneas que Más Acierta vs Líneas que Más Falla
            ld_col1, ld_col2 = st.columns(2)

            active_evaluated = [s for s in active_lines_dataset if s["total"] > 0]
            top_spec = active_evaluated[:5] if len(active_evaluated) >= 5 else active_evaluated
            bottom_spec = list(reversed(active_evaluated[-5:])) if len(active_evaluated) >= 5 else []

            with ld_col1:
                st.markdown("#### 🟢 Líneas Específicas con Mayor Tasa de Acierto (Alta Confianza)")
                st.caption("Líneas con porcentaje de éxito sobresaliente. Úsalas como base para asegurar tus apuestas:")
                for sl in top_spec:
                    st.markdown(
                        f"<div style='background:#064e3b; border-left:4px solid #10b981; padding:10px 14px; border-radius:6px; margin-bottom:8px;'>"
                        f"<div style='font-weight:700; color:#ffffff;'>{sl['line_name']}</div>"
                        f"<div style='color:#34d399; font-weight:800; font-size:1.05rem;'>{sl['win_rate']:.1f}% de Acierto ({sl['hits']}/{sl['total']} Aciertos)</div>"
                        f"<div style='font-size:0.80rem; color:#a7f3d0;'>{sl['status_tag']} • Prob. promedio: {sl['avg_prob']:.1f}% • Cuota justa: {round(100.0/max(1.0, sl['avg_prob']), 2):.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            with ld_col2:
                st.markdown("#### 🔴 Líneas Específicas que MÁS FALLA (Alto Riesgo)")
                st.caption("Líneas donde hay menor frecuencia o mayor varianza táctica. Se recomienda evitarlas o buscar líneas alternativas:")
                for bl in bottom_spec:
                    st.markdown(
                        f"<div style='background:#4c0519; border-left:4px solid #f43f5e; padding:10px 14px; border-radius:6px; margin-bottom:8px;'>"
                        f"<div style='font-weight:700; color:#ffffff;'>{bl['line_name']}</div>"
                        f"<div style='color:#fda4af; font-weight:800; font-size:1.05rem;'>{bl['win_rate']:.1f}% de Acierto ({bl['misses']} fallos de {bl['total']})</div>"
                        f"<div style='font-size:0.80rem; color:#fecdd3;'>{bl['status_tag']} • Alta volatilidad</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown("#### 📋 Tabla Resumen de Todas las Líneas Evaluadas")
            line_table_rows = []
            for sl in active_lines_dataset:
                if sl["total"] == 0:
                    continue
                line_table_rows.append({
                    "Línea de Apuesta": sl["line_name"],
                    "Categoría": sl["category"],
                    "Partidos Evaluados": sl["total"],
                    "Aciertos (✅)": sl["hits"],
                    "Fallos (❌)": sl["misses"],
                    "% Acierto Real": f"{sl['win_rate']:.1f}%",
                    "Prob. Modelo": f"{sl['avg_prob']:.1f}%",
                    "Cuota Justa": f"{round(100.0 / max(1.0, sl['avg_prob']), 2):.2f}",
                    "Veredicto": sl["badge"],
                })
            if line_table_rows:
                st.dataframe(pd.DataFrame(line_table_rows), use_container_width=True, hide_index=True)

            st.markdown("#### 🔍 Desglose de Partidos por Línea")
            for sl in active_lines_dataset:
                if sl["total"] == 0:
                    continue
                with st.expander(f"Ver partidos de: {sl['line_name']} ({sl['hits']}/{sl['total']} aciertos — {sl['win_rate']:.1f}%)"):
                    pick_rows = []
                    for pk_item in sl["picks"]:
                        pick_rows.append({
                            "Partido": pk_item["match"],
                            "Pronóstico / Línea": pk_item["pick"],
                            "Probabilidad Modelo": f"{pk_item['prob']:.1f}%",
                            "Dato Real": pk_item.get("actual") or "-",
                            "Dictamen": "✅ ACIERTO" if pk_item["hit"] is True else "❌ FALLO",
                        })
                    st.dataframe(pd.DataFrame(pick_rows), use_container_width=True, hide_index=True)

        with tab_general_cats:
            st.markdown("#### 📁 Resumen por 10 Categorías Principales")
            st.plotly_chart(plot_market_accuracy_chart(breakdown), use_container_width=True)

            d_col1, d_col2 = st.columns(2)
            active_stats = [s for s in breakdown["category_stats"] if s["total"] > 0]
            top_cats = active_stats[:4] if len(active_stats) >= 4 else active_stats
            bottom_cats = list(reversed(active_stats[-4:])) if len(active_stats) >= 4 else []

            with d_col1:
                st.markdown("#### 🟢 Categorías que MÁS ACIERTA")
                for s in top_cats:
                    st.markdown(
                        f"<div style='background:#064e3b; border-left:4px solid #10b981; padding:10px 14px; border-radius:6px; margin-bottom:8px;'>"
                        f"<div style='font-weight:700; color:#ffffff;'>{s['category']}</div>"
                        f"<div style='color:#34d399; font-weight:800; font-size:1.05rem;'>{s['win_rate']:.1f}% de Acierto ({s['hits']}/{s['total']})</div>"
                        f"<div style='font-size:0.80rem; color:#a7f3d0;'>{s['status_tag']} • Probabilidad promedio: {s['avg_prob']:.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            with d_col2:
                st.markdown("#### 🔴 Categorías que MÁS FALLA")
                for s in bottom_cats:
                    st.markdown(
                        f"<div style='background:#4c0519; border-left:4px solid #f43f5e; padding:10px 14px; border-radius:6px; margin-bottom:8px;'>"
                        f"<div style='font-weight:700; color:#ffffff;'>{s['category']}</div>"
                        f"<div style='color:#fda4af; font-weight:800; font-size:1.05rem;'>{s['win_rate']:.1f}% de Acierto ({s['misses']} fallos de {s['total']})</div>"
                        f"<div style='font-size:0.80rem; color:#fecdd3;'>{s['status_tag']} • Volatilidad táctica</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            table_rows = []
            for s in breakdown["category_stats"]:
                if s["total"] == 0:
                    continue
                table_rows.append({
                    "Mercado": s["category"],
                    "Evaluados": s["total"],
                    "Aciertos (✅)": s["hits"],
                    "Fallos (❌)": s["misses"],
                    "% Acierto": f"{s['win_rate']:.1f}%",
                    "Prob. Media": f"{s['avg_prob']:.1f}%",
                    "Cuota Justa": f"{round(100.0 / max(1.0, s['avg_prob']), 2):.2f}",
                    "Veredicto": s["badge"],
                })
            if table_rows:
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        return

    # MODO DEDICADO: AUTO-APRENDIZAJE E INTELIGENCIA ADAPTATIVA
    if view_mode == "🧠 Auto-Aprendizaje e IA Adaptativa":
        st.subheader("🧠 Centro de Auto-Aprendizaje y Calibración Continua de la IA")
        st.caption(
            "Este motor permite que el sistema **aprenda autónomamente de sus errores y aciertos en partidos finalizados**: "
            "ajusta los pesos del ensamble (Dixon-Coles vs Machine Learning), calibra las 28 líneas de apuesta para eliminar la sobreconfianza, "
            "optimiza las prioridades del Bandido Thompson Sampling para selecciones de alta confianza y reentrena el modelo Gradient Boosting con nuevos registros en la base de datos."
        )

        learner = get_auto_learner()
        state = learner.state
        w_dixon, w_ml = learner.get_ensemble_weights()
        b_d = state.get("recent_brier_scores", {}).get("dixon_coles_1x2", 0.20)
        b_m = state.get("recent_brier_scores", {}).get("ml_1x2", 0.21)
        tot_learned = state.get("total_matches_learned", 0)
        sessions_cnt = state.get("learned_sessions_count", 0)

        # 1. TARJETAS KPI DE ESTADO DEL CEREBRO
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "⚖️ Peso Dixon-Coles (Poisson)",
            f"{w_dixon * 100:.1f}%",
            f"{(w_dixon - 0.65) * 100:+.1f}% vs base",
        )
        k2.metric(
            "🤖 Peso Machine Learning (ML)",
            f"{w_ml * 100:.1f}%",
            f"{(w_ml - 0.35) * 100:+.1f}% vs base",
        )
        better_engine = "🏆 ML Menor Error" if b_m <= b_d else "🏆 Dixon Menor Error"
        k3.metric(
            "🎯 Brier Score (Error 1X2)",
            f"{b_m:.4f} (ML)",
            f"Dixon: {b_d:.4f} • {better_engine}",
        )
        k4.metric(
            "🏟️ Partidos en Memoria de IA",
            str(tot_learned),
            f"{sessions_cnt} ciclos de aprendizaje",
        )

        # Barra visual de balance del ensamble
        st.markdown(
            f"""
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px 16px; margin:10px 0 18px 0;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:700; margin-bottom:6px;">
                    <span style="color:#38bdf8;">🔵 Dixon-Coles / Opta: {w_dixon * 100:.1f}%</span>
                    <span style="color:#94a3b8;">⚖️ Balance Dinámico del Ensamble Híbrido</span>
                    <span style="color:#10b981;">🟢 Machine Learning (Gradient Boosting): {w_ml * 100:.1f}%</span>
                </div>
                <div style="width:100%; height:10px; background:#1e293b; border-radius:99px; display:flex; overflow:hidden;">
                    <div style="width:{w_dixon * 100:.1f}%; background:linear-gradient(90deg, #0284c7, #38bdf8); height:100%;"></div>
                    <div style="width:{w_ml * 100:.1f}%; background:linear-gradient(90deg, #10b981, #34d399); height:100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. CONTROL PARA EJECUTAR CICLO DE AUTO-APRENDIZAJE
        st.markdown("### 🚀 Ejecutar Nuevo Ciclo de Aprendizaje")
        l_col1, l_col2, l_col3 = st.columns([1.6, 1.8, 1.0])
        with l_col1:
            learn_scope = st.radio(
                "Selecciona los partidos finalizados a asimilar:",
                [
                    "📅 Partidos Finalizados de Hoy",
                    "📅 Últimos 3 Días (Hoy, Ayer y Antier)",
                    "📅 Última Semana (7 Días)",
                ],
                horizontal=False,
            )
        with l_col2:
            st.markdown(
                "La IA comparará cada pronóstico contra el marcador y estadísticas oficiales reales, "
                "actualizará los multiplicadores de calibración de las 28 líneas, ajustará los pesos de ensamble según Brier Score "
                "y reentrenará el modelo ML en SQLite."
            )
            btn_learn_now = st.button("🧠 Iniciar Ciclo de Auto-Aprendizaje", type="primary", use_container_width=True)
        with l_col3:
            st.write("")
            btn_reset_learner = st.button("🔄 Restablecer", use_container_width=True, help="Restablece los pesos a su valor de fábrica inicial")

        if btn_reset_learner:
            learner.reset_learning()
            st.success("✅ El cerebro de la IA ha sido restablecido a la configuración inicial.")
            st.rerun()

        if btn_learn_now:
            days_back = 1
            if "3 Días" in learn_scope:
                days_back = 3
            elif "Semana" in learn_scope:
                days_back = 7

            base_dt = selected_date
            scope_matches_to_learn: List[tuple[MatchDetailSchema, str]] = []
            seen_ids = set()

            for d_off in range(days_back):
                target_dt = base_dt - timedelta(days=d_off)
                target_str = target_dt.strftime("%Y-%m-%d")
                cat_d = catalog if (target_str == date_str) else load_day_catalog(target_str, False)
                for m_item in cat_d["matches"]:
                    if (
                        m_item.status_type == "finished"
                        and m_item.home_score is not None
                        and m_item.event_id not in seen_ids
                    ):
                        seen_ids.add(m_item.event_id)
                        fs_code = cat_d["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A")
                        scope_matches_to_learn.append((m_item, fs_code))

            if not scope_matches_to_learn:
                st.warning("No se encontraron partidos finalizados en este período para entrenar.")
                return

            prog_learn = st.progress(0, text="Extrayendo estadísticas y analizando partidos en paralelo...")
            audited_pairs_for_learning = []

            batch_res = analyze_matches_batch_parallel(
                scope_matches_to_learn,
                progress_bar=prog_learn,
                status_text_prefix="Analizando para IA",
                max_workers=MAX_PARALLEL_WORKERS,
            )
            for res_deep in batch_res:
                m_val = res_deep["match"]
                st.session_state["audited_predictions_cache"][m_val.event_id] = res_deep["prediction"]
                st.session_state["audited_matches_cache"][m_val.event_id] = m_val
                audited_pairs_for_learning.append((m_val, res_deep["prediction"]))

            prog_learn.progress(1.0, text="Calibrando probabilidades y reentrenando Gradient Boosting...")
            report = learner.learn_from_audited_matches(
                audited_pairs_for_learning, evaluate_all_systematic_lines_for_match
            )
            prog_learn.empty()

            st.session_state["last_learning_report"] = report
            st.success(
                f"🎉 ¡Ciclo de aprendizaje completado con éxito! Se procesaron {report['matches_processed']} partidos. "
                f"Pesos actualizados: Dixon-Coles {report['new_weights']['dixon']*100:.1f}% | ML {report['new_weights']['ml']*100:.1f}% "
                f"(Variación ML: {report['delta_ml_pct']:+.1f}%)."
            )
            st.rerun()

        # REPORTE DE LA ÚLTIMA SESIÓN SI EXISTE
        last_rep = st.session_state.get("last_learning_report")
        if last_rep:
            st.markdown("#### 📋 Reporte del Último Ciclo de Auto-Entrenamiento")
            rep_c1, rep_c2 = st.columns(2)
            with rep_c1:
                st.markdown("##### 🟢 Líneas Promovidas a 'Estrella' por la IA (Alta Efectividad Real)")
                for item in last_rep.get("top_promoted_lines", []):
                    st.markdown(
                        f"<div style='background:#064e3b; border-left:4px solid #10b981; padding:8px 12px; border-radius:6px; margin-bottom:6px; font-size:0.85rem;'>"
                        f"<b>{item['line_name']}</b><br>"
                        f"<span style='color:#34d399;'>Acierto acumulado: <b>{item['cum_win_rate']:.1f}%</b></span> • Multiplicador: <b>x{item['multiplier']:.3f}</b> • {item['status']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            with rep_c2:
                st.markdown("##### 🔴 Líneas Atenuadas / Penalizadas por Varianza")
                pen_list = last_rep.get("top_penalized_lines", [])
                if not pen_list:
                    st.info("No se registraron líneas penalizadas en este ciclo.")
                for item in pen_list:
                    st.markdown(
                        f"<div style='background:#4c0519; border-left:4px solid #f43f5e; padding:8px 12px; border-radius:6px; margin-bottom:6px; font-size:0.85rem;'>"
                        f"<b>{item['line_name']}</b><br>"
                        f"<span style='color:#fda4af;'>Acierto acumulado: <b>{item['cum_win_rate']:.1f}%</b></span> • Multiplicador de contención: <b>x{item['multiplier']:.3f}</b>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

        # 3. PESTAÑAS DETALLADAS DEL CEREBRO DE LA IA
        tab_bandit, tab_calib, tab_history = st.tabs(
            [
                "🎲 Bandido Thompson Sampling (Ranking de Confianza y Calibración Adaptativa)",
                "🎯 Calibración y Atenuación de las 28 Líneas",
                "📜 Historial de Sesiones y Evolución",
            ]
        )

        with tab_bandit:
            st.markdown("#### 🎲 Algoritmo de Bandido Multibrazo (Thompson Sampling)")
            st.caption(
                "El modelo consulta este ranking en tiempo real: las líneas que acumulan aciertos reciben recompensa positiva (+1) "
                "y ganan mayor prioridad estadística; las líneas con rachas de fallo son atenuadas para proteger tu banca."
            )

            bandit_rows = []
            for lname, bdata in state.get("bandit_priors", {}).items():
                alpha = bdata.get("alpha", 4.0)
                beta_val = bdata.get("beta", 2.5)
                score = round(alpha / (alpha + beta_val), 3)
                bandit_rows.append({
                    "Línea de Apuesta": lname,
                    "Puntuación de Recompensa (0-1)": score,
                    "Éxitos (Alpha)": alpha,
                    "Fallos (Beta)": beta_val,
                    "Nivel de Confianza": bdata.get("status", "Estándar"),
                })

            bandit_df = pd.DataFrame(bandit_rows).sort_values("Puntuación de Recompensa (0-1)", ascending=False)
            st.dataframe(bandit_df, use_container_width=True, hide_index=True)

        with tab_calib:
            st.markdown("#### 🎯 Calibración Empírica y Multiplicadores Bayesianos")
            st.caption(
                "Muestra el ajuste aplicado a cada mercado. Si el modelo teóricamente proyecta 75% pero la realidad histórica es 55%, "
                "el multiplicador bayesiano atempera la probabilidad hacia abajo para eliminar el sesgo de sobreestimación."
            )

            calib_rows = []
            for lname, cdata in state.get("lines_calibration", {}).items():
                calib_rows.append({
                    "Línea": lname,
                    "Evaluados": cdata.get("trials", 0),
                    "Aciertos": cdata.get("hits", 0),
                    "% Acierto Observado": f"{cdata.get('observed_win_rate', 0):.1f}%",
                    "Prob. Predicha Promedio": f"{cdata.get('avg_predicted_prob', 0):.1f}%",
                    "Multiplicador Bayesiano": f"x{cdata.get('multiplier', 1.0):.3f}",
                    "Veredicto IA": cdata.get("status", "-"),
                })

            calib_df = pd.DataFrame(calib_rows).sort_values("Evaluados", ascending=False)
            st.dataframe(calib_df, use_container_width=True, hide_index=True)

        with tab_history:
            st.markdown("#### 📜 Registro de Ciclos de Aprendizaje Realizados")
            hist_list = state.get("training_history", [])
            if not hist_list:
                st.info("Aún no se han ejecutado ciclos de entrenamiento registrados en el historial.")
            else:
                hist_rows = []
                for entry in reversed(hist_list):
                    hist_rows.append({
                        "Ciclo #": entry.get("session_id"),
                        "Fecha / Hora (UTC)": entry.get("timestamp", "").replace("T", " ")[:19],
                        "Partidos Asimilados": entry.get("matches_processed"),
                        "Peso Dixon": f"{entry.get('new_weights', {}).get('dixon', 0.65)*100:.1f}%",
                        "Peso ML": f"{entry.get('new_weights', {}).get('ml', 0.35)*100:.1f}%",
                        "Brier Dixon": entry.get("brier_dixon"),
                        "Brier ML": entry.get("brier_ml"),
                        "Muestras ML SQLite": entry.get("db_records_in_ml"),
                    })
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

        return

    # MODO 5: RADAR EN VIVO (IN-PLAY SCANNER & ATTACK MOMENTUM)
    if view_mode == "🔴 En Vivo & Momentum":
        st.subheader("🔴 Escáner de Partidos EN VIVO (Attack Momentum & Alertas de Gol Inminente)")
        sf_scraper, _ = get_scrapers()
        live_events = sf_scraper.get_live_events()

        if not live_events:
            st.info("No hay partidos de fútbol en juego en este exacto segundo.")
            return

        st.markdown(f"Se detectaron **`{len(live_events)}` partidos en juego ahora mismo**. Analizando presión ofensiva (*Momentum*) en los principales...")
        live_rows = []
        live_with_graphs: List[tuple[MatchDetailSchema, LiveMomentumSummarySchema]] = []

        for lm in live_events[:15]:
            mom = sf_scraper.get_match_momentum(lm.event_id, lm.home_team.name, lm.away_team.name)
            st_stats = sf_scraper.get_match_statistics(lm.event_id)
            xgh = st_stats.expected_goals_home if st_stats else None
            xga = st_stats.expected_goals_away if st_stats else None
            sot_h = st_stats.shots_on_target_home if st_stats else None
            sot_a = st_stats.shots_on_target_away if st_stats else None

            if mom:
                live_with_graphs.append((lm, mom))

            live_rows.append(
                {
                    "Torneo": f"{lm.tournament.name} ({lm.tournament.country or ''})",
                    "Partido en Juego": f"{lm.home_team.name} vs {lm.away_team.name}",
                    "Marcador Vivo": f"{lm.home_score or 0} - {lm.away_score or 0} ({lm.status_description or 'Vivo'})",
                    "xG en Vivo": f"{xgh:.2f} - {xga:.2f}" if (xgh is not None and xga is not None) else "S/D",
                    "Rem. Arco Vivo": f"{sot_h} - {sot_a}" if (sot_h is not None and sot_a is not None) else "S/D",
                    "Dominador Reciente": mom.dominant_team if mom else "S/D",
                    "Nivel Alerta de Gol": mom.goal_alert_level if mom else "Normal",
                    "Diagnóstico Momentum (Últ. 10m)": mom.goal_alert_reason if mom else "Sin gráfico de presión",
                }
            )

        st.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True)

        if live_with_graphs:
            st.markdown("#### 📈 Inspeccionar Gráfico de Presión Minuto a Minuto (Attack Momentum)")
            chosen_live_idx = st.selectbox(
                "Selecciona un partido en vivo con gráfico de presión:",
                options=range(len(live_with_graphs)),
                format_func=lambda idx: f"{live_with_graphs[idx][0].home_team.name} {live_with_graphs[idx][0].home_score}-{live_with_graphs[idx][0].away_score} {live_with_graphs[idx][0].away_team.name} ({live_with_graphs[idx][1].goal_alert_level})",
            )
            lm_sel, mom_sel = live_with_graphs[chosen_live_idx]
            st.plotly_chart(
                plot_momentum_chart(mom_sel, lm_sel.home_team.name, lm_sel.away_team.name),
                use_container_width=True,
            )
        return

    if not filtered_matches or selected_match_idx is None:
        st.warning("No se encontraron partidos con los filtros seleccionados. Prueba otra fecha o activa 'Cargar todas las ligas del mundo'.")
        return

    # MODO 4: CATÁLOGO COMPLETO DEL DÍA
    if view_mode == "📋 Catálogo Completo":
        schedule_rows = []
        for m in filtered_matches:
            if m.status_type == "finished":
                estado = f"Finalizado ({m.home_score} - {m.away_score})"
            elif m.status_type == "inprogress":
                estado = f"En Juego ({m.home_score} - {m.away_score})"
            else:
                col_time = format_colombia_time(m.start_timestamp or m.start_time_utc, "%H:%M")
                estado = f"{col_time} (Hora Colombia)"

            schedule_rows.append(
                {
                    "ID": str(m.event_id),
                    "Hora / Estado": estado,
                    "País / Región": m.tournament.country or "Internacional",
                    "Torneo": m.tournament.name,
                    "Equipo Local": m.home_team.name,
                    "Equipo Visitante": m.away_team.name,
                    "Ronda": str(m.round_number) if m.round_number is not None else "-",
                }
            )
        st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True, hide_index=True)
        return

    # MODO 2: RADAR AUTOMÁTICO DE MEJORES PICKS DEL DÍA & BACKTESTING GLOBAL
    if view_mode == "🔥 Radar Top Picks & Auditoría":
        st.subheader("🔥 Escáner Inteligente: Auditoría de Aciertos/Fallos en los 10 Mercados & Top Picks")
        
        finished_in_filter = [m for m in filtered_matches if m.status_type == "finished"]
        
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.3])
        with ctrl_col1:
            scan_count = st.slider(
                "Cantidad de partidos a procesar:",
                min_value=1,
                max_value=max(1, min(25, len(filtered_matches))),
                value=min(8, max(1, len(filtered_matches))),
            )
        with ctrl_col2:
            min_prob_filter = st.slider("Probabilidad mínima para Top Picks (%):", 55, 85, 65)
        with ctrl_col3:
            st.markdown(f"**Partidos finalizados disponibles hoy:** `{len(finished_in_filter)}`")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_audit_btn = st.button(
                f"📊 Auditar Partidos Finalizados ({min(scan_count, len(finished_in_filter))} P)",
                type="primary",
                use_container_width=True,
                help="Analiza partidos terminados para calcular la tasa de aciertos y fallos de cada uno de los 10 pronósticos.",
            )
        with btn_col2:
            run_scan_btn = st.button(
                f"🚀 Escanear Mejores Picks de la Jornada ({scan_count} P)",
                use_container_width=True,
                help="Escanea partidos para encontrar las mejores cuotas de valor (Value Bets) y selecciones más probables.",
            )

        if run_audit_btn:
            matches_to_audit = finished_in_filter[:scan_count] if finished_in_filter else filtered_matches[:scan_count]
            if not matches_to_audit:
                st.warning("No se encontraron partidos finalizados en la selección actual para auditar.")
            else:
                prog = st.progress(0, text="Auditando aciertos y fallos en paralelo...")
                audit_targets = [
                    (m_item, catalog["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A"))
                    for m_item in matches_to_audit
                ]
                batch_res = analyze_matches_batch_parallel(
                    audit_targets,
                    progress_bar=prog,
                    status_text_prefix="Auditando",
                    max_workers=MAX_PARALLEL_WORKERS,
                )
                for res_deep in batch_res:
                    m_val = res_deep["match"]
                    st.session_state["audited_predictions_cache"][m_val.event_id] = res_deep["prediction"]
                    if "audited_matches_cache" not in st.session_state:
                        st.session_state["audited_matches_cache"] = {}
                    st.session_state["audited_matches_cache"][m_val.event_id] = m_val
                prog.empty()
                st.success(f"✅ Se auditaron exitosamente {len(batch_res)} partidos finalizados a máxima velocidad.")

        if run_scan_btn:
            scan_targets = [
                (m_item, catalog["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A"))
                for m_item in filtered_matches[:scan_count]
            ]
            prog = st.progress(0, text="Escaneando partidos en paralelo...")
            scanned_results = analyze_matches_batch_parallel(
                scan_targets,
                progress_bar=prog,
                status_text_prefix="Escaneando Radar",
                max_workers=MAX_PARALLEL_WORKERS,
            )
            for res_deep in scanned_results:
                m_val = res_deep["match"]
                if m_val.status_type == "finished" and m_val.home_score is not None:
                    st.session_state["audited_predictions_cache"][m_val.event_id] = res_deep["prediction"]
            prog.empty()
            st.session_state["radar_results"] = scanned_results

        # Recopilar todas las predicciones auditadas disponibles
        all_audited_preds: List[MatchPrediction] = list(st.session_state.get("audited_predictions_cache", {}).values())
        radar_data = st.session_state.get("radar_results", [])
        for entry in radar_data:
            p_obj: MatchPrediction = entry["prediction"]
            if p_obj.verified_total > 0 and p_obj.event_id not in [ap.event_id for ap in all_audited_preds]:
                all_audited_preds.append(p_obj)

        audit_tab, picks_tab = st.tabs(
            [
                "📊 Auditoría de los 10 Mercados (Promedio, Ranking & Gráfica)",
                "🏆 Ranking Top Picks & Oportunidades de Valor",
            ]
        )

        with audit_tab:
            if not all_audited_preds:
                st.info(
                    "💡 Aún no se han auditado partidos finalizados en esta sesión. "
                    "Haz clic en el botón **'📊 Auditar Partidos Finalizados'** para procesar los marcadores "
                    "y descubrir cuáles de los 10 pronósticos son los más efectivos y cuáles fallan con mayor frecuencia."
                )
            else:
                breakdown = compute_market_accuracy_breakdown(all_audited_preds)
                
                # Tarjetas KPI Principales
                ak1, ak2, ak3, ak4 = st.columns(4)
                ak1.metric("🎯 Promedio Global de Acierto", f"{breakdown['overall_win_rate']:.1f}%", f"{breakdown['overall_hits']} aciertos de {breakdown['overall_total']}")
                
                best_name = breakdown['best_category']['category'] if breakdown['best_category'] else "N/A"
                best_wr = f"{breakdown['best_category']['win_rate']:.1f}%" if breakdown['best_category'] else "-"
                ak2.metric("🏆 Mercado que Más Acierta", best_wr, best_name[:26] + "..." if len(best_name) > 26 else best_name)
                
                worst_name = breakdown['worst_category']['category'] if breakdown['worst_category'] else "N/A"
                worst_wr = f"{breakdown['worst_category']['win_rate']:.1f}%" if breakdown['worst_category'] else "-"
                ak3.metric("⚠️ Mercado que Más Falla", worst_wr, worst_name[:26] + "..." if len(worst_name) > 26 else worst_name)
                
                ak4.metric("🏟️ Partidos Finalizados Auditados", str(breakdown['matches_audited']), f"{breakdown['overall_total']} pronósticos evaluados")

                st.markdown("---")
                
                # Gráfica Interactiva Plotly de los 10 Mercados
                st.plotly_chart(plot_market_accuracy_chart(breakdown), use_container_width=True)

                # Diagnóstico Táctico: ¿Cuáles acierta más y cuáles falla más?
                diag_col1, diag_col2 = st.columns(2)
                
                active_stats = [s for s in breakdown["category_stats"] if s["total"] > 0]
                top_markets = active_stats[:4] if len(active_stats) >= 4 else active_stats
                bottom_markets = list(reversed(active_stats[-4:])) if len(active_stats) >= 4 else []

                with diag_col1:
                    st.markdown("#### 🟢 ¿Cuáles son los que MÁS ACIERTA? (Top Fiabilidad)")
                    st.caption("Mercados con mayor solidez estadística recomendados para parleys o apuestas principales:")
                    for tm in top_markets:
                        tot_tm = tm["total"]
                        h_tm = tm["hits"]
                        wr_tm = tm["win_rate"]
                        st.markdown(
                            f"<div style='background:#132e27; border-left:4px solid #10b981; padding:8px 12px; border-radius:6px; margin-bottom:8px;'>"
                            f"<b>{tm['category']}</b>: <span style='color:#34d399; font-weight:800;'>{wr_tm:.1f}% de Acierto</span> ({h_tm}/{tot_tm})<br>"
                            f"<span style='font-size:0.80rem; color:#a7f3d0;'>{tm['status_tag']} — Prob. media pronosticada: {tm['avg_prob']:.1f}%</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                with diag_col2:
                    st.markdown("#### 🔴 ¿Cuáles son los que MÁS FALLA? (Mayor Volatilidad)")
                    st.caption("Mercados donde la varianza táctica o partidos atípicos (ej. 0-0) generan más fallos:")
                    for bm in bottom_markets:
                        tot_bm = bm["total"]
                        m_bm = bm["misses"]
                        wr_bm = bm["win_rate"]
                        st.markdown(
                            f"<div style='background:#3b141d; border-left:4px solid #f43f5e; padding:8px 12px; border-radius:6px; margin-bottom:8px;'>"
                            f"<b>{bm['category']}</b>: <span style='color:#fda4af; font-weight:800;'>{wr_bm:.1f}% de Acierto</span> ({m_bm} fallos de {tot_bm})<br>"
                            f"<span style='font-size:0.80rem; color:#fecdd3;'>{bm['status_tag']} — Se recomienda cautela en líneas altas</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
                st.markdown("#### 📋 Tabla Resumen de los 10 Mercados Evaluados")
                
                table_rows = []
                for s in breakdown["category_stats"]:
                    if s["total"] == 0:
                        continue
                    table_rows.append({
                        "Mercado Pronosticado": s["category"],
                        "Evaluados": s["total"],
                        "Aciertos (✅)": s["hits"],
                        "Fallos (❌)": s["misses"],
                        "Tasa de Acierto (%)": f"{s['win_rate']:.1f}%",
                        "Prob. Media": f"{s['avg_prob']:.1f}%",
                        "Cuota Justa": f"{round(100.0 / max(1.0, s['avg_prob']), 2):.2f}",
                        "Veredicto / Fiabilidad": s["badge"],
                    })
                if table_rows:
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

                # Desglose desplegable por cada mercado
                st.markdown("##### 🔍 Desglose Detallado de Partidos por Mercado")
                for s in breakdown["category_stats"]:
                    if s["total"] == 0:
                        continue
                    with st.expander(f"Ver partidos de: {s['category']} ({s['hits']}/{s['total']} aciertos — {s['win_rate']:.1f}%)"):
                        pick_rows = []
                        for pk_item in s["picks"]:
                            res_tag = "✅ ACIERTO" if pk_item["hit"] is True else "❌ FALLO"
                            pick_rows.append({
                                "Partido": pk_item["match"],
                                "Pronóstico Seleccionado": pk_item["pick"],
                                "Probabilidad (%)": f"{pk_item['prob']:.1f}%",
                                "Resultado Real": pk_item.get("actual") or "-",
                                "Dictamen": res_tag,
                            })
                        st.dataframe(pd.DataFrame(pick_rows), use_container_width=True, hide_index=True)

        with picks_tab:
            all_top_picks = []
            value_and_smart = []
            total_verified = 0
            total_hits = 0

            for entry in radar_data:
                m_obj: MatchDetailSchema = entry["match"]
                pred_obj: MatchPrediction = entry["prediction"]
                match_title = f"{m_obj.home_team.name} vs {m_obj.away_team.name}"

                for pk in pred_obj.all_market_picks:
                    if pk["hit"] is not None:
                        total_verified += 1
                        if pk["hit"] is True:
                            total_hits += 1

                    if pk["prob"] >= min_prob_filter:
                        hit_str = (
                            "✅ ACIERTO"
                            if pk["hit"] is True
                            else ("❌ FALLO" if pk["hit"] is False else "⏳ Pendiente")
                        )
                        all_top_picks.append(
                            {
                                "Torneo": m_obj.tournament.name,
                                "Partido": match_title,
                                "Mercado": pk["category"],
                                "Pronóstico": pk["pick"],
                                "Probabilidad (%)": pk["prob"],
                                "Cuota Justa": round(100.0 / max(1.0, pk["prob"]), 2),
                                "Verificación Real": f"{hit_str} ({pk['actual']})" if pk.get("actual") else hit_str,
                            }
                        )

                max_edge = max(
                    [
                        ("Local (1)", pred_obj.value_edge_home or -99.0),
                        ("Empate (X)", pred_obj.value_edge_draw or -99.0),
                        ("Visita (2)", pred_obj.value_edge_away or -99.0),
                    ],
                    key=lambda x: x[1],
                )
                smart_alert = m_obj.odds.smart_money_side if m_obj.odds else None
                if max_edge[1] >= 3.5 or smart_alert:
                    value_and_smart.append(
                        {
                            "Torneo": m_obj.tournament.name,
                            "Partido": match_title,
                            "Mejor Value Bet": f"{max_edge[0]} ({max_edge[1]:+.1f}%)" if max_edge[1] >= 3.5 else "Sin ventaja >3.5%",
                            "Alerta Smart Money (Caída de Cuota)": smart_alert or "Estable",
                            "Descanso (Loc vs Vis)": f"{pred_obj.rest_days_home}d vs {pred_obj.rest_days_away}d",
                            "Pronóstico Modelo": pred_obj.recommended_pick,
                        }
                    )

            win_rate = round((total_hits / total_verified) * 100.0, 1) if total_verified > 0 else 0.0
            k1, k2, k3 = st.columns(3)
            k1.metric("Picks de Alta Confianza Detectados", len(all_top_picks))
            k2.metric("Alertas Value Bet / Smart Money", len(value_and_smart))
            k3.metric(
                "Efectividad Verificada (Backtesting)",
                f"{win_rate:.1f}% ({total_hits}/{total_verified})" if total_verified > 0 else "Partidos por jugar",
            )

            all_top_picks.sort(key=lambda x: x["Probabilidad (%)"], reverse=True)
            st.markdown("#### 🏆 Ranking de Picks Más Seguros de la Jornada")
            if all_top_picks:
                st.dataframe(pd.DataFrame(all_top_picks), use_container_width=True, hide_index=True)

            if value_and_smart:
                st.markdown("#### 💎 Oportunidades de Valor (Value Bets) & Caídas de Cuota (Smart Money)")
                st.dataframe(pd.DataFrame(value_and_smart), use_container_width=True, hide_index=True)
        return

    # MODO 3: ASIGNADOR MAESTRO DE CAPITAL & GESTIÓN DE PORTAFOLIO CUANTITATIVO
    if view_mode == "💼 Asignador Maestro de Capital & Portafolio":
        st.subheader("💼 Asignador Maestro de Capital & Portafolio Cuantitativo")
        st.caption(
            "Arquitectura institucional para gestión de capital, arbitraje seguro (Surebets), "
            "optimización de valor esperado (+EV con Kelly Fraccional escalado), coberturas Dutching y auditoría contable determinista."
        )

        kpis = get_portfolio_kpis()
        current_bankroll = kpis["current_bankroll"]

        # 1. CABECERA DE BANCA Y PARÁMETROS DE RIESGO
        st.markdown("### 🏦 Estado de la Banca y Control de Riesgo")
        bank_c1, bank_c2, bank_c3, bank_c4, bank_c5 = st.columns([1.5, 1.2, 1.2, 1.1, 1.5])
        with bank_c1:
            st.metric("💰 Banca Total Actual", f"${current_bankroll:,.2f}")
        with bank_c2:
            pnl_val = kpis["total_pnl"]
            pnl_color = "normal" if pnl_val == 0 else ("inverse" if pnl_val < 0 else "off")
            st.metric("📈 P&L Neto Acumulado", f"{pnl_val:+,.2f} $")
        with bank_c3:
            st.metric("🎯 Yield / ROI Real", f"{kpis['roi_pct']:+.2f}%")
        with bank_c4:
            st.metric("🏆 Win Rate", f"{kpis['win_rate_pct']:.1f}%")
        with bank_c5:
            with st.popover("💳 Ajustar Fondos (Depósito / Retiro)"):
                st.markdown("**Registrar Transacción de Capital**")
                adj_type = st.selectbox("Tipo de Movimiento", ["DEPOSIT", "WITHDRAWAL"], format_func=lambda x: "🟢 Depósito (+)" if x == "DEPOSIT" else "🔴 Retiro (-)")
                adj_amount = st.number_input("Monto en $", min_value=1.0, value=100.0, step=25.0)
                adj_desc = st.text_input("Nota / Motivo", value="Ajuste manual de banca")
                if st.button("Confirmar Movimiento Contable", use_container_width=True):
                    record_bankroll_transaction(adj_type, adj_amount, adj_desc)
                    st.success("Transacción registrada en el libro diario.")
                    st.rerun()

        st.markdown("#### 🎛️ Disyuntores de Exposición y Perfil del Inversor")
        p_col1, p_col2, p_col3 = st.columns([1.2, 1.4, 1.4])
        with p_col1:
            risk_profile = st.selectbox(
                "Perfil de Riesgo",
                ["Conservador", "Equilibrado", "Agresivo"],
                index=1,
                help="Conservador = 1/8 Kelly | Equilibrado = 1/4 Kelly | Agresivo = 3/8 Kelly",
            )
        with p_col2:
            max_daily_exp = st.slider(
                "Límite de Exposición Diaria (% de la Banca)",
                min_value=5,
                max_value=30,
                value=15,
                step=1,
                format="%d%%",
                help="El total de apuestas del día no excederá este porcentaje acumulado.",
            )
        with p_col3:
            max_budget_usd = round(current_bankroll * (max_daily_exp / 100.0), 2)
            st.metric("Presupuesto Máximo Hoy", f"${max_budget_usd:,.2f}", f"Máx 4.0% (${current_bankroll * 0.04:,.2f}) por apuesta individual")

        st.divider()

        # 1. Carga ultra-rápida y thread-safe de cuotas (sin tocar st.session_state en hilos secundarios)
        if "attempted_odds_events" not in st.session_state:
            st.session_state["attempted_odds_events"] = set()

        missing_odds_matches = [
            m for m in filtered_matches
            if m.odds is None and m.event_id not in st.session_state["attempted_odds_events"]
        ]
        
        batch_to_fetch = missing_odds_matches[:20]
        # Registrar eventos en el hilo principal antes de iniciar el ThreadPool
        for m_item in batch_to_fetch:
            st.session_state["attempted_odds_events"].add(m_item.event_id)

        if batch_to_fetch:
            sf_scraper_inst, _ = get_scrapers()
            def _pull_odds(m_item):
                try:
                    m_item.odds = sf_scraper_inst.get_match_odds(
                        m_item.event_id, 
                        is_live=m_item.status_type in ("inprogress", "live")
                    )
                except Exception:
                    pass
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as o_exec:
                list(o_exec.map(_pull_odds, batch_to_fetch))

        # Banner de acción para enriquecer partidos de fechas futuras o no cacheadas
        is_today = (date_str == date.today().strftime("%Y-%m-%d"))
        uncached_matches = [m for m in filtered_matches if m.event_id not in st.session_state.get("audited_predictions_cache", {})]
        if uncached_matches:
            deep_col1, deep_col2 = st.columns([2.5, 1.5])
            with deep_col1:
                st.caption(f"💡 Se cargaron cuotas reales oficiales para los partidos de `{date_str}`. Para calcular el Valor (+EV) con máxima precisión matemática:")
            with deep_col2:
                if st.button("⚡ Analizar Top 8 con IA Opta (xG)", use_container_width=True):
                    scan_targets = [
                        (m_item, catalog["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A"))
                        for m_item in uncached_matches[:8]
                    ]
                    with st.spinner("Ejecutando análisis Opta, H2H y Monte Carlo en paralelo..."):
                        batch_res = analyze_matches_batch_parallel(scan_targets, max_workers=4)
                        for r in batch_res:
                            m_res = r["match"]
                            p_res = r["prediction"]
                            st.session_state["audited_predictions_cache"][m_res.event_id] = p_res
                            st.session_state.setdefault("deep_analysis_cache", {})[m_res.event_id] = r
                    st.rerun()

        # 2. PANEL DE GENERACIÓN DE CARTERA DEL DÍA
        st.markdown("### 📦 Generación de Cartera Óptima del Día")
        tab_sure, tab_value, tab_dutch = st.tabs([
            "🛡️ Arbitraje Seguro (0 Riesgo - Surebets)",
            "📈 Inversiones de Valor (+EV con Kelly)",
            "🎯 Coberturas Dutching",
        ])

        # Recolectar datos y predicciones de los partidos para alimentar los motores
        matches_odds_list: List[Dict[str, Any]] = []
        value_candidates: List[BetOpportunity] = []
        dutching_opportunities: List[DutchingOpportunity] = []

        for m_cand in filtered_matches:
            # Determinar si el partido está en juego (In-Play) o es Pre-Match
            is_live_match = m_cand.status_type in ("inprogress", "live")
            live_minute = 45
            live_momentum = 0.0

            if is_live_match:
                # Extraer minuto de juego
                import re
                min_match = re.search(r"(\d+)", m_cand.status_description or "")
                if min_match:
                    live_minute = int(min_match.group(1))
                elif "ht" in (m_cand.status_description or "").lower() or "half" in (m_cand.status_description or "").lower():
                    live_minute = 45

                # Extraer factor de momentum reciente si existe
                if m_cand.momentum and m_cand.momentum.points:
                    recent_pts = m_cand.momentum.points[-5:]
                    if recent_pts:
                        avg_val = sum(p.value for p in recent_pts) / len(recent_pts)
                        live_momentum = max(-1.0, min(1.0, avg_val / 50.0))

                # Calcular predicción adaptativa en vivo
                live_calc = predict_live_in_play(
                    home_score=m_cand.home_score if m_cand.home_score is not None else 0,
                    away_score=m_cand.away_score if m_cand.away_score is not None else 0,
                    current_minute=live_minute,
                    lambda_home_full=1.35,
                    lambda_away_full=1.10,
                    momentum_factor=live_momentum,
                )
                p_h = live_calc["prob_home_win"] / 100.0
                p_d = live_calc["prob_draw"] / 100.0
                p_a = live_calc["prob_away_win"] / 100.0
                p_o25 = live_calc["prob_over_2_5"] / 100.0
                p_btts = live_calc["prob_btts_yes"] / 100.0
                live_tag = f" [🔴 EN VIVO {live_minute}']"
            else:
                live_tag = ""
                # Obtener predicción precalculada pre-match si existe en caché
                pred_cand = st.session_state["audited_predictions_cache"].get(m_cand.event_id)
                if not pred_cand and m_cand.event_id in st.session_state.get("deep_analysis_cache", {}):
                    pred_cand = st.session_state["deep_analysis_cache"][m_cand.event_id].get("prediction")

                if pred_cand:
                    p_h = pred_cand.prob_home_win / 100.0
                    p_d = pred_cand.prob_draw / 100.0
                    p_a = pred_cand.prob_away_win / 100.0
                    p_o25 = getattr(pred_cand, "prob_over_25", 52.0) / 100.0
                    p_btts = getattr(pred_cand, "prob_btts_yes", 50.0) / 100.0
                else:
                    # Derivar probabilidades base equilibradas a partir de las cuotas reales del mercado
                    if m_cand.odds and m_cand.odds.home_win and m_cand.odds.draw and m_cand.odds.away_win:
                        inv_h = 1.0 / m_cand.odds.home_win
                        inv_d = 1.0 / m_cand.odds.draw
                        inv_a = 1.0 / m_cand.odds.away_win
                        tot_inv = inv_h + inv_d + inv_a
                        p_h = round(inv_h / tot_inv, 3)
                        p_d = round(inv_d / tot_inv, 3)
                        p_a = round(inv_a / tot_inv, 3)
                    else:
                        p_h, p_d, p_a = 0.44, 0.28, 0.28

                    if m_cand.odds and m_cand.odds.over_2_5 and m_cand.odds.under_2_5:
                        inv_o = 1.0 / m_cand.odds.over_2_5
                        inv_u = 1.0 / m_cand.odds.under_2_5
                        p_o25 = round(inv_o / (inv_o + inv_u), 3)
                    else:
                        p_o25 = 0.52

                    if m_cand.odds and getattr(m_cand.odds, "btts_yes", None) and getattr(m_cand.odds, "btts_no", None):
                        inv_by = 1.0 / m_cand.odds.btts_yes
                        inv_bn = 1.0 / m_cand.odds.btts_no
                        p_btts = round(inv_by / (inv_by + inv_bn), 3)
                    else:
                        p_btts = 0.50

            # Extraer ÚNICAMENTE cuotas reales oficiales de la casa de apuestas (Cero cuotas sintéticas)
            h_odd = m_cand.odds.home_win if (m_cand.odds and m_cand.odds.home_win) else None
            d_odd = m_cand.odds.draw if (m_cand.odds and m_cand.odds.draw) else None
            a_odd = m_cand.odds.away_win if (m_cand.odds and m_cand.odds.away_win) else None
            o25_odd = m_cand.odds.over_2_5 if (m_cand.odds and m_cand.odds.over_2_5) else None
            u25_odd = m_cand.odds.under_2_5 if (m_cand.odds and m_cand.odds.under_2_5) else None
            btts_y_odd = getattr(m_cand.odds, "btts_yes", None) if m_cand.odds else None
            btts_n_odd = getattr(m_cand.odds, "btts_no", None) if m_cand.odds else None

            event_label = f"{m_cand.home_team.name} vs {m_cand.away_team.name}{live_tag}"

            # Construir mapa de cuotas reales para Arbitraje
            odds_map: Dict[str, float] = {}
            if h_odd: odds_map["1X2_HOME"] = h_odd
            if d_odd: odds_map["1X2_DRAW"] = d_odd
            if a_odd: odds_map["1X2_AWAY"] = a_odd
            if o25_odd: odds_map["OU_2.5_OVER"] = o25_odd
            if u25_odd: odds_map["OU_2.5_UNDER"] = u25_odd
            if btts_y_odd: odds_map["BTTS_YES"] = btts_y_odd
            if btts_n_odd: odds_map["BTTS_NO"] = btts_n_odd

            if len(odds_map) >= 2:
                matches_odds_list.append({
                    "event_id": m_cand.event_id,
                    "event_name": event_label,
                    "tournament": m_cand.tournament.name,
                    "match_date": date_str,
                    "odds": odds_map,
                })

            # Generar candidatos para Value Bets (+EV) con CUOTAS REALES
            # 1. Victoria Local
            if h_odd and h_odd > 1.01:
                ev_h = (p_h * h_odd) - 1.0
                fair_h = round(1.0 / max(0.01, p_h), 2)
                value_candidates.append(
                    BetOpportunity(
                        event_id=m_cand.event_id,
                        event_name=event_label,
                        tournament=m_cand.tournament.name,
                        match_date=date_str,
                        strategy="VALUE_BET",
                        market_code="1X2_HOME",
                        pick_name=f"Victoria {m_cand.home_team.name}",
                        odds_taken=h_odd,
                        fair_odds=fair_h,
                        model_probability=round(p_h, 3),
                        ev_net=round(ev_h, 3),
                    )
                )

            # 2. Victoria Visitante
            if a_odd and a_odd > 1.01:
                ev_a = (p_a * a_odd) - 1.0
                fair_a = round(1.0 / max(0.01, p_a), 2)
                value_candidates.append(
                    BetOpportunity(
                        event_id=m_cand.event_id,
                        event_name=event_label,
                        tournament=m_cand.tournament.name,
                        match_date=date_str,
                        strategy="VALUE_BET",
                        market_code="1X2_AWAY",
                        pick_name=f"Victoria {m_cand.away_team.name}",
                        odds_taken=a_odd,
                        fair_odds=fair_a,
                        model_probability=round(p_a, 3),
                        ev_net=round(ev_a, 3),
                    )
                )

            # 3. Over 2.5 Goles (Solo si la casa cotiza Over 2.5)
            if o25_odd and o25_odd > 1.01:
                ev_o25 = (p_o25 * o25_odd) - 1.0
                fair_o25 = round(1.0 / max(0.01, p_o25), 2)
                value_candidates.append(
                    BetOpportunity(
                        event_id=m_cand.event_id,
                        event_name=event_label,
                        tournament=m_cand.tournament.name,
                        match_date=date_str,
                        strategy="VALUE_BET",
                        market_code="OU_2.5_OVER",
                        pick_name="Más de 2.5 Goles",
                        odds_taken=o25_odd,
                        fair_odds=fair_o25,
                        model_probability=round(p_o25, 3),
                        ev_net=round(ev_o25, 3),
                    )
                )

            # 4. Ambos Equipos Anotan (Solo si la casa cotiza BTTS)
            if btts_y_odd and btts_y_odd > 1.01:
                ev_btts = (p_btts * btts_y_odd) - 1.0
                fair_btts = round(1.0 / max(0.01, p_btts), 2)
                value_candidates.append(
                    BetOpportunity(
                        event_id=m_cand.event_id,
                        event_name=event_label,
                        tournament=m_cand.tournament.name,
                        match_date=date_str,
                        strategy="VALUE_BET",
                        market_code="BTTS_YES",
                        pick_name="Ambos Anotan: Sí",
                        odds_taken=btts_y_odd,
                        fair_odds=fair_btts,
                        model_probability=round(p_btts, 3),
                        ev_net=round(ev_btts, 3),
                    )
                )

            # 5. Cobertura Dutching de marcadores o doble oportunidad (si existen cuotas reales de 1 y X)
            if h_odd and d_odd and h_odd > 1.01 and d_odd > 1.01:
                dutch_picks = [
                    {"market_code": "1X2_HOME", "pick_name": f"1 ({m_cand.home_team.name})", "odds": h_odd, "model_prob": p_h},
                    {"market_code": "1X2_DRAW", "pick_name": "X (Empate)", "odds": d_odd, "model_prob": p_d},
                ]
                dutch_res = solve_dutching(
                    event_id=m_cand.event_id,
                    event_name=event_label,
                    tournament=m_cand.tournament.name,
                    match_date=date_str,
                    coverage_name="Cobertura 1X Sintética",
                    selections=dutch_picks,
                    target_budget=min(50.0, round(current_bankroll * 0.03, 2)),
                )
                if dutch_res and dutch_res.combined_probability >= 0.65:
                    dutching_opportunities.append(dutch_res)

        # TAB 1: ARBITRAJE SEGURO (SUREBETS)
        with tab_sure:
            st.markdown("#### 🛡️ Oportunidades de Arbitraje Libre de Riesgo")
            surebets = find_surebets(matches_odds_list, current_bankroll, min_roi=0.015)
            if not surebets:
                st.info("ℹ️ No se detectaron arbitrajes puros con ROI > 1.5% en las cuotas cargadas de hoy. El mercado está alineado.")
            else:
                for sb in surebets:
                    with st.container(border=True):
                        sb_c1, sb_c2, sb_c3, sb_c4 = st.columns([1.8, 1.2, 1.2, 1.2])
                        with sb_c1:
                            st.markdown(f"**⚽ {sb.event_name}** (`{sb.tournament}`)")
                            st.caption(f"Mercado: **{sb.market_type}** | Inversa: `{sb.inverse_sum:.4f}`")
                        with sb_c2:
                            st.metric("Stake Total", f"${sb.total_stake:.2f}")
                        with sb_c3:
                            st.metric("Retorno Seguro", f"${sb.guaranteed_payout:.2f}")
                        with sb_c4:
                            st.metric("Ganancia Neta", f"+${sb.guaranteed_profit:.2f}", f"+{sb.roi_pct:.2f}% ROI")

                        # Mostrar desglose de apuestas por rama
                        leg_cols = st.columns(len(sb.legs))
                        for idx_leg, leg in enumerate(sb.legs):
                            with leg_cols[idx_leg]:
                                st.markdown(f"**{leg.pick_name}**")
                                st.markdown(f"Cuota: `x{leg.odds:.2f}`")
                                st.markdown(f"Apostar: `💵 ${leg.stake_amount:.2f}`")
                                st.caption(f"Retorno si gana: ${leg.payout:.2f}")

        # TAB 2: INVERSIONES DE VALOR (+EV CON KELLY)
        with tab_value:
            st.markdown("#### 📈 Oportunidades +EV con Criterio de Kelly Fraccional Escalado")
            st.caption("Filtros aplicados: Descarte de varianza para cuotas > 3.50, Hard Cap del 4.0% por apuesta y corte de presupuesto diario.")

            allocated_value_bets = optimize_value_bets(
                value_candidates,
                bankroll=current_bankroll,
                risk_profile=risk_profile,
                max_exposure=max_daily_exp / 100.0,
            )

            if not allocated_value_bets:
                st.info(f"ℹ️ Aún no se han detectado selecciones con valor (+EV >= 2%) en las cuotas cargadas de `{date_str}`.")
                uncached_in_tab = [m for m in filtered_matches if m.event_id not in st.session_state.get("audited_predictions_cache", {})]
                if uncached_in_tab:
                    st.caption("Para encontrar ventaja matemática sobre la casa (+EV), la IA necesita contrastar las cuotas contra las métricas profundas de xG, fatiga y H2H:")
                    if st.button("⚡ Ejecutar Análisis de IA Opta (xG) en los Partidos de esta Fecha", type="primary", use_container_width=True, key="btn_run_ai_tab_val"):
                        scan_targets = [
                            (m_item, catalog["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A"))
                            for m_item in uncached_in_tab[:8]
                        ]
                        with st.spinner("Ejecutando análisis Opta, H2H y Monte Carlo en paralelo..."):
                            batch_res = analyze_matches_batch_parallel(scan_targets, max_workers=4)
                            for r in batch_res:
                                m_res = r["match"]
                                p_res = r["prediction"]
                                st.session_state["audited_predictions_cache"][m_res.event_id] = p_res
                                st.session_state.setdefault("deep_analysis_cache", {})[m_res.event_id] = r
                        st.rerun()
            else:
                total_plan_stake = sum(b.stake_amount for b in allocated_value_bets)
                total_plan_payout = sum(b.potential_payout for b in allocated_value_bets)

                st.markdown(f"**Cartera sugerida:** `{len(allocated_value_bets)} apuestas` | **Capital Total a Invertir:** `${total_plan_stake:,.2f}` | **Retorno Potencial:** `${total_plan_payout:,.2f}`")

                # Grilla de tarjetas de apuestas
                for b in allocated_value_bets:
                    with st.container(border=True):
                        v_c1, v_c2, v_c3, v_c4, v_c5 = st.columns([1.8, 1.0, 1.0, 1.2, 1.2])
                        with v_c1:
                            st.markdown(f"**{b.event_name}**")
                            st.caption(f"🏆 {b.tournament} | Mercado: `{b.market_code}`")
                        with v_c2:
                            st.markdown(f"**{b.pick_name}**")
                            st.caption(f"Prob. Modelo: **{b.model_probability * 100.0:.1f}%**")
                        with v_c3:
                            st.markdown(f"Cuota: **`{b.odds_taken:.2f}`**")
                            st.caption(f"Cuota Justa: `{b.fair_odds:.2f}`")
                        with v_c4:
                            st.metric("Monto a Apostar", f"${b.stake_amount:.2f}", f"{b.stake_percentage:.1f}% banca")
                        with v_c5:
                            st.metric("Retorno Potencial", f"${b.potential_payout:.2f}", f"EV: +{b.ev_net * 100.0:.1f}%")

                st.markdown("---")
                # Botón de confirmación con idempotencia
                if st.button("✅ Confirmar y Guardar Cartera de Valor en la Base de Datos", type="primary", use_container_width=True):
                    bets_payload = [
                        {
                            "event_id": b.event_id,
                            "event_name": b.event_name,
                            "tournament": b.tournament,
                            "match_date": b.match_date,
                            "strategy": b.strategy,
                            "market_code": b.market_code,
                            "pick_name": b.pick_name,
                            "odds_taken": b.odds_taken,
                            "fair_odds": b.fair_odds,
                            "model_probability": b.model_probability,
                            "stake_amount": b.stake_amount,
                            "stake_percentage": b.stake_percentage,
                            "potential_payout": b.potential_payout,
                        }
                        for b in allocated_value_bets
                    ]
                    saved, skipped = save_portfolio_bets(bets_payload)
                    if saved > 0:
                        st.success(f"🎉 **{saved} apuestas guardadas exitosamente** en el portafolio y registradas contablemente en el libro diario.")
                    if skipped > 0:
                        st.info(f"ℹ️ {skipped} apuestas ya existían previamente y se omitieron por protección idempotente (anti-duplicados).")
                    st.rerun()

        # TAB 3: COBERTURAS DUTCHING
        with tab_dutch:
            st.markdown("#### 🎯 Coberturas Dutching Balanceadas")
            st.caption("Distribuye el capital entre resultados excluyentes de alta probabilidad para garantizar un retorno idéntico si cualquiera se cumple.")
            if not dutching_opportunities:
                st.info("ℹ️ No se configuraron oportunidades de Dutching para la jornada actual.")
            else:
                for d_opp in dutching_opportunities[:5]:
                    with st.container(border=True):
                        d_c1, d_c2, d_c3, d_c4 = st.columns([1.8, 1.2, 1.2, 1.2])
                        with d_c1:
                            st.markdown(f"**⚽ {d_opp.event_name}** (`{d_opp.tournament}`)")
                            st.caption(f"Cobertura: **{d_opp.coverage_name}** | Prob. Combinada: `{d_opp.combined_probability * 100.0:.1f}%`")
                        with d_c2:
                            st.metric("Stake Total", f"${d_opp.total_stake:.2f}")
                        with d_c3:
                            st.metric("Retorno Objetivo", f"${d_opp.target_payout:.2f}")
                        with d_c4:
                            st.metric("Ganancia si Acierta", f"+${d_opp.net_profit_if_hit:.2f}", f"+{d_opp.roi_if_hit_pct:.1f}% ROI")

                        d_leg_cols = st.columns(len(d_opp.legs))
                        for idx_dl, leg in enumerate(d_opp.legs):
                            with d_leg_cols[idx_dl]:
                                st.markdown(f"**{leg.pick_name}**")
                                st.markdown(f"Cuota: `x{leg.odds:.2f}`")
                                st.markdown(f"Apostar: `💵 ${leg.stake_amount:.2f}`")

        st.divider()

        # 3. PANEL DE BITÁCORA, AUDITORÍA DE RESULTADOS Y EVOLUCIÓN DE BANCA
        st.markdown("### 📋 Bitácora y Auditoría Continua de Resultados")

        audit_top_c1, audit_top_c2 = st.columns([1.6, 2.4])
        with audit_top_c1:
            st.markdown(f"**Apuestas Activas Pendientes:** `{kpis['pending_count']}` | **Cerradas:** `{kpis['won_count'] + kpis['lost_count'] + kpis['void_count']}`")
        with audit_top_c2:
            if st.button("🔄 Auditar Partidos Finalizados & Calibrar Modelo", type="secondary", use_container_width=True):
                # Proveedor de resultados desde la base de datos o scraping
                def local_scores_provider(event_id: int, match_date: str) -> Optional[Dict[str, Any]]:
                    # Buscar en la caché de partidos o base de datos
                    match_obj = st.session_state["audited_matches_cache"].get(event_id)
                    if not match_obj:
                        session_q = get_db_session()
                        with session_q as sess:
                            from src.models.database import MatchORM
                            rec = sess.query(MatchORM).filter_by(event_id=event_id).first()
                            if rec and rec.home_score is not None:
                                return {
                                    "home_score": rec.home_score,
                                    "away_score": rec.away_score,
                                    "status": rec.status_type or "finished",
                                }
                        return None
                    return {
                        "home_score": match_obj.home_score,
                        "away_score": match_obj.away_score,
                        "status": match_obj.status_type,
                    }

                with get_db_session() as db_sess:
                    report = audit_and_settle_open_bets(db_sess, local_scores_provider)

                if report.settled_count > 0:
                    st.success(
                        f"✅ **Auditoría finalizada:** {report.settled_count} apuestas liquidadas "
                        f"({report.won_count} Ganadas, {report.lost_count} Perdidas, {report.void_count} Anuladas). "
                        f"**P&L de la ronda:** `${report.total_pnl:+,.2f}`."
                    )
                    # Retroalimentar al motor de AutoLearner
                    from src.analytics.auto_learner import get_auto_learner
                    learner = get_auto_learner()
                    settled_dicts = [
                        {
                            "market_code": s.market_code,
                            "status": s.status,
                            "model_probability": 0.60,
                            "pnl": s.pnl,
                        }
                        for s in report.settled_bets
                    ]
                    learn_res = learner.learn_from_settled_portfolio_bets(settled_dicts)
                    st.info(f"🧠 **AutoLearner calibrado:** {learn_res.get('updated_lines_count', 0)} líneas ajustaron sus multiplicadores empíricos.")
                else:
                    st.info(f"ℹ️ Se revisaron {report.total_audited} apuestas pendientes. Ningún partido ha concluido aún para liquidación.")
                st.rerun()

        # Curva de evolución de la banca
        ledger_history = get_bankroll_ledger_history(limit=200)
        if len(ledger_history) >= 2:
            st.markdown("#### 📈 Curva de Crecimiento de la Banca (Libro Diario Contable)")
            df_ledger = pd.DataFrame(ledger_history)
            fig_bankroll = go.Figure()
            fig_bankroll.add_trace(
                go.Scatter(
                    x=df_ledger["timestamp"],
                    y=df_ledger["resulting_balance"],
                    mode="lines+markers",
                    name="Saldo ($)",
                    line=dict(color="#38bdf8", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.12)",
                )
            )
            fig_bankroll.update_layout(
                paper_bgcolor="#0e1726",
                plot_bgcolor="#0e1726",
                font=dict(color="#e2e8f0"),
                margin=dict(l=30, r=30, t=30, b=30),
                height=320,
                xaxis=dict(gridcolor="#1e293b", title="Fecha y Hora de la Transacción"),
                yaxis=dict(gridcolor="#1e293b", title="Saldo Total ($)"),
            )
            st.plotly_chart(fig_bankroll, use_container_width=True)

        # Tabla interactiva con filtro de estado
        st.markdown("#### 📑 Historial Detallado de Apuestas del Portafolio")
        status_filter = st.selectbox("Filtrar por Estado", ["TODAS", "PENDING", "WON", "LOST", "VOID_PUSH"], format_func=lambda x: {
            "TODAS": "🔍 Todas las Apuestas",
            "PENDING": "⏳ Solo Pendientes",
            "WON": "✅ Solo Ganadas",
            "LOST": "❌ Solo Perdidas",
            "VOID_PUSH": "⚪ Solo Anuladas / Reembolsadas",
        }.get(x, x))

        portfolio_records = get_portfolio_bets(status=status_filter, limit=100)
        if not portfolio_records:
            st.caption("No hay registros en la base de datos para el filtro seleccionado.")
        else:
            df_bets = pd.DataFrame(portfolio_records)
            display_cols = [
                "match_date", "event_name", "strategy", "pick_name",
                "odds_taken", "stake_amount", "potential_payout", "status", "pnl"
            ]
            df_display = df_bets[display_cols].copy()
            df_display.columns = [
                "Fecha", "Partido", "Estrategia", "Selección",
                "Cuota", "Stake ($)", "Retorno ($)", "Estado", "P&L ($)"
            ]
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        return


    # MODO 1: ANÁLISIS PROFUNDO DEL PARTIDO SELECCIONADO
    target_match = filtered_matches[selected_match_idx]
    fs_id = catalog["fs_lookup"].get((target_match.home_team.id, target_match.away_team.id), "N/A")

    if "deep_analysis_cache" in st.session_state and target_match.event_id in st.session_state["deep_analysis_cache"]:
        deep = st.session_state["deep_analysis_cache"][target_match.event_id]
    else:
        deep = analyze_single_match_deep(target_match.model_dump(), fs_id)

    m: MatchDetailSchema = deep["match"]
    hf: TeamFormMetrics = deep["home_form"]
    af: TeamFormMetrics = deep["away_form"]
    p: MatchPrediction = deep["prediction"]
    h_players: List[PlayerPropSchema] = deep["home_players"]
    a_players: List[PlayerPropSchema] = deep["away_players"]
    s_map: Dict[int, Dict[str, Any]] = deep["standings_map"]

    # 1. CABECERA TIPO SCOREBOARD PRO
    if m.status_type == "finished":
        ht_txt = f" (1H: {m.home_score_ht}-{m.away_score_ht})" if m.home_score_ht is not None else ""
        score_center_html = f"<div style='font-size:2.5rem; font-weight:900; color:#4ade80;'>{m.home_score} - {m.away_score}</div><div style='font-size:0.80rem; color:#94a3b8;'>FINALIZADO{ht_txt}</div>"
    elif m.status_type == "inprogress":
        score_center_html = f"<div style='font-size:2.5rem; font-weight:900; color:#facc15;'>{m.home_score} - {m.away_score}</div><div style='font-size:0.80rem; color:#facc15;'>EN JUEGO</div>"
    else:
        col_time_12h = format_colombia_time(m.start_timestamp or m.start_time_utc, "%I:%M %p")
        col_time_24h = format_colombia_time(m.start_timestamp or m.start_time_utc, "%H:%M")
        score_center_html = (
            f"<div style='font-size:2.0rem; font-weight:900; color:#38bdf8;'>VS</div>"
            f"<div style='font-size:0.95rem; font-weight:700; color:#f8fafc;'>🇨🇴 {col_time_12h} ({col_time_24h})</div>"
            f"<div style='font-size:0.75rem; color:#94a3b8;'>Hora Oficial Colombia (UTC-5)</div>"
        )

    h_pos = s_map.get(m.home_team.id, {}).get("position")
    a_pos = s_map.get(m.away_team.id, {}).get("position")
    h_pos_str = f" • #{h_pos} Tabla" if h_pos else ""
    a_pos_str = f" • #{a_pos} Tabla" if a_pos else ""

    ref_badge_html = (
        f"<span class='ref-pill'>🟨 Árbitro: {m.referee.name} ({m.referee.yellow_per_game:.2f} tarj/P • {m.referee.strictness_label})</span>"
        if m.referee
        else "<span class='ref-pill'>🟨 Árbitro: Por designar</span>"
    )

    odds_banner = (
        f"Cuotas 1X2: <b>1: {m.odds.home_win:.2f}</b> | <b>X: {m.odds.draw:.2f}</b> | <b>2: {m.odds.away_win:.2f}</b>"
        if (m.odds and m.odds.home_win)
        else "Cuotas aún no disponibles"
    )

    scoreboard_html = (
        f"<div class='scoreboard-card notranslate' translate='no'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:14px;'>"
        f"<span class='league-pill'>🏆 {m.tournament.name} • {m.tournament.country or 'Int'} (Ronda {m.round_number or '-'})</span>"
        f"{ref_badge_html}"
        f"<span style='font-size:0.82rem; color:#94a3b8;'>{odds_banner}</span>"
        f"</div>"
        f"<div style='display:grid; grid-template-columns: 1fr 190px 1fr; align-items:center; text-align:center;'>"
        f"<div style='text-align:left;'>"
        f"<div class='team-title'>{m.home_team.name}</div>"
        f"<div class='elo-tag'>⚡ Elo: <b style='color:#38bdf8;'>{hf.elo_rating:.0f}</b>{h_pos_str} | {hf.venue_record} ({hf.venue_xg_for_avg:.2f} xG Casa)</div>"
        f"<div style='margin-top:4px; font-size:0.79rem; color:#cbd5e1;'>⏱️ {hf.fatigue_label}</div>"
        f"<div style='margin-top:7px;'>Racha General: {render_form_badges(hf.form_string)}</div>"
        f"</div>"
        f"<div>{score_center_html}<div style='margin-top:6px; font-size:0.80rem; color:#94a3b8;'>Marcador más prob: <b style='color:#fff;'>{p.most_likely_score}</b></div></div>"
        f"<div style='text-align:right;'>"
        f"<div class='team-title'>{m.away_team.name}</div>"
        f"<div class='elo-tag'>⚡ Elo: <b style='color:#f43f5e;'>{af.elo_rating:.0f}</b>{a_pos_str} | {af.venue_record} ({af.venue_xg_for_avg:.2f} xG Visita)</div>"
        f"<div style='margin-top:4px; font-size:0.79rem; color:#cbd5e1;'>⏱️ {af.fatigue_label}</div>"
        f"<div style='margin-top:7px;'>Racha General: {render_form_badges(af.form_string)}</div>"
        f"</div>"
        f"</div>"
        f"<div style='margin-top:18px;'>"
        f"<div style='display:flex; justify-content:space-between; font-size:0.84rem; font-weight:700; margin-bottom:5px;'>"
        f"<span style='color:#38bdf8;'>Local Ensamble (Dixon-Coles {p.stat_prob_home:.0f}% + ML {p.ml_prob_home:.0f}%): {p.prob_home_win:.1f}%</span>"
        f"<span style='color:#eab308;'>Empate: {p.prob_draw:.1f}%</span>"
        f"<span style='color:#f43f5e;'>Visita Ensamble (Dixon-Coles {p.stat_prob_away:.0f}% + ML {p.ml_prob_away:.0f}%): {p.prob_away_win:.1f}%</span>"
        f"</div>"
        f"<div style='width:100%; height:11px; border-radius:99px; overflow:hidden; display:flex; background:#0f172a;'>"
        f"<div style='width:{p.prob_home_win}%; background:#38bdf8;'></div>"
        f"<div style='width:{p.prob_draw}%; background:#eab308;'></div>"
        f"<div style='width:{p.prob_away_win}%; background:#f43f5e;'></div>"
        f"</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(scoreboard_html, unsafe_allow_html=True)

    # BANNER DE AUDITORÍA DE ACIERTOS (SI EL PARTIDO YA FINALIZÓ)
    if p.verified_total > 0:
        acc_pct = (p.verified_hits / p.verified_total) * 100.0
        roi_txt = f" | 📈 **ROI Simulado:** `{p.simulated_roi_pct:+.1f}%`" if p.simulated_roi_pct is not None else ""
        brier_txt = f" | 🎯 **Brier Score 1X2:** `{p.brier_score_1x2:.4f}` ({p.brier_calibration_label})" if p.brier_score_1x2 is not None else ""
        st.info(
            f"🎯 **Auditoría Automática de Predicciones (Partido Finalizado):** "
            f"Nuestro modelo acertó **{p.verified_hits} de {p.verified_total} mercados principales ({acc_pct:.1f}% de efectividad)** en este encuentro."
            f"{roi_txt}{brier_txt}"
        )

    # NUEVO v4: BANNER DE CHOQUE TÁCTICO & URGENCIA DE TABLA
    st.markdown(
        f"<div style='background:#1e293b; border-left: 4px solid #38bdf8; padding: 10px 14px; border-radius: 6px; margin: 10px 0; font-size: 0.88rem;'>"
        f"<div>⚔️ <b>Choque Táctico:</b> <span style='color:#38bdf8; font-weight:700;'>{m.home_team.name} ({hf.tactical_style})</span> vs <span style='color:#f43f5e; font-weight:700;'>{m.away_team.name} ({af.tactical_style})</span></div>"
        f"<div style='color:#cbd5e1; margin-top:2px;'>{p.tactical_clash_narrative}</div>"
        f"<div style='margin-top:6px; font-size:0.82rem; color:#94a3b8;'>🏆 <b>Contexto de Tabla & Necesidad:</b> {p.urgency_clash_summary} | 🎯 <b>Calidad de Definición:</b> {p.shot_quality_edge}</div>"
        f"<div style='margin-top:4px; font-size:0.82rem; color:#cbd5e1;'>🛡️ <b>Sinergia Zaga Defensiva:</b> {m.home_team.name}: {p.defense_status_home} ({p.defense_synergy_home_pct:.0f}%) | {m.away_team.name}: {p.defense_status_away} ({p.defense_synergy_away_pct:.0f}%) &nbsp;|&nbsp; 🌐 <b>Calibración de Liga:</b> <span style='color:#38bdf8;'>{p.league_profile_name}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # BANNER DE CAÍDA DE CUOTAS (DROPPING ODDS / SMART MONEY)
    if m.odds and m.odds.smart_money_side:
        st.warning(f"📉 **ALERTA DE MOVIMIENTO DE CUOTAS (Smart Money):** {m.odds.smart_money_side}")

    # BANNER DE PENALIZACIÓN POR BAJAS CLAVE
    if p.injury_penalty_home_pct > 0 or p.injury_penalty_away_pct > 0:
        msg_parts = []
        if p.injury_penalty_home_pct > 0:
            msg_parts.append(
                f"**{m.home_team.name}** (-{p.injury_penalty_home_pct:.1f}% potencial ofensivo por baja de: {', '.join(p.missing_key_home)})"
            )
        if p.injury_penalty_away_pct > 0:
            msg_parts.append(
                f"**{m.away_team.name}** (-{p.injury_penalty_away_pct:.1f}% potencial ofensivo por baja de: {', '.join(p.missing_key_away)})"
            )
        st.error("🩺 **Ajuste Automático por Bajas Ofensivas Clave:** " + " | ".join(msg_parts))

    # PESTAÑAS INTERNAS DEL PARTIDO
    tab_preds, tab_shots_players, tab_mc_ml, tab_stats, tab_history = st.tabs(
        [
            "🎯 Todos los Pronósticos (10 Mercados)",
            "🧤 Porteros, Remates al Arco & Jugadores",
            "🎲 Monte Carlo (Hándicaps) & IA (ML)",
            "📊 Estadísticas Opta, Momentum & Matriz",
            "📜 Historial H2H Profundo, Descanso & Bajas",
        ]
    )

    # TAB 1: LOS 10 MERCADOS PRINCIPALES + AGREGAR A BOLETO
    with tab_preds:
        if p.value_edge_home is not None:
            edges = [
                (f"Victoria Local ({m.home_team.name})", p.value_edge_home, m.odds.home_win if m.odds else None, p.prob_home_win),
                ("Empate (X)", p.value_edge_draw or -99.0, m.odds.draw if m.odds else None, p.prob_draw),
                (f"Victoria Visitante ({m.away_team.name})", p.value_edge_away or -99.0, m.odds.away_win if m.odds else None, p.prob_away_win),
            ]
            best_label, best_edge, best_odd, best_prob = max(edges, key=lambda x: x[1])
            if best_edge >= 3.5 and best_odd:
                st.success(
                    f"💎 **VALUE BET DETECTADA:** **{best_label}** — Probabilidad del modelo: **{best_prob:.1f}%** "
                    f"vs Cuota de mercado **{best_odd:.2f}** (Ventaja matemática: **+{best_edge:.1f}%**)"
                )

        if p.verified_total > 0:
            acc_pct = (p.verified_hits / p.verified_total) * 100.0
            st.markdown(
                f"<div style='background:linear-gradient(135deg, #1e293b, #0f172a); border:1px solid #334155; border-radius:10px; padding:12px 18px; margin-bottom:14px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
                f"<div><b>📊 Auditoría de este Partido:</b> Acertó <span style='color:#10b981; font-weight:800;'>{p.verified_hits}</span> de <span style='color:#f8fafc; font-weight:800;'>{p.verified_total}</span> pronósticos (<span style='color:#38bdf8; font-weight:800;'>{acc_pct:.1f}%</span> de efectividad)</div>"
                f"<div><span style='background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.35); padding:3px 8px; border-radius:5px; font-weight:700; font-size:0.80rem;'>🟢 {p.verified_hits} Aciertos</span> "
                f"<span style='background:rgba(244,63,94,0.15); color:#f43f5e; border:1px solid rgba(244,63,94,0.35); padding:3px 8px; border-radius:5px; font-weight:700; font-size:0.80rem;'>🔴 {p.verified_total - p.verified_hits} Fallos</span></div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("📈 Ver Gráfica de Rendimiento de Este Partido"):
                single_breakdown = compute_market_accuracy_breakdown([p])
                st.plotly_chart(plot_market_accuracy_chart(single_breakdown), use_container_width=True)

        picks_list = p.all_market_picks
        for row_start in range(0, len(picks_list), 3):
            cols = st.columns(3)
            for idx_col, item in enumerate(picks_list[row_start : row_start + 3]):
                prob_val = item["prob"]
                if item.get("hit") is True:
                    status_badge = "<span class='hit-yes'>✅ ACIERTO</span>"
                elif item.get("hit") is False:
                    status_badge = "<span class='hit-no'>❌ FALLO</span>"
                elif prob_val >= 64.0:
                    status_badge = "<span class='conf-high'>ALTA CONFIANZA</span>"
                elif prob_val >= 55.0:
                    status_badge = "<span class='conf-med'>CONFIANZA MEDIA</span>"
                else:
                    status_badge = "<span class='conf-low'>PAREJO</span>"

                actual_line = (
                    f"<div style='margin-top:5px; font-size:0.78rem; color:#e2e8f0; font-weight:700;'>📌 {item['actual']}</div>"
                    if item.get("actual")
                    else ""
                )

                with cols[idx_col]:
                    card_html = (
                        f"<div class='market-card notranslate' translate='no'>"
                        f"<div style='display:flex; justify-content:space-between; align-items:center; gap:4px;'>"
                        f"<span class='market-header'>{item['category']}</span>"
                        f"{status_badge}"
                        f"</div>"
                        f"<div class='market-pick'>{item['pick']}</div>"
                        f"<div style='font-size:0.88rem; font-weight:700; color:#38bdf8;'>Probabilidad: {prob_val:.1f}% (Cuota justa: {100.0/max(1.0, prob_val):.2f})</div>"
                        f"<div class='prob-bar-bg'>"
                        f"<div class='prob-bar-fill' style='width:{min(100.0, prob_val):.1f}%;'></div>"
                        f"</div>"
                        f"<div style='font-size:0.78rem; color:#94a3b8;'>{item['detail']}</div>"
                        f"{actual_line}"
                        f"</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

        add_col1, add_col2 = st.columns([3, 1])
        with add_col1:
            pick_to_add_idx = st.selectbox(
                "🎟️ Selecciona un pronóstico de este partido para añadir a tu Boleto / Combinada:",
                options=range(len(picks_list)),
                format_func=lambda i: f"{picks_list[i]['category']}: {picks_list[i]['pick']} ({picks_list[i]['prob']:.1f}%)",
            )
        with add_col2:
            st.write("")
            if st.button("➕ Añadir al Boleto", use_container_width=True):
                chosen_pk = picks_list[pick_to_add_idx]
                st.session_state["bet_slip"].append(
                    {
                        "Partido": f"{m.home_team.name} vs {m.away_team.name}",
                        "Mercado": chosen_pk["category"],
                        "Selección": chosen_pk["pick"],
                        "Probabilidad (%)": chosen_pk["prob"],
                        "Cuota Justa": round(100.0 / max(1.0, chosen_pk["prob"]), 2),
                    }
                )
                st.toast(f"Añadido: {chosen_pk['pick']} ({m.home_team.name} vs {m.away_team.name})")

        st.divider()
        st.markdown("#### 📋 Desglose Completo de Líneas (Goles, 1er Tiempo Opta, Remates, Atajadas del Portero, Córners y Tarjetas)")
        l_col1, l_col2, l_col3 = st.columns(3)

        with l_col1:
            st.markdown("##### ⚽ Goles FT, 1er Tiempo Opta & Cuotas")
            init_h = f"{m.odds.initial_home_win:.2f}" if (m.odds and m.odds.initial_home_win) else "-"
            curr_h = f"{m.odds.home_win:.2f}" if (m.odds and m.odds.home_win) else "-"
            init_a = f"{m.odds.initial_away_win:.2f}" if (m.odds and m.odds.initial_away_win) else "-"
            curr_a = f"{m.odds.away_win:.2f}" if (m.odds and m.odds.away_win) else "-"
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Mercado": f"1 - Gana {m.home_team.name}", "Prob. Modelo": f"{p.prob_home_win:.1f}%", "Cuota (Apert. ➔ Actual)": f"{init_h} ➔ {curr_h}"},
                        {"Mercado": "X - Empate", "Prob. Modelo": f"{p.prob_draw:.1f}%", "Cuota (Apert. ➔ Actual)": f"{m.odds.initial_draw if (m.odds and m.odds.initial_draw) else '-'} ➔ {m.odds.draw if (m.odds and m.odds.draw) else '-'}"},
                        {"Mercado": f"2 - Gana {m.away_team.name}", "Prob. Modelo": f"{p.prob_away_win:.1f}%", "Cuota (Apert. ➔ Actual)": f"{init_a} ➔ {curr_a}"},
                        {"Mercado": "Doble Oportunidad 1X / X2", "Prob. Modelo": f"1X: {p.prob_1x:.1f}% | X2: {p.prob_x2:.1f}%", "Cuota (Apert. ➔ Actual)": "-"},
                        {"Mercado": "Over 1.5 Goles (FT)", "Prob. Modelo": f"{p.prob_over_1_5:.1f}%", "Cuota (Apert. ➔ Actual)": f"Justa: {100/max(1, p.prob_over_1_5):.2f}"},
                        {"Mercado": "Over 2.5 Goles (FT)", "Prob. Modelo": f"{p.prob_over_2_5:.1f}%", "Cuota (Apert. ➔ Actual)": f"{m.odds.initial_over_2_5 if (m.odds and m.odds.initial_over_2_5) else '-'} ➔ {m.odds.over_2_5 if (m.odds and m.odds.over_2_5) else '-'}"},
                        {"Mercado": "Over 3.5 Goles (FT)", "Prob. Modelo": f"{p.prob_over_3_5:.1f}%", "Cuota (Apert. ➔ Actual)": f"Justa: {100/max(1, p.prob_over_3_5):.2f}"},
                        {"Mercado": "1er Tiempo: Over 0.5 Goles (+0.5 1H)", "Prob. Modelo": f"{p.prob_1h_over_0_5:.1f}%", "Cuota (Apert. ➔ Actual)": f"Justa: {100/max(1, p.prob_1h_over_0_5):.2f}"},
                        {"Mercado": "1er Tiempo: Over 1.5 Goles (+1.5 1H)", "Prob. Modelo": f"{p.prob_1h_over_1_5:.1f}%", "Cuota (Apert. ➔ Actual)": f"Justa: {100/max(1, p.prob_1h_over_1_5):.2f}"},
                        {"Mercado": "1er Tiempo 1X2 (Descanso)", "Prob. Modelo": f"1: {p.prob_1h_home_win:.0f}% | X: {p.prob_1h_draw:.0f}% | 2: {p.prob_1h_away_win:.0f}%", "Cuota (Apert. ➔ Actual)": "-"},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        with l_col2:
            st.markdown("##### 🎯 Remates al Arco & 🧤 Atajadas del Portero")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Mercado": "Remates al Arco Esperados (Total)", "Proyección / Prob.": f"{p.expected_sot_total:.1f} ({p.expected_sot_home:.1f} vs {p.expected_sot_away:.1f})"},
                        {"Mercado": "Over 7.5 / 8.5 Remates al Arco", "Proyección / Prob.": f"+7.5: {p.prob_over_7_5_sot:.1f}% | +8.5: {p.prob_over_8_5_sot:.1f}%"},
                        {"Mercado": "Remates Totales Esperados", "Proyección / Prob.": f"{p.expected_shots_total:.1f} (+24.5: {p.prob_over_24_5_shots:.1f}%)"},
                        {"Mercado": f"🧤 Atajadas {p.gk_home_name} (vs {p.expected_sot_away:.1f} SoT)", "Proyección / Prob.": f"{p.expected_saves_home_gk:.2f} ataj. (Hist: {p.gk_home_hist_saves_avg:.2f}/P)"},
                        {"Mercado": f"🧤 {p.gk_home_name}: 2+ / 3+ / 4+ Atajadas", "Proyección / Prob.": f"2+: {p.prob_home_gk_2plus_saves:.0f}% | 3+: {p.prob_home_gk_3plus_saves:.0f}% | 4+: {p.prob_home_gk_4plus_saves:.0f}%"},
                        {"Mercado": f"🧤 Atajadas {p.gk_away_name} (vs {p.expected_sot_home:.1f} SoT)", "Proyección / Prob.": f"{p.expected_saves_away_gk:.2f} ataj. (Hist: {p.gk_away_hist_saves_avg:.2f}/P)"},
                        {"Mercado": f"🧤 {p.gk_away_name}: 2+ / 3+ / 4+ Atajadas", "Proyección / Prob.": f"2+: {p.prob_away_gk_2plus_saves:.0f}% | 3+: {p.prob_away_gk_3plus_saves:.0f}% | 4+: {p.prob_away_gk_4plus_saves:.0f}%"},
                        {"Mercado": "🧤 Atajadas Totales del Partido", "Proyección / Prob.": f"{p.expected_saves_total:.1f} atajadas combinadas"},
                        {"Mercado": "🧤 Over 4.5 / 5.5 / 6.5 Atajadas Totales", "Proyección / Prob.": f"+4.5: {p.prob_over_4_5_total_saves:.0f}% | +5.5: {p.prob_over_5_5_total_saves:.0f}% | +6.5: {p.prob_over_6_5_total_saves:.0f}%"},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        with l_col3:
            st.markdown("##### 🚩 Córners (FT / 1H Opta) & Tarjetas (Árbitro)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Mercado": "Córners Esperados (FT)", "Proyección / Prob.": f"{p.expected_corners_total:.1f} ({p.expected_corners_home:.1f} vs {p.expected_corners_away:.1f})"},
                        {"Mercado": "Córners 1er Tiempo (Opta 1ST)", "Proyección / Prob.": f"{p.expected_1h_corners:.1f} (+4.5 1H: {p.prob_1h_over_4_5_corners:.1f}%)"},
                        {"Mercado": "Over 8.5 Córners (FT)", "Proyección / Prob.": f"{p.prob_over_8_5_corners:.1f}%"},
                        {"Mercado": "Over 9.5 Córners (FT)", "Proyección / Prob.": f"{p.prob_over_9_5_corners:.1f}%"},
                        {"Mercado": "Over 10.5 Córners (FT)", "Proyección / Prob.": f"{p.prob_over_10_5_corners:.1f}%"},
                        {"Mercado": "Tarjetas Amarillas Esperadas", "Proyección / Prob.": f"{p.expected_cards_total:.1f} ({p.expected_cards_home:.1f} vs {p.expected_cards_away:.1f})"},
                        {"Mercado": "Over 3.5 Tarjetas", "Proyección / Prob.": f"{p.prob_over_3_5_cards:.1f}%"},
                        {"Mercado": "Over 4.5 Tarjetas", "Proyección / Prob.": f"{p.prob_over_4_5_cards:.1f}%"},
                        {"Mercado": "Over 5.5 Tarjetas", "Proyección / Prob.": f"{p.prob_over_5_5_cards:.1f}%"},
                        {"Mercado": "Árbitro Asignado", "Proyección / Prob.": f"{p.referee_name or 'Por designar'} ({p.referee_cards_avg or '-'} tarj/P)"},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    # TAB 2: PORTEROS (ATAJADAS), REMATES AL ARCO Y JUGADORES (PLAYER PROPS)
    with tab_shots_players:
        st.markdown("#### 🧤 Duelo de Porteros (Opta GK Model): Atajadas Proyectadas según Disparos al Arco del Rival")
        st.caption(
            "El modelo cruza un **65% los Remates al Arco Proyectados del Contrincante × % de Atajada Efectiva** "
            "(ajustado por el % de tiros lejanos fuera del área del rival) + **35% el Promedio Histórico de Atajadas del Arquero** y sus Goles Evitados Opta (`goalsPrevented`)."
        )
        gk_col1, gk_col2 = st.columns(2)

        with gk_col1:
            st.markdown(
                f"<div class='market-card notranslate' translate='no'>"
                f"<div class='market-header'>🧤 PORTERO LOCAL — {m.home_team.name}</div>"
                f"<div class='market-pick'>{p.gk_home_name} — Proyección: <span style='color:#38bdf8;'>{p.expected_saves_home_gk:.2f} Atajadas</span></div>"
                f"<div style='font-size:0.84rem; color:#cbd5e1; margin-bottom:6px;'>"
                f"• Disparos al Arco Proyectados de <b>{m.away_team.name}</b> hoy: <b>{p.expected_sot_away:.2f} SoT</b> ({100.0 - af.inside_box_shot_pct:.0f}% desde fuera del área)<br>"
                f"• Promedio Histórico del Arquero: <b>{p.gk_home_hist_saves_avg:.2f} atajadas/P</b> (Efectividad: <b>{p.gk_home_save_pct:.1f}%</b>)<br>"
                f"• Goles Evitados Opta (PSxG - Goles): <b>{p.gk_home_goals_prevented:+.2f} / P</b>"
                f"</div>"
                f"<div style='font-size:0.85rem; font-weight:700; color:#4ade80;'>"
                f"Probabilidades: 2+ Atajadas: {p.prob_home_gk_2plus_saves:.1f}% | 3+ Atajadas: {p.prob_home_gk_3plus_saves:.1f}% | 4+ Atajadas: {p.prob_home_gk_4plus_saves:.1f}% | 5+ Atajadas: {p.prob_home_gk_5plus_saves:.1f}%"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with gk_col2:
            st.markdown(
                f"<div class='market-card notranslate' translate='no'>"
                f"<div class='market-header'>🧤 PORTERO VISITANTE — {m.away_team.name}</div>"
                f"<div class='market-pick'>{p.gk_away_name} — Proyección: <span style='color:#f43f5e;'>{p.expected_saves_away_gk:.2f} Atajadas</span></div>"
                f"<div style='font-size:0.84rem; color:#cbd5e1; margin-bottom:6px;'>"
                f"• Disparos al Arco Proyectados de <b>{m.home_team.name}</b> hoy: <b>{p.expected_sot_home:.2f} SoT</b> ({100.0 - hf.inside_box_shot_pct:.0f}% desde fuera del área)<br>"
                f"• Promedio Histórico del Arquero: <b>{p.gk_away_hist_saves_avg:.2f} atajadas/P</b> (Efectividad: <b>{p.gk_away_save_pct:.1f}%</b>)<br>"
                f"• Goles Evitados Opta (PSxG - Goles): <b>{p.gk_away_goals_prevented:+.2f} / P</b>"
                f"</div>"
                f"<div style='font-size:0.85rem; font-weight:700; color:#4ade80;'>"
                f"Probabilidades: 2+ Atajadas: {p.prob_away_gk_2plus_saves:.1f}% | 3+ Atajadas: {p.prob_away_gk_3plus_saves:.1f}% | 4+ Atajadas: {p.prob_away_gk_4plus_saves:.1f}% | 5+ Atajadas: {p.prob_away_gk_5plus_saves:.1f}%"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("#### 👟 Jugadores Más Probables que Rematen a Puerta, Rematen Total o Anoten Gol (Calibrado con Titularidad y Rival)")
        p_col1, p_col2 = st.columns(2)

        def _render_player_table(players: List[PlayerPropSchema], team_title: str) -> None:
            st.markdown(f"##### 🛡️ {team_title}")
            if not players:
                st.info("No hay estadísticas individuales de jugadores registradas para este torneo.")
                return
            rows_p = []
            for pl in players[:10]:
                if pl.is_missing:
                    name_lbl = f"⚠️ {pl.name} (BAJA)"
                elif pl.is_starter is True:
                    name_lbl = f"🟢 {pl.name} (Titular)"
                elif pl.is_starter is False:
                    name_lbl = f"🟡 {pl.name} (Supl.)"
                else:
                    name_lbl = pl.name

                rows_p.append(
                    {
                        "Jugador": name_lbl,
                        "Pos": pl.position or "-",
                        "PJ": str(pl.appearances),
                        "Rem. Arco / P": pl.sot_per_match,
                        "Prob. 1+ Rem. Arco": f"{pl.prob_1plus_sot:.0f}%",
                        "Prob. 2+ Rem. Arco": f"{pl.prob_2plus_sot:.0f}%",
                        "Rem. Totales / P": pl.shots_per_match,
                        "Prob. 2+ Rem. Totales": f"{pl.prob_2plus_shots:.0f}%",
                        "Prob. Gol (xG/xGOT)": f"{pl.prob_anytime_goal:.0f}% ({pl.xg_per_match:.2f} xG)",
                        "Pases Clave / P": pl.key_passes_per_match,
                    }
                )
            st.dataframe(pd.DataFrame(rows_p), use_container_width=True, hide_index=True)

        with p_col1:
            _render_player_table(h_players, m.home_team.name)
        with p_col2:
            _render_player_table(a_players, m.away_team.name)

        combined_shooters = [pl for pl in (h_players + a_players) if not pl.is_missing and pl.sot_per_match > 0]
        combined_shooters.sort(key=lambda x: x.prob_1plus_sot, reverse=True)
        if combined_shooters:
            st.markdown("##### 🎯 Top 10 Jugadores del Partido con Mayor Probabilidad de Rematar al Arco (+0.5 SoT)")
            top10_df = pd.DataFrame(
                [
                    {
                        "Jugador": f"{pl.name} ({pl.team_name})",
                        "Equipo": pl.team_name,
                        "Probabilidad 1+ Remate al Arco (%)": pl.prob_1plus_sot,
                        "Promedio Remates al Arco / P": pl.sot_per_match,
                    }
                    for pl in combined_shooters[:10]
                ]
            )
            fig_shooters = px.bar(
                top10_df,
                x="Probabilidad 1+ Remate al Arco (%)",
                y="Jugador",
                color="Equipo",
                orientation="h",
                text_auto=".1f",
                hover_data=["Promedio Remates al Arco / P"],
            )
            fig_shooters.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_shooters, use_container_width=True)

    # TAB 3: SIMULADOR MONTE CARLO (10,000 PARTIDOS) + ENSAMBLE MACHINE LEARNING
    with tab_mc_ml:
        mc_col1, mc_col2 = st.columns(2)

        with mc_col1:
            st.markdown("#### 🎲 Simulador Monte Carlo (10,000 Iteraciones): Hándicap Asiático & Equipos")
            st.caption("Simulación estocástica Gamma-Poisson con sobredispersión para mercados de Hándicap Asiático, DNB y Goles por Equipo.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Mercado Monte Carlo (10,000 sims)": f"Apuesta Sin Empate — {m.home_team.name} (DNB 1)", "Probabilidad": f"{p.prob_dnb_home:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_dnb_home), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"Apuesta Sin Empate — {m.away_team.name} (DNB 2)", "Probabilidad": f"{p.prob_dnb_away:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_dnb_away), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"Hándicap Asiático {m.home_team.name} -1.5 (Gana por 2+)", "Probabilidad": f"{p.prob_ah_home_minus_1_5:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_ah_home_minus_1_5), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"Hándicap Asiático {m.home_team.name} -1.0 (Gana o Nulo)", "Probabilidad": f"{p.prob_ah_home_minus_1_0_win_or_push:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_ah_home_minus_1_0_win_or_push), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"Hándicap Asiático {m.away_team.name} -1.5 (Gana por 2+)", "Probabilidad": f"{p.prob_ah_away_minus_1_5:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_ah_away_minus_1_5), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"Hándicap Asiático {m.away_team.name} +1.5 (Pierde máx por 1)", "Probabilidad": f"{p.prob_ah_away_plus_1_5:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_ah_away_plus_1_5), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"{m.home_team.name} Anota +0.5 Goles", "Probabilidad": f"{p.prob_home_over_0_5:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_home_over_0_5), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"{m.home_team.name} Anota +1.5 Goles", "Probabilidad": f"{p.prob_home_over_1_5:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_home_over_1_5), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"{m.home_team.name} Anota +2.5 Goles", "Probabilidad": f"{p.prob_home_over_2_5:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_home_over_2_5), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"{m.away_team.name} Anota +0.5 Goles", "Probabilidad": f"{p.prob_away_over_0_5:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_away_over_0_5), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"{m.away_team.name} Anota +1.5 Goles", "Probabilidad": f"{p.prob_away_over_1_5:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_away_over_1_5), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"Portería a Cero (Clean Sheet) {m.home_team.name}", "Probabilidad": f"{p.prob_home_clean_sheet:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_home_clean_sheet), 2)},
                        {"Mercado Monte Carlo (10,000 sims)": f"Portería a Cero (Clean Sheet) {m.away_team.name}", "Probabilidad": f"{p.prob_away_clean_sheet:.1f}%", "Cuota Justa": round(100.0 / max(1.0, p.prob_away_clean_sheet), 2)},
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        with mc_col2:
            st.markdown("#### 🤖 Ensamble Machine Learning (`GradientBoostingClassifier`)")
            st.caption(
                f"Combina 65% Modelo Matemático Dixon-Coles (Opta + H2H) + 35% Gradient Boosting entrenado con `{p.ml_db_samples}` partidos reales en SQLite + Prior Calibrado."
            )
            ens_df = pd.DataFrame(
                [
                    {
                        "Resultado": f"1 - {m.home_team.name}",
                        "Modelo Dixon-Coles (%)": p.stat_prob_home,
                        "Modelo IA Gradient Boosting (%)": p.ml_prob_home,
                        "Ensamble Final (%)": p.prob_home_win,
                    },
                    {
                        "Resultado": "X - Empate",
                        "Modelo Dixon-Coles (%)": p.stat_prob_draw,
                        "Modelo IA Gradient Boosting (%)": p.ml_prob_draw,
                        "Ensamble Final (%)": p.prob_draw,
                    },
                    {
                        "Resultado": f"2 - {m.away_team.name}",
                        "Modelo Dixon-Coles (%)": p.stat_prob_away,
                        "Modelo IA Gradient Boosting (%)": p.ml_prob_away,
                        "Ensamble Final (%)": p.prob_away_win,
                    },
                ]
            )
            st.dataframe(ens_df, use_container_width=True, hide_index=True)

            if p.ml_feature_importances:
                st.markdown("##### 📊 Peso de las Variables en la Decisión del Modelo IA (Feature Importance)")
                imp_df = pd.DataFrame(p.ml_feature_importances)
                fig_imp = px.bar(
                    imp_df,
                    x="Importancia (%)",
                    y="Variable",
                    orientation="h",
                    text_auto=".1f",
                )
                fig_imp.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=300,
                    margin=dict(l=20, r=20, t=15, b=15),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

    # TAB 4: ANÁLISIS ESTADÍSTICO OPTA, MOMENTUM Y MATRIZ DE MARCADOR EXACTO
    with tab_stats:
        if m.momentum and m.momentum.points:
            st.markdown(f"#### ⚡ Análisis de Presión Minuto a Minuto ({m.momentum.goal_alert_level} — {m.momentum.goal_alert_reason})")
            st.plotly_chart(
                plot_momentum_chart(m.momentum, m.home_team.name, m.away_team.name),
                use_container_width=True,
            )
            st.divider()

        s_col1, s_col2 = st.columns([1.05, 1.1])

        with s_col1:
            st.markdown("#### ⚖️ Comparativa Lado a Lado: Micro-Métricas Opta, Portería & Descanso")
            render_stat_comparison_bar("Índice de Sinergia Zaga Defensiva (%)", p.defense_synergy_home_pct, p.defense_synergy_away_pct, "%")
            render_stat_comparison_bar("Presión Alta PPDA Opta (Menor = Más Asfixiante)", p.ppda_home, p.ppda_away)
            render_stat_comparison_bar("Índice de Urgencia en la Tabla (0-100)", hf.urgency_index, af.urgency_index)
            render_stat_comparison_bar("Calidad de Disparo (xG por Remate)", hf.xg_per_shot_for, af.xg_per_shot_for)
            render_stat_comparison_bar("Días de Descanso Previos", hf.rest_days, af.rest_days, "d")
            render_stat_comparison_bar("Índice de Peligro en el Área Opta (%)", p.opta_danger_home, p.opta_danger_away, "%")
            render_stat_comparison_bar("xG Específico en esta Condición (Casa vs Visita)", hf.venue_xg_for_avg, af.venue_xg_for_avg)
            render_stat_comparison_bar("Calidad de Remate al Arco Opta (xGOT / P)", hf.xgot_for_avg, af.xgot_for_avg)
            render_stat_comparison_bar("Toques en el Área Rival / P (Opta Box Touches)", hf.touches_in_box_avg, af.touches_in_box_avg)
            render_stat_comparison_bar("Remates Dentro del Área / P", hf.shots_in_box_for_avg, af.shots_in_box_for_avg)
            render_stat_comparison_bar("Entradas al Último Tercio / P (Dominio Territorial)", hf.final_third_entries_avg, af.final_third_entries_avg)
            render_stat_comparison_bar("Recuperaciones de Balón / P (Presión Alta)", hf.ball_recoveries_avg, af.ball_recoveries_avg)
            render_stat_comparison_bar("xG Real en el 1er Tiempo (Opta 1ST Period)", hf.ht_xg_for_avg, af.ht_xg_for_avg)
            render_stat_comparison_bar("Atajadas Promedio del Portero / P", p.gk_home_hist_saves_avg, p.gk_away_hist_saves_avg)
            render_stat_comparison_bar("Efectividad de Atajada del Portero (%)", p.gk_home_save_pct, p.gk_away_save_pct, "%")
            render_stat_comparison_bar("Córners a Favor / P", hf.corners_for_avg, af.corners_for_avg)
            render_stat_comparison_bar("Posesión Promedio (%)", hf.possession_avg, af.possession_avg, "%")

        with s_col2:
            st.markdown(f"#### 🔥 Mapa de Calor: Matriz de Marcador Exacto ({p.league_profile_name})")
            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=p.score_matrix,
                    x=[f"{a} ({m.away_team.short_name or m.away_team.name})" for a in range(6)],
                    y=[f"{h} ({m.home_team.short_name or m.home_team.name})" for h in range(6)],
                    colorscale="Tealgrn",
                    text=[[f"{val:.1f}%" for val in row] for row in p.score_matrix],
                    texttemplate="%{text}",
                    hoverinfo="x+y+text",
                )
            )
            fig_hm.update_layout(
                xaxis_title=f"Goles Visitante ({m.away_team.name})",
                yaxis_title=f"Goles Local ({m.home_team.name})",
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown("##### 🏅 Top 6 Marcadores Exactos Más Probables")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Marcador Exacto (Local - Visita)": item["score"],
                            "Probabilidad (%)": f"{item['prob']:.1f}%",
                            "Cuota Equivalente": round(100.0 / max(0.5, item["prob"]), 2),
                        }
                        for item in p.top_scores
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.divider()
            st.markdown(f"##### ⏱️ Desglose de Probabilidad de Gol por Tramos de Tiempo ({p.prob_most_active_half})")
            st.caption("Distribución estadística proyectada de goles a lo largo de los 90 minutos de juego:")
            st.dataframe(pd.DataFrame(p.time_windows), use_container_width=True, hide_index=True)

    # TAB 5: HISTORIAL H2H PROFUNDO, RECIENTES, DESCANSO Y BAJAS
    with tab_history:
        if m.h2h:
            st.markdown(
                f"#### 🤝 Historial de Enfrentamientos Directos (H2H Total: `{m.h2h.total_matches}` partidos | Recientes Analizados a Fondo: `{m.h2h.recent_matches_analyzed}`)\n"
                f"- **Victorias {m.home_team.name}:** `{m.h2h.team_wins_home}` | **Empates:** `{m.h2h.draws}` | **Victorias {m.away_team.name}:** `{m.h2h.team_wins_away}`"
            )
            if m.h2h.recent_matches_analyzed > 0:
                hk1, hk2, hk3, hk4, hk5 = st.columns(5)
                hk1.metric("Goles Promedio H2H", f"{m.h2h.h2h_avg_goals:.2f}" if m.h2h.h2h_avg_goals is not None else "-")
                hk2.metric("Ambos Anotan (BTTS H2H)", f"{m.h2h.h2h_btts_pct:.0f}%" if m.h2h.h2h_btts_pct is not None else "-")
                hk3.metric("Over 2.5 Goles H2H", f"{m.h2h.h2h_over_2_5_pct:.0f}%" if m.h2h.h2h_over_2_5_pct is not None else "-")
                hk4.metric("Córners Promedio H2H", f"{m.h2h.h2h_corners_avg:.1f}" if m.h2h.h2h_corners_avg is not None else "-")
                hk5.metric("Tarjetas Promedio H2H", f"{m.h2h.h2h_cards_avg:.1f}" if m.h2h.h2h_cards_avg is not None else "-")

            if m.h2h.recent_h2h_events:
                st.markdown("##### ⚔️ Últimos Enfrentamientos Directos (H2H Partido a Partido)")
                h2h_rows = []
                for hev in m.h2h.recent_h2h_events:
                    h2h_rows.append(
                        {
                            "Fecha": hev.date_str,
                            "Torneo": hev.tournament_name,
                            "Equipo Local": hev.home_team_name,
                            "Marcador": f"{hev.home_score} - {hev.away_score}",
                            "Equipo Visitante": hev.away_team_name,
                            "xG (Loc - Vis)": f"{hev.xg_home:.2f} - {hev.xg_away:.2f}" if (hev.xg_home is not None and hev.xg_away is not None) else "-",
                            "Rem. Arco": str(hev.sot_total) if hev.sot_total is not None else "-",
                            "Atajadas Tot.": str(hev.saves_total) if hev.saves_total is not None else "-",
                            "Córners Tot.": str(hev.corners_total) if hev.corners_total is not None else "-",
                            "Tarjetas Tot.": str(hev.yellow_cards_total) if hev.yellow_cards_total is not None else "-",
                        }
                    )
                st.dataframe(pd.DataFrame(h2h_rows), use_container_width=True, hide_index=True)

        st.divider()
        h_col1, h_col2 = st.columns(2)

        def _format_recent_df(recents: List[MatchDetailSchema], tid: int) -> pd.DataFrame:
            rows_r = []
            for rm in recents:
                is_h = rm.home_team.id == tid
                opp = rm.away_team.name if is_h else rm.home_team.name
                venue = "🏠 Casa" if is_h else "✈️ Visita"
                gf = rm.home_score if is_h else rm.away_score
                ga = rm.away_score if is_h else rm.home_score
                xg_f = (rm.stats.expected_goals_home if is_h else rm.stats.expected_goals_away) if rm.stats else None
                xgot_f = (rm.stats.xg_on_target_home if is_h else rm.stats.xg_on_target_away) if rm.stats else None
                sot_f = (rm.stats.shots_on_target_home if is_h else rm.stats.shots_on_target_away) if rm.stats else None
                sv_f = (rm.stats.saves_home if is_h else rm.stats.saves_away) if rm.stats else None
                crn_f = (rm.stats.corner_kicks_home if is_h else rm.stats.corner_kicks_away) if rm.stats else None
                rows_r.append(
                    {
                        "Fecha": format_colombia_time(rm.start_timestamp or rm.start_time_utc, "%Y-%m-%d"),
                        "Condición": venue,
                        "Rival": opp,
                        "Resultado": f"{gf}-{ga}",
                        "xG / xGOT": f"{xg_f:.2f} / {xgot_f:.2f}" if (xg_f is not None and xgot_f is not None) else (f"{xg_f:.2f}" if xg_f is not None else "-"),
                        "Rem. Arco": str(sot_f) if sot_f is not None else "-",
                        "Atajadas Arq.": str(sv_f) if sv_f is not None else "-",
                        "Córners": str(crn_f) if crn_f is not None else "-",
                    }
                )
            return pd.DataFrame(rows_r)

        with h_col1:
            st.markdown(f"##### 📅 {m.home_team.name} ({hf.venue_record} • {hf.fatigue_label} • Rival Medio: {hf.sos_rating:.2f} PPG)")
            st.dataframe(_format_recent_df(deep["home_recent"], m.home_team.id), use_container_width=True, hide_index=True)

        with h_col2:
            st.markdown(f"##### 📅 {m.away_team.name} ({af.venue_record} • {af.fatigue_label} • Rival Medio: {af.sos_rating:.2f} PPG)")
            st.dataframe(_format_recent_df(deep["away_recent"], m.away_team.id), use_container_width=True, hide_index=True)

        st.divider()
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("##### ⚡ Rachas Oficiales del Partido (Sofascore Streaks)")
            if m.streaks:
                st.dataframe(pd.DataFrame(m.streaks), use_container_width=True, hide_index=True)
            else:
                st.info("No hay rachas registradas para este partido.")

        with b_col2:
            st.markdown("##### 🩺 Bajas por Lesión / Suspensión y Alineaciones")
            if m.home_lineup or m.away_lineup:
                if m.home_lineup:
                    st.write(f"**{m.home_team.name}** — Formación: `{m.home_lineup.formation or 'Por confirmar'}` | Portero: `{p.gk_home_name}`")
                    if m.home_lineup.missing_players:
                        st.error("Bajas / Ausentes: " + ", ".join(m.home_lineup.missing_players))
                    else:
                        st.success("Sin bajas reportadas en la plantilla local.")
                if m.away_lineup:
                    st.write(f"**{m.away_team.name}** — Formación: `{m.away_lineup.formation or 'Por confirmar'}` | Portero: `{p.gk_away_name}`")
                    if m.away_lineup.missing_players:
                        st.error("Bajas / Ausentes: " + ", ".join(m.away_lineup.missing_players))
                    else:
                        st.success("Sin bajas reportadas en la plantilla visitante.")
            else:
                st.info("Alineaciones y parte médico aún no publicados para este encuentro.")


if __name__ == "__main__":
    main()
