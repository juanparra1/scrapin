"""Dashboard Deportivo Pro v3: Radar EN VIVO (Momentum), Monte Carlo 10k (Hándicaps), Fatiga/Descanso, Ensamble ML (Gradient Boosting), Jugadores, Árbitro y Kelly."""

import logging
import sys

logger = logging.getLogger(__name__)
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
    get_precomputed_deep_for_event,
    get_precomputed_deep_for_events,
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
from src.scrapers.odds_api_client import (
    OddsAPIClient,
    calculate_three_way_surebet,
    calculate_two_way_surebet,
    match_and_cross_arbitrage,
)
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

from src.ui.styles import inject_custom_styles_and_scripts
inject_custom_styles_and_scripts()


@st.cache_resource
def get_scrapers() -> tuple[SofascoreScraper, FlashscoreScraper]:
    return (
        SofascoreScraper(rate_limit_seconds=0.30, use_cache=True),
        FlashscoreScraper(rate_limit_seconds=0.45, use_cache=True),
    )


from src.ui.audit_view import (
    compute_market_accuracy_breakdown,
    evaluate_all_systematic_lines_for_match,
    plot_market_accuracy_chart,
    plot_specific_lines_chart,
)


_PRECOMPUTED_DEEP_MEMORY_CACHE: Dict[int, Dict[str, Any]] = {}


def _decode_and_cache_deep_record(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Deserializa bajo demanda el análisis profundo de un registro precalculado y lo guarda en caché de memoria."""
    try:
        eid = int(r["event_id"])
        deep_data = r.get("deep_analysis") or {}
        if "home_form" not in deep_data or "away_form" not in deep_data:
            return None
        pred_obj = MatchPrediction.model_validate(r["prediction"]) if r.get("prediction") else None
        m_obj_cached = MatchDetailSchema.model_validate(r["match_detail"]) if r.get("match_detail") else None
        if not m_obj_cached:
            return None
        fs_id = deep_data.get("flashscore_id", "N/A")
        deep_entry = {
            "match": m_obj_cached,
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
        if len(_PRECOMPUTED_DEEP_MEMORY_CACHE) > 350:
            _PRECOMPUTED_DEEP_MEMORY_CACHE.pop(next(iter(_PRECOMPUTED_DEEP_MEMORY_CACHE)), None)
        _PRECOMPUTED_DEEP_MEMORY_CACHE[eid] = deep_entry
        return deep_entry
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def load_day_catalog(date_str: str, include_all_leagues: bool) -> Dict[str, Any]:
    """
    Carga ultrarrápida de todos los partidos de la fecha con enriquecimiento de Supabase Turbo Cache.
    Omite la columna pesada `deep_analysis_json` al listar la fecha (Lazy Loading) para reducir un 80% el tráfico y uso de CPU.
    """
    precomputed_records = []
    precomputed_deep_cache: Dict[int, Dict[str, Any]] = {}
    precomputed_preds: Dict[int, MatchPrediction] = {}
    fs_lookup: Dict[tuple[int, int], str] = {}

    # 0. Consultar catálogo ligero de Supabase (match_detail + prediction, sin deep_analysis_json)
    try:
        precomputed_records = get_precomputed_matches_for_date(date_str, include_deep=False)
        for r in precomputed_records:
            try:
                eid = r["event_id"]
                if r.get("prediction"):
                    try:
                        pred_obj = MatchPrediction.model_validate(r["prediction"])
                        precomputed_preds[eid] = pred_obj
                    except Exception as pred_err:
                        logger.debug("Error parseando prediccion para %s: %s", eid, pred_err)

                if r.get("match_detail"):
                    m_raw = r["match_detail"]
                    h_id = m_raw.get("home_team", {}).get("id")
                    a_id = m_raw.get("away_team", {}).get("id")
                    if h_id and a_id:
                        fs_lookup[(h_id, a_id)] = m_raw.get("custom_id") or "N/A"

                if eid in _PRECOMPUTED_DEEP_MEMORY_CACHE:
                    precomputed_deep_cache[eid] = _PRECOMPUTED_DEEP_MEMORY_CACHE[eid]
            except Exception as err:
                logger.debug("Error deserializando partido precalculado: %s", err)
    except Exception as e:
        logger.warning(f"Error consultando caché de Supabase: {e}")

    sf_scraper, fs_scraper = get_scrapers()

    # FAST-PATH INSTANTÁNEO (< 0.05s): Si Supabase ya tiene la jornada precalculada y no se piden ligas menores
    if precomputed_records and not include_all_leagues:
        matches_cached: List[MatchDetailSchema] = []
        for r in precomputed_records:
            if r.get("match_detail"):
                try:
                    m_obj = MatchDetailSchema.model_validate(r["match_detail"])
                    if m_obj.event_id in precomputed_deep_cache:
                        cached_m = precomputed_deep_cache[m_obj.event_id]["match"]
                        if m_obj.odds is None and cached_m.odds:
                            m_obj.odds = cached_m.odds
                    matches_cached.append(m_obj)
                except Exception:
                    pass
        if matches_cached:
            matches_cached.sort(key=lambda m: m.start_timestamp)
            return {
                "matches": matches_cached,
                "fs_total": len(matches_cached),
                "fs_lookup": fs_lookup,
                "entity_links": [],
                "source": "⚡ Supabase Turbo Cache (Instantáneo)",
                "precomputed_predictions": precomputed_preds,
                "precomputed_deep_cache": precomputed_deep_cache,
            }

    resolver = EntityResolver(confidence_threshold=78.0)

    # 1. Consultar partidos de Sofascore (con Circuit Breaker instantáneo si la IP en la nube está bloqueada)
    matches: List[MatchDetailSchema] = []
    try:
        matches = sf_scraper.get_scheduled_events(
            date_str=date_str, only_top_tournaments=not include_all_leagues
        )
    except Exception as e:
        logger.warning(f"Error consultando Sofascore: {e}")
        matches = []

    source_label = "⚡ Sofascore + Supabase Turbo Cache" if precomputed_records else "Sofascore Opta"

    # Fallback si Sofascore está bloqueado pero hay datos en Supabase
    if not matches and precomputed_records:
        matches = [MatchDetailSchema.model_validate(r["match_detail"]) for r in precomputed_records if r.get("match_detail")]
        source_label = "⚡ Supabase Cloud Cache (Fallback)"

    # Enriquecer cuotas y análisis precalculados de Supabase
    if precomputed_deep_cache and matches:
        for m in matches:
            if m.event_id in precomputed_deep_cache:
                cached_m = precomputed_deep_cache[m.event_id]["match"]
                if m.odds is None and cached_m.odds:
                    m.odds = cached_m.odds

    # 2. Extraer partidos de Flashscore (solo cuando se necesita cruce o como fallback si Sofascore está bloqueado)
    fs_matches = []
    try:
        col_today = datetime.now(timezone(timedelta(hours=-5))).date()
        t_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        day_offset = max(-7, min(7, (t_dt - col_today).days))
        fs_matches = list(fs_scraper.get_daily_matches(day_offset=day_offset))
        # Si es fallback (sin partidos de Sofascore), consultar también day_offset + 1 por el corte de medianoche CET vs Colombia UTC-5
        if not matches and day_offset < 7:
            seen_fs_ids = {f.flashscore_id for f in fs_matches}
            for f_next in fs_scraper.get_daily_matches(day_offset=day_offset + 1):
                if f_next.flashscore_id not in seen_fs_ids:
                    fs_matches.append(f_next)
    except Exception as e:
        logger.warning(f"Error consultando Flashscore: {e}")
        fs_matches = []

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
            "international", "internacional", "concacaf", "conmebol", "uefa",
            "nations league"
        }

        for fsm in fs_matches:
            try:
                parts = fsm.tournament_title.split(":")
                raw_country = parts[0].strip() if len(parts) > 1 else fsm.country
                country_clean = (raw_country + " " + fsm.tournament_title).lower()

                if not include_all_leagues and not any(c in country_clean for c in curated_countries):
                    continue

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

    entity_links: List[Dict[str, Any]] = []

    if source_label == "Flashscore Cloud":
        for m in matches:
            if m.custom_id:
                fs_lookup[(m.home_team.id, m.away_team.id)] = m.custom_id
    elif matches and fs_matches:
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

    matches.sort(key=lambda m: m.start_timestamp)
    return {
        "matches": matches,
        "fs_total": len(fs_matches),
        "fs_lookup": fs_lookup,
        "entity_links": entity_links,
        "source": source_label,
        "precomputed_predictions": precomputed_preds,
        "precomputed_deep_cache": precomputed_deep_cache,
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
    eid_lookup = match_dict.get("event_id")
    if eid_lookup:
        if eid_lookup in _PRECOMPUTED_DEEP_MEMORY_CACHE:
            cached_entry = _PRECOMPUTED_DEEP_MEMORY_CACHE[eid_lookup]
            if cached_entry.get("prediction") is not None and cached_entry.get("home_form") is not None:
                return cached_entry
        # Carga Perezosa (Lazy Load) desde Supabase por primary key (< 15ms)
        db_rec = get_precomputed_deep_for_event(int(eid_lookup))
        if db_rec:
            decoded_entry = _decode_and_cache_deep_record(db_rec)
            if decoded_entry and decoded_entry.get("prediction") is not None:
                return decoded_entry

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
        if "deep_analysis_cache" in st.session_state:
            st.session_state["deep_analysis_cache"][match.event_id] = {
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
        if "audited_predictions_cache" in st.session_state:
            st.session_state["audited_predictions_cache"][match.event_id] = prediction
        if "audited_matches_cache" in st.session_state:
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

    # Precarga en bloque (1 sola consulta SQL IN) de partidos que estén en Supabase pero aún no en RAM
    missing_eids = []
    for item in match_pairs:
        m_obj = item[0] if isinstance(item, (tuple, list)) else item
        eid = getattr(m_obj, "event_id", None) or (m_obj.get("event_id") if isinstance(m_obj, dict) else None)
        if eid and int(eid) not in _PRECOMPUTED_DEEP_MEMORY_CACHE:
            missing_eids.append(int(eid))
    if missing_eids:
        bulk_records = get_precomputed_deep_for_events(missing_eids)
        for rec_item in bulk_records.values():
            _decode_and_cache_deep_record(rec_item)

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


@st.cache_resource
def get_or_start_scheduler():
    from src.services.scheduler import start_heartbeat_scheduler
    return start_heartbeat_scheduler(interval_seconds=840)


def main() -> None:
    scheduler = get_or_start_scheduler()
    if "bet_slip" not in st.session_state:
        st.session_state["bet_slip"] = []
    if "audited_predictions_cache" not in st.session_state:
        st.session_state["audited_predictions_cache"] = {}
    if "audited_matches_cache" not in st.session_state:
        st.session_state["audited_matches_cache"] = {}

    with st.sidebar:
        st.markdown("## ⚽ ProSports AI v3")
        st.caption("Live Momentum • Monte Carlo 10k • Hándicaps • Fatiga • Ensamble ML • Kelly")

        sched_status = scheduler.get_status()
        status_color = "#4ade80" if sched_status["is_running"] else "#f87171"
        st.markdown(
            f"<div style='background-color:#1e293b; padding:8px 12px; border-radius:6px; border-left:4px solid {status_color}; font-size:0.80rem; margin:6px 0;'>"
            f"<b>Heartbeat 24/7:</b> {'🟢 Activo' if sched_status['is_running'] else '🔴 Inactivo'} ({sched_status['cycles_count']} ciclos)<br/>"
            f"<span style='color:#94a3b8;'>Último: {sched_status['last_run_str']} • Próx: en {sched_status['mins_until_next']}m</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("⚡ Sincronizar Ahora", help="Fuerza un ciclo de actualización y ping a Supabase", use_container_width=True):
            with st.spinner("Sincronizando con Supabase y Sofascore..."):
                scheduler.trigger_now()
            st.rerun()

        st.divider()

        col_today = datetime.now(COLOMBIA_TZ).date()
        selected_date = st.date_input("📅 1. Selecciona la Fecha", value=col_today)
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
            st.cache_resource.clear()
            # Limpiar caché de archivos temporales en disco
            from src.config import CACHE_DIR
            try:
                for f in CACHE_DIR.glob("*.json"):
                    f.unlink(missing_ok=True)
            except Exception:
                pass
            # Limpiar memoria de la sesión
            st.session_state.pop("audited_matches_cache", None)
            st.session_state.pop("audited_predictions_cache", None)
            st.session_state.pop("deep_analysis_cache", None)
            st.session_state.pop("attempted_odds_events", None)
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
                "💼 Asignador Maestro de Capital & Portafolio",
                "📊 Auditoría de Pronósticos (Aciertos/Fallos)",
                "🧠 Auto-Aprendizaje e IA Adaptativa",
                "🔴 En Vivo & Momentum",
                "🔥 Radar Top Picks",
                "📋 Catálogo Completo",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )

    if view_mode == "📊 Auditoría de Pronósticos (Aciertos/Fallos)":
        from src.ui.audit_view import render_audit_view
        render_audit_view(selected_date, date_str, catalog, load_day_catalog, analyze_matches_batch_parallel)
        return

    if view_mode == "🧠 Auto-Aprendizaje e IA Adaptativa":
        from src.ui.learner_view import render_learner_view
        render_learner_view(selected_date, date_str, catalog, load_day_catalog, analyze_matches_batch_parallel)
        return

    if view_mode == "🔴 En Vivo & Momentum":
        from src.ui.live_view import render_live_view
        sf_scraper, _ = get_scrapers()
        render_live_view(sf_scraper)
        return

    if not filtered_matches or selected_match_idx is None:
        st.warning("No se encontraron partidos con los filtros seleccionados.")
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

    if view_mode in ("🔥 Radar Top Picks", "🔥 Radar Top Picks & Auditoría"):
        from src.ui.audit_view import render_radar_top_picks_view
        render_radar_top_picks_view(filtered_matches, catalog, analyze_matches_batch_parallel)
        return

    # MODO 3: ASIGNADOR MAESTRO DE CAPITAL & GESTIÓN DE PORTAFOLIO CUANTITATIVO
    if view_mode == "💼 Asignador Maestro de Capital & Portafolio":
        from src.ui.portfolio_view import render_portfolio_allocator_view
        sf_scraper_inst, _ = get_scrapers()
        render_portfolio_allocator_view(
            filtered_matches=filtered_matches,
            catalog=catalog,
            date_str=date_str,
            sf_scraper=sf_scraper_inst,
            analyze_matches_batch_parallel_fn=analyze_matches_batch_parallel,
        )
        return

    # MODO 1: ANÁLISIS PROFUNDO DEL PARTIDO SELECCIONADO
    if view_mode != "🎯 Análisis del Partido":
        return

    target_match = filtered_matches[selected_match_idx]
    fs_id = catalog["fs_lookup"].get((target_match.home_team.id, target_match.away_team.id), "N/A")

    if "deep_analysis_cache" in st.session_state and target_match.event_id in st.session_state["deep_analysis_cache"]:
        deep = st.session_state["deep_analysis_cache"][target_match.event_id]
    else:
        deep = analyze_single_match_deep(target_match.model_dump(), fs_id)

    from src.ui.match_detail_view import render_match_detail_view
    render_match_detail_view(deep, format_colombia_time)


if __name__ == "__main__":
    main()
