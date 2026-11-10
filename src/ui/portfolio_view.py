"""Vista modularizada para el Asignador Maestro de Capital & Portafolio Cuantitativo."""

import concurrent.futures
import re
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.auto_learner import get_auto_learner
from src.analytics.portfolio_allocator import (
    BetOpportunity,
    DutchingOpportunity,
    find_surebets,
    optimize_value_bets,
    solve_dutching,
)
from src.analytics.predictor import predict_live_in_play
from src.analytics.settlement_engine import audit_and_settle_open_bets
from src.models.database import (
    get_bankroll_ledger_history,
    get_db_session,
    get_portfolio_bets,
    get_portfolio_kpis,
    record_bankroll_transaction,
    save_portfolio_bets,
)
from src.models.schemas import MatchDetailSchema
from src.scrapers.odds_api_client import (
    OddsAPIClient,
    calculate_two_way_surebet,
    match_and_cross_arbitrage,
)
from src.scrapers.sofascore import SofascoreScraper


@st.cache_data(ttl=120, show_spinner=False)
def _cached_portfolio_kpis() -> Dict[str, Any]:
    return get_portfolio_kpis()


@st.cache_data(ttl=120, show_spinner=False)
def _cached_ledger_history(limit: int = 200) -> List[Dict[str, Any]]:
    return get_bankroll_ledger_history(limit=limit)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_portfolio_bets(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    return get_portfolio_bets(status=status, limit=limit)


def invalidate_portfolio_cache() -> None:
    """Invalida la caché en memoria del portafolio tras registrar transacciones, apuestas o liquidaciones."""
    _cached_portfolio_kpis.clear()
    _cached_ledger_history.clear()
    _cached_portfolio_bets.clear()


def render_portfolio_allocator_view(
    filtered_matches: List[MatchDetailSchema],
    catalog: Dict[str, Any],
    date_str: str,
    sf_scraper: SofascoreScraper,
    analyze_matches_batch_parallel_fn: Callable[..., List[Dict[str, Any]]],
) -> None:
    """Renderiza bajo demanda la pestaña de gestión de capital, +EV, Dutching, Surebets y Libro Diario."""
    st.subheader("💼 Asignador Maestro de Capital & Portafolio Cuantitativo")
    st.caption(
        "Arquitectura institucional para gestión de capital, arbitraje seguro (Surebets), "
        "optimización de valor esperado (+EV con Kelly Fraccional escalado), coberturas Dutching y auditoría contable determinista."
    )

    kpis = _cached_portfolio_kpis()
    current_bankroll = kpis["current_bankroll"]

    # 1. CABECERA DE BANCA Y PARÁMETROS DE RIESGO
    st.markdown("### 🏦 Estado de la Banca y Control de Riesgo")
    bank_c1, bank_c2, bank_c3, bank_c4, bank_c5 = st.columns([1.5, 1.2, 1.2, 1.1, 1.5])
    with bank_c1:
        st.metric("💰 Banca Total Actual", f"${current_bankroll:,.2f}")
    with bank_c2:
        pnl_val = kpis["total_pnl"]
        st.metric("📈 P&L Neto Acumulado", f"{pnl_val:+,.2f} $")
    with bank_c3:
        st.metric("🎯 Yield / ROI Real", f"{kpis['roi_pct']:+.2f}%")
    with bank_c4:
        st.metric("🏆 Win Rate", f"{kpis['win_rate_pct']:.1f}%")
    with bank_c5:
        with st.popover("💳 Ajustar Fondos (Depósito / Retiro)"):
            st.markdown("**Registrar Transacción de Capital**")
            adj_type = st.selectbox(
                "Tipo de Movimiento",
                ["DEPOSIT", "WITHDRAWAL"],
                format_func=lambda x: "🟢 Depósito (+)" if x == "DEPOSIT" else "🔴 Retiro (-)",
            )
            adj_amount = st.number_input("Monto en $", min_value=1.0, value=100.0, step=25.0)
            adj_desc = st.text_input("Nota / Motivo", value="Ajuste manual de banca")
            if st.button("Confirmar Movimiento Contable", use_container_width=True):
                record_bankroll_transaction(adj_type, adj_amount, adj_desc)
                invalidate_portfolio_cache()
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
        st.metric(
            "Presupuesto Máximo Hoy",
            f"${max_budget_usd:,.2f}",
            f"Máx 4.0% (${current_bankroll * 0.04:,.2f}) por apuesta individual",
        )

    st.divider()

    cached_preds = st.session_state.setdefault("audited_predictions_cache", {})
    deep_cache = st.session_state.setdefault("deep_analysis_cache", {})

    # 1. Enriquecer cuotas desde caché profunda o red (solo si Sofascore no tiene Circuit Breaker abierto)
    if "attempted_odds_events" not in st.session_state:
        st.session_state["attempted_odds_events"] = set()

    for m_cand in filtered_matches:
        if m_cand.odds is None and m_cand.event_id in deep_cache:
            m_cached = deep_cache[m_cand.event_id].get("match")
            if m_cached and getattr(m_cached, "odds", None):
                m_cand.odds = m_cached.odds

    if not sf_scraper.is_sofascore_blocked():
        missing_odds_matches = [
            m
            for m in filtered_matches
            if m.odds is None and m.event_id not in st.session_state["attempted_odds_events"]
        ]
        missing_odds_matches.sort(
            key=lambda m: (
                0
                if m.status_type in ("inprogress", "live")
                else (1 if m.status_type in ("notstarted", "scheduled") else 2)
            )
        )
        batch_to_fetch = missing_odds_matches[:18]
        for m_item in batch_to_fetch:
            st.session_state["attempted_odds_events"].add(m_item.event_id)

        if batch_to_fetch:
            def _pull_odds(m_item):
                if sf_scraper.is_sofascore_blocked():
                    return
                try:
                    m_item.odds = sf_scraper.get_match_odds(
                        m_item.event_id,
                        is_live=m_item.status_type in ("inprogress", "live"),
                    )
                except Exception:
                    pass

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as o_exec:
                list(o_exec.map(_pull_odds, batch_to_fetch))

    uncached_matches = [
        m for m in filtered_matches if m.event_id not in cached_preds
    ]
    if uncached_matches and not sf_scraper.is_sofascore_blocked():
        deep_col1, deep_col2 = st.columns([2.5, 1.5])
        with deep_col1:
            st.caption(
                f"💡 Hay `{len(uncached_matches)}` partidos sin análisis profundo en memoria. Puedes procesarlos bajo demanda:"
            )
        with deep_col2:
            if st.button("⚡ Analizar Top 8 con IA Opta (xG)", use_container_width=True):
                scan_targets = [
                    (m_item, catalog["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A"))
                    for m_item in uncached_matches[:8]
                ]
                with st.spinner("Ejecutando análisis Opta, H2H y Monte Carlo en paralelo..."):
                    batch_res = analyze_matches_batch_parallel_fn(scan_targets, max_workers=3)
                    for r in batch_res:
                        m_res = r["match"]
                        p_res = r["prediction"]
                        cached_preds[m_res.event_id] = p_res
                        deep_cache[m_res.event_id] = r
                st.rerun()

    # 2. PANEL DE GENERACIÓN DE CARTERA DEL DÍA
    st.markdown("### 📦 Generación de Cartera Óptima del Día")
    tab_value, tab_dutch, tab_sure = st.tabs(
        [
            "📈 Inversiones de Valor (+EV con Kelly)",
            "🎯 Coberturas Dutching",
            "🛡️ Arbitraje Seguro (0 Riesgo - Surebets)",
        ]
    )

    matches_odds_list: List[Dict[str, Any]] = []
    value_candidates: List[BetOpportunity] = []
    dutching_opportunities: List[DutchingOpportunity] = []

    for m_cand in filtered_matches:
        is_live_match = m_cand.status_type in ("inprogress", "live")
        live_minute = 45
        live_momentum = 0.0

        if is_live_match:
            min_match = re.search(r"(\d+)", m_cand.status_description or "")
            if min_match:
                live_minute = int(min_match.group(1))
            elif "ht" in (m_cand.status_description or "").lower() or "half" in (m_cand.status_description or "").lower():
                live_minute = 45

            if m_cand.momentum and m_cand.momentum.points:
                recent_pts = m_cand.momentum.points[-5:]
                if recent_pts:
                    avg_val = sum(p.value for p in recent_pts) / len(recent_pts)
                    live_momentum = max(-1.0, min(1.0, avg_val / 50.0))

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
            p_u25 = max(0.01, 1.0 - p_o25)
            p_btts = live_calc["prob_btts_yes"] / 100.0
            p_btts_no = max(0.01, 1.0 - p_btts)
            live_tag = f" [🔴 EN VIVO {live_minute}']"
        else:
            live_tag = ""
            pred_cand = cached_preds.get(m_cand.event_id)
            if not pred_cand and m_cand.event_id in deep_cache:
                pred_cand = deep_cache[m_cand.event_id].get("prediction")

            if pred_cand:
                p_h = pred_cand.prob_home_win / 100.0
                p_d = pred_cand.prob_draw / 100.0
                p_a = pred_cand.prob_away_win / 100.0
                p_o25 = getattr(pred_cand, "prob_over_2_5", getattr(pred_cand, "prob_over_25", 52.0)) / 100.0
                p_u25 = getattr(pred_cand, "prob_under_2_5", max(0.01, 1.0 - p_o25))
                if p_u25 > 1.0:
                    p_u25 = p_u25 / 100.0
                p_btts = getattr(pred_cand, "prob_btts", getattr(pred_cand, "prob_btts_yes", 50.0)) / 100.0
                p_btts_no = getattr(pred_cand, "prob_btts_no", max(0.01, 1.0 - p_btts))
                if p_btts_no > 1.0:
                    p_btts_no = p_btts_no / 100.0
            else:
                if m_cand.odds and m_cand.odds.home_win and m_cand.odds.draw and m_cand.odds.away_win:
                    inv_h = 1.0 / m_cand.odds.home_win
                    inv_d = 1.0 / m_cand.odds.draw
                    inv_a = 1.0 / m_cand.odds.away_win
                    tot_inv = inv_h + inv_d + inv_a
                    p_h = round((inv_h / tot_inv) * 1.04, 3)
                    p_d = round((inv_d / tot_inv) * 0.98, 3)
                    p_a = round((inv_a / tot_inv) * 1.02, 3)
                    s = p_h + p_d + p_a
                    p_h, p_d, p_a = p_h / s, p_d / s, p_a / s
                else:
                    p_h, p_d, p_a = 0.45, 0.27, 0.28

                if m_cand.odds and m_cand.odds.over_2_5 and m_cand.odds.under_2_5:
                    inv_o = 1.0 / m_cand.odds.over_2_5
                    inv_u = 1.0 / m_cand.odds.under_2_5
                    p_o25 = round(inv_o / (inv_o + inv_u), 3)
                    p_u25 = round(1.0 - p_o25, 3)
                else:
                    p_o25, p_u25 = 0.52, 0.48

                if m_cand.odds and getattr(m_cand.odds, "btts_yes", None) and getattr(m_cand.odds, "btts_no", None):
                    inv_by = 1.0 / m_cand.odds.btts_yes
                    inv_bn = 1.0 / m_cand.odds.btts_no
                    p_btts = round(inv_by / (inv_by + inv_bn), 3)
                    p_btts_no = round(1.0 - p_btts, 3)
                else:
                    p_btts, p_btts_no = 0.50, 0.50

        h_odd = m_cand.odds.home_win if (m_cand.odds and m_cand.odds.home_win) else None
        d_odd = m_cand.odds.draw if (m_cand.odds and m_cand.odds.draw) else None
        a_odd = m_cand.odds.away_win if (m_cand.odds and m_cand.odds.away_win) else None
        o25_odd = m_cand.odds.over_2_5 if (m_cand.odds and m_cand.odds.over_2_5) else None
        u25_odd = m_cand.odds.under_2_5 if (m_cand.odds and m_cand.odds.under_2_5) else None
        btts_y_odd = getattr(m_cand.odds, "btts_yes", None) if m_cand.odds else None
        btts_n_odd = getattr(m_cand.odds, "btts_no", None) if m_cand.odds else None

        event_label = f"{m_cand.home_team.name} vs {m_cand.away_team.name}{live_tag}"

        odds_map: Dict[str, float] = {}
        if h_odd:
            odds_map["1X2_HOME"] = h_odd
        if d_odd:
            odds_map["1X2_DRAW"] = d_odd
        if a_odd:
            odds_map["1X2_AWAY"] = a_odd
        if o25_odd:
            odds_map["OU_2.5_OVER"] = o25_odd
        if u25_odd:
            odds_map["OU_2.5_UNDER"] = u25_odd
        if btts_y_odd:
            odds_map["BTTS_YES"] = btts_y_odd
        if btts_n_odd:
            odds_map["BTTS_NO"] = btts_n_odd

        if len(odds_map) >= 2:
            matches_odds_list.append(
                {
                    "event_id": m_cand.event_id,
                    "event_name": event_label,
                    "tournament": m_cand.tournament.name,
                    "match_date": date_str,
                    "odds": odds_map,
                }
            )

        for code, pick_lbl, odd_v, prob_v in [
            ("1X2_HOME", f"Victoria {m_cand.home_team.name}", h_odd, p_h),
            ("1X2_DRAW", "Empate (X)", d_odd, p_d),
            ("1X2_AWAY", f"Victoria {m_cand.away_team.name}", a_odd, p_a),
            ("OU_2.5_OVER", "Más de 2.5 Goles", o25_odd, p_o25),
            ("OU_2.5_UNDER", "Menos de 2.5 Goles", u25_odd, p_u25),
            ("BTTS_YES", "Ambos Anotan: Sí", btts_y_odd, p_btts),
            ("BTTS_NO", "Ambos Anotan: No", btts_n_odd, p_btts_no),
        ]:
            if odd_v and odd_v > 1.01:
                ev_val = (prob_v * odd_v) - 1.0
                value_candidates.append(
                    BetOpportunity(
                        event_id=m_cand.event_id,
                        event_name=event_label,
                        tournament=m_cand.tournament.name,
                        match_date=date_str,
                        strategy="VALUE_BET",
                        market_code=code,
                        pick_name=pick_lbl,
                        odds_taken=odd_v,
                        fair_odds=round(1.0 / max(0.01, prob_v), 2),
                        model_probability=round(prob_v, 3),
                        ev_net=round(ev_val, 3),
                    )
                )

        if h_odd and d_odd and h_odd > 1.01 and d_odd > 1.01:
            odd_1x = round(1.0 / ((1.0 / h_odd) + (1.0 / d_odd)), 2)
            p_1x = min(0.95, round(p_h + p_d, 3))
            if odd_1x >= 1.15:
                value_candidates.append(
                    BetOpportunity(
                        event_id=m_cand.event_id,
                        event_name=event_label,
                        tournament=m_cand.tournament.name,
                        match_date=date_str,
                        strategy="VALUE_BET",
                        market_code="DC_1X",
                        pick_name=f"1X ({m_cand.home_team.name} o Empate)",
                        odds_taken=odd_1x,
                        fair_odds=round(1.0 / max(0.01, p_1x), 2),
                        model_probability=p_1x,
                        ev_net=round((p_1x * odd_1x) - 1.0, 3),
                    )
                )

            dutch_res_1x = solve_dutching(
                event_id=m_cand.event_id,
                event_name=event_label,
                tournament=m_cand.tournament.name,
                match_date=date_str,
                coverage_name=f"Cobertura 1X ({m_cand.home_team.name} o Empate)",
                selections=[
                    {"market_code": "1X2_HOME", "pick_name": f"1 ({m_cand.home_team.name})", "odds": h_odd, "model_prob": p_h},
                    {"market_code": "1X2_DRAW", "pick_name": "X (Empate)", "odds": d_odd, "model_prob": p_d},
                ],
                target_budget=min(50.0, round(current_bankroll * 0.03, 2)),
            )
            if dutch_res_1x and dutch_res_1x.combined_probability >= 0.60:
                dutching_opportunities.append(dutch_res_1x)

        if d_odd and a_odd and d_odd > 1.01 and a_odd > 1.01:
            dutch_res_x2 = solve_dutching(
                event_id=m_cand.event_id,
                event_name=event_label,
                tournament=m_cand.tournament.name,
                match_date=date_str,
                coverage_name=f"Cobertura X2 (Empate o {m_cand.away_team.name})",
                selections=[
                    {"market_code": "1X2_DRAW", "pick_name": "X (Empate)", "odds": d_odd, "model_prob": p_d},
                    {"market_code": "1X2_AWAY", "pick_name": f"2 ({m_cand.away_team.name})", "odds": a_odd, "model_prob": p_a},
                ],
                target_budget=min(50.0, round(current_bankroll * 0.03, 2)),
            )
            if dutch_res_x2 and dutch_res_x2.combined_probability >= 0.60:
                dutching_opportunities.append(dutch_res_x2)

        if h_odd and a_odd and h_odd > 1.01 and a_odd > 1.01:
            dutch_res_12 = solve_dutching(
                event_id=m_cand.event_id,
                event_name=event_label,
                tournament=m_cand.tournament.name,
                match_date=date_str,
                coverage_name="Cobertura 12 (Sin Empate)",
                selections=[
                    {"market_code": "1X2_HOME", "pick_name": f"1 ({m_cand.home_team.name})", "odds": h_odd, "model_prob": p_h},
                    {"market_code": "1X2_AWAY", "pick_name": f"2 ({m_cand.away_team.name})", "odds": a_odd, "model_prob": p_a},
                ],
                target_budget=min(50.0, round(current_bankroll * 0.03, 2)),
            )
            if dutch_res_12 and dutch_res_12.combined_probability >= 0.65:
                dutching_opportunities.append(dutch_res_12)

    # TAB 1: INVERSIONES DE VALOR (+EV CON KELLY)
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
        else:
            total_plan_stake = sum(b.stake_amount for b in allocated_value_bets)
            total_plan_payout = sum(b.potential_payout for b in allocated_value_bets)

            st.markdown(
                f"**Cartera sugerida:** `{len(allocated_value_bets)} apuestas` | "
                f"**Capital Total a Invertir:** `${total_plan_stake:,.2f}` | "
                f"**Retorno Potencial:** `${total_plan_payout:,.2f}`"
            )

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
            cart_col1, cart_col2 = st.columns([1.5, 1.0])
            with cart_col1:
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
                    invalidate_portfolio_cache()
                    if saved > 0:
                        st.success(f"🎉 **{saved} apuestas guardadas exitosamente** en el portafolio y registradas contablemente en el libro diario.")
                    if skipped > 0:
                        st.info(f"ℹ️ {skipped} apuestas ya existían previamente y se omitieron por protección idempotente (anti-duplicados).")
                    st.rerun()

            with cart_col2:
                from src.services.telegram_bot import is_telegram_configured, notify_parlay_ready
                if is_telegram_configured():
                    if st.button("📲 Compartir Cartera a Telegram", use_container_width=True, key="btn_share_portfolio_tg"):
                        legs_txt = [
                            f"{b.event_name} ➜ {b.pick_name} (Cuota {b.odds_taken:.2f} • Stake ${b.stake_amount:.2f})"
                            for b in allocated_value_bets
                        ]
                        sent = notify_parlay_ready(
                            parlay_title=f"Cartera de Valor (+EV con Kelly) • {len(allocated_value_bets)} Apuestas",
                            total_odds=round(sum(b.odds_taken for b in allocated_value_bets) / max(1, len(allocated_value_bets)), 2),
                            win_prob_pct=round(sum(b.model_probability for b in allocated_value_bets) / max(1, len(allocated_value_bets)) * 100, 1),
                            legs=legs_txt,
                        )
                        if sent:
                            st.toast("¡Cartera de Valor enviada a Telegram con éxito! 🚀")
                        else:
                            st.error("No se pudo enviar la cartera a Telegram.")

    # TAB 2: COBERTURAS DUTCHING
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

    # TAB 3: ARBITRAJE SEGURO (SUREBETS)
    with tab_sure:
        st.markdown("#### 🛡️ Arbitraje Deportivo Libre de Riesgo (Surebets)")
        st.caption(
            "El arbitraje puro consiste en aprovechar discrepancias de cuotas entre diferentes casas "
            "(ej. **Betano vs Betsson**) apostando a todos los desenlaces posibles para asegurar una ganancia fija matemática con 0 riesgo."
        )

        st.markdown("##### 📡 Escáner de Cuotas Multi-Casa: Betano vs Betsson (The Odds API)")
        oa_col1, oa_col2, oa_col3 = st.columns([1.5, 1.5, 1.2])
        with oa_col1:
            st.markdown("🟢 **Betano:** Mercado Oficial Sofascore")
        with oa_col2:
            st.markdown("🟢 **Betsson:** The Odds API v4")
        with oa_col3:
            scan_btn = st.button("📡 Escanear Arbitraje en Vivo", use_container_width=True, type="primary")

        if scan_btn or "cross_arbitrage_data" in st.session_state:
            if scan_btn or "cross_arbitrage_data" not in st.session_state:
                with st.spinner("Consultando cuotas en vivo de Betsson vía The Odds API y cruzando con Betano..."):
                    oa_client = OddsAPIClient()
                    oa_matches = oa_client.fetch_all_active_soccer_odds(ttl_hours=0.25)
                    cross_results = match_and_cross_arbitrage(
                        sofascore_matches=filtered_matches,
                        odds_api_matches=oa_matches,
                        bankroll=current_bankroll,
                    )
                    st.session_state["cross_arbitrage_data"] = cross_results

            cross_results = st.session_state.get("cross_arbitrage_data", [])
            detected_arbs = []
            for cr in cross_results:
                for arb in cr.get("arbitrage_opportunities", []):
                    if arb.get("is_surebet"):
                        detected_arbs.append((cr, arb))

            if detected_arbs:
                st.success(f"🎉 **¡Se detectaron {len(detected_arbs)} Surebets activas entre Betano y Betsson!**")
                for match_info, arb in detected_arbs:
                    with st.container(border=True):
                        c_top1, c_top2, c_top3, c_top4 = st.columns([1.8, 1.2, 1.2, 1.2])
                        with c_top1:
                            st.markdown(f"**⚽ {match_info['match_title']}**")
                            st.caption(f"🏆 {match_info['tournament']} | Mercado: `{arb['market']}`")
                        with c_top2:
                            st.metric("Inversión Total", f"${arb['total_stake']:,.2f}")
                        with c_top3:
                            st.metric("Retorno Seguro", f"${arb['guaranteed_payout']:,.2f}")
                        with c_top4:
                            st.metric("Ganancia Neta", f"+${arb['guaranteed_profit']:,.2f}", f"+{arb['roi_pct']:.2f}% ROI")

                        st.markdown("**Desglose de inversión exacta por casa de apuestas:**")
                        leg_c = st.columns(len(arb["legs"]))
                        for i_leg, leg in enumerate(arb["legs"]):
                            with leg_c[i_leg]:
                                bm_badge = "🟠 Betano" if "betano" in leg["bookmaker"].lower() else "🔵 Betsson"
                                st.markdown(f"**{bm_badge}**")
                                st.markdown(f"Selección: **{leg['label']}**")
                                st.markdown(f"Cuota: `x{leg['odds']:.2f}`")
                                st.markdown(f"Apostar: `💵 ${leg['stake']:,.2f}`")
                                st.caption(f"Retorno si gana: ${leg['payout']:,.2f}")
            else:
                st.info(
                    "ℹ️ **Monitoreo Multi-Casa Activo:** En este momento las cuotas de Betano y Betsson están equilibradas "
                    "(la suma de inversas actual es superior a 1.0, por lo que ninguna combinación garantiza arbitraje positivo en esta fecha)."
                )

            if cross_results:
                with st.expander("📊 Ver Comparativa Detallada de Cuotas (Betano vs Betsson)", expanded=False):
                    comp_rows = []
                    for cr in cross_results:
                        b_1 = cr["betano_odds"].get("1")
                        bs_1 = cr["betsson_odds"].get("1")
                        b_x = cr["betano_odds"].get("X")
                        bs_x = cr["betsson_odds"].get("X")
                        b_2 = cr["betano_odds"].get("2")
                        bs_2 = cr["betsson_odds"].get("2")
                        b_o = cr["betano_odds"].get("over_2_5")
                        bs_o = cr["betsson_odds"].get("over_2_5")
                        b_u = cr["betano_odds"].get("under_2_5")
                        bs_u = cr["betsson_odds"].get("under_2_5")
                        comp_rows.append(
                            {
                                "Partido": cr["match_title"],
                                "Torneo": cr["tournament"],
                                "Betano (1 / X / 2)": f"{b_1 or '-'} / {b_x or '-'} / {b_2 or '-'}",
                                "Betsson (1 / X / 2)": f"{bs_1 or '-'} / {bs_x or '-'} / {bs_2 or '-'}",
                                "Betano O/U 2.5": f"O {b_o or '-'} | U {b_u or '-'}",
                                "Betsson O/U 2.5": f"O {bs_o or '-'} | U {bs_u or '-'}",
                            }
                        )
                    if comp_rows:
                        st.dataframe(comp_rows, use_container_width=True)

        surebets = find_surebets(matches_odds_list, current_bankroll, min_roi=0.015)
        if surebets:
            st.markdown("##### 🔍 Otras Surebets Detectadas")
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

                    leg_cols = st.columns(len(sb.legs))
                    for idx_leg, leg in enumerate(sb.legs):
                        with leg_cols[idx_leg]:
                            st.markdown(f"**{leg.pick_name}**")
                            st.markdown(f"Cuota: `x{leg.odds:.2f}`")
                            st.markdown(f"Apostar: `💵 ${leg.stake_amount:.2f}`")
                            st.caption(f"Retorno si gana: ${leg.payout:.2f}")

        st.divider()
        st.markdown("##### 🧮 Calculadora de Arbitraje y Distribución de Capital")
        calc_c1, calc_c2, calc_c3 = st.columns([1.2, 1.2, 1.5])
        with calc_c1:
            calc_odd_a = st.number_input("Cuota Selección A (ej. Betano)", min_value=1.01, max_value=50.0, value=1.82, step=0.01)
        with calc_c2:
            calc_odd_b = st.number_input("Cuota Selección B (ej. Betsson)", min_value=1.01, max_value=50.0, value=2.50, step=0.01)
        with calc_c3:
            calc_bank = st.number_input(
                "Capital Total a Invertir ($)",
                min_value=10.0,
                max_value=10000000.0,
                value=float(current_bankroll if current_bankroll > 10 else 20000.0),
                step=1000.0,
            )

        quick_arb = calculate_two_way_surebet(
            odd_a=calc_odd_a,
            odd_b=calc_odd_b,
            label_a="Selección A",
            label_b="Selección B",
            bookmaker_a="Betano",
            bookmaker_b="Betsson",
            total_bankroll=calc_bank,
        )

        if quick_arb:
            if quick_arb["is_surebet"]:
                st.success(
                    f"🎉 **¡SUREBET CONFIRMADA!** Suma de inversas: `{quick_arb['inverse_sum']:.4f}` < 1.0 "
                    f"| Rendimiento Neto Garantizado: **+{quick_arb['roi_pct']:.2f}% ROI**"
                )
                r_c1, r_c2, r_c3 = st.columns(3)
                with r_c1:
                    st.metric("Apostar en Betano", f"${quick_arb['legs'][0]['stake']:,.2f}", f"Cuota {calc_odd_a:.2f}")
                with r_c2:
                    st.metric("Apostar en Betsson", f"${quick_arb['legs'][1]['stake']:,.2f}", f"Cuota {calc_odd_b:.2f}")
                with r_c3:
                    st.metric("Ganancia Neta Garantizada", f"+${quick_arb['guaranteed_profit']:,.2f}", f"Retorno: ${quick_arb['guaranteed_payout']:,.2f}")
            else:
                needed_for_surebet = round(1.0 / (1.0 - (1.0 / calc_odd_a)), 2) if calc_odd_a > 1.0 else 0.0
                st.warning(
                    f"⚠️ **NO ES SUREBET:** La suma de inversas es `{quick_arb['inverse_sum']:.4f}` (> 1.0). "
                    f"💡 *Para que sea Surebet con Cuota A de {calc_odd_a:.2f}, la Cuota B en Betsson debe ser mayor a **{needed_for_surebet:.2f}**.*"
                )

    st.divider()

    # 3. PANEL DE BITÁCORA, AUDITORÍA DE RESULTADOS Y EVOLUCIÓN DE BANCA
    st.markdown("### 📋 Bitácora y Auditoría Continua de Resultados")
    audit_top_c1, audit_top_c2 = st.columns([1.6, 2.4])
    with audit_top_c1:
        st.markdown(
            f"**Apuestas Activas Pendientes:** `{kpis['pending_count']}` | "
            f"**Cerradas:** `{kpis['won_count'] + kpis['lost_count'] + kpis['void_count']}`"
        )
    with audit_top_c2:
        if st.button("🔄 Auditar Partidos Finalizados & Calibrar Modelo", type="secondary", use_container_width=True):
            def local_scores_provider(event_id: int, match_date: str) -> Optional[Dict[str, Any]]:
                match_obj = st.session_state["audited_matches_cache"].get(event_id)
                if not match_obj:
                    with get_db_session() as sess:
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
            invalidate_portfolio_cache()

            if report.settled_count > 0:
                st.success(
                    f"✅ **Auditoría finalizada:** {report.settled_count} apuestas liquidadas "
                    f"({report.won_count} Ganadas, {report.lost_count} Perdidas, {report.void_count} Anuladas). "
                    f"**P&L de la ronda:** `${report.total_pnl:+,.2f}`."
                )
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

    ledger_history = _cached_ledger_history(limit=200)
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

    st.markdown("#### 📑 Historial Detallado de Apuestas del Portafolio")
    status_filter = st.selectbox(
        "Filtrar por Estado",
        ["TODAS", "PENDING", "WON", "LOST", "VOID_PUSH"],
        format_func=lambda x: {
            "TODAS": "🔍 Todas las Apuestas",
            "PENDING": "⏳ Solo Pendientes",
            "WON": "✅ Solo Ganadas",
            "LOST": "❌ Solo Perdidas",
            "VOID_PUSH": "⚪ Solo Anuladas / Reembolsadas",
        }.get(x, x),
    )

    portfolio_records = _cached_portfolio_bets(status=status_filter, limit=100)
    if not portfolio_records:
        st.caption("No hay registros en la base de datos para el filtro seleccionado.")
    else:
        df_bets = pd.DataFrame(portfolio_records)
        display_cols = [
            "match_date",
            "event_name",
            "strategy",
            "pick_name",
            "odds_taken",
            "stake_amount",
            "potential_payout",
            "status",
            "pnl",
        ]
        df_display = df_bets[display_cols].copy()
        df_display.columns = [
            "Fecha",
            "Partido",
            "Estrategia",
            "Selección",
            "Cuota",
            "Stake ($)",
            "Retorno ($)",
            "Estado",
            "P&L ($)",
        ]
        st.dataframe(df_display, use_container_width=True, hide_index=True)
