"""Vista modularizada para el Análisis Profundo del Partido Seleccionado (Scoreboard, 10 Mercados, Porteros, Monte Carlo, ML y H2H)."""

from typing import Any, Callable, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.predictor import MatchPrediction, TeamFormMetrics
from src.models.schemas import MatchDetailSchema, PlayerPropSchema
from src.ui.audit_view import compute_market_accuracy_breakdown, plot_market_accuracy_chart
from src.ui.live_view import plot_momentum_chart


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


def render_match_detail_view(
    deep: Dict[str, Any],
    format_colombia_time: Callable[..., str],
) -> None:
    """Renderiza el análisis profundo del partido seleccionado."""
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
        st.markdown(
            f"<div class='smart-money-badge'>"
            f"<span>🔥</span> <div><b>ALERTA DE FLUJO INSTITUCIONAL (SMART MONEY):</b> {m.odds.smart_money_side} "
            f"<span style='color:#38bdf8; font-size:0.78rem;'>• Las casas de apuestas ajustaron la cuota a la baja por volumen inusual de dinero profesional.</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

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


