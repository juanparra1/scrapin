"""Motor estadístico v3.5 (Opta + H2H Profundo + Porteros): Monte Carlo (10,000 sims), Atajadas de Arquero, Hándicaps Asiáticos, Fatiga/Descanso, ML Ensemble, Árbitro, Jugadores y Backtesting."""

import math
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.analytics.auto_learner import get_auto_learner
from src.analytics.ml_ensemble import get_ml_predictor
from src.models.schemas import (
    MatchDetailSchema,
    PlayerLineupSchema,
    PlayerPropSchema,
    TeamLineupSchema,
    TournamentSchema,
)


class TeamFormMetrics(BaseModel):
    team_id: int
    team_name: str
    matches_analyzed: int
    wins: int
    draws: int
    losses: int
    goals_scored_avg: float
    goals_conceded_avg: float
    xg_for_avg: float
    xg_against_avg: float
    venue_xg_for_avg: float
    venue_xg_against_avg: float
    venue_record: str
    total_shots_for_avg: float
    total_shots_against_avg: float
    shots_on_target_avg: float
    shots_on_target_against_avg: float
    big_chances_avg: float
    possession_avg: float
    corners_for_avg: float
    corners_against_avg: float
    yellow_cards_for_avg: float
    yellow_cards_against_avg: float
    fouls_for_avg: float
    points_per_match: float
    elo_rating: float
    form_string: str
    # Fatiga y Días de Descanso
    rest_days: float = 6.0
    matches_last_14d: int = 2
    fatigue_index: float = 20.0
    fatigue_label: str = "Descanso Óptimo 🟢"
    # Micro-Métricas Avanzadas Opta & 1er Tiempo Real (1ST)
    xgot_for_avg: float = 1.28
    xgot_against_avg: float = 1.15
    shots_in_box_for_avg: float = 7.8
    shots_in_box_against_avg: float = 7.2
    inside_box_shot_pct: float = 62.0
    touches_in_box_avg: float = 19.5
    final_third_entries_avg: float = 48.0
    ball_recoveries_avg: float = 46.0
    ht_xg_for_avg: float = 0.60
    ht_xg_against_avg: float = 0.54
    ht_corners_for_avg: float = 2.25
    ht_corners_against_avg: float = 2.15
    # Métricas de Portería (Goalkeeper Saves & Opta Goals Prevented)
    gk_saves_avg: float = 2.85
    gk_save_pct: float = 70.5
    gk_goals_prevented_avg: float = 0.0
    sos_rating: float = 1.35
    # NUEVO v4: Calidad de Disparo & Eficiencia de Peligro Opta
    xg_per_shot_for: float = 0.11
    xg_per_shot_against: float = 0.10
    # NUEVO v4: Estilo Táctico Identificado
    tactical_style: str = "Equilibrado / Mixto"
    # NUEVO v4: Índice de Urgencia Competitiva y Necesidad de Puntos (0 a 100)
    urgency_index: float = 50.0
    urgency_label: str = "Urgencia Media 🟡"
    table_position: Optional[int] = None
    table_points: Optional[int] = None
    table_matches: Optional[int] = None
    # NUEVO v4.5: PPDA (Presión Alta Opta) y Sinergia de Centrales Titulares
    ppda_avg: float = 11.5
    center_back_synergy_score: float = 100.0  # 100.0 = pareja titular consolidada
    defensive_line_status: str = "Zaga Central Titular Confirmada 🛡️"


class MatchPrediction(BaseModel):
    event_id: int
    home_team: str
    away_team: str
    elo_home: float
    elo_away: float
    elo_diff: float

    # Impacto de Bajas, Fatiga y Árbitro
    injury_penalty_home_pct: float = 0.0
    injury_penalty_away_pct: float = 0.0
    missing_key_home: List[str] = Field(default_factory=list)
    missing_key_away: List[str] = Field(default_factory=list)
    rest_days_home: float = 6.0
    rest_days_away: float = 6.0
    fatigue_home_label: str = "Descanso Óptimo 🟢"
    fatigue_away_label: str = "Descanso Óptimo 🟢"
    referee_name: Optional[str] = None
    referee_cards_avg: Optional[float] = None
    referee_strictness: Optional[str] = None

    # 1X2: Ensemble (Dixon-Coles/MonteCarlo + Gradient Boosting ML)
    expected_goals_home: float
    expected_goals_away: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    stat_prob_home: float
    stat_prob_draw: float
    stat_prob_away: float
    ml_prob_home: float
    ml_prob_draw: float
    ml_prob_away: float
    ml_db_samples: int = 0
    ml_feature_importances: List[Dict[str, Any]] = Field(default_factory=list)

    prob_1x: float
    prob_x2: float
    prob_12: float

    # Simulación Monte Carlo (10,000 iteraciones): Hándicap Asiático, DNB y Goles por Equipo
    mc_simulations: int = 10000
    prob_dnb_home: float
    prob_dnb_away: float
    prob_ah_home_minus_1_5: float
    prob_ah_home_minus_1_0_win_or_push: float
    prob_ah_away_minus_1_5: float
    prob_ah_away_plus_1_5: float
    prob_home_over_0_5: float
    prob_home_over_1_5: float
    prob_home_over_2_5: float
    prob_away_over_0_5: float
    prob_away_over_1_5: float
    prob_home_clean_sheet: float
    prob_away_clean_sheet: float

    # Mercados de Goles & BTTS
    prob_over_1_5: float
    prob_over_2_5: float
    prob_over_3_5: float
    prob_under_2_5: float
    prob_btts: float
    prob_btts_no: float

    # Mercados de 1er Tiempo (1H Calibrado con Opta 1ST Period)
    expected_1h_goals_total: float
    prob_1h_over_0_5: float
    prob_1h_over_1_5: float
    prob_1h_home_win: float
    prob_1h_draw: float
    prob_1h_away_win: float
    expected_1h_corners: float
    prob_1h_over_4_5_corners: float

    # Mercados de Remates Totales y Remates al Arco
    expected_shots_home: float
    expected_shots_away: float
    expected_shots_total: float
    prob_over_22_5_shots: float
    prob_over_24_5_shots: float
    prob_over_26_5_shots: float

    expected_sot_home: float
    expected_sot_away: float
    expected_sot_total: float
    prob_over_7_5_sot: float
    prob_over_8_5_sot: float
    prob_over_9_5_sot: float

    # NUEVO: Mercados de Atajadas del Portero (Goalkeeper Saves)
    gk_home_name: str = "Portero Local"
    gk_away_name: str = "Portero Visitante"
    gk_home_hist_saves_avg: float = 2.85
    gk_away_hist_saves_avg: float = 2.95
    gk_home_save_pct: float = 70.5
    gk_away_save_pct: float = 70.5
    gk_home_goals_prevented: float = 0.0
    gk_away_goals_prevented: float = 0.0
    expected_saves_home_gk: float = 2.8
    expected_saves_away_gk: float = 3.1
    expected_saves_total: float = 5.9
    prob_home_gk_2plus_saves: float = 0.0
    prob_home_gk_3plus_saves: float = 0.0
    prob_home_gk_4plus_saves: float = 0.0
    prob_home_gk_5plus_saves: float = 0.0
    prob_away_gk_2plus_saves: float = 0.0
    prob_away_gk_3plus_saves: float = 0.0
    prob_away_gk_4plus_saves: float = 0.0
    prob_away_gk_5plus_saves: float = 0.0
    prob_over_4_5_total_saves: float = 0.0
    prob_over_5_5_total_saves: float = 0.0
    prob_over_6_5_total_saves: float = 0.0

    # Mercados de Córners
    expected_corners_home: float
    expected_corners_away: float
    expected_corners_total: float
    prob_over_8_5_corners: float
    prob_over_9_5_corners: float
    prob_over_10_5_corners: float

    # Mercados de Tarjetas y Faltas
    expected_cards_home: float
    expected_cards_away: float
    expected_cards_total: float
    prob_over_3_5_cards: float
    prob_over_4_5_cards: float
    prob_over_5_5_cards: float
    expected_fouls_total: float

    # Posesión e Índice de Peligro Opta
    expected_possession_home: float
    expected_possession_away: float
    opta_danger_home: float = 50.0
    opta_danger_away: float = 50.0

    # NUEVO v4: Urgencia Competitiva y Necesidad de Tabla
    urgency_home_index: float = 50.0
    urgency_away_index: float = 50.0
    urgency_home_label: str = "Urgencia Media 🟡"
    urgency_away_label: str = "Urgencia Media 🟡"
    urgency_clash_summary: str = "Duelo de intensidad regular de temporada"

    # NUEVO v4: Calidad Real de Ocasiones (xG / Disparo & xGOT / SoT)
    shot_quality_home: float = 0.11
    shot_quality_away: float = 0.10
    shot_quality_edge: str = "Calidad de remate similar"

    # NUEVO v4: Choque Táctico de Estilos
    tactical_style_home: str = "Equilibrado"
    tactical_style_away: str = "Equilibrado"
    tactical_clash_narrative: str = "Partido equilibrado en propuesta táctica"

    # NUEVO v4.5: Sinergia Defensiva & Presión PPDA
    ppda_home: float = 11.5
    ppda_away: float = 11.5
    defense_synergy_home_pct: float = 100.0
    defense_synergy_away_pct: float = 100.0
    defense_status_home: str = "Zaga Central Titular 🛡️"
    defense_status_away: str = "Zaga Central Titular 🛡️"

    # NUEVO v4.5: Calibración Dinámica de Liga (Platt Scaling & Rho Específico)
    league_rho_calibration: float = -0.09
    league_profile_name: str = "Estándar UEFA"

    # NUEVO v4: Desglose de Probabilidad de Goles por Tramos de Tiempo (Minutos)
    prob_goal_early_1_15: float = 28.5
    prob_goal_mid_15_45: float = 48.0
    prob_goal_late_75_90: float = 42.0
    prob_most_active_half: str = "2do Tiempo (Min 46-90)"
    time_windows: List[Dict[str, Any]] = Field(default_factory=list)

    # NUEVO v4: Backtesting de ROI Real Simulado & Brier Score de Calibración
    brier_score_1x2: Optional[float] = None
    brier_score_goals: Optional[float] = None
    brier_calibration_label: str = "Calibración Óptima (Modelo Fiable)"
    simulated_roi_pct: Optional[float] = None

    # Marcadores, Value Bets y Auditoría
    most_likely_score: str
    top_scores: List[Dict[str, Any]] = Field(default_factory=list)
    score_matrix: List[List[float]] = Field(default_factory=list)
    value_edge_home: Optional[float] = None
    value_edge_draw: Optional[float] = None
    value_edge_away: Optional[float] = None
    recommended_pick: str
    all_market_picks: List[Dict[str, Any]] = Field(default_factory=list)
    verified_hits: int = 0
    verified_total: int = 0


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)


def _poisson_prob_over(threshold: float, lam: float) -> float:
    max_k = int(math.floor(threshold))
    cdf = sum(_poisson_pmf(k, lam) for k in range(max_k + 1))
    return max(0.0, min(1.0, 1.0 - cdf))


def calculate_kelly_stake(
    prob_pct: float, decimal_odds: float, bankroll: float, fraction: float = 0.25
) -> Dict[str, float]:
    """Calcula el Valor Esperado (EV%) y el Stake óptimo mediante Criterio de Kelly Fraccional."""
    p = max(0.001, min(0.999, prob_pct / 100.0))
    q = 1.0 - p
    b = max(0.01, decimal_odds - 1.0)
    ev_pct = ((p * decimal_odds) - 1.0) * 100.0
    raw_kelly = (b * p - q) / b
    stake_pct = max(0.0, min(15.0, raw_kelly * fraction * 100.0))
    stake_amount = round((stake_pct / 100.0) * bankroll, 2)
    return {
        "ev_pct": round(ev_pct, 2),
        "stake_pct": round(stake_pct, 2),
        "stake_amount": stake_amount,
        "fair_odds": round(1.0 / p, 2),
    }


def _run_monte_carlo_10k(
    lambda_home: float, lambda_away: float, seed: int = 12345, n_sims: int = 10000
) -> Dict[str, float]:
    """
    Ejecuta 10,000 simulaciones estocásticas del partido (Gamma-Poisson con sobredispersión
    para capturar expulsiones, rachas y varianza táctica) y calcula Hándicaps Asiáticos,
    Draw No Bet (DNB), Goles por Equipo y Porterías a Cero (Clean Sheets).
    """
    rng = np.random.default_rng(seed)
    tempo_shock = rng.gamma(shape=16.0, scale=1.0 / 16.0, size=n_sims)
    sim_h = rng.poisson(np.clip(lambda_home * tempo_shock, 0.05, 10.0))
    sim_a = rng.poisson(np.clip(lambda_away * tempo_shock, 0.05, 10.0))
    diff = sim_h - sim_a

    non_draws = max(1, int(np.sum(diff != 0)))
    dnb_home = float(np.sum(diff > 0)) / non_draws * 100.0
    dnb_away = float(np.sum(diff < 0)) / non_draws * 100.0

    return {
        "dnb_home": round(dnb_home, 1),
        "dnb_away": round(dnb_away, 1),
        "ah_home_m1_5": round(float(np.mean(diff >= 2)) * 100.0, 1),
        "ah_home_m1_0_win_or_push": round(float(np.mean(diff >= 1)) * 100.0, 1),
        "ah_away_m1_5": round(float(np.mean(diff <= -2)) * 100.0, 1),
        "ah_away_p1_5": round(float(np.mean(diff > -2)) * 100.0, 1),
        "home_o0_5": round(float(np.mean(sim_h >= 1)) * 100.0, 1),
        "home_o1_5": round(float(np.mean(sim_h >= 2)) * 100.0, 1),
        "home_o2_5": round(float(np.mean(sim_h >= 3)) * 100.0, 1),
        "away_o0_5": round(float(np.mean(sim_a >= 1)) * 100.0, 1),
        "away_o1_5": round(float(np.mean(sim_a >= 2)) * 100.0, 1),
        "home_cs": round(float(np.mean(sim_a == 0)) * 100.0, 1),
        "away_cs": round(float(np.mean(sim_h == 0)) * 100.0, 1),
    }


def compute_team_form(
    team_id: int,
    team_name: str,
    recent_matches: List[MatchDetailSchema],
    standing_info: Optional[Dict[str, Any]] = None,
    target_venue: str = "home",
    reference_timestamp: Optional[int] = None,
    standings_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> TeamFormMetrics:
    """
    Calcula métricas ponderadas de estado de forma separando condición Local vs Visitante,
    integra micro-métricas Opta (xGOT, toques en área, tiros dentro del área, entradas a último tercio,
    estadísticas reales de 1er tiempo y atajadas del portero) y ajusta por calidad del rival (Strength of Schedule).
    """
    if not recent_matches:
        return TeamFormMetrics(
            team_id=team_id,
            team_name=team_name,
            matches_analyzed=0,
            wins=0,
            draws=0,
            losses=0,
            goals_scored_avg=1.35,
            goals_conceded_avg=1.20,
            xg_for_avg=1.35,
            xg_against_avg=1.20,
            venue_xg_for_avg=1.35,
            venue_xg_against_avg=1.20,
            venue_record="S/D",
            total_shots_for_avg=12.5,
            total_shots_against_avg=11.8,
            shots_on_target_avg=4.4,
            shots_on_target_against_avg=4.1,
            big_chances_avg=2.1,
            possession_avg=50.0,
            corners_for_avg=4.8,
            corners_against_avg=4.5,
            yellow_cards_for_avg=2.1,
            yellow_cards_against_avg=2.1,
            fouls_for_avg=11.5,
            points_per_match=1.35,
            elo_rating=1500.0,
            form_string="N/A",
        )

    valid_recents = [
        rm
        for rm in recent_matches
        if (reference_timestamp is None or rm.start_timestamp < reference_timestamp - 1800)
    ]
    if not valid_recents:
        valid_recents = recent_matches

    rest_days = 6.0
    matches_14d = 2
    if reference_timestamp and valid_recents:
        last_ts = valid_recents[0].start_timestamp
        diff_sec = max(43200, reference_timestamp - last_ts)
        rest_days = round(min(21.0, diff_sec / 86400.0), 1)
        matches_14d = sum(
            1
            for rm in valid_recents
            if 0 < (reference_timestamp - rm.start_timestamp) <= (14 * 86400)
        )

    if rest_days < 3.4 or matches_14d >= 4:
        fatigue_idx = 78.0
        fatigue_lbl = f"Fatiga Alta 🔴 ({rest_days}d descanso • {matches_14d}P/14d)"
    elif rest_days <= 4.3 or matches_14d == 3:
        fatigue_idx = 48.0
        fatigue_lbl = f"Carga Media 🟡 ({rest_days}d descanso • {matches_14d}P/14d)"
    elif rest_days <= 8.5:
        fatigue_idx = 18.0
        fatigue_lbl = f"Descanso Óptimo 🟢 ({rest_days}d descanso)"
    else:
        fatigue_idx = 32.0
        fatigue_lbl = f"Inactividad Larga 🔵 ({rest_days}d sin jugar)"

    wins, draws, losses = 0, 0, 0
    v_wins, v_draws, v_losses = 0, 0, 0
    gf_w, ga_w, w_sum = 0.0, 0.0, 0.0
    xgf_w, xga_w, xg_w_sum = 0.0, 0.0, 0.0
    xgotf_w, xgota_w, xgot_w_sum = 0.0, 0.0, 0.0
    v_xgf_w, v_xga_w, v_xg_sum = 0.0, 0.0, 0.0
    tsf_w, tsa_w, ts_w_sum = 0.0, 0.0, 0.0
    sotf_w, sota_w, sot_w_sum = 0.0, 0.0, 0.0
    sib_f_w, sib_a_w, sib_w_sum = 0.0, 0.0, 0.0
    tib_w, tib_w_sum = 0.0, 0.0
    fte_w, fte_w_sum = 0.0, 0.0
    brec_w, brec_w_sum = 0.0, 0.0
    bc_w, bc_w_sum = 0.0, 0.0
    pos_w, pos_w_sum = 0.0, 0.0
    cf_w, ca_w, c_w_sum = 0.0, 0.0, 0.0
    yc_f_w, yc_a_w, yc_w_sum = 0.0, 0.0, 0.0
    fl_w, fl_w_sum = 0.0, 0.0
    ppda_w, ppda_w_sum = 0.0, 0.0

    # 1er Tiempo (1ST) y Portería
    ht_xgf_w, ht_xga_w, ht_xg_sum = 0.0, 0.0, 0.0
    ht_cf_w, ht_ca_w, ht_c_sum = 0.0, 0.0, 0.0
    saves_w, saves_w_sum = 0.0, 0.0
    save_ratio_num, save_ratio_den = 0.0, 0.0
    gp_w, gp_w_sum = 0.0, 0.0
    opp_ppg_list: List[float] = []

    letters: List[str] = []
    elo_delta = 0.0

    for idx, m in enumerate(valid_recents):
        if m.home_score is None or m.away_score is None:
            continue

        is_home = m.home_team.id == team_id
        opp_id = m.away_team.id if is_home else m.home_team.id
        same_venue = (target_venue == "home" and is_home) or (
            target_venue == "away" and not is_home
        )
        gf = m.home_score if is_home else m.away_score
        ga = m.away_score if is_home else m.home_score

        # Ajuste por Calidad del Rival (Strength of Schedule)
        opp_ppg = 1.35
        if standings_map and opp_id in standings_map:
            opp_ppg = float(standings_map[opp_id].get("ppg", 1.35))
        opp_ppg_list.append(opp_ppg)
        opp_attack_merit = max(0.88, min(1.14, 1.0 + (opp_ppg - 1.35) * 0.12))
        opp_def_merit = max(0.88, min(1.14, 1.0 - (opp_ppg - 1.35) * 0.10))

        venue_boost = 1.40 if same_venue else 0.90
        w = math.exp(-0.12 * idx) * venue_boost
        gf_w += gf * opp_attack_merit * w
        ga_w += ga * opp_def_merit * w
        w_sum += w

        if same_venue:
            if gf > ga:
                v_wins += 1
            elif gf == ga:
                v_draws += 1
            else:
                v_losses += 1

        if m.stats:
            xg_f = m.stats.expected_goals_home if is_home else m.stats.expected_goals_away
            xg_a = m.stats.expected_goals_away if is_home else m.stats.expected_goals_home
            if xg_f is not None and xg_a is not None:
                xgf_w += xg_f * opp_attack_merit * w
                xga_w += xg_a * opp_def_merit * w
                xg_w_sum += w
                if same_venue:
                    v_xgf_w += xg_f * opp_attack_merit
                    v_xga_w += xg_a * opp_def_merit
                    v_xg_sum += 1.0

            xgot_f = m.stats.xg_on_target_home if is_home else m.stats.xg_on_target_away
            xgot_a = m.stats.xg_on_target_away if is_home else m.stats.xg_on_target_home
            if xgot_f is not None and xgot_a is not None:
                xgotf_w += xgot_f * opp_attack_merit * w
                xgota_w += xgot_a * opp_def_merit * w
                xgot_w_sum += w

            gp_own = m.stats.goals_prevented_home if is_home else m.stats.goals_prevented_away
            if gp_own is not None:
                gp_w += gp_own * w
                gp_w_sum += w

            ts_f = m.stats.total_shots_home if is_home else m.stats.total_shots_away
            ts_a = m.stats.total_shots_away if is_home else m.stats.total_shots_home
            if ts_f is not None and ts_a is not None:
                tsf_w += ts_f * opp_attack_merit * w
                tsa_w += ts_a * opp_def_merit * w
                ts_w_sum += w

            sot_f = m.stats.shots_on_target_home if is_home else m.stats.shots_on_target_away
            sot_a = m.stats.shots_on_target_away if is_home else m.stats.shots_on_target_home
            if sot_f is not None and sot_a is not None:
                sotf_w += sot_f * opp_attack_merit * w
                sota_w += sot_a * opp_def_merit * w
                sot_w_sum += w

            sib_f = m.stats.shots_inside_box_home if is_home else m.stats.shots_inside_box_away
            sib_a = m.stats.shots_inside_box_away if is_home else m.stats.shots_inside_box_home
            if sib_f is not None and sib_a is not None:
                sib_f_w += sib_f * opp_attack_merit * w
                sib_a_w += sib_a * opp_def_merit * w
                sib_w_sum += w

            tib_f = m.stats.touches_in_box_home if is_home else m.stats.touches_in_box_away
            if tib_f is not None:
                tib_w += tib_f * opp_attack_merit * w
                tib_w_sum += w

            fte_f = m.stats.final_third_entries_home if is_home else m.stats.final_third_entries_away
            if fte_f is not None:
                fte_w += fte_f * w
                fte_w_sum += w

            brec_f = m.stats.ball_recoveries_home if is_home else m.stats.ball_recoveries_away
            if brec_f is not None:
                brec_w += brec_f * w
                brec_w_sum += w

            sv_own = m.stats.saves_home if is_home else m.stats.saves_away
            if sv_own is not None:
                saves_w += sv_own * w
                saves_w_sum += w
                denom_shots = max(float(sv_own + ga), float(sot_a or (sv_own + ga)))
                if denom_shots > 0:
                    save_ratio_num += sv_own * w
                    save_ratio_den += denom_shots * w

            bc = m.stats.big_chances_home if is_home else m.stats.big_chances_away
            if bc is not None:
                bc_w += bc * opp_attack_merit * w
                bc_w_sum += w

            pos = m.stats.ball_possession_home if is_home else m.stats.ball_possession_away
            if pos is not None:
                pos_w += pos * w
                pos_w_sum += w

            c_f = m.stats.corner_kicks_home if is_home else m.stats.corner_kicks_away
            c_a = m.stats.corner_kicks_away if is_home else m.stats.corner_kicks_home
            if c_f is not None and c_a is not None:
                cf_w += c_f * w
                ca_w += c_a * w
                c_w_sum += w

            yc_f = m.stats.yellow_cards_home if is_home else m.stats.yellow_cards_away
            yc_a = m.stats.yellow_cards_away if is_home else m.stats.yellow_cards_home
            if yc_f is not None and yc_a is not None:
                yc_f_w += yc_f * w
                yc_a_w += yc_a * w
                yc_w_sum += w

            fl_f = m.stats.fouls_home if is_home else m.stats.fouls_away
            if fl_f is not None:
                fl_w += fl_f * w
                fl_w_sum += w

            # Métrica Opta de Presión Alta (PPDA)
            ppda_f = m.stats.ppda_home if is_home else m.stats.ppda_away
            if ppda_f is not None and ppda_f > 0:
                ppda_w += ppda_f * w
                ppda_w_sum += w

            # Estadísticas Reales de 1er Tiempo (Opta 1ST Period)
            ht_xg_f = m.stats.ht_xg_home if is_home else m.stats.ht_xg_away
            ht_xg_a = m.stats.ht_xg_away if is_home else m.stats.ht_xg_home
            if ht_xg_f is not None and ht_xg_a is not None:
                ht_xgf_w += ht_xg_f * w
                ht_xga_w += ht_xg_a * w
                ht_xg_sum += w

            ht_c_f = m.stats.ht_corners_home if is_home else m.stats.ht_corners_away
            ht_c_a = m.stats.ht_corners_away if is_home else m.stats.ht_corners_home
            if ht_c_f is not None and ht_c_a is not None:
                ht_cf_w += ht_c_f * w
                ht_ca_w += ht_c_a * w
                ht_c_sum += w

        if gf > ga:
            wins += 1
            letters.append("W")
            margin_mult = math.log(abs(gf - ga) + 1.0)
            elo_delta += (18.0 if not is_home else 14.0) * margin_mult * opp_attack_merit * math.exp(-0.12 * idx)
        elif gf == ga:
            draws += 1
            letters.append("D")
            elo_delta += (2.5 if not is_home else -1.5) * math.exp(-0.12 * idx)
        else:
            losses += 1
            letters.append("L")
            margin_mult = math.log(abs(gf - ga) + 1.0)
            elo_delta -= (14.0 if not is_home else 18.0) * margin_mult * opp_def_merit * math.exp(-0.12 * idx)

    n = wins + draws + losses
    if n == 0 or w_sum == 0:
        return TeamFormMetrics(
            team_id=team_id,
            team_name=team_name,
            matches_analyzed=0,
            wins=0,
            draws=0,
            losses=0,
            goals_scored_avg=1.35,
            goals_conceded_avg=1.20,
            xg_for_avg=1.35,
            xg_against_avg=1.20,
            venue_xg_for_avg=1.35,
            venue_xg_against_avg=1.20,
            venue_record="S/D",
            total_shots_for_avg=12.5,
            total_shots_against_avg=11.8,
            shots_on_target_avg=4.4,
            shots_on_target_against_avg=4.1,
            big_chances_avg=2.1,
            possession_avg=50.0,
            corners_for_avg=4.8,
            corners_against_avg=4.5,
            yellow_cards_for_avg=2.1,
            yellow_cards_against_avg=2.1,
            fouls_for_avg=11.5,
            points_per_match=1.35,
            elo_rating=1500.0,
            form_string="N/A",
        )

    gf_avg = round(gf_w / w_sum, 2)
    ga_avg = round(ga_w / w_sum, 2)
    xg_for = round(xgf_w / xg_w_sum, 2) if xg_w_sum > 0 else gf_avg
    xg_against = round(xga_w / xg_w_sum, 2) if xg_w_sum > 0 else ga_avg
    xgot_for = round(xgotf_w / xgot_w_sum, 2) if xgot_w_sum > 0 else round(xg_for * 0.96, 2)
    xgot_against = round(xgota_w / xgot_w_sum, 2) if xgot_w_sum > 0 else round(xg_against * 0.96, 2)
    v_xg_for = round(v_xgf_w / v_xg_sum, 2) if v_xg_sum > 0 else xg_for
    v_xg_against = round(v_xga_w / v_xg_sum, 2) if v_xg_sum > 0 else xg_against

    sot_for = round(sotf_w / sot_w_sum, 2) if sot_w_sum > 0 else round(xg_for * 3.1, 2)
    sot_against = round(sota_w / sot_w_sum, 2) if sot_w_sum > 0 else round(xg_against * 3.1, 2)
    ts_for = round(tsf_w / ts_w_sum, 2) if ts_w_sum > 0 else round(sot_for * 2.75, 2)
    ts_against = round(tsa_w / ts_w_sum, 2) if ts_w_sum > 0 else round(sot_against * 2.75, 2)

    sib_for = round(sib_f_w / sib_w_sum, 2) if sib_w_sum > 0 else round(ts_for * 0.62, 2)
    sib_against = round(sib_a_w / sib_w_sum, 2) if sib_w_sum > 0 else round(ts_against * 0.62, 2)
    in_box_pct = round(min(85.0, max(35.0, (sib_for / max(1.0, ts_for)) * 100.0)), 1)
    tib_avg = round(tib_w / tib_w_sum, 1) if tib_w_sum > 0 else round(sib_for * 2.4, 1)
    fte_avg = round(fte_w / fte_w_sum, 1) if fte_w_sum > 0 else 48.0
    brec_avg = round(brec_w / brec_w_sum, 1) if brec_w_sum > 0 else 46.0

    bc_avg = round(bc_w / bc_w_sum, 2) if bc_w_sum > 0 else round(xg_for * 1.4, 2)
    pos_avg = round(pos_w / pos_w_sum, 1) if pos_w_sum > 0 else 50.0
    corners_for = round(cf_w / c_w_sum, 2) if c_w_sum > 0 else 4.8
    corners_against = round(ca_w / c_w_sum, 2) if c_w_sum > 0 else 4.5
    yc_for = round(yc_f_w / yc_w_sum, 2) if yc_w_sum > 0 else 2.1
    yc_against = round(yc_a_w / yc_w_sum, 2) if yc_w_sum > 0 else 2.1
    fouls_for = round(fl_w / fl_w_sum, 1) if fl_w_sum > 0 else 11.5
    ppda_avg = round(ppda_w / ppda_w_sum, 1) if ppda_w_sum > 0 else 11.5

    ht_xg_for = round(ht_xgf_w / ht_xg_sum, 2) if ht_xg_sum > 0 else round(xg_for * 0.44, 2)
    ht_xg_against = round(ht_xga_w / ht_xg_sum, 2) if ht_xg_sum > 0 else round(xg_against * 0.44, 2)
    ht_c_for = round(ht_cf_w / ht_c_sum, 2) if ht_c_sum > 0 else round(corners_for * 0.47, 2)
    ht_c_against = round(ht_ca_w / ht_c_sum, 2) if ht_c_sum > 0 else round(corners_against * 0.47, 2)

    gk_saves = round(saves_w / saves_w_sum, 2) if saves_w_sum > 0 else round(max(1.5, sot_against - ga_avg), 2)
    gk_sv_pct = (
        round(min(86.0, max(52.0, (save_ratio_num / save_ratio_den) * 100.0)), 1)
        if save_ratio_den > 0
        else round(min(84.0, max(55.0, (gk_saves / max(1.0, sot_against)) * 100.0)), 1)
    )
    gk_gp = round(gp_w / gp_w_sum, 2) if gp_w_sum > 0 else round(xgot_against - ga_avg, 2)
    sos_val = round(sum(opp_ppg_list) / len(opp_ppg_list), 2) if opp_ppg_list else 1.35

    elo = 1500.0 + elo_delta + ((xg_for - xg_against) * 28.0) + ((xgot_for - xgot_against) * 12.0)
    if standing_info:
        ppg_bonus = (standing_info.get("ppg", 1.35) - 1.35) * 65.0
        gd_bonus = standing_info.get("gd_per_match", 0.0) * 22.0
        elo += ppg_bonus + gd_bonus

    v_label = "Local" if target_venue == "home" else "Visita"
    venue_rec_str = f"{v_label}: {v_wins}V-{v_draws}E-{v_losses}D"

    # NUEVO v4: Calidad Real de Ocasiones (xG / Disparo & xGOT / SoT)
    xg_per_shot_f = round(xg_for / max(1.0, ts_for), 3)
    xg_per_shot_a = round(xg_against / max(1.0, ts_against), 3)

    # NUEVO v4: Clasificador de Estilos Tácticos
    # 1. Posesión y Asedio
    # 2. Contragolpe Vertical
    # 3. Bloque Bajo Defensivo
    # 4. Presión Alta e Intensidad
    if pos_avg >= 56.5 and tib_avg >= 21.0:
        style_desc = "Posesión y Asedio en Área Rival"
    elif brec_avg >= 48.0 and fouls_for >= 13.0 and fte_avg >= 50.0:
        style_desc = "Presión Alta e Intensidad Asfixiante"
    elif pos_avg <= 44.5 and xg_per_shot_f >= 0.11:
        style_desc = "Contragolpe Vertical Rápido"
    elif ga_avg <= 1.05 and sib_against <= 6.5 and pos_avg <= 48.0:
        style_desc = "Bloque Bajo y Muralla Defensiva"
    else:
        style_desc = "Juego Equilibrado y Transición Mixta"

    # NUEVO v4: Índice de Urgencia Competitiva y Necesidad de Tabla (0 a 100)
    urg_idx = 50.0
    urg_lbl = "Urgencia Media 🟡"
    pos_num = None
    pts_num = None
    pj_num = None
    if standing_info:
        pos_num = standing_info.get("position")
        pts_num = standing_info.get("points")
        pj_num = standing_info.get("matches")
        ppg_val = standing_info.get("ppg", 1.35)

        # Peleando Descenso (fondo de tabla)
        if pos_num and pos_num >= 17:
            urg_idx = 90.0
            urg_lbl = "Urgencia Extrema (Pelea Descenso) 🔴"
        # Peleando Título o Puestos Champions (Top 4)
        elif pos_num and pos_num <= 4:
            urg_idx = 82.0
            urg_lbl = "Alta Necesidad (Pelea Título/Champions) 🔥"
        # Zona Media-Alta (Pelea Copas / Europa League)
        elif pos_num and 5 <= pos_num <= 8:
            urg_idx = 72.0
            urg_lbl = "Urgencia Alta (Clasificación a Copas) 🟠"
        # Zona Cómoda Media Tabla (sin peligro ni copas directas)
        elif pos_num and 9 <= pos_num <= 16:
            urg_idx = 45.0
            urg_lbl = "Zona Cómoda de Tabla 🟢"

    return TeamFormMetrics(
        team_id=team_id,
        team_name=team_name,
        matches_analyzed=n,
        wins=wins,
        draws=draws,
        losses=losses,
        goals_scored_avg=gf_avg,
        goals_conceded_avg=ga_avg,
        xg_for_avg=xg_for,
        xg_against_avg=xg_against,
        venue_xg_for_avg=v_xg_for,
        venue_xg_against_avg=v_xg_against,
        venue_record=venue_rec_str,
        total_shots_for_avg=ts_for,
        total_shots_against_avg=ts_against,
        shots_on_target_avg=sot_for,
        shots_on_target_against_avg=sot_against,
        big_chances_avg=bc_avg,
        possession_avg=pos_avg,
        corners_for_avg=corners_for,
        corners_against_avg=corners_against,
        yellow_cards_for_avg=yc_for,
        yellow_cards_against_avg=yc_against,
        fouls_for_avg=fouls_for,
        points_per_match=round((wins * 3 + draws) / n, 2),
        elo_rating=round(elo, 1),
        form_string="-".join(letters[:5]),
        rest_days=rest_days,
        matches_last_14d=matches_14d,
        fatigue_index=fatigue_idx,
        fatigue_label=fatigue_lbl,
        xgot_for_avg=xgot_for,
        xgot_against_avg=xgot_against,
        shots_in_box_for_avg=sib_for,
        shots_in_box_against_avg=sib_against,
        inside_box_shot_pct=in_box_pct,
        touches_in_box_avg=tib_avg,
        final_third_entries_avg=fte_avg,
        ball_recoveries_avg=brec_avg,
        ht_xg_for_avg=ht_xg_for,
        ht_xg_against_avg=ht_xg_against,
        ht_corners_for_avg=ht_c_for,
        ht_corners_against_avg=ht_c_against,
        gk_saves_avg=gk_saves,
        gk_save_pct=gk_sv_pct,
        gk_goals_prevented_avg=gk_gp,
        sos_rating=sos_val,
        xg_per_shot_for=xg_per_shot_f,
        xg_per_shot_against=xg_per_shot_a,
        tactical_style=style_desc,
        urgency_index=urg_idx,
        urgency_label=urg_lbl,
        table_position=pos_num,
        table_points=pts_num,
        table_matches=pj_num,
        ppda_avg=ppda_avg,
        center_back_synergy_score=100.0,
        defensive_line_status="Zaga Central Consolidada 🛡️",
    )


def _compute_injury_penalty(
    players: Optional[List[PlayerPropSchema]],
    lineup: Optional[TeamLineupSchema] = None,
    gk_info: Optional[Dict[str, Any]] = None,
) -> tuple[float, float, List[str]]:
    """
    Calcula la penalización multidimensional por ausencias (Ofensiva + Defensiva):
    - Analiza goleadores y generadores de xG ausentes.
    - Inspecciona el reporte oficial de lesiones/sanciones de la alineación de Sofascore.
    - Evalúa vulnerabilidad en el arco si el arquero titular está ausente o tiene bajo % de atajadas.
    """
    missing_sot = 0.0
    key_missing_names: List[str] = []
    total_team_sot = max(1.0, sum(p.sot_per_match for p in (players or [])))

    # 1. Bajas de Jugadores Clave según estadísticas de remates y xG
    if players:
        for p in players:
            if p.is_missing and (p.sot_per_match >= 0.38 or p.xg_per_match >= 0.15 or p.goals >= 1):
                missing_sot += p.sot_per_match
                key_missing_names.append(f"{p.name} ({p.sot_per_match:.1f} SoT/P, {p.goals}G)")

    # 2. Bajas Oficiales Reportadas en la Alineación de Sofascore (Lesiones / Sanciones)
    if lineup and lineup.missing_players:
        for raw_name in lineup.missing_players:
            name_clean = raw_name.split("(")[0].strip()
            if not any(name_clean.lower() in kn.lower() for kn in key_missing_names):
                key_missing_names.append(f"{raw_name} 🚑")
                missing_sot += 0.42

    # 3. Penalización Defensiva por Incertidumbre en el Arco
    defensive_penalty_pct = 0.0
    if gk_info:
        save_pct = gk_info.get("save_pct", 70.0)
        gp = gk_info.get("goals_prevented", 0.0)
        if save_pct < 64.0 or gp < -0.15:
            defensive_penalty_pct = 10.0
            gk_name = gk_info.get("name", "Portero")
            key_missing_names.append(f"Inseguridad en el Arco: {gk_name} (Atajadas: {save_pct:.0f}%) 🧤⚠️")

    # Tasa ofensiva: penalización máxima del 25%
    penalty_ratio = min(0.25, (missing_sot / total_team_sot) * 0.70)
    return round(penalty_ratio * 100.0, 1), defensive_penalty_pct, key_missing_names


def _extract_starting_goalkeeper(
    lineup_players: Optional[List[PlayerLineupSchema]],
    fallback_team_name: str,
    form_saves_avg: float,
    form_save_pct: float,
    form_gp_avg: float,
) -> Dict[str, Any]:
    """
    Identifica al portero titular desde la alineación de Sofascore y combina sus
    métricas Opta de temporada (savesPerGame, goalsPrevented) con el rendimiento reciente del arco.
    """
    default_name = f"Arquero {fallback_team_name}"
    if not lineup_players:
        return {
            "name": default_name,
            "saves_avg": form_saves_avg,
            "save_pct": form_save_pct,
            "goals_prevented": form_gp_avg,
            "match_saves": None,
        }

    gks = [p for p in lineup_players if p.position == "G"]
    starter_gks = [p for p in gks if not p.substitute]
    chosen_gk = starter_gks[0] if starter_gks else (gks[0] if gks else None)
    if not chosen_gk:
        return {
            "name": default_name,
            "saves_avg": form_saves_avg,
            "save_pct": form_save_pct,
            "goals_prevented": form_gp_avg,
            "match_saves": None,
        }

    if chosen_gk.gk_saves_per_game is not None and chosen_gk.gk_saves_per_game > 0:
        blended_saves = round((0.55 * chosen_gk.gk_saves_per_game) + (0.45 * form_saves_avg), 2)
    else:
        blended_saves = form_saves_avg

    if chosen_gk.gk_goals_prevented is not None:
        # gk_goals_prevented en seasonRatingBreakdown es acumulado de temporada (~15-25 PJ), normalizar a por partido
        per_match_gp = max(-0.65, min(0.65, chosen_gk.gk_goals_prevented / 16.0))
        blended_gp = round((0.55 * per_match_gp) + (0.45 * form_gp_avg), 2)
    else:
        blended_gp = form_gp_avg

    adj_save_pct = min(85.0, max(54.0, form_save_pct + (blended_gp * 6.5)))

    return {
        "name": chosen_gk.name,
        "saves_avg": blended_saves,
        "save_pct": round(adj_save_pct, 1),
        "goals_prevented": blended_gp,
        "match_saves": chosen_gk.match_saves,
    }


def _compute_defensive_synergy(
    lineup_players: Optional[List[PlayerLineupSchema]],
    missing_players: Optional[List[PlayerPropSchema]],
    team_name: str,
) -> tuple[float, str, float]:
    """
    Calcula el índice de Sinergia de la Zaga Defensiva (0-100%).
    - Evalúa si defensores clave (position == 'D') están ausentes en missing_players.
    - Si la pareja de centrales habitual se rompe o faltan defensores titulares:
      penaliza la sinergia e incrementa el multiplicador de goles concedidos (+10% a +16%).
    Retorna: (synergy_pct, status_label, penalty_goals_conceded_mult)
    """
    missing_defs: List[str] = []
    if missing_players:
        for p in missing_players:
            if p.is_missing and p.position == "D":
                missing_defs.append(p.name)

    starters_def = 0
    if lineup_players:
        starters_def = sum(1 for p in lineup_players if not p.substitute and p.position == "D")

    if len(missing_defs) >= 2:
        synergy_pct = 72.0
        label = f"Zaga Improvizada ⚠️ (Bajas: {', '.join(missing_defs[:2])})"
        penalty_mult = 1.15
    elif len(missing_defs) == 1:
        synergy_pct = 84.0
        label = f"Defensa Mermada 🟡 (Baja clave: {missing_defs[0]})"
        penalty_mult = 1.09
    elif starters_def in (3, 4, 5):
        synergy_pct = 100.0
        label = f"Zaga Titular Consolidada 🛡️ ({starters_def} Defensores)"
        penalty_mult = 1.00
    else:
        synergy_pct = 95.0
        label = "Zaga Estable / Regular 🛡️"
        penalty_mult = 1.00

    return synergy_pct, label, penalty_mult


def _get_league_profile(tournament: Optional[TournamentSchema]) -> tuple[float, str]:
    """
    Calibra dinámicamente el parámetro rho de Dixon-Coles y el perfil competitivo según la liga.
    - Ligas Over / juego vertical (Bundesliga, UCL, Eredivisie, Premier League): rho = -0.06
    - Ligas Under / tácticas y de alta fricción (Argentina, Colombia, Uruguay, Segunda División): rho = -0.14
    - Ligas estándar UEFA y Conference League: rho = -0.09
    """
    if not tournament or not tournament.name:
        return -0.09, "Estándar Competitivo UEFA (Rho -0.09) ⚖️"

    t_name = tournament.name.lower()
    t_country = (tournament.country or "").lower()

    under_keywords = [
        "argentina", "colombia", "uruguay", "segunda", "laliga 2", "serie b",
        "paraguay", "chile", "bolivia", "peru", "liga betplay", "liga profesional",
        "copa de la liga profesional", "primera b"
    ]
    if any(kw in t_name or kw in t_country for kw in under_keywords):
        return -0.14, "Liga de Alta Fricción y Tendencia Under (Rho -0.14) 🔒"

    over_keywords = [
        "bundesliga", "champions league", "eredivisie", "premier league",
        "norway", "eliteserien", "austria", "belgium", "pro league", "switzerland"
    ]
    if any(kw in t_name or kw in t_country for kw in over_keywords):
        return -0.06, "Liga Abierta de Alta Producción Ofensiva (Rho -0.06) ⚡"

    if "conference" in t_name:
        return -0.08, "UEFA Europa Conference League (Rho -0.08) 🏆"
    if "europa league" in t_name:
        return -0.07, "UEFA Europa League (Rho -0.07) 🏆"

    return -0.09, "Estándar Táctico Internacional (Rho -0.09) ⚖️"


def predict_match_poisson(
    match: MatchDetailSchema,
    home_form: TeamFormMetrics,
    away_form: TeamFormMetrics,
    home_players: Optional[List[PlayerPropSchema]] = None,
    away_players: Optional[List[PlayerPropSchema]] = None,
    max_goals: int = 6,
) -> MatchPrediction:
    """
    Motor Predictivo Híbrido v3.5 (Opta + H2H Profundo + Goalkeeper Saves):
    - Calidad de Ocasiones Opta (xG + xGOT + Toques en Área Rival + Tiros Dentro del Área)
    - Efecto Portero Rival (Opta Goals Prevented & % Atajadas)
    - Historial H2H Profundo (Goles, Córners y Tarjetas en duelos directos recientes)
    - Modelo Especializado de Atajadas del Portero (Remates al Arco del Rival × Save% + Media del Arquero)
    - Dixon-Coles + Simulación Monte Carlo (10,000 partidos) + Gradient Boosting ML
    """
    # Extraer perfil de ambos porteros titulares (Opta Goalkeeping)
    gk_home_info = _extract_starting_goalkeeper(
        match.home_lineup.players if match.home_lineup else None,
        match.home_team.short_name or match.home_team.name,
        home_form.gk_saves_avg,
        home_form.gk_save_pct,
        home_form.gk_goals_prevented_avg,
    )
    gk_away_info = _extract_starting_goalkeeper(
        match.away_lineup.players if match.away_lineup else None,
        match.away_team.short_name or match.away_team.name,
        away_form.gk_saves_avg,
        away_form.gk_save_pct,
        away_form.gk_goals_prevented_avg,
    )

    pen_home_pct, def_pen_home, key_missing_home = _compute_injury_penalty(
        home_players, match.home_lineup, gk_home_info
    )
    pen_away_pct, def_pen_away, key_missing_away = _compute_injury_penalty(
        away_players, match.away_lineup, gk_away_info
    )
    injury_mult_home = 1.0 - (pen_home_pct / 100.0)
    injury_mult_away = 1.0 - (pen_away_pct / 100.0)
    def_leak_home = 1.0 + (def_pen_home / 100.0)
    def_leak_away = 1.0 + (def_pen_away / 100.0)

    rest_diff = home_form.rest_days - away_form.rest_days
    fatigue_mult_home = 0.94 if home_form.fatigue_index >= 75.0 else (1.03 if rest_diff >= 2.5 else 1.0)
    fatigue_mult_away = 0.94 if away_form.fatigue_index >= 75.0 else (1.03 if rest_diff <= -2.5 else 1.0)

    # Índice de Peligro en el Área de Opta (Toques en área rival + Tiros dentro del área + Entradas a 3/4)
    raw_danger_h = (home_form.touches_in_box_avg * 1.4) + (home_form.shots_in_box_for_avg * 2.5) + (home_form.final_third_entries_avg * 0.35)
    raw_danger_a = (away_form.touches_in_box_avg * 1.4) + (away_form.shots_in_box_for_avg * 2.5) + (away_form.final_third_entries_avg * 0.35)
    danger_tot = max(1.0, raw_danger_h + raw_danger_a)
    opta_danger_home = round((raw_danger_h / danger_tot) * 100.0, 1)
    opta_danger_away = round(100.0 - opta_danger_home, 1)

    box_boost_home = max(0.92, min(1.10, 1.0 + ((home_form.touches_in_box_avg - 19.5) * 0.004)))
    box_boost_away = max(0.92, min(1.10, 1.0 + ((away_form.touches_in_box_avg - 19.5) * 0.004)))

    # Escudo del portero rival (si el arquero rival evita goles según Opta, frena el xG esperado)
    away_gk_shield_mult = max(0.91, min(1.09, 1.0 - (gk_away_info["goals_prevented"] * 0.14)))
    home_gk_shield_mult = max(0.91, min(1.09, 1.0 - (gk_home_info["goals_prevented"] * 0.14)))

    # Ataque y Defensa integrando xG + xGOT (Post-Shot xG de Opta) + Condición Local/Visita + Goles
    home_attack = (
        (0.32 * home_form.xg_for_avg)
        + (0.20 * home_form.xgot_for_avg)
        + (0.24 * home_form.venue_xg_for_avg)
        + (0.24 * home_form.goals_scored_avg)
    ) * box_boost_home
    home_defense = (
        (0.35 * home_form.xg_against_avg)
        + (0.18 * home_form.xgot_against_avg)
        + (0.24 * home_form.venue_xg_against_avg)
        + (0.23 * home_form.goals_conceded_avg)
    )

    away_attack = (
        (0.32 * away_form.xg_for_avg)
        + (0.20 * away_form.xgot_for_avg)
        + (0.24 * away_form.venue_xg_for_avg)
        + (0.24 * away_form.goals_scored_avg)
    ) * box_boost_away
    away_defense = (
        (0.35 * away_form.xg_against_avg)
        + (0.18 * away_form.xgot_against_avg)
        + (0.24 * away_form.venue_xg_against_avg)
        + (0.23 * away_form.goals_conceded_avg)
    )

    elo_diff = (home_form.elo_rating + 45.0) - away_form.elo_rating
    elo_multiplier_home = math.pow(10.0, max(-250.0, min(250.0, elo_diff)) / 1200.0)
    elo_multiplier_away = math.pow(10.0, max(-250.0, min(250.0, -elo_diff)) / 1200.0)

    # NUEVO v4: Impacto del Choque Táctico y Factor de Urgencia en la Producción Ofensiva/Defensiva
    # Si un equipo tiene urgencia extrema (descenso o título), aumenta su intensidad ofensiva y de faltas
    urgency_boost_home = 1.0 + ((home_form.urgency_index - 50.0) / 50.0) * 0.04
    urgency_boost_away = 1.0 + ((away_form.urgency_index - 50.0) / 50.0) * 0.04

    # Síntesis del Choque Táctico
    style_h = home_form.tactical_style
    style_a = away_form.tactical_style
    if "Posesión y Asedio" in style_h and "Contragolpe" in style_a:
        tactical_narrative = f"{match.home_team.name} dominará la posesión y el territorio; {match.away_team.name} buscará lastimar en transiciones rápidas (alto potencial de córners para el local)."
    elif "Posesión y Asedio" in style_a and "Contragolpe" in style_h:
        tactical_narrative = f"{match.away_team.name} tendrá el control del balón; {match.home_team.name} apostará a la verticalidad al espacio."
    elif "Presión Alta" in style_h or "Presión Alta" in style_a:
        tactical_narrative = "Partido de alta fricción física en mediocampo, duelos individuales constantes y probabilidad elevada de tarjetas y faltas."
    elif "Bloque Bajo" in style_h and "Bloque Bajo" in style_a:
        tactical_narrative = "Duelo táctico cerrado y de cautela mutua; tendencia clara a marcadores cortos y ritmo bajo."
    else:
        tactical_narrative = f"Choque de propuestas balanceadas: {style_h} vs {style_a}."

    # Síntesis de Urgencia Competitiva
    if home_form.urgency_index >= 80.0 and away_form.urgency_index >= 80.0:
        urgency_summary = "🔥 Duelo de máxima tensión competitiva: Ambos equipos se juegan objetivos cruciales de la temporada."
    elif home_form.urgency_index >= 80.0:
        urgency_summary = f"⚡ Mayor necesidad para {match.home_team.name} ({home_form.urgency_label}), que llega obligado a sumar."
    elif away_form.urgency_index >= 80.0:
        urgency_summary = f"⚡ Mayor necesidad para {match.away_team.name} ({away_form.urgency_label}), que no puede ceder puntos."
    else:
        urgency_summary = "Compromiso de exigencia estándar de temporada regular."

    # Calidad de Remate (Shot Quality): xG por disparo
    sq_h = home_form.xg_per_shot_for
    sq_a = away_form.xg_per_shot_for
    if sq_h >= sq_a + 0.025:
        sq_edge = f"{match.home_team.name} genera ocasiones de remate más claras ({sq_h:.3f} vs {sq_a:.3f} xG/tiro)"
    elif sq_a >= sq_h + 0.025:
        sq_edge = f"{match.away_team.name} genera disparos con mayor probabilidad de gol ({sq_a:.3f} vs {sq_h:.3f} xG/tiro)"
    else:
        sq_edge = f"Calidad de disparo similar en ambos clubes ({sq_h:.3f} vs {sq_a:.3f} xG/tiro)"

    # NUEVO v4.5: Evaluación de Sinergia de Zaga Central y Bajas Defensivas
    syn_home_pct, syn_home_lbl, syn_home_penalty_mult = _compute_defensive_synergy(
        match.home_lineup.players if match.home_lineup else None,
        home_players,
        match.home_team.name,
    )
    syn_away_pct, syn_away_lbl, syn_away_penalty_mult = _compute_defensive_synergy(
        match.away_lineup.players if match.away_lineup else None,
        away_players,
        match.away_team.name,
    )

    # NUEVO v4.5: Calibración Dinámica de Liga (Rho específico de Dixon-Coles)
    league_rho, league_profile = _get_league_profile(match.tournament)

    # NUEVO v4.5: Presión Alta Opta (PPDA)
    ppda_h = home_form.ppda_avg if home_form.ppda_avg > 0 else 11.5
    ppda_a = away_form.ppda_avg if away_form.ppda_avg > 0 else 11.5
    ppda_boost_h = 1.04 if (ppda_h <= 9.5 and ppda_a >= 13.0) else 1.0
    ppda_boost_a = 1.04 if (ppda_a <= 9.5 and ppda_h >= 13.0) else 1.0

    lambda_home = max(
        0.32,
        ((home_attack * 0.56) + (away_defense * 0.44))
        * 1.06
        * elo_multiplier_home
        * injury_mult_home
        * def_leak_away
        * fatigue_mult_home
        * away_gk_shield_mult
        * urgency_boost_home
        * syn_away_penalty_mult
        * ppda_boost_h,
    )
    lambda_away = max(
        0.28,
        ((away_attack * 0.56) + (home_defense * 0.44))
        * 0.94
        * elo_multiplier_away
        * injury_mult_away
        * def_leak_home
        * fatigue_mult_away
        * home_gk_shield_mult
        * urgency_boost_away
        * syn_home_penalty_mult
        * ppda_boost_a,
    )

    # Integración del Historial H2H Profundo (Goles/xG en enfrentamientos directos recientes + Dominancia)
    if match.h2h:
        if (
            match.h2h.recent_matches_analyzed >= 2
            and match.h2h.h2h_home_goals_avg is not None
            and match.h2h.h2h_away_goals_avg is not None
        ):
            h2h_weight = min(0.20, 0.05 * match.h2h.recent_matches_analyzed)
            lambda_home = ((1.0 - h2h_weight) * lambda_home) + (
                h2h_weight * max(0.35, match.h2h.h2h_home_goals_avg)
            )
            lambda_away = ((1.0 - h2h_weight) * lambda_away) + (
                h2h_weight * max(0.30, match.h2h.h2h_away_goals_avg)
            )
        elif match.h2h.total_matches >= 3:
            h2h_home_ratio = match.h2h.team_wins_home / match.h2h.total_matches
            h2h_away_ratio = match.h2h.team_wins_away / match.h2h.total_matches
            lambda_home *= 0.95 + (0.10 * h2h_home_ratio)
            lambda_away *= 0.95 + (0.10 * h2h_away_ratio)

    # Matriz de Tiempo Completo con corrección Dixon-Coles Calibrada por Liga
    rho = league_rho
    matrix: Dict[tuple[int, int], float] = {}
    total_prob = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            base_p = _poisson_pmf(h, lambda_home) * _poisson_pmf(a, lambda_away)
            if h == 0 and a == 0:
                tau = 1.0 - (lambda_home * lambda_away * rho)
            elif h == 0 and a == 1:
                tau = 1.0 + (lambda_home * rho)
            elif h == 1 and a == 0:
                tau = 1.0 + (lambda_away * rho)
            elif h == 1 and a == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0

            p = max(0.0, base_p * tau)
            matrix[(h, a)] = p
            total_prob += p

    for k in matrix:
        matrix[k] /= total_prob

    stat_p_home = sum(p for (h, a), p in matrix.items() if h > a)
    stat_p_draw = sum(p for (h, a), p in matrix.items() if h == a)
    stat_p_away = sum(p for (h, a), p in matrix.items() if h < a)

    p_over_1_5 = sum(p for (h, a), p in matrix.items() if (h + a) > 1)
    p_over_2_5 = sum(p for (h, a), p in matrix.items() if (h + a) > 2)
    p_over_3_5 = sum(p for (h, a), p in matrix.items() if (h + a) > 3)
    p_btts = sum(p for (h, a), p in matrix.items() if h >= 1 and a >= 1)

    mc = _run_monte_carlo_10k(lambda_home, lambda_away, seed=match.event_id)

    # Remates Totales y Remates al Arco Esperados (ajustados con peligro Opta y H2H)
    exp_shots_home = round(
        ((home_form.total_shots_for_avg * 0.56) + (away_form.total_shots_against_avg * 0.44))
        * 1.05
        * injury_mult_home
        * box_boost_home,
        2,
    )
    exp_shots_away = round(
        ((away_form.total_shots_for_avg * 0.56) + (home_form.total_shots_against_avg * 0.44))
        * 0.95
        * injury_mult_away
        * box_boost_away,
        2,
    )
    exp_shots_total = round(exp_shots_home + exp_shots_away, 2)
    prob_o22_5_shots = round(_poisson_prob_over(22.5, exp_shots_total) * 100, 1)
    prob_o24_5_shots = round(_poisson_prob_over(24.5, exp_shots_total) * 100, 1)
    prob_o26_5_shots = round(_poisson_prob_over(26.5, exp_shots_total) * 100, 1)

    exp_sot_home = round(
        ((home_form.shots_on_target_avg * 0.56) + (away_form.shots_on_target_against_avg * 0.44))
        * 1.05
        * injury_mult_home,
        2,
    )
    exp_sot_away = round(
        ((away_form.shots_on_target_avg * 0.56) + (home_form.shots_on_target_against_avg * 0.44))
        * 0.95
        * injury_mult_away,
        2,
    )
    if match.h2h and match.h2h.h2h_sot_avg is not None and match.h2h.recent_matches_analyzed >= 2:
        sot_raw_tot = max(1.0, exp_sot_home + exp_sot_away)
        blended_sot_tot = (0.82 * sot_raw_tot) + (0.18 * match.h2h.h2h_sot_avg)
        sot_scale = blended_sot_tot / sot_raw_tot
        exp_sot_home = round(exp_sot_home * sot_scale, 2)
        exp_sot_away = round(exp_sot_away * sot_scale, 2)

    exp_sot_total = round(exp_sot_home + exp_sot_away, 2)
    prob_o7_5_sot = round(_poisson_prob_over(7.5, exp_sot_total) * 100, 1)
    prob_o8_5_sot = round(_poisson_prob_over(8.5, exp_sot_total) * 100, 1)
    prob_o9_5_sot = round(_poisson_prob_over(9.5, exp_sot_total) * 100, 1)

    # =========================================================================
    # MOTOR ESPECIALIZADO DE ATAJADAS DEL PORTERO (GOALKEEPER SAVES MODEL)
    # =========================================================================
    # 1. El Portero Local recibe los disparos al arco del Visitante (exp_sot_away).
    #    Cuantos más tiros desde fuera del área realiza el rival, mayor es el % de atajada.
    away_outside_box_share = max(0.15, min(0.65, (100.0 - away_form.inside_box_shot_pct) / 100.0))
    home_outside_box_share = max(0.15, min(0.65, (100.0 - home_form.inside_box_shot_pct) / 100.0))

    eff_save_rate_home_gk = min(
        0.84,
        max(
            0.56,
            (gk_home_info["save_pct"] / 100.0) + ((away_outside_box_share - 0.38) * 0.14),
        ),
    )
    eff_save_rate_away_gk = min(
        0.84,
        max(
            0.56,
            (gk_away_info["save_pct"] / 100.0) + ((home_outside_box_share - 0.38) * 0.14),
        ),
    )

    # Cruce: 65% Remates a puerta proyectados del rival × % Atajada + 35% Promedio histórico del arquero
    sot_based_saves_home = max(0.8, exp_sot_away * eff_save_rate_home_gk)
    sot_based_saves_away = max(0.8, exp_sot_home * eff_save_rate_away_gk)

    exp_saves_home_gk = round(
        (0.65 * sot_based_saves_home) + (0.35 * gk_home_info["saves_avg"]), 2
    )
    exp_saves_away_gk = round(
        (0.65 * sot_based_saves_away) + (0.35 * gk_away_info["saves_avg"]), 2
    )

    if match.h2h and match.h2h.h2h_saves_avg is not None and match.h2h.recent_matches_analyzed >= 2:
        raw_sv_tot = max(1.0, exp_saves_home_gk + exp_saves_away_gk)
        blended_sv_tot = (0.85 * raw_sv_tot) + (0.15 * match.h2h.h2h_saves_avg)
        sv_scale = blended_sv_tot / raw_sv_tot
        exp_saves_home_gk = round(exp_saves_home_gk * sv_scale, 2)
        exp_saves_away_gk = round(exp_saves_away_gk * sv_scale, 2)

    exp_saves_total = round(exp_saves_home_gk + exp_saves_away_gk, 2)

    prob_h_gk_2p = round(_poisson_prob_over(1.5, exp_saves_home_gk) * 100, 1)
    prob_h_gk_3p = round(_poisson_prob_over(2.5, exp_saves_home_gk) * 100, 1)
    prob_h_gk_4p = round(_poisson_prob_over(3.5, exp_saves_home_gk) * 100, 1)
    prob_h_gk_5p = round(_poisson_prob_over(4.5, exp_saves_home_gk) * 100, 1)

    prob_a_gk_2p = round(_poisson_prob_over(1.5, exp_saves_away_gk) * 100, 1)
    prob_a_gk_3p = round(_poisson_prob_over(2.5, exp_saves_away_gk) * 100, 1)
    prob_a_gk_4p = round(_poisson_prob_over(3.5, exp_saves_away_gk) * 100, 1)
    prob_a_gk_5p = round(_poisson_prob_over(4.5, exp_saves_away_gk) * 100, 1)

    prob_o4_5_saves = round(_poisson_prob_over(4.5, exp_saves_total) * 100, 1)
    prob_o5_5_saves = round(_poisson_prob_over(5.5, exp_saves_total) * 100, 1)
    prob_o6_5_saves = round(_poisson_prob_over(6.5, exp_saves_total) * 100, 1)

    # Córners (con ajuste por entradas al último tercio Opta y H2H profundo)
    exp_corners_home = round(
        ((home_form.corners_for_avg * 0.58) + (away_form.corners_against_avg * 0.42)) * 1.05,
        2,
    )
    exp_corners_away = round(
        ((away_form.corners_for_avg * 0.58) + (home_form.corners_against_avg * 0.42)) * 0.95,
        2,
    )
    if match.h2h and match.h2h.h2h_corners_avg is not None and match.h2h.recent_matches_analyzed >= 2:
        raw_crn_tot = max(1.0, exp_corners_home + exp_corners_away)
        blended_crn_tot = (0.82 * raw_crn_tot) + (0.18 * match.h2h.h2h_corners_avg)
        crn_scale = blended_crn_tot / raw_crn_tot
        exp_corners_home = round(exp_corners_home * crn_scale, 2)
        exp_corners_away = round(exp_corners_away * crn_scale, 2)

    exp_corners_total = round(exp_corners_home + exp_corners_away, 2)
    prob_o8_5_corners = round(_poisson_prob_over(8.5, exp_corners_total) * 100, 1)
    prob_o9_5_corners = round(_poisson_prob_over(9.5, exp_corners_total) * 100, 1)
    prob_o10_5_corners = round(_poisson_prob_over(10.5, exp_corners_total) * 100, 1)

    # Córners de 1er Tiempo (usando estadísticas reales 1ST Period de Opta)
    raw_1h_crn_h = (home_form.ht_corners_for_avg * 0.58) + (away_form.ht_corners_against_avg * 0.42)
    raw_1h_crn_a = (away_form.ht_corners_for_avg * 0.58) + (home_form.ht_corners_against_avg * 0.42)
    exp_1h_corners = round(
        (0.55 * (raw_1h_crn_h + raw_1h_crn_a)) + (0.45 * (exp_corners_total * 0.47)), 2
    )
    prob_1h_o4_5_corners = round(_poisson_prob_over(4.5, exp_1h_corners) * 100, 1)

    # Ensamble Machine Learning (Gradient Boosting)
    ml_engine = get_ml_predictor()
    ml_res = ml_engine.predict_proba_match(
        elo_diff=elo_diff,
        xg_home=lambda_home,
        xg_away=lambda_away,
        sot_diff=exp_sot_home - exp_sot_away,
        corners_diff=exp_corners_home - exp_corners_away,
        rest_days_diff=rest_diff,
        implied_home=match.odds.implied_prob_home if match.odds else None,
        implied_draw=match.odds.implied_prob_draw if match.odds else None,
        implied_away=match.odds.implied_prob_away if match.odds else None,
    )

    learner = get_auto_learner()
    w_dixon, w_ml = learner.get_ensemble_weights()
    p_home = (w_dixon * stat_p_home) + (w_ml * (ml_res["ml_prob_home"] / 100.0))
    p_draw = (w_dixon * stat_p_draw) + (w_ml * (ml_res["ml_prob_draw"] / 100.0))
    p_away = (w_dixon * stat_p_away) + (w_ml * (ml_res["ml_prob_away"] / 100.0))
    norm_ens = max(0.001, p_home + p_draw + p_away)
    p_home, p_draw, p_away = p_home / norm_ens, p_draw / norm_ens, p_away / norm_ens

    # Mercados de 1er Tiempo (1H calibrado con xG real del 1ST Period de Opta)
    opta_1h_h = (home_form.ht_xg_for_avg * 0.56) + (away_form.ht_xg_against_avg * 0.44)
    opta_1h_a = (away_form.ht_xg_for_avg * 0.56) + (home_form.ht_xg_against_avg * 0.44)
    lam_1h_home = (0.50 * (lambda_home * 0.44)) + (0.50 * opta_1h_h)
    lam_1h_away = (0.50 * (lambda_away * 0.44)) + (0.50 * opta_1h_a)
    lam_1h_total = round(lam_1h_home + lam_1h_away, 2)
    prob_1h_o0_5 = round(_poisson_prob_over(0.5, lam_1h_total) * 100, 1)
    prob_1h_o1_5 = round(_poisson_prob_over(1.5, lam_1h_total) * 100, 1)

    p1h_h, p1h_d, p1h_a = 0.0, 0.0, 0.0
    for h1 in range(5):
        for a1 in range(5):
            p1 = _poisson_pmf(h1, lam_1h_home) * _poisson_pmf(a1, lam_1h_away)
            if h1 > a1:
                p1h_h += p1
            elif h1 == a1:
                p1h_d += p1
            else:
                p1h_a += p1
    sum_1h = max(0.001, p1h_h + p1h_d + p1h_a)
    prob_1h_home = round((p1h_h / sum_1h) * 100, 1)
    prob_1h_draw = round((p1h_d / sum_1h) * 100, 1)
    prob_1h_away = round((p1h_a / sum_1h) * 100, 1)

    sorted_scores = sorted(matrix.items(), key=lambda x: x[1], reverse=True)
    most_likely_score = f"{sorted_scores[0][0][0]}-{sorted_scores[0][0][1]}"
    top_scores = [
        {"score": f"{h}-{a}", "prob": round(prob * 100, 1)}
        for (h, a), prob in sorted_scores[:6]
    ]

    heatmap_grid: List[List[float]] = []
    for h in range(6):
        row = [round(matrix.get((h, a), 0.0) * 100, 2) for a in range(6)]
        heatmap_grid.append(row)

    # Tarjetas Amarillas (ajustadas con Árbitro, Fatiga y Fricción Histórica H2H)
    raw_cards_home = (home_form.yellow_cards_for_avg * 0.60) + (away_form.yellow_cards_against_avg * 0.40)
    raw_cards_away = (away_form.yellow_cards_for_avg * 0.60) + (home_form.yellow_cards_against_avg * 0.40)
    if home_form.fatigue_index >= 75.0:
        raw_cards_home *= 1.08
    if away_form.fatigue_index >= 75.0:
        raw_cards_away *= 1.08
    team_cards_total = raw_cards_home + raw_cards_away

    if match.h2h and match.h2h.h2h_cards_avg is not None and match.h2h.recent_matches_analyzed >= 2:
        team_cards_total = (0.78 * team_cards_total) + (0.22 * match.h2h.h2h_cards_avg)

    ref_name, ref_avg, ref_strict = None, None, None
    if match.referee and match.referee.games >= 5:
        ref_name = match.referee.name
        ref_avg = match.referee.yellow_per_game
        ref_strict = match.referee.strictness_label
        blended_total = (0.55 * team_cards_total) + (0.45 * ref_avg)
        scale_c = blended_total / max(0.5, raw_cards_home + raw_cards_away)
        exp_cards_home = round(raw_cards_home * scale_c, 2)
        exp_cards_away = round(raw_cards_away * scale_c, 2)
    else:
        scale_c = team_cards_total / max(0.5, raw_cards_home + raw_cards_away)
        exp_cards_home = round(raw_cards_home * scale_c, 2)
        exp_cards_away = round(raw_cards_away * scale_c, 2)

    exp_cards_total = round(exp_cards_home + exp_cards_away, 2)
    prob_o3_5_cards = round(_poisson_prob_over(3.5, exp_cards_total) * 100, 1)
    prob_o4_5_cards = round(_poisson_prob_over(4.5, exp_cards_total) * 100, 1)
    prob_o5_5_cards = round(_poisson_prob_over(5.5, exp_cards_total) * 100, 1)
    exp_fouls_total = round(home_form.fouls_for_avg + away_form.fouls_for_avg, 1)

    pos_sum = max(1.0, home_form.possession_avg + away_form.possession_avg)
    exp_pos_home = round((home_form.possession_avg / pos_sum) * 100.0, 1)
    exp_pos_away = round(100.0 - exp_pos_home, 1)

    edge_home, edge_draw, edge_away = None, None, None
    if match.odds and match.odds.implied_prob_home:
        edge_home = round((p_home - match.odds.implied_prob_home) * 100, 1)
        edge_draw = round((p_draw - (match.odds.implied_prob_draw or 0)) * 100, 1)
        edge_away = round((p_away - (match.odds.implied_prob_away or 0)) * 100, 1)

    h_short = match.home_team.short_name or match.home_team.name
    a_short = match.away_team.short_name or match.away_team.name

    recs: List[str] = []
    if p_home >= 0.55:
        recs.append(f"Victoria {h_short} ({p_home*100:.0f}%)")
    elif p_away >= 0.52:
        recs.append(f"Victoria {a_short} ({p_away*100:.0f}%)")
    elif p_home + p_draw >= 0.74:
        recs.append(f"1X ({h_short} o Empate: {(p_home+p_draw)*100:.0f}%)")
    elif p_away + p_draw >= 0.74:
        recs.append(f"X2 ({a_short} o Empate: {(p_away+p_draw)*100:.0f}%)")

    if p_over_2_5 >= 0.58:
        recs.append(f"Over 2.5 Goles ({p_over_2_5*100:.0f}%)")
    elif (1.0 - p_over_2_5) >= 0.60:
        recs.append(f"Under 2.5 Goles ({(1.0-p_over_2_5)*100:.0f}%)")

    if p_btts >= 0.58:
        recs.append(f"Ambos Anotan: Sí ({p_btts*100:.0f}%)")

    if prob_a_gk_3p >= 60.0:
        recs.append(f"🧤 {gk_away_info['name']} 3+ Atajadas ({prob_a_gk_3p:.0f}%)")
    elif prob_h_gk_3p >= 60.0:
        recs.append(f"🧤 {gk_home_info['name']} 3+ Atajadas ({prob_h_gk_3p:.0f}%)")

    # Construcción de las 10 tarjetas de mercado + Auditoría Automática (Backtesting ✅/❌)
    is_finished = match.status_type == "finished" and match.home_score is not None and match.away_score is not None
    hs = match.home_score or 0
    as_ = match.away_score or 0
    total_g = hs + as_

    if p_home >= 0.53:
        res_pick = f"Victoria {h_short}"
        res_prob = round(p_home * 100, 1)
        res_hit = (hs > as_) if is_finished else None
    elif p_away >= 0.52:
        res_pick = f"Victoria {a_short}"
        res_prob = round(p_away * 100, 1)
        res_hit = (as_ > hs) if is_finished else None
    elif (p_home + p_draw) >= (p_away + p_draw):
        res_pick = f"Doble Oportunidad 1X ({h_short}/Empate)"
        res_prob = round((p_home + p_draw) * 100, 1)
        res_hit = (hs >= as_) if is_finished else None
    else:
        res_pick = f"Doble Oportunidad X2 ({a_short}/Empate)"
        res_prob = round((p_away + p_draw) * 100, 1)
        res_hit = (as_ >= hs) if is_finished else None

    if p_over_2_5 >= 0.55:
        goals_pick = "Más de 2.5 Goles (Over 2.5)"
        goals_prob = round(p_over_2_5 * 100, 1)
        goals_hit = (total_g > 2) if is_finished else None
    elif (1.0 - p_over_2_5) >= 0.56:
        goals_pick = "Menos de 2.5 Goles (Under 2.5)"
        goals_prob = round((1.0 - p_over_2_5) * 100, 1)
        goals_hit = (total_g < 3) if is_finished else None
    else:
        goals_pick = "Más de 1.5 Goles (Over 1.5)"
        goals_prob = round(p_over_1_5 * 100, 1)
        goals_hit = (total_g > 1) if is_finished else None

    btts_pick = "Ambos Anotan: SÍ" if p_btts >= 0.52 else "Ambos Anotan: NO"
    btts_prob = round((p_btts if p_btts >= 0.52 else (1.0 - p_btts)) * 100, 1)
    btts_hit = ((hs >= 1 and as_ >= 1) if p_btts >= 0.52 else not (hs >= 1 and as_ >= 1)) if is_finished else None

    ht_goals_real = (
        (match.home_score_ht + match.away_score_ht)
        if (match.home_score_ht is not None and match.away_score_ht is not None)
        else None
    )
    if prob_1h_o0_5 >= 62.0:
        ht_pick = "1er Tiempo: Más de 0.5 Goles (+0.5 1H)"
        ht_prob = prob_1h_o0_5
        ht_hit = (ht_goals_real >= 1) if (is_finished and ht_goals_real is not None) else None
    else:
        ht_pick = "1er Tiempo: Menos de 1.5 Goles (-1.5 1H)"
        ht_prob = round(100.0 - prob_1h_o1_5, 1)
        ht_hit = (ht_goals_real < 2) if (is_finished and ht_goals_real is not None) else None

    real_sot = (
        (match.stats.shots_on_target_home + match.stats.shots_on_target_away)
        if (match.stats and match.stats.shots_on_target_home is not None and match.stats.shots_on_target_away is not None)
        else None
    )
    if prob_o8_5_sot >= 55.0:
        sot_pick = "Más de 8.5 Remates al Arco"
        sot_prob = prob_o8_5_sot
        sot_hit = (real_sot > 8) if (is_finished and real_sot is not None) else None
    elif prob_o7_5_sot >= 55.0:
        sot_pick = "Más de 7.5 Remates al Arco"
        sot_prob = prob_o7_5_sot
        sot_hit = (real_sot > 7) if (is_finished and real_sot is not None) else None
    else:
        sot_pick = "Menos de 8.5 Remates al Arco"
        sot_prob = round(100.0 - prob_o8_5_sot, 1)
        sot_hit = (real_sot <= 8) if (is_finished and real_sot is not None) else None

    # 6ª Tarjeta: Atajadas del Portero (GK Saves) con Verificación Real
    real_saves_h = (
        match.stats.saves_home
        if (match.stats and match.stats.saves_home is not None)
        else gk_home_info["match_saves"]
    )
    real_saves_a = (
        match.stats.saves_away
        if (match.stats and match.stats.saves_away is not None)
        else gk_away_info["match_saves"]
    )
    real_saves_tot = (
        (real_saves_h + real_saves_a)
        if (real_saves_h is not None and real_saves_a is not None)
        else None
    )

    # Elegir la selección más sólida de Atajadas del Portero
    if prob_a_gk_3p >= prob_h_gk_3p and prob_a_gk_3p >= 56.0:
        gk_pick = f"🧤 {gk_away_info['name']} ({a_short}): 3+ Atajadas (+2.5)"
        gk_prob = prob_a_gk_3p
        gk_hit = (real_saves_a >= 3) if (is_finished and real_saves_a is not None) else None
        gk_actual = f"Atajadas {gk_away_info['name']}: {real_saves_a} (Total P: {real_saves_tot})" if (is_finished and real_saves_a is not None) else None
    elif prob_h_gk_3p >= 56.0:
        gk_pick = f"🧤 {gk_home_info['name']} ({h_short}): 3+ Atajadas (+2.5)"
        gk_prob = prob_h_gk_3p
        gk_hit = (real_saves_h >= 3) if (is_finished and real_saves_h is not None) else None
        gk_actual = f"Atajadas {gk_home_info['name']}: {real_saves_h} (Total P: {real_saves_tot})" if (is_finished and real_saves_h is not None) else None
    elif prob_o5_5_saves >= 55.0:
        gk_pick = "🧤 Más de 5.5 Atajadas Totales en el Partido"
        gk_prob = prob_o5_5_saves
        gk_hit = (real_saves_tot >= 6) if (is_finished and real_saves_tot is not None) else None
        gk_actual = f"Atajadas Reales: {real_saves_tot} ({h_short}: {real_saves_h} | {a_short}: {real_saves_a})" if (is_finished and real_saves_tot is not None) else None
    elif prob_a_gk_2p >= prob_h_gk_2p:
        gk_pick = f"🧤 {gk_away_info['name']} ({a_short}): 2+ Atajadas (+1.5)"
        gk_prob = prob_a_gk_2p
        gk_hit = (real_saves_a >= 2) if (is_finished and real_saves_a is not None) else None
        gk_actual = f"Atajadas {gk_away_info['name']}: {real_saves_a} (Total P: {real_saves_tot})" if (is_finished and real_saves_a is not None) else None
    else:
        gk_pick = f"🧤 {gk_home_info['name']} ({h_short}): 2+ Atajadas (+1.5)"
        gk_prob = prob_h_gk_2p
        gk_hit = (real_saves_h >= 2) if (is_finished and real_saves_h is not None) else None
        gk_actual = f"Atajadas {gk_home_info['name']}: {real_saves_h} (Total P: {real_saves_tot})" if (is_finished and real_saves_h is not None) else None

    real_shots = (
        (match.stats.total_shots_home + match.stats.total_shots_away)
        if (match.stats and match.stats.total_shots_home is not None and match.stats.total_shots_away is not None)
        else None
    )
    if prob_o24_5_shots >= 54.0:
        sh_pick = "Más de 24.5 Remates Totales"
        sh_prob = prob_o24_5_shots
        sh_hit = (real_shots > 24) if (is_finished and real_shots is not None) else None
    elif prob_o22_5_shots >= 55.0:
        sh_pick = "Más de 22.5 Remates Totales"
        sh_prob = prob_o22_5_shots
        sh_hit = (real_shots > 22) if (is_finished and real_shots is not None) else None
    else:
        sh_pick = "Menos de 24.5 Remates Totales"
        sh_prob = round(100.0 - prob_o24_5_shots, 1)
        sh_hit = (real_shots <= 24) if (is_finished and real_shots is not None) else None

    real_corners = (
        (match.stats.corner_kicks_home + match.stats.corner_kicks_away)
        if (match.stats and match.stats.corner_kicks_home is not None and match.stats.corner_kicks_away is not None)
        else None
    )
    if prob_o9_5_corners >= 55.0:
        crn_pick = "Más de 9.5 Córners"
        crn_prob = prob_o9_5_corners
        crn_hit = (real_corners > 9) if (is_finished and real_corners is not None) else None
    elif prob_o8_5_corners >= 55.0:
        crn_pick = "Más de 8.5 Córners"
        crn_prob = prob_o8_5_corners
        crn_hit = (real_corners > 8) if (is_finished and real_corners is not None) else None
    else:
        crn_pick = "Menos de 9.5 Córners"
        crn_prob = round(100.0 - prob_o9_5_corners, 1)
        crn_hit = (real_corners <= 9) if (is_finished and real_corners is not None) else None

    real_cards = (
        (match.stats.yellow_cards_home + match.stats.yellow_cards_away)
        if (match.stats and match.stats.yellow_cards_home is not None and match.stats.yellow_cards_away is not None)
        else None
    )
    if prob_o4_5_cards >= 54.0:
        crd_pick = "Más de 4.5 Tarjetas"
        crd_prob = prob_o4_5_cards
        crd_hit = (real_cards > 4) if (is_finished and real_cards is not None) else None
    elif prob_o3_5_cards >= 55.0:
        crd_pick = "Más de 3.5 Tarjetas"
        crd_prob = prob_o3_5_cards
        crd_hit = (real_cards > 3) if (is_finished and real_cards is not None) else None
    else:
        crd_pick = "Menos de 4.5 Tarjetas"
        crd_prob = round(100.0 - prob_o4_5_cards, 1)
        crd_hit = (real_cards <= 4) if (is_finished and real_cards is not None) else None

    if mc["home_o1_5"] >= 58.0:
        mc_pick = f"{h_short} Más de 1.5 Goles (+1.5 Equ.)"
        mc_prob = mc["home_o1_5"]
        mc_hit = (hs >= 2) if is_finished else None
    elif mc["away_o1_5"] >= 55.0:
        mc_pick = f"{a_short} Más de 1.5 Goles (+1.5 Equ.)"
        mc_prob = mc["away_o1_5"]
        mc_hit = (as_ >= 2) if is_finished else None
    elif mc["dnb_home"] >= 62.0:
        mc_pick = f"Apuesta Sin Empate: {h_short} (DNB 1)"
        mc_prob = mc["dnb_home"]
        mc_hit = (hs > as_) if (is_finished and hs != as_) else None
    elif mc["dnb_away"] >= 60.0:
        mc_pick = f"Apuesta Sin Empate: {a_short} (DNB 2)"
        mc_prob = mc["dnb_away"]
        mc_hit = (as_ > hs) if (is_finished and hs != as_) else None
    else:
        mc_pick = f"Hándicap Asiático {a_short} +1.5"
        mc_prob = mc["ah_away_p1_5"]
        mc_hit = ((as_ - hs) > -2) if is_finished else None

    ref_detail_str = f" | Árbitro: {ref_name} ({ref_avg:.2f} tarj/P)" if ref_name and ref_avg else ""

    all_picks = [
        {
            "category": "Resultado (Ensamble Opta + H2H + ML)",
            "pick": res_pick,
            "prob": res_prob,
            "detail": f"Dixon-Coles: {stat_p_home*100:.0f}/{stat_p_draw*100:.0f}/{stat_p_away*100:.0f}% | ML: {ml_res['ml_prob_home']:.0f}/{ml_res['ml_prob_draw']:.0f}/{ml_res['ml_prob_away']:.0f}%",
            "hit": res_hit,
            "actual": f"Resultado Final: {hs}-{as_}" if is_finished else None,
        },
        {
            "category": "Mercado de Goles (Tiempo Completo)",
            "pick": goals_pick,
            "prob": goals_prob,
            "detail": f"Over 1.5: {p_over_1_5*100:.0f}% | Over 2.5: {p_over_2_5*100:.0f}% | Over 3.5: {p_over_3_5*100:.0f}%",
            "hit": goals_hit,
            "actual": f"Total Goles: {total_g} ({hs}-{as_})" if is_finished else None,
        },
        {
            "category": "Ambos Equipos Anotan (BTTS)",
            "pick": btts_pick,
            "prob": btts_prob,
            "detail": f"xG Local: {lambda_home:.2f} | xG Visitante: {lambda_away:.2f}",
            "hit": btts_hit,
            "actual": f"Marcador: {hs}-{as_}" if is_finished else None,
        },
        {
            "category": "Goles en el 1er Tiempo (Opta 1ST)",
            "pick": ht_pick,
            "prob": ht_prob,
            "detail": f"xG 1H Esperado: {lam_1h_total:.2f} | +0.5 1H: {prob_1h_o0_5:.0f}% | +1.5 1H: {prob_1h_o1_5:.0f}%",
            "hit": ht_hit,
            "actual": f"Descanso 1H: {match.home_score_ht}-{match.away_score_ht}" if (is_finished and ht_goals_real is not None) else None,
        },
        {
            "category": "Remates al Arco (Shots on Target)",
            "pick": sot_pick,
            "prob": sot_prob,
            "detail": f"Esperados: {exp_sot_total:.1f} al arco ({h_short}: {exp_sot_home:.1f} | {a_short}: {exp_sot_away:.1f})",
            "hit": sot_hit,
            "actual": f"Remates al Arco Reales: {real_sot}" if (is_finished and real_sot is not None) else None,
        },
        {
            "category": "Atajadas del Portero (GK Saves)",
            "pick": gk_pick,
            "prob": gk_prob,
            "detail": f"Proy: {gk_home_info['name']} {exp_saves_home_gk:.1f} ataj. (vs {exp_sot_away:.1f} SoT) | {gk_away_info['name']} {exp_saves_away_gk:.1f} ataj. (vs {exp_sot_home:.1f} SoT)",
            "hit": gk_hit,
            "actual": gk_actual,
        },
        {
            "category": "Remates Totales del Partido",
            "pick": sh_pick,
            "prob": sh_prob,
            "detail": f"Esperados: {exp_shots_total:.1f} remates ({h_short}: {exp_shots_home:.1f} | {a_short}: {exp_shots_away:.1f})",
            "hit": sh_hit,
            "actual": f"Remates Totales Reales: {real_shots}" if (is_finished and real_shots is not None) else None,
        },
        {
            "category": "Córners (Saques de Esquina)",
            "pick": crn_pick,
            "prob": crn_prob,
            "detail": f"Esperados: {exp_corners_total:.1f} FT ({exp_1h_corners:.1f} en 1H)",
            "hit": crn_hit,
            "actual": f"Córners Reales: {real_corners}" if (is_finished and real_corners is not None) else None,
        },
        {
            "category": "Tarjetas Amarillas & Árbitro",
            "pick": crd_pick,
            "prob": crd_prob,
            "detail": f"Esperadas: {exp_cards_total:.1f} tarjetas{ref_detail_str}",
            "hit": crd_hit,
            "actual": f"Tarjetas Reales: {real_cards}" if (is_finished and real_cards is not None) else None,
        },
        {
            "category": "Hándicap Asiático & Equipos (10k Sims)",
            "pick": mc_pick,
            "prob": mc_prob,
            "detail": f"DNB 1: {mc['dnb_home']:.0f}% | AH -1.5 Loc: {mc['ah_home_m1_5']:.0f}% | Loc +1.5G: {mc['home_o1_5']:.0f}%",
            "hit": mc_hit,
            "actual": f"Marcador: {hs}-{as_}" if is_finished else None,
        },
    ]

    # =========================================================================
    # NUEVO v4: DESGLOSE DE PROBABILIDAD POR TRAMOS DE TIEMPO (MINUTOS)
    # =========================================================================
    # Calibrado según distribución típica de goles:
    # 0-15m (13%), 16-30m (16%), 31-45m (18%), 46-60m (17%), 61-75m (17%), 76-90m+ (19%)
    exp_g_match = lambda_home + lambda_away
    lam_0_15 = exp_g_match * 0.135
    lam_16_45 = exp_g_match * 0.335
    lam_46_75 = exp_g_match * 0.330
    lam_76_90 = exp_g_match * 0.200

    prob_early_15 = round(_poisson_prob_over(0.5, lam_0_15) * 100.0, 1)
    prob_mid_45 = round(_poisson_prob_over(0.5, lam_16_45) * 100.0, 1)
    prob_late_90 = round(_poisson_prob_over(0.5, lam_76_90) * 100.0, 1)

    time_windows_data = [
        {"Tramo": "Min 01 – 15 (Arranque)", "xG Esperado": round(lam_0_15, 2), "Prob. 1+ Gol (%)": prob_early_15, "Intensidad": "Media 🟡" if prob_early_15 >= 28.0 else "Baja 🟢"},
        {"Tramo": "Min 16 – 45 (Final 1H)", "xG Esperado": round(lam_16_45, 2), "Prob. 1+ Gol (%)": prob_mid_45, "Intensidad": "Alta 🔥" if prob_mid_45 >= 45.0 else "Media 🟡"},
        {"Tramo": "Min 46 – 75 (Reanudación)", "xG Esperado": round(lam_46_75, 2), "Prob. 1+ Gol (%)": round(_poisson_prob_over(0.5, lam_46_75) * 100.0, 1), "Intensidad": "Alta 🔥"},
        {"Tramo": "Min 76 – 90+ (Cierre / Agónico)", "xG Esperado": round(lam_76_90, 2), "Prob. 1+ Gol (%)": prob_late_90, "Intensidad": "Crítica ⚡" if prob_late_90 >= 40.0 else "Media 🟡"},
    ]
    half_active_label = "2do Tiempo (Min 46–90+) 🔥" if (lam_46_75 + lam_76_90) > (lam_0_15 + lam_16_45) else "1er Tiempo (Min 01–45) ⚡"

    # =========================================================================
    # NUEVO v4: CALIBRACIÓN BRIER SCORE & ROI SIMULADO (PARTIDOS FINALIZADOS)
    # =========================================================================
    brier_1x2_val = None
    brier_goals_val = None
    brier_label = "Calibración Óptima (Modelo Fiable)"
    sim_roi = None

    if is_finished:
        # Vector One-Hot del resultado real [Home, Draw, Away]
        y_real_1x2 = [1.0 if hs > as_ else 0.0, 1.0 if hs == as_ else 0.0, 1.0 if hs < as_ else 0.0]
        p_pred_1x2 = [p_home, p_draw, p_away]
        # Brier Score = sum((p - y)^2) / 3
        brier_1x2_val = round(sum((p - y) ** 2 for p, y in zip(p_pred_1x2, y_real_1x2)) / 3.0, 4)

        # Brier para Over 2.5
        y_real_o25 = 1.0 if total_g > 2 else 0.0
        brier_goals_val = round((p_over_2_5 - y_real_o25) ** 2, 4)

        if brier_1x2_val <= 0.18:
            brier_label = "Calibración Sobresaliente 🟢 (Brier Score Bajo / Alta Precisión)"
        elif brier_1x2_val <= 0.28:
            brier_label = "Calibración Adecuada 🟡 (Desempeño Estable)"
        else:
            brier_label = "Desvío Estadístico Inesperado 🔴 (Partido con anomalías)"

        # Simulación de ROI sobre los picks del partido
        # Cada pick acertado suma (cuota - 1), cada fallido resta 1 unidad
        roi_units_won = 0.0
        roi_bets_count = 0
        for item in all_picks:
            if item["hit"] is not None:
                prob_item = item["prob"]
                odd_item = 100.0 / max(1.0, prob_item)
                roi_bets_count += 1
                if item["hit"] is True:
                    roi_units_won += (odd_item - 1.0)
                else:
                    roi_units_won -= 1.0
        if roi_bets_count > 0:
            sim_roi = round((roi_units_won / roi_bets_count) * 100.0, 1)

    v_hits = sum(1 for item in all_picks if item["hit"] is True)
    v_tot = sum(1 for item in all_picks if item["hit"] is not None)

    return MatchPrediction(
        event_id=match.event_id,
        home_team=match.home_team.name,
        away_team=match.away_team.name,
        elo_home=home_form.elo_rating,
        elo_away=away_form.elo_rating,
        elo_diff=round(elo_diff, 1),
        injury_penalty_home_pct=pen_home_pct,
        injury_penalty_away_pct=pen_away_pct,
        missing_key_home=key_missing_home,
        missing_key_away=key_missing_away,
        rest_days_home=home_form.rest_days,
        rest_days_away=away_form.rest_days,
        fatigue_home_label=home_form.fatigue_label,
        fatigue_away_label=away_form.fatigue_label,
        referee_name=ref_name,
        referee_cards_avg=ref_avg,
        referee_strictness=ref_strict,
        expected_goals_home=round(lambda_home, 2),
        expected_goals_away=round(lambda_away, 2),
        prob_home_win=round(p_home * 100, 1),
        prob_draw=round(p_draw * 100, 1),
        prob_away_win=round(p_away * 100, 1),
        stat_prob_home=round(stat_p_home * 100, 1),
        stat_prob_draw=round(stat_p_draw * 100, 1),
        stat_prob_away=round(stat_p_away * 100, 1),
        ml_prob_home=ml_res["ml_prob_home"],
        ml_prob_draw=ml_res["ml_prob_draw"],
        ml_prob_away=ml_res["ml_prob_away"],
        ml_db_samples=ml_res["db_samples_used"],
        ml_feature_importances=ml_res["feature_importances"],
        prob_1x=round((p_home + p_draw) * 100, 1),
        prob_x2=round((p_away + p_draw) * 100, 1),
        prob_12=round((p_home + p_away) * 100, 1),
        prob_dnb_home=mc["dnb_home"],
        prob_dnb_away=mc["dnb_away"],
        prob_ah_home_minus_1_5=mc["ah_home_m1_5"],
        prob_ah_home_minus_1_0_win_or_push=mc["ah_home_m1_0_win_or_push"],
        prob_ah_away_minus_1_5=mc["ah_away_m1_5"],
        prob_ah_away_plus_1_5=mc["ah_away_p1_5"],
        prob_home_over_0_5=mc["home_o0_5"],
        prob_home_over_1_5=mc["home_o1_5"],
        prob_home_over_2_5=mc["home_o2_5"],
        prob_away_over_0_5=mc["away_o0_5"],
        prob_away_over_1_5=mc["away_o1_5"],
        prob_home_clean_sheet=mc["home_cs"],
        prob_away_clean_sheet=mc["away_cs"],
        prob_over_1_5=round(p_over_1_5 * 100, 1),
        prob_over_2_5=round(p_over_2_5 * 100, 1),
        prob_over_3_5=round(p_over_3_5 * 100, 1),
        prob_under_2_5=round((1.0 - p_over_2_5) * 100, 1),
        prob_btts=round(p_btts * 100, 1),
        prob_btts_no=round((1.0 - p_btts) * 100, 1),
        expected_1h_goals_total=lam_1h_total,
        prob_1h_over_0_5=prob_1h_o0_5,
        prob_1h_over_1_5=prob_1h_o1_5,
        prob_1h_home_win=prob_1h_home,
        prob_1h_draw=prob_1h_draw,
        prob_1h_away_win=prob_1h_away,
        expected_1h_corners=exp_1h_corners,
        prob_1h_over_4_5_corners=prob_1h_o4_5_corners,
        expected_shots_home=exp_shots_home,
        expected_shots_away=exp_shots_away,
        expected_shots_total=exp_shots_total,
        prob_over_22_5_shots=prob_o22_5_shots,
        prob_over_24_5_shots=prob_o24_5_shots,
        prob_over_26_5_shots=prob_o26_5_shots,
        expected_sot_home=exp_sot_home,
        expected_sot_away=exp_sot_away,
        expected_sot_total=exp_sot_total,
        prob_over_7_5_sot=prob_o7_5_sot,
        prob_over_8_5_sot=prob_o8_5_sot,
        prob_over_9_5_sot=prob_o9_5_sot,
        gk_home_name=gk_home_info["name"],
        gk_away_name=gk_away_info["name"],
        gk_home_hist_saves_avg=gk_home_info["saves_avg"],
        gk_away_hist_saves_avg=gk_away_info["saves_avg"],
        gk_home_save_pct=gk_home_info["save_pct"],
        gk_away_save_pct=gk_away_info["save_pct"],
        gk_home_goals_prevented=gk_home_info["goals_prevented"],
        gk_away_goals_prevented=gk_away_info["goals_prevented"],
        expected_saves_home_gk=exp_saves_home_gk,
        expected_saves_away_gk=exp_saves_away_gk,
        expected_saves_total=exp_saves_total,
        prob_home_gk_2plus_saves=prob_h_gk_2p,
        prob_home_gk_3plus_saves=prob_h_gk_3p,
        prob_home_gk_4plus_saves=prob_h_gk_4p,
        prob_home_gk_5plus_saves=prob_h_gk_5p,
        prob_away_gk_2plus_saves=prob_a_gk_2p,
        prob_away_gk_3plus_saves=prob_a_gk_3p,
        prob_away_gk_4plus_saves=prob_a_gk_4p,
        prob_away_gk_5plus_saves=prob_a_gk_5p,
        prob_over_4_5_total_saves=prob_o4_5_saves,
        prob_over_5_5_total_saves=prob_o5_5_saves,
        prob_over_6_5_total_saves=prob_o6_5_saves,
        expected_corners_home=exp_corners_home,
        expected_corners_away=exp_corners_away,
        expected_corners_total=exp_corners_total,
        prob_over_8_5_corners=prob_o8_5_corners,
        prob_over_9_5_corners=prob_o9_5_corners,
        prob_over_10_5_corners=prob_o10_5_corners,
        expected_cards_home=exp_cards_home,
        expected_cards_away=exp_cards_away,
        expected_cards_total=exp_cards_total,
        prob_over_3_5_cards=prob_o3_5_cards,
        prob_over_4_5_cards=prob_o4_5_cards,
        prob_over_5_5_cards=prob_o5_5_cards,
        expected_fouls_total=exp_fouls_total,
        expected_possession_home=exp_pos_home,
        expected_possession_away=exp_pos_away,
        opta_danger_home=opta_danger_home,
        opta_danger_away=opta_danger_away,
        most_likely_score=most_likely_score,
        top_scores=top_scores,
        score_matrix=heatmap_grid,
        value_edge_home=edge_home,
        value_edge_draw=edge_draw,
        value_edge_away=edge_away,
        recommended_pick=" | ".join(recs) if recs else "Partido parejo / Sin ventaja clara",
        all_market_picks=all_picks,
        verified_hits=v_hits,
        verified_total=v_tot,
        urgency_home_index=home_form.urgency_index,
        urgency_away_index=away_form.urgency_index,
        urgency_home_label=home_form.urgency_label,
        urgency_away_label=away_form.urgency_label,
        urgency_clash_summary=urgency_summary,
        shot_quality_home=sq_h,
        shot_quality_away=sq_a,
        shot_quality_edge=sq_edge,
        tactical_style_home=style_h,
        tactical_style_away=style_a,
        tactical_clash_narrative=tactical_narrative,
        prob_goal_early_1_15=prob_early_15,
        prob_goal_mid_15_45=prob_mid_45,
        prob_goal_late_75_90=prob_late_90,
        prob_most_active_half=half_active_label,
        time_windows=time_windows_data,
        brier_score_1x2=brier_1x2_val,
        brier_score_goals=brier_goals_val,
        brier_calibration_label=brier_label,
        simulated_roi_pct=sim_roi,
        ppda_home=ppda_h,
        ppda_away=ppda_a,
        defense_synergy_home_pct=syn_home_pct,
        defense_synergy_away_pct=syn_away_pct,
        defense_status_home=syn_home_lbl,
        defense_status_away=syn_away_lbl,
        league_rho_calibration=league_rho,
        league_profile_name=league_profile,
    )



def predict_live_in_play(
    home_score: int,
    away_score: int,
    current_minute: int,
    lambda_home_full: float = 1.35,
    lambda_away_full: float = 1.10,
    momentum_factor: float = 0.0,
) -> Dict[str, float]:
    """
    Calcula probabilidades EN VIVO (In-Play) mediante Poisson con decaimiento temporal
    y ajuste dinámico de presión de ataque (Momentum).
    """
    minute_clamped = max(1, min(92, current_minute))
    rem_ratio = max(0.01, (90.0 - minute_clamped) / 90.0)

    # Ajuste suave por momentum reciente (+/- hasta 25%)
    mom_home_boost = max(0.6, min(1.4, 1.0 + (momentum_factor * 0.25)))
    mom_away_boost = max(0.6, min(1.4, 1.0 - (momentum_factor * 0.25)))

    rem_lam_home = max(0.01, lambda_home_full * rem_ratio * mom_home_boost)
    rem_lam_away = max(0.01, lambda_away_full * rem_ratio * mom_away_boost)

    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0
    p_over_1_5 = 0.0
    p_over_2_5 = 0.0
    p_over_3_5 = 0.0
    p_btts_yes = 0.0

    max_rem_goals = 7
    for g_h in range(max_rem_goals):
        p_gh = _poisson_pmf(g_h, rem_lam_home)
        for g_a in range(max_rem_goals):
            p_ga = _poisson_pmf(g_a, rem_lam_away)
            p_joint = p_gh * p_ga

            final_h = home_score + g_h
            final_a = away_score + g_a
            tot_g = final_h + final_a

            if final_h > final_a:
                p_home_win += p_joint
            elif final_h == final_a:
                p_draw += p_joint
            else:
                p_away_win += p_joint

            if tot_g > 1:
                p_over_1_5 += p_joint
            if tot_g > 2:
                p_over_2_5 += p_joint
            if tot_g > 3:
                p_over_3_5 += p_joint

            if final_h > 0 and final_a > 0:
                p_btts_yes += p_joint

    tot_1x2 = p_home_win + p_draw + p_away_win
    if tot_1x2 > 0:
        p_home_win /= tot_1x2
        p_draw /= tot_1x2
        p_away_win /= tot_1x2

    return {
        "prob_home_win": round(p_home_win * 100.0, 1),
        "prob_draw": round(p_draw * 100.0, 1),
        "prob_away_win": round(p_away_win * 100.0, 1),
        "prob_over_1_5": round(min(99.0, max(1.0, p_over_1_5 * 100.0)), 1),
        "prob_over_2_5": round(min(99.0, max(1.0, p_over_2_5 * 100.0)), 1),
        "prob_over_3_5": round(min(99.0, max(1.0, p_over_3_5 * 100.0)), 1),
        "prob_btts_yes": round(min(99.0, max(1.0, p_btts_yes * 100.0)), 1),
        "rem_lambda_home": round(rem_lam_home, 2),
        "rem_lambda_away": round(rem_lam_away, 2),
        "time_remaining_pct": round(rem_ratio * 100.0, 1),
    }
