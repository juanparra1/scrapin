"""Motor de asignación de capital, gestión de riesgo cuantitativo y optimización de portafolio.

Biblioteca de lógica pura (SIN dependencias de Streamlit) para cálculo de:
- Filtros de varianza y disyuntores (Circuit Breakers).
- Arbitraje seguro (Surebets).
- Optimización de valor esperado (+EV) con Criterio de Kelly Fraccional escalado.
- Coberturas Dutching con balanceo de retorno sobre selecciones excluyentes.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# CONSTANTES Y CIRCUIT BREAKERS CUANTITATIVOS
# ============================================================================

HARD_CAP_PER_BET: float = 0.04            # Ninguna apuesta puede exceder el 4.0% de la banca total
DEFAULT_MAX_DAILY_EXPOSURE: float = 0.15  # Exposición máxima diaria acumulada (15.0% por defecto)
MIN_VARIANCE_ODDS_THRESHOLD: float = 3.50 # Cuotas superiores a 3.50 requieren filtro estricto de varianza
HIGH_ODDS_MIN_EV: float = 0.12            # Para cuotas > 3.50, el EV neto debe ser al menos +12.0%
MIN_EV_THRESHOLD: float = 0.02            # Umbral mínimo de EV (+2.0%) para calificar como Value Bet
MIN_SUREBET_ROI: float = 0.015            # Ganancia neta mínima del 1.5% para considerarse Surebet válida


@dataclass
class BetOpportunity:
    """Representa una oportunidad de apuesta individual evaluada."""
    event_id: int
    event_name: str
    tournament: str
    match_date: str
    strategy: str  # VALUE_BET, SUREBET, DUTCHING
    market_code: str
    pick_name: str
    odds_taken: float
    fair_odds: float
    model_probability: float
    ev_net: float


@dataclass
class AllocatedBet(BetOpportunity):
    """Representa una apuesta con asignación de capital exacto calculada."""
    stake_amount: float = 0.0
    stake_percentage: float = 0.0
    potential_payout: float = 0.0


@dataclass
class SurebetLeg:
    """Una rama de una apuesta de arbitraje."""
    market_code: str
    pick_name: str
    odds: float
    stake_amount: float
    payout: float


@dataclass
class SurebetOpportunity:
    """Oportunidad de arbitraje libre de riesgo (Surebet)."""
    event_id: int
    event_name: str
    tournament: str
    match_date: str
    market_type: str  # "1X2", "OU_2.5", "BTTS"
    legs: List[SurebetLeg]
    total_stake: float
    guaranteed_payout: float
    guaranteed_profit: float
    roi_pct: float
    inverse_sum: float


@dataclass
class DutchingLeg:
    """Una selección de una estrategia de Dutching."""
    market_code: str
    pick_name: str
    odds: float
    model_prob: float
    stake_amount: float
    payout: float


@dataclass
class DutchingOpportunity:
    """Estrategia de Dutching sobre resultados excluyentes."""
    event_id: int
    event_name: str
    tournament: str
    match_date: str
    coverage_name: str
    legs: List[DutchingLeg]
    total_stake: float
    target_payout: float
    net_profit_if_hit: float
    combined_probability: float
    roi_if_hit_pct: float


# ============================================================================
# GESTOR DE RIESGO: CIRCUIT BREAKERS Y FILTROS DE VARIANZA
# ============================================================================


def passes_variance_filter(odds: float, ev_net: float) -> bool:
    """
    Filtro de Varianza Mínima:
    Descartar apuestas con cuotas longshot (> 3.50) a menos que su valor esperado sea extraordinario (> 12.0%).
    Evita drawdowns prolongados en estrategias de valor diario.
    """
    if odds > MIN_VARIANCE_ODDS_THRESHOLD:
        return ev_net >= HIGH_ODDS_MIN_EV
    return ev_net >= MIN_EV_THRESHOLD


def get_kelly_multiplier(risk_profile: str) -> float:
    """
    Fracción de Kelly según el perfil de aversión al riesgo del inversor.
    - Conservador: 1/8 de Kelly (0.125)
    - Equilibrado (Default): 1/4 de Kelly (0.250)
    - Agresivo: 3/8 de Kelly (0.375)
    """
    profile = risk_profile.lower().strip()
    if profile in ("conservador", "conservative"):
        return 0.125
    if profile in ("agresivo", "aggressive"):
        return 0.375
    return 0.250  # Equilibrado por defecto


# ============================================================================
# MOTOR 1: OPTIMIZACIÓN DE PORTAFOLIO VALUE (+EV) CON CRITERIO DE KELLY
# ============================================================================


def optimize_value_bets(
    candidates: List[BetOpportunity],
    bankroll: float,
    risk_profile: str = "equilibrado",
    max_exposure: float = DEFAULT_MAX_DAILY_EXPOSURE,
) -> List[AllocatedBet]:
    """
    Optimiza la cartera de apuestas de valor aplicando:
    1. Filtro de varianza mínima (descarta cuotas > 3.50 con EV insuficiente).
    2. Cálculo de Kelly Fraccional individual.
    3. Hard Cap de seguridad por apuesta (máx 4.0% de la banca).
    4. Escalado proporcional si la suma total de stakes excede el límite de exposición diaria.
    """
    if bankroll <= 0.0 or not candidates:
        return []

    kelly_multiplier = get_kelly_multiplier(risk_profile)
    max_daily_budget = round(bankroll * max_exposure, 2)
    hard_cap_amount = round(bankroll * HARD_CAP_PER_BET, 2)

    pre_allocated: List[AllocatedBet] = []

    for opp in candidates:
        # 1. Filtro de varianza y corte de EV positivo
        if not passes_variance_filter(opp.odds_taken, opp.ev_net):
            continue

        b = opp.odds_taken - 1.0
        p = opp.model_probability
        q = 1.0 - p

        if b <= 0.0 or p <= 0.0:
            continue

        # 2. Fórmula pura de Kelly: f* = (b*p - q) / b
        full_kelly = (b * p - q) / b
        if full_kelly <= 0.0:
            continue

        # Aplicar fracción de Kelly (ej. 0.25 = 1/4 Kelly)
        target_fraction = full_kelly * kelly_multiplier

        # 3. Disyuntor: Hard Cap individual por apuesta
        capped_fraction = min(target_fraction, HARD_CAP_PER_BET)
        raw_stake = round(bankroll * capped_fraction, 2)

        if raw_stake < 1.0:
            continue

        allocated = AllocatedBet(
            event_id=opp.event_id,
            event_name=opp.event_name,
            tournament=opp.tournament,
            match_date=opp.match_date,
            strategy="VALUE_BET",
            market_code=opp.market_code,
            pick_name=opp.pick_name,
            odds_taken=opp.odds_taken,
            fair_odds=opp.fair_odds,
            model_probability=opp.model_probability,
            ev_net=opp.ev_net,
            stake_amount=raw_stake,
            stake_percentage=round((raw_stake / bankroll) * 100.0, 2),
            potential_payout=round(raw_stake * opp.odds_taken, 2),
        )
        pre_allocated.append(allocated)

    if not pre_allocated:
        return []

    # 4. Disyuntor de Exposición Diaria Acumulada
    total_raw_stakes = sum(b.stake_amount for b in pre_allocated)
    if total_raw_stakes > max_daily_budget and total_raw_stakes > 0:
        scaling_factor = max_daily_budget / total_raw_stakes
        final_allocated: List[AllocatedBet] = []
        for b in pre_allocated:
            scaled_stake = round(b.stake_amount * scaling_factor, 2)
            if scaled_stake >= 1.0:
                b.stake_amount = scaled_stake
                b.stake_percentage = round((scaled_stake / bankroll) * 100.0, 2)
                b.potential_payout = round(scaled_stake * b.odds_taken, 2)
                final_allocated.append(b)
        return final_allocated

    return pre_allocated


# ============================================================================
# MOTOR 2: ARBITRAJE SEGURO (SUREBETS) LIBRE DE RIESGO
# ============================================================================


def find_surebets(
    matches_odds_data: List[Dict[str, Any]],
    bankroll: float,
    min_roi: float = MIN_SUREBET_ROI,
    max_budget_pct: float = 0.10,
) -> List[SurebetOpportunity]:
    """
    Identifica oportunidades de arbitraje deportivo donde la suma de las inversas de las cuotas
    es estrictamente menor a 1.0 (ej. < 0.985 para al menos +1.5% de ganancia neta garantizada).

    Soporta mercados de 2 vías (Over/Under, BTTS) y 3 vías (1X2).
    """
    if bankroll <= 0.0 or not matches_odds_data:
        return []

    max_stake_total = round(bankroll * max_budget_pct, 2)
    surebets: List[SurebetOpportunity] = []

    for item in matches_odds_data:
        event_id = item.get("event_id", 0)
        event_name = item.get("event_name", "Partido")
        tournament = item.get("tournament", "Torneo")
        match_date = item.get("match_date", "")
        odds_dict = item.get("odds", {})

        # 1. Evaluar Mercado 1X2 (3 vías)
        h_odd = odds_dict.get("1X2_HOME")
        d_odd = odds_dict.get("1X2_DRAW")
        a_odd = odds_dict.get("1X2_AWAY")

        if h_odd and d_odd and a_odd and h_odd > 1.0 and d_odd > 1.0 and a_odd > 1.0:
            inv_sum = (1.0 / h_odd) + (1.0 / d_odd) + (1.0 / a_odd)
            if inv_sum < (1.0 - min_roi):
                roi = round(((1.0 / inv_sum) - 1.0) * 100.0, 2)
                # Distribuir capital proporcional a la inversa de cada cuota
                s_h = round(max_stake_total * ((1.0 / h_odd) / inv_sum), 2)
                s_d = round(max_stake_total * ((1.0 / d_odd) / inv_sum), 2)
                s_a = round(max_stake_total * ((1.0 / a_odd) / inv_sum), 2)
                tot_s = round(s_h + s_d + s_a, 2)

                payout_h = round(s_h * h_odd, 2)
                payout_d = round(s_d * d_odd, 2)
                payout_a = round(s_a * a_odd, 2)
                min_payout = min(payout_h, payout_d, payout_a)
                net_profit = round(min_payout - tot_s, 2)

                if net_profit > 0:
                    surebets.append(
                        SurebetOpportunity(
                            event_id=event_id,
                            event_name=event_name,
                            tournament=tournament,
                            match_date=match_date,
                            market_type="1X2 (3 Vías)",
                            legs=[
                                SurebetLeg("1X2_HOME", f"1 ({event_name.split(' vs ')[0]})", h_odd, s_h, payout_h),
                                SurebetLeg("1X2_DRAW", "X (Empate)", d_odd, s_d, payout_d),
                                SurebetLeg("1X2_AWAY", f"2 ({event_name.split(' vs ')[-1]})", a_odd, s_a, payout_a),
                            ],
                            total_stake=tot_s,
                            guaranteed_payout=min_payout,
                            guaranteed_profit=net_profit,
                            roi_pct=roi,
                            inverse_sum=round(inv_sum, 4),
                        )
                    )

        # 2. Evaluar Over/Under 2.5 (2 vías)
        over_odd = odds_dict.get("OU_2.5_OVER")
        under_odd = odds_dict.get("OU_2.5_UNDER")
        if over_odd and under_odd and over_odd > 1.0 and under_odd > 1.0:
            inv_sum = (1.0 / over_odd) + (1.0 / under_odd)
            if inv_sum < (1.0 - min_roi):
                roi = round(((1.0 / inv_sum) - 1.0) * 100.0, 2)
                s_o = round(max_stake_total * ((1.0 / over_odd) / inv_sum), 2)
                s_u = round(max_stake_total * ((1.0 / under_odd) / inv_sum), 2)
                tot_s = round(s_o + s_u, 2)
                p_o = round(s_o * over_odd, 2)
                p_u = round(s_u * under_odd, 2)
                min_payout = min(p_o, p_u)
                net_profit = round(min_payout - tot_s, 2)

                if net_profit > 0:
                    surebets.append(
                        SurebetOpportunity(
                            event_id=event_id,
                            event_name=event_name,
                            tournament=tournament,
                            match_date=match_date,
                            market_type="Más/Menos 2.5 Goles",
                            legs=[
                                SurebetLeg("OU_2.5_OVER", "Más de 2.5 Goles", over_odd, s_o, p_o),
                                SurebetLeg("OU_2.5_UNDER", "Menos de 2.5 Goles", under_odd, s_u, p_u),
                            ],
                            total_stake=tot_s,
                            guaranteed_payout=min_payout,
                            guaranteed_profit=net_profit,
                            roi_pct=roi,
                            inverse_sum=round(inv_sum, 4),
                        )
                    )

        # 3. Evaluar Ambos Anotan (BTTS) (2 vías)
        btts_yes = odds_dict.get("BTTS_YES")
        btts_no = odds_dict.get("BTTS_NO")
        if btts_yes and btts_no and btts_yes > 1.0 and btts_no > 1.0:
            inv_sum = (1.0 / btts_yes) + (1.0 / btts_no)
            if inv_sum < (1.0 - min_roi):
                roi = round(((1.0 / inv_sum) - 1.0) * 100.0, 2)
                s_y = round(max_stake_total * ((1.0 / btts_yes) / inv_sum), 2)
                s_n = round(max_stake_total * ((1.0 / btts_no) / inv_sum), 2)
                tot_s = round(s_y + s_n, 2)
                p_y = round(s_y * btts_yes, 2)
                p_n = round(s_n * btts_no, 2)
                min_payout = min(p_y, p_n)
                net_profit = round(min_payout - tot_s, 2)

                if net_profit > 0:
                    surebets.append(
                        SurebetOpportunity(
                            event_id=event_id,
                            event_name=event_name,
                            tournament=tournament,
                            match_date=match_date,
                            market_type="Ambos Equipos Anotan",
                            legs=[
                                SurebetLeg("BTTS_YES", "Ambos Equipos Marcan: SÍ", btts_yes, s_y, p_y),
                                SurebetLeg("BTTS_NO", "Ambos Equipos Marcan: NO", btts_no, s_n, p_n),
                            ],
                            total_stake=tot_s,
                            guaranteed_payout=min_payout,
                            guaranteed_profit=net_profit,
                            roi_pct=roi,
                            inverse_sum=round(inv_sum, 4),
                        )
                    )

    # Ordenar de mayor a menor rentabilidad
    surebets.sort(key=lambda x: x.roi_pct, reverse=True)
    return surebets


# ============================================================================
# MOTOR 3: COBERTURAS DUTCHING BALANCEADAS
# ============================================================================


def solve_dutching(
    event_id: int,
    event_name: str,
    tournament: str,
    match_date: str,
    coverage_name: str,
    selections: List[Dict[str, Any]],
    target_budget: float,
) -> Optional[DutchingOpportunity]:
    """
    Calcula los montos exactos a apostar sobre un subconjunto de resultados mutuamente
    excluyentes (ej. Marcadores Exactos Top de Dixon-Coles o Doble Oportunidad sintética)
    para que, en caso de que acierte cualquiera de ellos, el retorno bruto sea idéntico.

    selections: Lista de dicts con {"market_code", "pick_name", "odds", "model_prob"}
    """
    if not selections or target_budget <= 0:
        return None

    # Filtrar cuotas válidas > 1.0
    valid_picks = [s for s in selections if s.get("odds", 0.0) > 1.0]
    if len(valid_picks) < 2:
        return None

    inv_sum = sum(1.0 / s["odds"] for s in valid_picks)
    if inv_sum <= 0:
        return None

    legs: List[DutchingLeg] = []
    total_allocated = 0.0

    for s in valid_picks:
        odds = s["odds"]
        # Stake proporcional a la inversa de la cuota
        stake = round(target_budget * ((1.0 / odds) / inv_sum), 2)
        payout = round(stake * odds, 2)
        total_allocated += stake
        legs.append(
            DutchingLeg(
                market_code=s["market_code"],
                pick_name=s["pick_name"],
                odds=odds,
                model_prob=s.get("model_prob", 0.0),
                stake_amount=stake,
                payout=payout,
            )
        )

    if not legs or total_allocated <= 0:
        return None

    avg_payout = sum(l.payout for l in legs) / len(legs)
    net_profit = round(avg_payout - total_allocated, 2)
    roi_if_hit = round((net_profit / total_allocated) * 100.0, 2) if total_allocated > 0 else 0.0
    combined_prob = round(sum(l.model_prob for l in legs), 3)

    return DutchingOpportunity(
        event_id=event_id,
        event_name=event_name,
        tournament=tournament,
        match_date=match_date,
        coverage_name=coverage_name,
        legs=legs,
        total_stake=round(total_allocated, 2),
        target_payout=round(avg_payout, 2),
        net_profit_if_hit=net_profit,
        combined_probability=combined_prob,
        roi_if_hit_pct=roi_if_hit,
    )
