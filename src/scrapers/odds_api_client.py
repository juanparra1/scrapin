"""Cliente para The Odds API para monitoreo automatizado de cuotas multi-casa (Betsson, Pinnacle, Betano).

Permite consultar cuotas de casas de apuestas internacionales en tiempo real y cruzarlas
con las cuotas oficiales de Betano obtenidas vía Sofascore para detección de Arbitraje (Surebets).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz

from src.analytics.entity_resolver import normalize_team_name
from src.config import CACHE_DIR, ODDS_API_BASE, ODDS_API_KEY
from src.scrapers.base import BaseScraperClient

logger = logging.getLogger(__name__)

# Mapeo de torneos top a keys de The Odds API
TOURNAMENT_TO_ODDS_API: Dict[str, str] = {
    "Premier League": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "UEFA Champions League": "soccer_uefa_champs_league",
    "UEFA Europa League": "soccer_uefa_europa_league",
    "UEFA Conference League": "soccer_uefa_europa_conference_league",
    "CONMEBOL Libertadores": "soccer_conmebol_copa_libertadores",
    "CONMEBOL Sudamericana": "soccer_conmebol_copa_sudamericana",
    "Liga Profesional de Fútbol": "soccer_argentina_primera_division",
    "Brasileirão Série A": "soccer_brazil_campeonato",
    "Liga Portugal Betclic": "soccer_portugal_primeira_liga",
    "Trendyol Süper Lig": "soccer_turkey_super_league",
    "Major League Soccer (MLS)": "soccer_usa_mls",
    "Championship": "soccer_efl_champ",
    "Liga MX": "soccer_mexico_ligamx",
}


class OddsAPIClient(BaseScraperClient):
    """Cliente especializado para The Odds API con control de cuota y caché inteligente."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(rate_limit_seconds=0.5, use_cache=True)
        self.api_key = api_key or ODDS_API_KEY
        self.last_remaining_requests: Optional[int] = None

    def get_remaining_requests(self) -> Optional[int]:
        """Devuelve el número de solicitudes restantes en la cuota mensual de The Odds API."""
        return self.last_remaining_requests

    def fetch_odds_for_sport(
        self,
        sport_key: str,
        regions: str = "eu",
        bookmakers: str = "betsson,pinnacle,1xBet,coolbet",
        markets: str = "h2h,totals",
        ttl_hours: float = 0.15,  # Caché de 9 minutos para proteger cuota
    ) -> List[Dict[str, Any]]:
        """Obtiene eventos y cuotas para un deporte/liga específico."""
        if not self.api_key:
            logger.warning("No se ha configurado ODDS_API_KEY.")
            return []

        url = (
            f"{ODDS_API_BASE}/sports/{sport_key}/odds/"
            f"?apiKey={self.api_key}"
            f"&regions={regions}"
            f"&bookmakers={bookmakers}"
            f"&markets={markets}"
        )

        try:
            data = self.get_json(url, ttl_hours=ttl_hours)
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.error(f"Error consultando The Odds API para {sport_key}: {e}")
            return []

    def fetch_all_active_soccer_odds(
        self,
        target_sports: Optional[List[str]] = None,
        bookmakers: str = "betsson,pinnacle",
        ttl_hours: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Recupera cuotas de todas las ligas seleccionadas y las normaliza.
        """
        if target_sports is None:
            # Ligas principales por defecto para optimizar cuota
            target_sports = [
                "soccer_epl",
                "soccer_spain_la_liga",
                "soccer_italy_serie_a",
                "soccer_germany_bundesliga",
                "soccer_france_ligue_one",
                "soccer_uefa_champs_league",
                "soccer_uefa_europa_league",
                "soccer_argentina_primera_division",
                "soccer_brazil_campeonato",
            ]

        all_matches: List[Dict[str, Any]] = []
        for s_key in target_sports:
            events = self.fetch_odds_for_sport(
                s_key,
                regions="eu",
                bookmakers=bookmakers,
                markets="h2h,totals",
                ttl_hours=ttl_hours,
            )
            for ev in events:
                normalized = self._normalize_event(ev, s_key)
                if normalized:
                    all_matches.append(normalized)

        return all_matches

    def _normalize_event(self, ev: Dict[str, Any], sport_key: str) -> Optional[Dict[str, Any]]:
        """Normaliza un evento de The Odds API a una estructura limpia con mercados indexados."""
        home_team = ev.get("home_team")
        away_team = ev.get("away_team")
        if not home_team or not away_team:
            return None

        event_id = ev.get("id")
        commence_time = ev.get("commence_time")
        bms_data: Dict[str, Dict[str, Any]] = {}

        for bm in ev.get("bookmakers", []):
            bm_key = bm.get("key", "").lower()
            bm_title = bm.get("title", bm_key)

            markets_dict: Dict[str, Any] = {}
            for mkt in bm.get("markets", []):
                m_type = mkt.get("key")
                outcomes = mkt.get("outcomes", [])

                if m_type == "h2h":
                    h2h_map = {}
                    for o in outcomes:
                        name = o.get("name")
                        price = float(o.get("price", 0.0))
                        if name == home_team:
                            h2h_map["home"] = price
                        elif name == away_team:
                            h2h_map["away"] = price
                        elif name.lower() == "draw" or "draw" in name.lower() or "tie" in name.lower():
                            h2h_map["draw"] = price
                    markets_dict["h2h"] = h2h_map

                elif m_type == "totals":
                    totals_map = {}
                    for o in outcomes:
                        name = (o.get("name") or "").lower()
                        point = float(o.get("point", 2.5))
                        price = float(o.get("price", 0.0))
                        if point == 2.5:
                            if "over" in name:
                                totals_map["over_2_5"] = price
                            elif "under" in name:
                                totals_map["under_2_5"] = price
                    markets_dict["totals"] = totals_map

            bms_data[bm_key] = {
                "title": bm_title,
                "markets": markets_dict,
            }

        return {
            "external_id": event_id,
            "sport_key": sport_key,
            "home_team": home_team,
            "away_team": away_team,
            "norm_home": normalize_team_name(home_team),
            "norm_away": normalize_team_name(away_team),
            "commence_time": commence_time,
            "bookmakers": bms_data,
        }


def calculate_two_way_surebet(
    odd_a: float,
    odd_b: float,
    label_a: str,
    label_b: str,
    bookmaker_a: str,
    bookmaker_b: str,
    total_bankroll: float,
) -> Optional[Dict[str, Any]]:
    """Calcula arbitraje matemático exacto para un mercado de 2 vías (ej. Over/Under)."""
    if odd_a <= 1.0 or odd_b <= 1.0 or total_bankroll <= 0:
        return None

    inv_sum = (1.0 / odd_a) + (1.0 / odd_b)
    roi_pct = round(((1.0 / inv_sum) - 1.0) * 100.0, 2)
    is_surebet = inv_sum < 1.0

    # Asignación de stakes para igualar retornos exactos
    stake_a = round(total_bankroll * ((1.0 / odd_a) / inv_sum), 2)
    stake_b = round(total_bankroll * ((1.0 / odd_b) / inv_sum), 2)
    tot_stake = round(stake_a + stake_b, 2)

    payout_a = round(stake_a * odd_a, 2)
    payout_b = round(stake_b * odd_b, 2)
    min_payout = min(payout_a, payout_b)
    net_profit = round(min_payout - tot_stake, 2)

    return {
        "is_surebet": is_surebet,
        "inverse_sum": round(inv_sum, 4),
        "roi_pct": roi_pct,
        "total_stake": tot_stake,
        "guaranteed_payout": min_payout,
        "guaranteed_profit": net_profit,
        "legs": [
            {
                "label": label_a,
                "bookmaker": bookmaker_a,
                "odds": odd_a,
                "stake": stake_a,
                "payout": payout_a,
            },
            {
                "label": label_b,
                "bookmaker": bookmaker_b,
                "odds": odd_b,
                "stake": stake_b,
                "payout": payout_b,
            },
        ],
    }


def calculate_three_way_surebet(
    odd_1: float,
    odd_x: float,
    odd_2: float,
    bm_1: str,
    bm_x: str,
    bm_2: str,
    home_name: str,
    away_name: str,
    total_bankroll: float,
) -> Optional[Dict[str, Any]]:
    """Calcula arbitraje matemático exacto para un mercado 1X2 (3 vías)."""
    if odd_1 <= 1.0 or odd_x <= 1.0 or odd_2 <= 1.0 or total_bankroll <= 0:
        return None

    inv_sum = (1.0 / odd_1) + (1.0 / odd_x) + (1.0 / odd_2)
    roi_pct = round(((1.0 / inv_sum) - 1.0) * 100.0, 2)
    is_surebet = inv_sum < 1.0

    stake_1 = round(total_bankroll * ((1.0 / odd_1) / inv_sum), 2)
    stake_x = round(total_bankroll * ((1.0 / odd_x) / inv_sum), 2)
    stake_2 = round(total_bankroll * ((1.0 / odd_2) / inv_sum), 2)
    tot_stake = round(stake_1 + stake_x + stake_2, 2)

    payout_1 = round(stake_1 * odd_1, 2)
    payout_x = round(stake_x * odd_x, 2)
    payout_2 = round(stake_2 * odd_2, 2)
    min_payout = min(payout_1, payout_x, payout_2)
    net_profit = round(min_payout - tot_stake, 2)

    return {
        "is_surebet": is_surebet,
        "inverse_sum": round(inv_sum, 4),
        "roi_pct": roi_pct,
        "total_stake": tot_stake,
        "guaranteed_payout": min_payout,
        "guaranteed_profit": net_profit,
        "legs": [
            {
                "label": f"1 ({home_name})",
                "bookmaker": bm_1,
                "odds": odd_1,
                "stake": stake_1,
                "payout": payout_1,
            },
            {
                "label": "X (Empate)",
                "bookmaker": bm_x,
                "odds": odd_x,
                "stake": stake_x,
                "payout": payout_x,
            },
            {
                "label": f"2 ({away_name})",
                "bookmaker": bm_2,
                "odds": odd_2,
                "stake": stake_2,
                "payout": payout_2,
            },
        ],
    }


def match_and_cross_arbitrage(
    sofascore_matches: List[Any],
    odds_api_matches: List[Dict[str, Any]],
    bankroll: float = 20000.0,
    min_similarity: float = 72.0,
) -> List[Dict[str, Any]]:
    """
    Cruza los partidos de Sofascore (cuotas oficiales de Betano) con los de The Odds API
    (Betsson, Pinnacle, etc.) usando EntityResolver y detecta todas las combinaciones de arbitraje.
    """
    cross_comparisons: List[Dict[str, Any]] = []

    for sf_m in sofascore_matches:
        if not getattr(sf_m, "odds", None):
            continue

        sf_home_raw = sf_m.home_team.name
        sf_away_raw = sf_m.away_team.name
        norm_sf_h = normalize_team_name(sf_home_raw)
        norm_sf_a = normalize_team_name(sf_away_raw)

        # Buscar el mejor candidato en The Odds API
        best_match = None
        best_score = 0.0

        for oa_m in odds_api_matches:
            s_h = fuzz.token_sort_ratio(norm_sf_h, oa_m["norm_home"])
            s_a = fuzz.token_sort_ratio(norm_sf_a, oa_m["norm_away"])
            avg_s = (s_h + s_a) / 2.0

            if avg_s > best_score and s_h >= min_similarity and s_a >= min_similarity:
                best_score = avg_s
                best_match = oa_m

        if not best_match:
            continue

        # Extraer cuotas de Betano (Sofascore)
        betano_odds = {
            "1": getattr(sf_m.odds, "home_win", None),
            "X": getattr(sf_m.odds, "draw", None),
            "2": getattr(sf_m.odds, "away_win", None),
            "over_2_5": getattr(sf_m.odds, "over_2_5", None),
            "under_2_5": getattr(sf_m.odds, "under_2_5", None),
        }

        # Extraer cuotas de Betsson de The Odds API
        oa_bms = best_match.get("bookmakers", {})
        betsson_bm = oa_bms.get("betsson", {})
        betsson_h2h = betsson_bm.get("markets", {}).get("h2h", {})
        betsson_totals = betsson_bm.get("markets", {}).get("totals", {})

        betsson_odds = {
            "1": betsson_h2h.get("home"),
            "X": betsson_h2h.get("draw"),
            "2": betsson_h2h.get("away"),
            "over_2_5": betsson_totals.get("over_2_5"),
            "under_2_5": betsson_totals.get("under_2_5"),
        }

        # Si Betsson no tiene cuotas pero hay Pinnacle o 1xBet, guardamos también como alternativa
        pinnacle_bm = oa_bms.get("pinnacle", {})
        pinnacle_h2h = pinnacle_bm.get("markets", {}).get("h2h", {})
        pinnacle_totals = pinnacle_bm.get("markets", {}).get("totals", {})

        pinnacle_odds = {
            "1": pinnacle_h2h.get("home"),
            "X": pinnacle_h2h.get("draw"),
            "2": pinnacle_h2h.get("away"),
            "over_2_5": pinnacle_totals.get("over_2_5"),
            "under_2_5": pinnacle_totals.get("under_2_5"),
        }

        comparison_entry = {
            "event_id": sf_m.event_id,
            "match_title": f"{sf_home_raw} vs {sf_away_raw}",
            "tournament": sf_m.tournament.name,
            "betano_odds": betano_odds,
            "betsson_odds": betsson_odds,
            "pinnacle_odds": pinnacle_odds,
            "arbitrage_opportunities": [],
            "best_cross_odds": {},
        }

        # 1. EVALUAR ARBITRAJE EN OVER/UNDER 2.5 ENTRE BETANO Y BETSSON
        # Combinación 1: Over en Betano vs Under en Betsson
        if betano_odds.get("over_2_5") and betsson_odds.get("under_2_5"):
            arb_ou1 = calculate_two_way_surebet(
                odd_a=betano_odds["over_2_5"],
                odd_b=betsson_odds["under_2_5"],
                label_a="Más de 2.5 Goles",
                label_b="Menos de 2.5 Goles",
                bookmaker_a="Betano",
                bookmaker_b="Betsson",
                total_bankroll=bankroll,
            )
            if arb_ou1:
                comparison_entry["arbitrage_opportunities"].append({
                    "market": "Over/Under 2.5 (Betano Over / Betsson Under)",
                    **arb_ou1,
                })

        # Combinación 2: Over en Betsson vs Under en Betano
        if betsson_odds.get("over_2_5") and betano_odds.get("under_2_5"):
            arb_ou2 = calculate_two_way_surebet(
                odd_a=betsson_odds["over_2_5"],
                odd_b=betano_odds["under_2_5"],
                label_a="Más de 2.5 Goles",
                label_b="Menos de 2.5 Goles",
                bookmaker_a="Betsson",
                bookmaker_b="Betano",
                total_bankroll=bankroll,
            )
            if arb_ou2:
                comparison_entry["arbitrage_opportunities"].append({
                    "market": "Over/Under 2.5 (Betsson Over / Betano Under)",
                    **arb_ou2,
                })

        # 2. EVALUAR ARBITRAJE EN MERCADO 1X2 CRUZADO (Tomando la mejor cuota entre Betano y Betsson)
        odds_1_cand = []
        if betano_odds.get("1"): odds_1_cand.append((betano_odds["1"], "Betano"))
        if betsson_odds.get("1"): odds_1_cand.append((betsson_odds["1"], "Betsson"))

        odds_x_cand = []
        if betano_odds.get("X"): odds_x_cand.append((betano_odds["X"], "Betano"))
        if betsson_odds.get("X"): odds_x_cand.append((betsson_odds["X"], "Betsson"))

        odds_2_cand = []
        if betano_odds.get("2"): odds_2_cand.append((betano_odds["2"], "Betano"))
        if betsson_odds.get("2"): odds_2_cand.append((betsson_odds["2"], "Betsson"))

        if odds_1_cand and odds_x_cand and odds_2_cand:
            best_1, bm_1 = max(odds_1_cand, key=lambda x: x[0])
            best_x, bm_x = max(odds_x_cand, key=lambda x: x[0])
            best_2, bm_2 = max(odds_2_cand, key=lambda x: x[0])

            comparison_entry["best_cross_odds"] = {
                "1": (best_1, bm_1),
                "X": (best_x, bm_x),
                "2": (best_2, bm_2),
            }

            arb_1x2 = calculate_three_way_surebet(
                odd_1=best_1,
                odd_x=best_x,
                odd_2=best_2,
                bm_1=bm_1,
                bm_x=bm_x,
                bm_2=bm_2,
                home_name=sf_home_raw,
                away_name=sf_away_raw,
                total_bankroll=bankroll,
            )
            if arb_1x2:
                comparison_entry["arbitrage_opportunities"].append({
                    "market": "1X2 Multi-Casa (Mejor Cuota Cruzada)",
                    **arb_1x2,
                })

        cross_comparisons.append(comparison_entry)

    return cross_comparisons
