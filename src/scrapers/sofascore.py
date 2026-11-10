"""Extractor de datos de Sofascore + Opta (partidos del día, xG, xGOT, porteros, cuotas, alineaciones, H2H profundo e historial)."""

import concurrent.futures
import json
import math
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config import (
    CACHE_DIR,
    PRIORITY_FALLBACK_TOURNAMENTS,
    SOFASCORE_API_BASE,
    TOP_TOURNAMENTS,
    TOURNAMENT_ESPN_SLUGS,
)
from src.models.schemas import (
    H2HMatchItemSchema,
    H2HSummarySchema,
    LiveMomentumSummarySchema,
    MatchDetailSchema,
    MatchStatsSchema,
    MomentumPointSchema,
    OddsSchema,
    PlayerLineupSchema,
    PlayerPropSchema,
    RefereeSchema,
    TeamLineupSchema,
    TeamSchema,
    TournamentSchema,
)
from src.scrapers.base import BaseScraperClient


def _fractional_to_decimal(fractional_str: Optional[str]) -> Optional[float]:
    """Convierte cuota fraccionaria (ej. '6/5') en cuota decimal (ej. 2.20)."""
    if not fractional_str or "/" not in fractional_str:
        return None
    try:
        num, den = fractional_str.split("/", 1)
        return round(1.0 + (float(num) / float(den)), 2)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_stat_number(val: Any) -> Optional[float]:
    """Convierte cadenas como '58%', '1.42' o enteros en float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip().replace("%", "")
    if "/" in text:
        text = text.split("/")[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_team_label(name: str) -> str:
    """Normaliza nombres de equipos (sin tildes ni sufijos genéricos) para cruce exacto entre Sofascore y ESPN."""
    if not name:
        return ""
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    for token in ("fc", "cf", "sc", "ac", "ud", "cd", "bk", "fk", "sk", "if", "women", "femenino", "wfc"):
        norm = norm.replace(f" {token} ", " ").replace(f" {token}", "").replace(f"{token} ", "")
    norm = " ".join(norm.split()).strip()
    aliases = {
        "internazionale": "inter",
        "inter milan": "inter",
        "paris saint-germain": "psg",
        "paris saint germain": "psg",
        "ol lyonnes": "lyon",
        "olympique lyonnais": "lyon",
        "turkiye": "turkey",
        "czechia": "czech republic",
        "bosnia & herzegovina": "bosnia and herzegovina",
        "atletico madrid": "atletico",
        "club atletico de madrid": "atletico",
        "bayern munchen": "bayern munich",
    }
    return aliases.get(norm, norm)


class SofascoreScraper(BaseScraperClient):
    """Cliente especializado para extraer datos estructurados de Sofascore / Opta con respaldo automático ESPN."""

    _top_players_cache_indexed: bool = False
    _team_top_players_files: Dict[int, Path] = {}
    _espn_team_meta: Dict[int, tuple[str, str]] = {}

    def get_scheduled_events(
        self,
        date_str: str,
        only_top_tournaments: bool = True,
    ) -> List[MatchDetailSchema]:
        """
        Obtiene los partidos programados o jugados en una fecha ('YYYY-MM-DD' hora Colombia UTC-5).
        Incluye disyuntor inmediato si Sofascore bloquea la IP (HTTP 403 en Render Cloud)
        y soporte multi-huso horario para partidos nocturnos de América y Selecciones (Nations League).
        """
        if self.is_sofascore_blocked():
            cat_payload = None
        else:
            cat_url = f"{SOFASCORE_API_BASE}/sport/football/{date_str}/-18000/categories"
            cat_payload = self.get_json(cat_url, ttl_hours=0.5)
            if not self.is_sofascore_blocked() and (
                not cat_payload or "categories" not in cat_payload or not cat_payload["categories"]
            ):
                cat_url_utc0 = f"{SOFASCORE_API_BASE}/sport/football/{date_str}/0/categories"
                cat_payload = self.get_json(cat_url_utc0, ttl_hours=0.5)

        matches: List[MatchDetailSchema] = []
        seen_event_ids: set[int] = set()

        active_top_tournaments: List[int] = []
        active_categories: List[int] = []

        if cat_payload and "categories" in cat_payload and cat_payload["categories"]:
            sorted_cats = sorted(
                cat_payload["categories"],
                key=lambda c: (
                    c.get("category", {}).get("priority", 0),
                    c.get("totalEvents", 0),
                ),
                reverse=True,
            )

            for item in sorted_cats:
                cat_id = item.get("category", {}).get("id")
                if cat_id:
                    active_categories.append(int(cat_id))
                for ut_id in item.get("uniqueTournamentIds", []):
                    if ut_id in TOP_TOURNAMENTS and ut_id not in active_top_tournaments:
                        active_top_tournaments.append(int(ut_id))

        # Asegurar siempre los torneos prioritarios (incluyendo UEFA Women's Champions League 692 y Nations League 10783)
        for p_id in PRIORITY_FALLBACK_TOURNAMENTS:
            if p_id not in active_top_tournaments:
                active_top_tournaments.append(p_id)

        tz_local = timezone(timedelta(hours=-5))
        try:
            next_date_str = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            next_date_str = date_str

        # Torneos cuyos partidos vespertinos/nocturnos pueden cruzar medianoche UTC
        americas_and_intl_ids = {
            692, 7, 679, 17015, 11539, 11542, 1335, 10825, 155, 1024, 11707, 325, 373,
            242, 11621, 384, 480, 490, 377, 120, 133, 14100, 130, 10783, 1077, 851
        }

        def _fetch_tourney(ut_id: int) -> List[Dict[str, Any]]:
            if self.is_sofascore_blocked():
                return []
            events_out: List[Dict[str, Any]] = []
            t_url = f"{SOFASCORE_API_BASE}/unique-tournament/{ut_id}/scheduled-events/{date_str}"
            t_data = self.get_json(t_url, ttl_hours=0.5)
            if t_data and "events" in t_data:
                events_out.extend(t_data["events"])
            if ut_id in americas_and_intl_ids and next_date_str != date_str and not self.is_sofascore_blocked():
                t_url_next = f"{SOFASCORE_API_BASE}/unique-tournament/{ut_id}/scheduled-events/{next_date_str}"
                t_data_next = self.get_json(t_url_next, ttl_hours=0.5)
                if t_data_next and "events" in t_data_next:
                    events_out.extend(t_data_next["events"])
            return events_out

        found_ut_ids: set[int] = set()
        if not self.is_sofascore_blocked():
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as exec_t:
                results = exec_t.map(_fetch_tourney, active_top_tournaments)
                for raw_list in results:
                    for raw in raw_list:
                        match_schema = self._parse_event_summary(raw)
                        if match_schema and match_schema.event_id not in seen_event_ids:
                            event_date_local = datetime.fromtimestamp(match_schema.start_timestamp, tz=tz_local).strftime("%Y-%m-%d")
                            if event_date_local == date_str:
                                seen_event_ids.add(match_schema.event_id)
                                if match_schema.tournament.unique_id:
                                    found_ut_ids.add(match_schema.tournament.unique_id)
                                matches.append(match_schema)

        # Respaldo automático con ESPN API para UEFA Women's Champions League (692) y torneos prioritarios si Sofascore no devolvió eventos
        espn_check_ids = [p_id for p_id in PRIORITY_FALLBACK_TOURNAMENTS if p_id not in found_ut_ids]
        if espn_check_ids:
            seen_pair_keys = {
                (_normalize_team_label(m.home_team.name), _normalize_team_label(m.away_team.name))
                for m in matches
            }
            for ut_id in espn_check_ids:
                for em in self._fetch_espn_scheduled_events(date_str, ut_id):
                    pair_key = (_normalize_team_label(em.home_team.name), _normalize_team_label(em.away_team.name))
                    if em.event_id not in seen_event_ids and pair_key not in seen_pair_keys:
                        seen_event_ids.add(em.event_id)
                        seen_pair_keys.add(pair_key)
                        matches.append(em)

        if not only_top_tournaments and not self.is_sofascore_blocked():
            for cat_id in active_categories[:12]:
                if self.is_sofascore_blocked():
                    break
                c_url = f"{SOFASCORE_API_BASE}/category/{cat_id}/scheduled-events/{date_str}"
                c_data = self.get_json(c_url, ttl_hours=0.5)
                if not c_data or "events" not in c_data:
                    continue
                for raw in c_data["events"]:
                    match_schema = self._parse_event_summary(raw)
                    if match_schema and match_schema.event_id not in seen_event_ids:
                        event_date_local = datetime.fromtimestamp(match_schema.start_timestamp, tz=tz_local).strftime("%Y-%m-%d")
                        if event_date_local == date_str:
                            seen_event_ids.add(match_schema.event_id)
                            matches.append(match_schema)

        matches.sort(key=lambda m: m.start_timestamp)
        return matches

    def get_live_events(self) -> List[MatchDetailSchema]:
        """Obtiene todos los partidos de fútbol que se están jugando EN VIVO en este instante."""
        url = f"{SOFASCORE_API_BASE}/sport/football/events/live"
        data = self.get_json(url, ttl_hours=0.015)
        if not data or "events" not in data:
            return []

        live_matches: List[MatchDetailSchema] = []
        for raw in data["events"]:
            parsed = self._parse_event_summary(raw)
            if parsed:
                live_matches.append(parsed)
        return live_matches

    def _parse_event_summary(self, raw: Dict[str, Any]) -> Optional[MatchDetailSchema]:
        """Transforma un nodo de evento crudo de Sofascore en MatchDetailSchema."""
        try:
            event_id = int(raw["id"])
            start_ts = int(raw.get("startTimestamp", 0))
            start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)

            tournament_raw = raw.get("tournament", {})
            unique_t = tournament_raw.get("uniqueTournament", {})
            category = tournament_raw.get("category", {})

            tournament = TournamentSchema(
                id=int(tournament_raw.get("id", 0)),
                unique_id=unique_t.get("id"),
                name=unique_t.get("name") or tournament_raw.get("name", "Unknown"),
                country=category.get("name"),
            )

            home_raw = raw.get("homeTeam", {})
            away_raw = raw.get("awayTeam", {})

            home_team = TeamSchema(
                id=int(home_raw["id"]),
                name=home_raw.get("name", "Home"),
                short_name=home_raw.get("shortName"),
                slug=home_raw.get("slug"),
                country=home_raw.get("country", {}).get("name"),
            )
            away_team = TeamSchema(
                id=int(away_raw["id"]),
                name=away_raw.get("name", "Away"),
                short_name=away_raw.get("shortName"),
                slug=away_raw.get("slug"),
                country=away_raw.get("country", {}).get("name"),
            )

            home_score_obj = raw.get("homeScore", {})
            away_score_obj = raw.get("awayScore", {})
            home_score = home_score_obj.get("current")
            away_score = away_score_obj.get("current")
            home_ht = home_score_obj.get("period1")
            away_ht = away_score_obj.get("period1")
            status = raw.get("status", {})

            return MatchDetailSchema(
                event_id=event_id,
                custom_id=raw.get("customId"),
                slug=raw.get("slug", f"match-{event_id}"),
                start_timestamp=start_ts,
                start_time_utc=start_dt,
                status_type=status.get("type", "notstarted"),
                status_description=status.get("description"),
                tournament=tournament,
                season_id=raw.get("season", {}).get("id"),
                round_number=raw.get("roundInfo", {}).get("round"),
                home_team=home_team,
                away_team=away_team,
                home_score=int(home_score) if home_score is not None else None,
                away_score=int(away_score) if away_score is not None else None,
                home_score_ht=int(home_ht) if home_ht is not None else None,
                away_score_ht=int(away_ht) if away_ht is not None else None,
            )
        except (KeyError, ValueError, TypeError):
            return None

    def enrich_match(
        self,
        match: MatchDetailSchema,
        include_stats: bool = True,
        include_odds: bool = True,
        include_h2h: bool = True,
        include_lineups: bool = True,
        include_streaks: bool = True,
        include_referee: bool = True,
    ) -> MatchDetailSchema:
        """Enriquece un partido en paralelo (5x más rápido) con estadísticas Opta, árbitro, cuotas, H2H, rachas y alineaciones."""
        eid = match.event_id

        # Ejecución paralela de las consultas independientes del partido
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            fut_ref = None
            if include_referee or not match.custom_id:
                fut_ref = executor.submit(self._get_event_referee_and_custom_id, eid)

            fut_stats = None
            fut_mom = None
            if include_stats and match.status_type in ("finished", "inprogress"):
                fut_stats = executor.submit(self.get_match_statistics, eid)
                fut_mom = executor.submit(
                    self.get_match_momentum, eid, match.home_team.name, match.away_team.name
                )

            fut_odds = None
            if include_odds:
                is_live_match = match.status_type in ("inprogress", "live")
                fut_odds = executor.submit(self.get_match_odds, eid, is_live_match)

            fut_streaks = None
            if include_streaks:
                fut_streaks = executor.submit(self.get_match_streaks, eid)

            fut_lineups = None
            if include_lineups:
                fut_lineups = executor.submit(self.get_match_lineups, eid)

            if fut_ref:
                try:
                    ref_obj, cid = fut_ref.result()
                    if include_referee:
                        match.referee = ref_obj
                    if cid and not match.custom_id:
                        match.custom_id = cid
                except Exception:
                    pass

            if fut_stats:
                try:
                    match.stats = fut_stats.result()
                except Exception:
                    pass

            if fut_mom:
                try:
                    match.momentum = fut_mom.result()
                except Exception:
                    pass

            if fut_odds:
                try:
                    match.odds = fut_odds.result()
                except Exception:
                    pass

            if fut_streaks:
                try:
                    match.streaks = fut_streaks.result()
                except Exception:
                    pass

            if fut_lineups:
                try:
                    lineups = fut_lineups.result()
                    if lineups:
                        match.home_lineup = lineups.get("home")
                        match.away_lineup = lineups.get("away")
                except Exception:
                    pass

        # H2H (requiere custom_id que pudo ser descubierto en fut_ref)
        if include_h2h:
            try:
                match.h2h = self.get_match_h2h(
                    eid,
                    custom_id=match.custom_id,
                    home_team_id=match.home_team.id,
                    away_team_id=match.away_team.id,
                    reference_timestamp=match.start_timestamp,
                )
            except Exception:
                pass

        return match

    def _get_event_referee_and_custom_id(
        self, event_id: int
    ) -> tuple[Optional[RefereeSchema], Optional[str]]:
        """Obtiene el árbitro asignado y el customId del evento en una sola petición."""
        url = f"{SOFASCORE_API_BASE}/event/{event_id}"
        data = self.get_json(url, ttl_hours=6.0)
        if not data or "event" not in data:
            return None, None

        ev = data["event"]
        custom_id = ev.get("customId")
        ref = ev.get("referee")
        if not ref or not ref.get("name"):
            return None, custom_id

        games = max(1, int(ref.get("games", 0) or 1))
        yc = int(ref.get("yellowCards", 0) or 0)
        rc = int(ref.get("redCards", 0) or 0)
        yrc = int(ref.get("yellowRedCards", 0) or 0)

        y_per_game = round(yc / games, 2) if ref.get("games") else 4.2
        r_per_game = round((rc + yrc) / games, 2) if ref.get("games") else 0.18

        if y_per_game >= 4.65:
            strictness = "Muy Estricto 🔴"
        elif y_per_game >= 3.85:
            strictness = "Promedio 🟡"
        else:
            strictness = "Permisivo 🟢"

        return (
            RefereeSchema(
                id=ref.get("id"),
                name=ref.get("name", "Desconocido"),
                country=ref.get("country", {}).get("name"),
                games=int(ref.get("games", 0) or 0),
                yellow_cards=yc,
                red_cards=rc,
                yellow_red_cards=yrc,
                yellow_per_game=y_per_game,
                red_per_game=r_per_game,
                strictness_label=strictness,
            ),
            custom_id,
        )

    def get_match_referee(self, event_id: int) -> Optional[RefereeSchema]:
        """Obtiene el árbitro asignado al partido y su historial de tarjetas por encuentro."""
        ref_obj, _ = self._get_event_referee_and_custom_id(event_id)
        return ref_obj

    def get_match_momentum(
        self, event_id: int, home_name: str, away_name: str
    ) -> Optional[LiveMomentumSummarySchema]:
        """
        Extrae el gráfico de presión minuto a minuto (Attack Momentum) de Sofascore
        y detecta asedio ofensivo / alertas de gol en los últimos 10 minutos.
        """
        url = f"{SOFASCORE_API_BASE}/event/{event_id}/graph"
        data = self.get_json(url, ttl_hours=0.05)
        if not data or "graphPoints" not in data or not data["graphPoints"]:
            return None

        pts: List[MomentumPointSchema] = []
        home_vals: List[float] = []
        away_vals: List[float] = []

        for raw_pt in data["graphPoints"]:
            minute = float(raw_pt.get("minute", 0))
            val = float(raw_pt.get("value", 0))
            pts.append(MomentumPointSchema(minute=minute, value=val))
            if val > 0:
                home_vals.append(val)
            elif val < 0:
                away_vals.append(abs(val))

        if not pts:
            return None

        h_avg = round(sum(home_vals) / max(1, len(pts)), 1)
        a_avg = round(sum(away_vals) / max(1, len(pts)), 1)

        recent_slice = pts[-10:]
        recent_net = round(sum(p.value for p in recent_slice) / len(recent_slice), 1)
        recent_abs = round(sum(abs(p.value) for p in recent_slice) / len(recent_slice), 1)

        if recent_net >= 15.0:
            dom = home_name
        elif recent_net <= -15.0:
            dom = away_name
        elif h_avg > a_avg * 1.25:
            dom = home_name
        elif a_avg > h_avg * 1.25:
            dom = away_name
        else:
            dom = "Equilibrado"

        if abs(recent_net) >= 32.0 or recent_abs >= 42.0:
            alert_lvl = "ALTA 🔥"
            attacking_side = home_name if recent_net > 0 else away_name
            alert_reason = (
                f"Asedio intenso de {attacking_side} en los últimos 10 min "
                f"(Presión media: {abs(recent_net):.0f} pts)"
            )
        elif abs(recent_net) >= 20.0 or recent_abs >= 28.0:
            alert_lvl = "Media 🟡"
            attacking_side = home_name if recent_net > 0 else away_name
            alert_reason = f"Dominio territorial de {attacking_side} en el tramo reciente ({abs(recent_net):.0f} pts)"
        else:
            alert_lvl = "Normal"
            alert_reason = "Ritmo controlado sin asedio extremo en los últimos 10 min"

        return LiveMomentumSummarySchema(
            home_pressure_avg=h_avg,
            away_pressure_avg=a_avg,
            recent_10m_net=recent_net,
            recent_10m_abs_intensity=recent_abs,
            dominant_team=dom,
            goal_alert_level=alert_lvl,
            goal_alert_reason=alert_reason,
            points=pts,
        )

    def get_match_statistics(self, event_id: int) -> Optional[MatchStatsSchema]:
        """Extrae todas las estadísticas Opta del partido (períodos ALL y 1ST), incluyendo xG, xGOT, toques en área y atajadas."""
        url = f"{SOFASCORE_API_BASE}/event/{event_id}/statistics"
        data = self.get_json(url, ttl_hours=24.0)
        if not data or "statistics" not in data:
            return None

        all_period = next(
            (p for p in data["statistics"] if p.get("period") == "ALL"),
            data["statistics"][0] if data["statistics"] else None,
        )
        first_half = next(
            (p for p in data["statistics"] if p.get("period") == "1ST"),
            None,
        )
        if not all_period:
            return None

        stats = MatchStatsSchema()
        for group in all_period.get("groups", []):
            for item in group.get("statisticsItems", []):
                key = item.get("key") or item.get("name", "").lower().replace(" ", "_")
                h_val = _parse_stat_number(item.get("homeValue", item.get("home")))
                a_val = _parse_stat_number(item.get("awayValue", item.get("away")))

                stats.raw_metrics[key] = {"home": h_val, "away": a_val}

                if key == "expectedGoals":
                    stats.expected_goals_home = h_val
                    stats.expected_goals_away = a_val
                elif key == "expectedGoalsOnTarget":
                    stats.xg_on_target_home = h_val
                    stats.xg_on_target_away = a_val
                elif key == "goalsPrevented":
                    stats.goals_prevented_home = h_val
                    stats.goals_prevented_away = a_val
                elif key == "ballPossession":
                    stats.ball_possession_home = h_val
                    stats.ball_possession_away = a_val
                elif key == "totalShotsOnGoal":
                    stats.total_shots_home = int(h_val) if h_val is not None else None
                    stats.total_shots_away = int(a_val) if a_val is not None else None
                elif key == "shotsOnGoal":
                    stats.shots_on_target_home = int(h_val) if h_val is not None else None
                    stats.shots_on_target_away = int(a_val) if a_val is not None else None
                elif key == "totalShotsInsideBox":
                    stats.shots_inside_box_home = int(h_val) if h_val is not None else None
                    stats.shots_inside_box_away = int(a_val) if a_val is not None else None
                elif key == "totalShotsOutsideBox":
                    stats.shots_outside_box_home = int(h_val) if h_val is not None else None
                    stats.shots_outside_box_away = int(a_val) if a_val is not None else None
                elif key == "touchesInOppBox":
                    stats.touches_in_box_home = int(h_val) if h_val is not None else None
                    stats.touches_in_box_away = int(a_val) if a_val is not None else None
                elif key == "bigChanceCreated":
                    stats.big_chances_home = int(h_val) if h_val is not None else None
                    stats.big_chances_away = int(a_val) if a_val is not None else None
                elif key == "bigChanceMissed":
                    stats.big_chances_missed_home = int(h_val) if h_val is not None else None
                    stats.big_chances_missed_away = int(a_val) if a_val is not None else None
                elif key == "finalThirdEntries":
                    stats.final_third_entries_home = int(h_val) if h_val is not None else None
                    stats.final_third_entries_away = int(a_val) if a_val is not None else None
                elif key == "ballRecovery":
                    stats.ball_recoveries_home = int(h_val) if h_val is not None else None
                    stats.ball_recoveries_away = int(a_val) if a_val is not None else None
                elif key == "interceptionWon":
                    stats.interceptions_home = int(h_val) if h_val is not None else None
                    stats.interceptions_away = int(a_val) if a_val is not None else None
                elif key in ("clearanceTotal", "clearances"):
                    stats.clearances_home = int(h_val) if h_val is not None else None
                    stats.clearances_away = int(a_val) if a_val is not None else None
                elif key in ("tacklesWon", "totalTackles"):
                    stats.tackles_won_home = int(h_val) if h_val is not None else None
                    stats.tackles_won_away = int(a_val) if a_val is not None else None
                elif key == "passes":
                    pass_h = int(h_val) if h_val is not None else None
                    pass_a = int(a_val) if a_val is not None else None
                    # Cálculo de PPDA (Pases del rival divididos entre acciones defensivas: tackles + intercepciones)
                    def_actions_h = max(1, (stats.interceptions_home or 0) + (stats.ball_recoveries_home or 0))
                    def_actions_a = max(1, (stats.interceptions_away or 0) + (stats.ball_recoveries_away or 0))
                    if pass_a:
                        stats.ppda_home = round(pass_a / def_actions_h, 1)
                    if pass_h:
                        stats.ppda_away = round(pass_h / def_actions_a, 1)
                elif key == "cornerKicks":
                    stats.corner_kicks_home = int(h_val) if h_val is not None else None
                    stats.corner_kicks_away = int(a_val) if a_val is not None else None
                elif key == "fouls":
                    stats.fouls_home = int(h_val) if h_val is not None else None
                    stats.fouls_away = int(a_val) if a_val is not None else None
                elif key == "yellowCards":
                    stats.yellow_cards_home = int(h_val) if h_val is not None else None
                    stats.yellow_cards_away = int(a_val) if a_val is not None else None
                elif key == "redCards":
                    stats.red_cards_home = int(h_val) if h_val is not None else None
                    stats.red_cards_away = int(a_val) if a_val is not None else None
                elif key == "goalkeeperSaves":
                    stats.saves_home = int(h_val) if h_val is not None else None
                    stats.saves_away = int(a_val) if a_val is not None else None
                elif key == "diveSaves":
                    stats.dive_saves_home = int(h_val) if h_val is not None else None
                    stats.dive_saves_away = int(a_val) if a_val is not None else None
                elif key == "offsides":
                    stats.offsides_home = int(h_val) if h_val is not None else None
                    stats.offsides_away = int(a_val) if a_val is not None else None

        if first_half:
            for group in first_half.get("groups", []):
                for item in group.get("statisticsItems", []):
                    key = item.get("key") or item.get("name", "").lower().replace(" ", "_")
                    h_val = _parse_stat_number(item.get("homeValue", item.get("home")))
                    a_val = _parse_stat_number(item.get("awayValue", item.get("away")))

                    if key == "expectedGoals":
                        stats.ht_xg_home = h_val
                        stats.ht_xg_away = a_val
                    elif key == "cornerKicks":
                        stats.ht_corners_home = int(h_val) if h_val is not None else None
                        stats.ht_corners_away = int(a_val) if a_val is not None else None
                    elif key == "shotsOnGoal":
                        stats.ht_sot_home = int(h_val) if h_val is not None else None
                        stats.ht_sot_away = int(a_val) if a_val is not None else None
                    elif key == "goalkeeperSaves":
                        stats.ht_saves_home = int(h_val) if h_val is not None else None
                        stats.ht_saves_away = int(a_val) if a_val is not None else None

        return stats

    def get_match_odds(self, event_id: int, is_live: bool = False) -> Optional[OddsSchema]:
        """Extrae cuotas de apertura vs actuales (Dropping Odds / Smart Money) y probabilidades implícitas.
        Ajusta el TTL de caché dinámicamente: 30 seg para en vivo, 5 min para pre-match."""
        ttl = 0.008 if is_live else 0.08  # ~30s en vivo, ~5min pre-match
        url = f"{SOFASCORE_API_BASE}/event/{event_id}/odds/1/all"
        data = self.get_json(url, ttl_hours=ttl)
        if not data or "markets" not in data:
            return None

        odds = OddsSchema()
        odds.is_live = is_live
        raw_mkts: Dict[str, Dict[str, float]] = {}
        for market in data["markets"]:
            market_id = market.get("marketId")
            market_name_raw = market.get("marketName") or ""
            market_name = market_name_raw.lower()
            choice_grp = str(market.get("choiceGroup") or "")
            choices = market.get("choices", [])

            # Guardar mapa completo de cuotas de la casa
            m_key = f"{market_name_raw} ({choice_grp})" if choice_grp else market_name_raw
            dec_map = {}
            for c in choices:
                c_name = c.get("name")
                c_dec = _fractional_to_decimal(c.get("fractionalValue"))
                if c_name and c_dec:
                    dec_map[c_name] = c_dec
            if dec_map:
                raw_mkts[m_key] = dec_map

            # 1. 1X2 Tiempo Completo
            if market_id == 1 or market_name == "full time":
                for c in choices:
                    name = c.get("name")
                    dec = _fractional_to_decimal(c.get("fractionalValue"))
                    init_dec = _fractional_to_decimal(c.get("initialFractionalValue")) or dec
                    if name == "1":
                        odds.home_win = dec
                        odds.initial_home_win = init_dec
                        if init_dec and dec and init_dec > 0:
                            odds.drop_pct_home = round(((init_dec - dec) / init_dec) * 100.0, 1)
                    elif name == "X":
                        odds.draw = dec
                        odds.initial_draw = init_dec
                        if init_dec and dec and init_dec > 0:
                            odds.drop_pct_draw = round(((init_dec - dec) / init_dec) * 100.0, 1)
                    elif name == "2":
                        odds.away_win = dec
                        odds.initial_away_win = init_dec
                        if init_dec and dec and init_dec > 0:
                            odds.drop_pct_away = round(((init_dec - dec) / init_dec) * 100.0, 1)

            # 2. Goles Totales (Over / Under 1.5, 2.5, 3.5)
            elif "match goals" in market_name or "total goals" in market_name:
                for c in choices:
                    name = (c.get("name") or "").lower()
                    dec = _fractional_to_decimal(c.get("fractionalValue"))
                    init_dec = _fractional_to_decimal(c.get("initialFractionalValue")) or dec
                    if choice_grp == "1.5":
                        if "over" in name:
                            odds.over_1_5 = dec
                        elif "under" in name:
                            odds.under_1_5 = dec
                    elif choice_grp == "2.5":
                        if "over" in name:
                            odds.over_2_5 = dec
                            odds.initial_over_2_5 = init_dec
                            if init_dec and dec and init_dec > 0:
                                odds.drop_pct_over_2_5 = round(((init_dec - dec) / init_dec) * 100.0, 1)
                        elif "under" in name:
                            odds.under_2_5 = dec
                    elif choice_grp == "3.5":
                        if "over" in name:
                            odds.over_3_5 = dec
                        elif "under" in name:
                            odds.under_3_5 = dec

            # 3. Ambos Equipos Anotan (BTTS)
            elif "both teams to score" in market_name or market_name == "btts":
                for c in choices:
                    name = (c.get("name") or "").lower()
                    dec = _fractional_to_decimal(c.get("fractionalValue"))
                    if "yes" in name or "si" in name or name == "y":
                        odds.btts_yes = dec
                    elif "no" in name or name == "n":
                        odds.btts_no = dec

            # 4. Doble Oportunidad
            elif "double chance" in market_name:
                for c in choices:
                    name = (c.get("name") or "").lower().replace(" ", "").replace("/", "")
                    dec = _fractional_to_decimal(c.get("fractionalValue"))
                    if "1x" in name:
                        odds.dc_1x = dec
                    elif "x2" in name:
                        odds.dc_x2 = dec
                    elif "12" in name:
                        odds.dc_12 = dec

        drops = [
            ("Local (1)", odds.drop_pct_home, odds.initial_home_win, odds.home_win),
            ("Empate (X)", odds.drop_pct_draw, odds.initial_draw, odds.draw),
            ("Visitante (2)", odds.drop_pct_away, odds.initial_away_win, odds.away_win),
        ]
        valid_drops = [d for d in drops if 6.5 <= d[1] <= 45.0 and d[2] and d[3]]
        if valid_drops:
            best_drop = max(valid_drops, key=lambda x: x[1])
            odds.smart_money_side = (
                f"{best_drop[0]} cayó un -{best_drop[1]:.1f}% "
                f"(Apertura {best_drop[2]:.2f} ➔ Actual {best_drop[3]:.2f})"
            )

        if odds.home_win and odds.draw and odds.away_win:
            inv_sum = (1.0 / odds.home_win) + (1.0 / odds.draw) + (1.0 / odds.away_win)
            if inv_sum > 0:
                odds.implied_prob_home = round((1.0 / odds.home_win) / inv_sum, 4)
                odds.implied_prob_draw = round((1.0 / odds.draw) / inv_sum, 4)
                odds.implied_prob_away = round((1.0 / odds.away_win) / inv_sum, 4)

        odds.raw_markets = raw_mkts
        return odds

    def get_match_h2h(
        self,
        event_id: int,
        custom_id: Optional[str] = None,
        home_team_id: Optional[int] = None,
        away_team_id: Optional[int] = None,
        reference_timestamp: Optional[int] = None,
    ) -> Optional[H2HSummarySchema]:
        """
        Extrae el historial de enfrentamientos directos (Head-to-Head):
        1) Resumen histórico general (/event/{id}/h2h)
        2) Últimos enfrentamientos directos partido a partido (/event/{customId}/h2h/events)
           con goles, BTTS, córners, tarjetas y atajadas en sus cruces recientes.
        """
        url = f"{SOFASCORE_API_BASE}/event/{event_id}/h2h"
        data = self.get_json(url, ttl_hours=12.0)

        h_wins, draws, a_wins = 0, 0, 0
        if data and "teamDuel" in data and data["teamDuel"]:
            duel = data["teamDuel"]
            h_wins = int(duel.get("homeWins", 0))
            draws = int(duel.get("draws", 0))
            a_wins = int(duel.get("awayWins", 0))

        if not custom_id:
            _, custom_id = self._get_event_referee_and_custom_id(event_id)

        recent_items: List[H2HMatchItemSchema] = []
        if custom_id:
            ev_url = f"{SOFASCORE_API_BASE}/event/{custom_id}/h2h/events"
            ev_data = self.get_json(ev_url, ttl_hours=12.0)
            if ev_data and "events" in ev_data:
                finished_h2h = []
                for raw_ev in ev_data["events"]:
                    eid_item = int(raw_ev.get("id", 0))
                    if eid_item == event_id:
                        continue
                    st_type = raw_ev.get("status", {}).get("type")
                    ts_item = int(raw_ev.get("startTimestamp", 0))
                    if st_type != "finished":
                        continue
                    if reference_timestamp and ts_item >= reference_timestamp - 600:
                        continue
                    # Filtrar enfrentamientos de los últimos ~4.5 años (142M seg) para evitar planteles obsoletos
                    if reference_timestamp and (reference_timestamp - ts_item) > 142_000_000:
                        continue
                    finished_h2h.append(raw_ev)

                finished_h2h.sort(key=lambda x: int(x.get("startTimestamp", 0)), reverse=True)
                for idx, raw_ev in enumerate(finished_h2h[:6]):
                    parsed_ev = self._parse_event_summary(raw_ev)
                    if not parsed_ev or parsed_ev.home_score is None or parsed_ev.away_score is None:
                        continue

                    ev_stats: Optional[MatchStatsSchema] = None
                    if idx < 3:
                        ev_stats = self.get_match_statistics(parsed_ev.event_id)

                    crn_tot = (
                        (ev_stats.corner_kicks_home + ev_stats.corner_kicks_away)
                        if (ev_stats and ev_stats.corner_kicks_home is not None and ev_stats.corner_kicks_away is not None)
                        else None
                    )
                    yc_tot = (
                        (ev_stats.yellow_cards_home + ev_stats.yellow_cards_away)
                        if (ev_stats and ev_stats.yellow_cards_home is not None and ev_stats.yellow_cards_away is not None)
                        else None
                    )
                    sot_tot = (
                        (ev_stats.shots_on_target_home + ev_stats.shots_on_target_away)
                        if (ev_stats and ev_stats.shots_on_target_home is not None and ev_stats.shots_on_target_away is not None)
                        else None
                    )
                    sv_tot = (
                        (ev_stats.saves_home + ev_stats.saves_away)
                        if (ev_stats and ev_stats.saves_home is not None and ev_stats.saves_away is not None)
                        else None
                    )

                    recent_items.append(
                        H2HMatchItemSchema(
                            event_id=parsed_ev.event_id,
                            date_str=parsed_ev.start_time_utc.strftime("%Y-%m-%d"),
                            timestamp=parsed_ev.start_timestamp,
                            tournament_name=parsed_ev.tournament.name,
                            home_team_id=parsed_ev.home_team.id,
                            home_team_name=parsed_ev.home_team.name,
                            away_team_id=parsed_ev.away_team.id,
                            away_team_name=parsed_ev.away_team.name,
                            home_score=parsed_ev.home_score,
                            away_score=parsed_ev.away_score,
                            xg_home=ev_stats.expected_goals_home if ev_stats else None,
                            xg_away=ev_stats.expected_goals_away if ev_stats else None,
                            corners_total=crn_tot,
                            yellow_cards_total=yc_tot,
                            sot_total=sot_tot,
                            saves_total=sv_tot,
                        )
                    )

        if (h_wins + draws + a_wins) == 0 and not recent_items:
            return None

        # Si teamDuel no trajo conteo o queremos enriquecer con las métricas de recent_items
        h2h_summary = H2HSummarySchema(
            team_wins_home=h_wins,
            draws=draws,
            team_wins_away=a_wins,
            total_matches=max(h_wins + draws + a_wins, len(recent_items)),
            recent_matches_analyzed=len(recent_items),
            recent_h2h_events=recent_items,
        )

        if recent_items:
            w_goals_h, w_goals_a, w_tot = 0.0, 0.0, 0.0
            btts_hits, o25_hits = 0, 0
            crn_vals, crd_vals, sot_vals, sv_vals = [], [], [], []

            for i, it in enumerate(recent_items):
                decay = math.exp(-0.18 * i)
                same_host = home_team_id is not None and it.home_team_id == home_team_id
                venue_w = 1.35 if same_host else 0.85
                w = decay * venue_w

                if home_team_id is not None:
                    gh_for_today_home = it.home_score if it.home_team_id == home_team_id else it.away_score
                    ga_for_today_away = it.away_score if it.home_team_id == home_team_id else it.home_score
                    if it.xg_home is not None and it.xg_away is not None:
                        xgh_today = it.xg_home if it.home_team_id == home_team_id else it.xg_away
                        xga_today = it.xg_away if it.home_team_id == home_team_id else it.xg_home
                        gh_for_today_home = (0.55 * xgh_today) + (0.45 * gh_for_today_home)
                        ga_for_today_away = (0.55 * xga_today) + (0.45 * ga_for_today_away)
                else:
                    gh_for_today_home = float(it.home_score)
                    ga_for_today_away = float(it.away_score)

                w_goals_h += gh_for_today_home * w
                w_goals_a += ga_for_today_away * w
                w_tot += w

                if it.home_score >= 1 and it.away_score >= 1:
                    btts_hits += 1
                if (it.home_score + it.away_score) >= 3:
                    o25_hits += 1
                if it.corners_total is not None:
                    crn_vals.append(it.corners_total)
                if it.yellow_cards_total is not None:
                    crd_vals.append(it.yellow_cards_total)
                if it.sot_total is not None:
                    sot_vals.append(it.sot_total)
                if it.saves_total is not None:
                    sv_vals.append(it.saves_total)

            n_rec = len(recent_items)
            h2h_summary.h2h_home_goals_avg = round(w_goals_h / max(0.01, w_tot), 2)
            h2h_summary.h2h_away_goals_avg = round(w_goals_a / max(0.01, w_tot), 2)
            h2h_summary.h2h_avg_goals = round(
                h2h_summary.h2h_home_goals_avg + h2h_summary.h2h_away_goals_avg, 2
            )
            h2h_summary.h2h_btts_pct = round((btts_hits / n_rec) * 100.0, 1)
            h2h_summary.h2h_over_2_5_pct = round((o25_hits / n_rec) * 100.0, 1)
            if crn_vals:
                h2h_summary.h2h_corners_avg = round(sum(crn_vals) / len(crn_vals), 2)
            if crd_vals:
                h2h_summary.h2h_cards_avg = round(sum(crd_vals) / len(crd_vals), 2)
            if sot_vals:
                h2h_summary.h2h_sot_avg = round(sum(sot_vals) / len(sot_vals), 2)
            if sv_vals:
                h2h_summary.h2h_saves_avg = round(sum(sv_vals) / len(sv_vals), 2)

        return h2h_summary

    def get_match_streaks(self, event_id: int) -> List[Dict[str, str]]:
        """Obtiene las rachas estadísticas previas al partido (goles, córners, tarjetas, victorias)."""
        url = f"{SOFASCORE_API_BASE}/event/{event_id}/team-streaks"
        data = self.get_json(url, ttl_hours=6.0)
        if not data:
            return []

        streaks: List[Dict[str, str]] = []
        for section in ("general", "head2head"):
            for item in data.get(section, []):
                streaks.append(
                    {
                        "category": section,
                        "name": str(item.get("name", "")),
                        "value": str(item.get("value", "")),
                        "team": str(item.get("team", "")),
                    }
                )
        return streaks

    def get_match_lineups(self, event_id: int) -> Optional[Dict[str, TeamLineupSchema]]:
        """Obtiene alineaciones confirmadas/probables, desglose Opta (seasonRatingBreakdown) de arqueros y jugadores, y bajas."""
        url = f"{SOFASCORE_API_BASE}/event/{event_id}/lineups"
        data = self.get_json(url, ttl_hours=1.0)
        if not data:
            return None

        confirmed = bool(data.get("confirmed", False))
        result: Dict[str, TeamLineupSchema] = {}

        for side in ("home", "away"):
            side_data = data.get(side, {})
            if not side_data:
                continue

            players: List[PlayerLineupSchema] = []
            for p_entry in side_data.get("players", []):
                p_info = p_entry.get("player", {})
                stats_info = p_entry.get("statistics", {}) or {}
                breakdown = p_entry.get("seasonRatingBreakdown", {}) or {}
                if not p_info.get("id"):
                    continue

                gk_bd = breakdown.get("goalkeeping", {}) or {}
                shot_bd = breakdown.get("shot", {}) or {}
                pass_bd = breakdown.get("pass", {}) or {}
                drib_bd = breakdown.get("dribble", {}) or {}

                avg_r = p_entry.get("avgRating")
                gk_spg = gk_bd.get("savesPerGame")
                gk_gp = gk_bd.get("goalsPrevented")
                m_saves = stats_info.get("saves")

                players.append(
                    PlayerLineupSchema(
                        player_id=int(p_info["id"]),
                        name=p_info.get("name", "Unknown"),
                        position=p_entry.get("position") or p_info.get("position"),
                        jersey_number=str(p_entry.get("shirtNumber", "")),
                        substitute=bool(p_entry.get("substitute", False)),
                        rating=stats_info.get("rating"),
                        avg_season_rating=float(avg_r) if avg_r is not None else None,
                        gk_saves_per_game=round(float(gk_spg), 2) if gk_spg is not None else None,
                        gk_goals_prevented=round(float(gk_gp), 2) if gk_gp is not None else None,
                        match_saves=int(m_saves) if m_saves is not None else None,
                        season_shots_per_game=round(float(shot_bd["totalShotsPerGame"]), 2)
                        if shot_bd.get("totalShotsPerGame") is not None
                        else None,
                        season_sot_per_game=round(float(shot_bd["shotsOnTargetPerGame"]), 2)
                        if shot_bd.get("shotsOnTargetPerGame") is not None
                        else None,
                        season_xg=round(float(shot_bd["expectedGoals"]), 2)
                        if shot_bd.get("expectedGoals") is not None
                        else None,
                        season_xgot=round(float(shot_bd["expectedGoalsOnTarget"]), 2)
                        if shot_bd.get("expectedGoalsOnTarget") is not None
                        else None,
                        season_key_passes_per_game=round(float(pass_bd["keyPassesPerGame"]), 2)
                        if pass_bd.get("keyPassesPerGame") is not None
                        else None,
                        season_touches_per_game=round(float(drib_bd["touchesPerGame"]), 1)
                        if drib_bd.get("touchesPerGame") is not None
                        else None,
                    )
                )

            missing: List[str] = []
            for m_entry in side_data.get("missingPlayers", []):
                p_name = m_entry.get("player", {}).get("name")
                if p_name:
                    missing.append(p_name)

            result[side] = TeamLineupSchema(
                formation=side_data.get("formation"),
                confirmed=confirmed,
                players=players,
                missing_players=missing,
            )

        return result if result else None

    def _fetch_espn_scheduled_events(self, date_str: str, ut_id: int) -> List[MatchDetailSchema]:
        """Obtiene partidos programados desde ESPN Public API (ej. UEFA Women's Champions League 692, Nations League 10783) sin bloqueo 403."""
        espn_slug = TOURNAMENT_ESPN_SLUGS.get(ut_id)
        if not espn_slug:
            return []
        ymd = date_str.replace("-", "")
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_slug}/scoreboard?dates={ymd}"
        data = self.get_json(url, ttl_hours=1.0)
        if not data or "events" not in data:
            return []

        t_meta = TOP_TOURNAMENTS.get(ut_id, {"name": "Torneo Internacional", "country": "World"})
        tz_local = timezone(timedelta(hours=-5))
        results: List[MatchDetailSchema] = []

        for ev in data.get("events", []):
            try:
                ev_id = int(ev.get("id", 0))
                if not ev_id:
                    continue
                iso_date = ev.get("date", "")
                dt_utc = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
                if dt_utc.astimezone(tz_local).strftime("%Y-%m-%d") != date_str:
                    continue

                comp = (ev.get("competitions") or [{}])[0]
                competitors = comp.get("competitors", [])
                home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home_c or not away_c:
                    continue

                ht = home_c.get("team", {})
                at = away_c.get("team", {})
                h_id = int(ht.get("id", 0))
                a_id = int(at.get("id", 0))
                SofascoreScraper._espn_team_meta[h_id] = (espn_slug, str(h_id))
                SofascoreScraper._espn_team_meta[a_id] = (espn_slug, str(a_id))

                st_state = comp.get("status", {}).get("type", {}).get("state", "pre")
                st_type = "finished" if st_state == "post" else ("inprogress" if st_state == "in" else "notstarted")

                hs_val = home_c.get("score")
                as_val = away_c.get("score")
                h_score = int(float(hs_val)) if (st_type != "notstarted" and hs_val is not None and str(hs_val).replace(".", "").isdigit()) else None
                a_score = int(float(as_val)) if (st_type != "notstarted" and as_val is not None and str(as_val).replace(".", "").isdigit()) else None

                results.append(
                    MatchDetailSchema(
                        event_id=ev_id,
                        custom_id=f"espn_{ev_id}",
                        slug=ev.get("shortName", f"match-{ev_id}").lower().replace(" ", "-"),
                        start_timestamp=int(dt_utc.timestamp()),
                        start_time_utc=dt_utc,
                        status_type=st_type,
                        status_description=comp.get("status", {}).get("type", {}).get("shortDetail"),
                        tournament=TournamentSchema(
                            id=ut_id,
                            unique_id=ut_id,
                            name=t_meta["name"],
                            country=t_meta["country"],
                        ),
                        season_id=int(ev.get("season", {}).get("year", 2026)),
                        home_team=TeamSchema(
                            id=h_id,
                            name=ht.get("displayName") or ht.get("name", "Home"),
                            short_name=ht.get("shortDisplayName") or ht.get("name"),
                        ),
                        away_team=TeamSchema(
                            id=a_id,
                            name=at.get("displayName") or at.get("name", "Away"),
                            short_name=at.get("shortDisplayName") or at.get("name"),
                        ),
                        home_score=h_score,
                        away_score=a_score,
                    )
                )
            except Exception:
                continue
        return results

    def _fetch_espn_team_recent_matches(
        self,
        team_id: int,
        limit: int = 8,
        with_stats: bool = True,
        max_stat_matches: int = 5,
    ) -> List[MatchDetailSchema]:
        """Recupera los últimos partidos jugados y sus estadísticas Opta desde ESPN Public API."""
        meta = SofascoreScraper._espn_team_meta.get(team_id)
        slugs_to_try = [meta[0]] if meta else ["uefa.wchampions", "uefa.nations", "uefa.champions"]
        espn_tid = meta[1] if meta else str(team_id)

        raw_events: List[Dict[str, Any]] = []
        used_slug = slugs_to_try[0]
        for slug in slugs_to_try:
            s_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams/{espn_tid}/schedule"
            s_data = self.get_json(s_url, ttl_hours=6.0)
            if s_data and s_data.get("events"):
                raw_events = s_data["events"]
                used_slug = slug
                break

        if not raw_events:
            return []

        finished = []
        for e in raw_events:
            comp = (e.get("competitions") or [{}])[0]
            if comp.get("status", {}).get("type", {}).get("state") == "post":
                finished.append(e)

        finished.sort(key=lambda x: x.get("date", ""), reverse=True)
        results: List[MatchDetailSchema] = []

        for idx, ev in enumerate(finished[:limit]):
            try:
                ev_id = int(ev.get("id", 0))
                dt_utc = datetime.fromisoformat(ev.get("date", "").replace("Z", "+00:00"))
                comp = (ev.get("competitions") or [{}])[0]
                comps = comp.get("competitors", [])
                hc = next((c for c in comps if c.get("homeAway") == "home"), None)
                ac = next((c for c in comps if c.get("homeAway") == "away"), None)
                if not hc or not ac:
                    continue

                def _extract_score(c_obj: Dict[str, Any]) -> int:
                    sc = c_obj.get("score")
                    if isinstance(sc, dict):
                        return int(float(sc.get("value", 0)))
                    return int(float(sc or 0))

                hs = _extract_score(hc)
                as_ = _extract_score(ac)
                ht = hc.get("team", {})
                at = ac.get("team", {})

                m_obj = MatchDetailSchema(
                    event_id=ev_id,
                    slug=f"espn-{ev_id}",
                    start_timestamp=int(dt_utc.timestamp()),
                    start_time_utc=dt_utc,
                    status_type="finished",
                    tournament=TournamentSchema(
                        id=692 if used_slug == "uefa.wchampions" else 10783,
                        unique_id=692 if used_slug == "uefa.wchampions" else 10783,
                        name=ev.get("season", {}).get("displayName", "Torneo"),
                    ),
                    home_team=TeamSchema(id=int(ht.get("id", 0)), name=ht.get("displayName", "Home")),
                    away_team=TeamSchema(id=int(at.get("id", 0)), name=at.get("displayName", "Away")),
                    home_score=hs,
                    away_score=as_,
                )

                if with_stats and idx < max_stat_matches:
                    sum_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{used_slug}/summary?event={ev_id}"
                    sum_data = self.get_json(sum_url, ttl_hours=24.0)
                    b_teams = (sum_data or {}).get("boxscore", {}).get("teams", [])
                    if len(b_teams) == 2:
                        h_tid_str = str(ht.get("id"))
                        bt_h = next((bt for bt in b_teams if str(bt.get("team", {}).get("id")) == h_tid_str), b_teams[0])
                        bt_a = next((bt for bt in b_teams if bt is not bt_h), b_teams[1])
                        sh_map = {s.get("name"): _parse_stat_number(s.get("displayValue")) for s in bt_h.get("statistics", [])}
                        sa_map = {s.get("name"): _parse_stat_number(s.get("displayValue")) for s in bt_a.get("statistics", [])}
                        sot_h = int(sh_map.get("shotsOnTarget") or 0)
                        sot_a = int(sa_map.get("shotsOnTarget") or 0)
                        ts_h = int(sh_map.get("totalShots") or max(sot_h, 0))
                        ts_a = int(sa_map.get("totalShots") or max(sot_a, 0))
                        m_obj.stats = MatchStatsSchema(
                            ball_possession_home=sh_map.get("possessionPct"),
                            ball_possession_away=sa_map.get("possessionPct"),
                            total_shots_home=ts_h,
                            total_shots_away=ts_a,
                            shots_on_target_home=sot_h,
                            shots_on_target_away=sot_a,
                            corner_kicks_home=int(sh_map.get("wonCorners") or 0),
                            corner_kicks_away=int(sa_map.get("wonCorners") or 0),
                            fouls_home=int(sh_map.get("foulsCommitted") or 0),
                            fouls_away=int(sa_map.get("foulsCommitted") or 0),
                            yellow_cards_home=int(sh_map.get("yellowCards") or 0),
                            yellow_cards_away=int(sa_map.get("yellowCards") or 0),
                            red_cards_home=int(sh_map.get("redCards") or 0),
                            red_cards_away=int(sa_map.get("redCards") or 0),
                            saves_home=int(sh_map.get("saves") or max(0, sot_a - as_)),
                            saves_away=int(sa_map.get("saves") or max(0, sot_h - hs)),
                            expected_goals_home=round((sot_h * 0.31) + (max(0, ts_h - sot_h) * 0.05), 2),
                            expected_goals_away=round((sot_a * 0.31) + (max(0, ts_a - sot_a) * 0.05), 2),
                        )
                results.append(m_obj)
            except Exception:
                continue
        return results

    def get_team_recent_matches(
        self,
        team_id: int,
        limit: int = 8,
        with_stats: bool = True,
        max_stat_matches: int = 5,
    ) -> List[MatchDetailSchema]:
        """
        Obtiene los últimos N partidos finalizados de un equipo y opcionalmente
        enriquece los más recientes con sus estadísticas Opta reales (xG, xGOT, córners, tiros, atajadas).
        """
        if team_id in SofascoreScraper._espn_team_meta:
            espn_rec = self._fetch_espn_team_recent_matches(team_id, limit, with_stats, max_stat_matches)
            if espn_rec:
                return espn_rec

        url = f"{SOFASCORE_API_BASE}/team/{team_id}/events/last/0"
        data = self.get_json(url, ttl_hours=6.0)
        if not data or "events" not in data:
            return self._fetch_espn_team_recent_matches(team_id, limit, with_stats, max_stat_matches)

        finished_events = [
            e
            for e in data["events"]
            if e.get("status", {}).get("type") == "finished"
        ]
        recent_raw = finished_events[-limit:][::-1]

        results: List[MatchDetailSchema] = []
        for idx, raw in enumerate(recent_raw):
            parsed = self._parse_event_summary(raw)
            if not parsed:
                continue
            if with_stats and idx < max_stat_matches:
                parsed.stats = self.get_match_statistics(parsed.event_id)
            results.append(parsed)
        return results

    def get_tournament_standings_map(
        self, unique_tournament_id: int, season_id: int
    ) -> Dict[int, Dict[str, Any]]:
        """
        Obtiene la tabla de posiciones de la temporada y retorna métricas por team_id
        (posición, puntos, goles, partidos jugados) para ajuste de Rating Elo y Fuerza de Calendario.
        """
        url = (
            f"{SOFASCORE_API_BASE}/unique-tournament/{unique_tournament_id}"
            f"/season/{season_id}/standings/total"
        )
        data = self.get_json(url, ttl_hours=6.0)
        if not data or "standings" not in data or not data["standings"]:
            return {}

        table_map: Dict[int, Dict[str, Any]] = {}
        for group in data["standings"]:
            for row in group.get("rows", []):
                team_id = row.get("team", {}).get("id")
                if not team_id:
                    continue
                matches_played = max(1, int(row.get("matches", 1)))
                points = int(row.get("points", 0))
                scores_for = int(row.get("scoresFor", 0))
                scores_against = int(row.get("scoresAgainst", 0))
                position = int(row.get("position", 10))
                table_map[int(team_id)] = {
                    "position": position,
                    "matches": matches_played,
                    "points": points,
                    "ppg": round(points / matches_played, 2),
                    "gd_per_match": round((scores_for - scores_against) / matches_played, 2),
                }
        return table_map

    @classmethod
    def _get_cached_top_players_for_team(cls, team_id: int) -> Optional[Dict[str, Any]]:
        """Índice perezoso de archivos `topPlayers` en `data/cache` para recuperar líderes por team_id en < 1ms."""
        if not cls._top_players_cache_indexed:
            cls._top_players_cache_indexed = True
            try:
                for cf in CACHE_DIR.glob("*.json"):
                    try:
                        txt = cf.read_text(encoding="utf-8", errors="ignore")
                        if '"topPlayers"' not in txt:
                            continue
                        d = json.loads(txt)
                        tp_obj = d.get("topPlayers", {})
                        for cat_list in tp_obj.values():
                            if isinstance(cat_list, list) and cat_list:
                                tid = cat_list[0].get("team", {}).get("id")
                                if tid:
                                    cls._team_top_players_files[int(tid)] = cf
                                    break
                    except Exception:
                        continue
            except Exception:
                pass

        cf_match = cls._team_top_players_files.get(int(team_id))
        if cf_match and cf_match.exists():
            try:
                return json.loads(cf_match.read_text(encoding="utf-8")).get("topPlayers", {})
            except Exception:
                return None
        return None

    def _populate_players_from_espn_roster(
        self,
        team_id: int,
        team_name: str,
        unique_tournament_id: int,
        players_map: Dict[int, PlayerPropSchema],
        missing_set: set[str],
        starter_ids: set[int],
        sub_ids: set[int],
    ) -> None:
        """
        Respaldo 100% Cloud-Safe vía ESPN Public API (`/teams/{id}/roster`) cuando Sofascore no tiene
        `top-players/overall` para la temporada actual o aplica bloqueo 403.
        Extrae partidos jugados, remates totales, remates al arco, goles, asistencias y tarjetas de la temporada.
        """
        candidate_slugs: List[str] = []
        meta = SofascoreScraper._espn_team_meta.get(team_id)
        if meta and meta[0] not in candidate_slugs:
            candidate_slugs.append(meta[0])
        primary_slug = TOURNAMENT_ESPN_SLUGS.get(unique_tournament_id)
        if primary_slug and primary_slug not in candidate_slugs:
            candidate_slugs.append(primary_slug)
        for fallback_slug in (
            "uefa.nations",
            "uefa.wchampions",
            "uefa.champions",
            "uefa.europa",
            "eng.1",
            "esp.1",
            "ita.1",
            "ger.1",
            "fra.1",
            "col.1",
            "arg.1",
            "bra.1",
            "por.1",
            "tur.1",
            "concacaf.nations.league",
            "fifa.friendly",
        ):
            if fallback_slug not in candidate_slugs:
                candidate_slugs.append(fallback_slug)

        target_norm = _normalize_team_label(team_name)
        espn_tid: Optional[str] = meta[1] if meta else None
        matched_slug: Optional[str] = meta[0] if meta else None

        if not espn_tid:
            for slug in candidate_slugs[:6]:
                t_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams?limit=120"
                t_data = self.get_json(t_url, ttl_hours=24.0)
                if not t_data:
                    continue
                teams_list = [
                    t.get("team", {})
                    for s in t_data.get("sports", [])
                    for l in s.get("leagues", [])
                    for t in l.get("teams", [])
                ]
                for t_obj in teams_list:
                    cand_names = [
                        t_obj.get("displayName", ""),
                        t_obj.get("name", ""),
                        t_obj.get("shortDisplayName", ""),
                        t_obj.get("location", ""),
                    ]
                    norms = {_normalize_team_label(cn) for cn in cand_names if cn}
                    if target_norm in norms or any(
                        target_norm and cn_norm and (target_norm in cn_norm or cn_norm in target_norm)
                        for cn_norm in norms
                    ):
                        espn_tid = str(t_obj.get("id"))
                        matched_slug = slug
                        SofascoreScraper._espn_team_meta[team_id] = (slug, espn_tid)
                        break
                if espn_tid:
                    break

        if not espn_tid or not matched_slug:
            return

        r_url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{matched_slug}/teams/{espn_tid}/roster"
        r_data = self.get_json(r_url, ttl_hours=12.0)
        athletes = (r_data or {}).get("athletes", [])
        if not athletes:
            return

        fallback_attackers: List[tuple[int, str, str, int]] = []
        for a in athletes:
            try:
                pid = int(a.get("id", 0))
                if not pid:
                    continue
                pname = a.get("displayName") or a.get("fullName") or "Jugador"
                pos_abbr = (a.get("position") or {}).get("abbreviation") or "M"
                if pos_abbr == "G":
                    continue

                cats = a.get("statistics", {}).get("splits", {}).get("categories", [])
                s_map: Dict[str, float] = {}
                for c in cats:
                    for st_item in c.get("stats", []):
                        val = st_item.get("value")
                        if isinstance(val, (int, float)):
                            s_map[st_item.get("name", "")] = float(val)

                apps = int(s_map.get("appearances") or 0)
                shots = int(s_map.get("totalShots") or 0)
                sot = int(s_map.get("shotsOnTarget") or 0)
                goals = int(s_map.get("totalGoals") or 0)
                assists = int(s_map.get("goalAssists") or 0)
                yc = int(s_map.get("yellowCards") or 0)

                if pos_abbr in ("F", "M"):
                    fallback_attackers.append((pid, pname, pos_abbr, max(1, apps)))

                if shots <= 0 and sot <= 0 and goals <= 0:
                    continue

                eff_apps = max(1, apps)
                eff_shots = max(shots, sot, goals)
                eff_sot = max(sot, goals)
                est_xg = round((eff_sot * 0.30) + (max(0, eff_shots - eff_sot) * 0.06) + (goals * 0.20), 2)
                is_st = True if pid in starter_ids else (False if pid in sub_ids else None)

                players_map[pid] = PlayerPropSchema(
                    player_id=pid,
                    name=pname,
                    position=pos_abbr,
                    team_name=team_name,
                    appearances=eff_apps,
                    goals=goals,
                    expected_goals=est_xg,
                    shots_on_target=eff_sot,
                    total_shots=eff_shots,
                    yellow_cards=yc,
                    key_passes_per_match=round((assists * 1.8) / eff_apps, 2) if assists > 0 else 0.0,
                    is_missing=pname.lower() in missing_set,
                    is_starter=is_st,
                )
            except Exception:
                continue

        # Si el torneo está en Jornada 1 y ningún jugador acumula tiros aún, proyectar delanteros/mediocampistas ofensivos del plantel
        if not players_map and fallback_attackers:
            fallback_attackers.sort(key=lambda x: (x[2] == "F", x[3]), reverse=True)
            for pid, pname, pos_abbr, eff_apps in fallback_attackers[:8]:
                base_sot = 1.15 if pos_abbr == "F" else 0.65
                base_shots = 2.60 if pos_abbr == "F" else 1.55
                base_xg = 0.34 if pos_abbr == "F" else 0.16
                players_map[pid] = PlayerPropSchema(
                    player_id=pid,
                    name=pname,
                    position=pos_abbr,
                    team_name=team_name,
                    appearances=eff_apps,
                    shots_on_target=max(1, int(round(base_sot * eff_apps))),
                    total_shots=max(2, int(round(base_shots * eff_apps))),
                    expected_goals=round(base_xg * eff_apps, 2),
                    sot_per_match=base_sot,
                    shots_per_match=base_shots,
                    xg_per_match=base_xg,
                    is_missing=pname.lower() in missing_set,
                    is_starter=True if pid in starter_ids else None,
                )

    def get_team_top_players(
        self,
        team_id: int,
        team_name: str,
        unique_tournament_id: int,
        season_id: int,
        missing_names: Optional[List[str]] = None,
        lineup: Optional[TeamLineupSchema] = None,
        opp_sot_conceded_avg: Optional[float] = None,
    ) -> List[PlayerPropSchema]:
        """
        Obtiene los líderes estadísticos de un equipo en la temporada actual (Opta),
        los cruza con la alineación del partido (Titular vs Suplente y seasonRatingBreakdown)
        y ajusta sus probabilidades según la permisividad defensiva del rival.
        Incluye respaldo automático de caché local e integración con ESPN Roster API.
        """
        players_map: Dict[int, PlayerPropSchema] = {}
        missing_set = {n.lower() for n in (missing_names or [])}

        starter_ids: set[int] = set()
        sub_ids: set[int] = set()
        lineup_map: Dict[int, PlayerLineupSchema] = {}
        if lineup and lineup.players:
            for lp in lineup.players:
                lineup_map[lp.player_id] = lp
                if lp.substitute:
                    sub_ids.add(lp.player_id)
                else:
                    starter_ids.add(lp.player_id)

        seasons_list: List[Dict[str, Any]] = []
        if unique_tournament_id and not season_id and not self.is_sofascore_blocked():
            try:
                seasons_url = f"{SOFASCORE_API_BASE}/unique-tournament/{unique_tournament_id}/seasons"
                seasons_data = self.get_json(seasons_url, ttl_hours=24.0)
                if seasons_data and "seasons" in seasons_data and seasons_data["seasons"]:
                    seasons_list = seasons_data["seasons"]
                    season_id = seasons_list[0].get("id")
            except Exception:
                pass

        if unique_tournament_id and season_id:
            url = (
                f"{SOFASCORE_API_BASE}/team/{team_id}/unique-tournament/{unique_tournament_id}"
                f"/season/{season_id}/top-players/overall"
            )
            data = self.get_json(url, ttl_hours=12.0)
        else:
            data = None

        tp = data.get("topPlayers", {}) if data else {}

        # Si la temporada nueva aún no registra estadísticas individuales en Sofascore, probar temporada previa o caché local
        if not tp and unique_tournament_id and not self.is_sofascore_blocked():
            try:
                if not seasons_list:
                    seasons_url = f"{SOFASCORE_API_BASE}/unique-tournament/{unique_tournament_id}/seasons"
                    seasons_data = self.get_json(seasons_url, ttl_hours=24.0)
                    seasons_list = (seasons_data or {}).get("seasons", [])
                for s_item in seasons_list[:3]:
                    alt_sid = s_item.get("id")
                    if alt_sid and alt_sid != season_id:
                        alt_url = (
                            f"{SOFASCORE_API_BASE}/team/{team_id}/unique-tournament/{unique_tournament_id}"
                            f"/season/{alt_sid}/top-players/overall"
                        )
                        alt_data = self.get_json(alt_url, ttl_hours=12.0)
                        if alt_data and alt_data.get("topPlayers"):
                            tp = alt_data["topPlayers"]
                            break
            except Exception:
                pass

        if not tp:
            cached_tp = self._get_cached_top_players_for_team(team_id)
            if cached_tp:
                tp = cached_tp

        def _get_or_create(entry: Dict[str, Any]) -> Optional[PlayerPropSchema]:
            p_info = entry.get("player", {})
            pid = p_info.get("id")
            if not pid:
                return None
            pid = int(pid)
            if pid not in players_map:
                pname = p_info.get("name", "Unknown")
                apps = max(1, int(entry.get("statistics", {}).get("appearances", 1)))
                is_st = True if pid in starter_ids else (False if pid in sub_ids else None)
                players_map[pid] = PlayerPropSchema(
                    player_id=pid,
                    name=pname,
                    position=p_info.get("position"),
                    team_name=team_name,
                    appearances=apps,
                    is_missing=pname.lower() in missing_set,
                    is_starter=is_st,
                )
            else:
                apps = int(entry.get("statistics", {}).get("appearances", 0))
                if apps > players_map[pid].appearances:
                    players_map[pid].appearances = apps
            return players_map[pid]

        for item in tp.get("shotsOnTarget", []):
            p = _get_or_create(item)
            if p:
                p.shots_on_target = int(item.get("statistics", {}).get("shotsOnTarget", 0))

        for item in tp.get("totalShots", []):
            p = _get_or_create(item)
            if p:
                p.total_shots = int(item.get("statistics", {}).get("totalShots", 0))

        for item in tp.get("goals", []):
            p = _get_or_create(item)
            if p:
                p.goals = int(item.get("statistics", {}).get("goals", 0))

        for item in tp.get("expectedGoals", []):
            p = _get_or_create(item)
            if p:
                p.expected_goals = round(
                    float(item.get("statistics", {}).get("expectedGoals", 0.0)), 2
                )

        for item in tp.get("keyPasses", []):
            p = _get_or_create(item)
            if p:
                kp_tot = float(item.get("statistics", {}).get("keyPasses", 0.0))
                p.key_passes_per_match = round(kp_tot / max(1, p.appearances), 2)

        for item in tp.get("yellowCards", []):
            p = _get_or_create(item)
            if p:
                p.yellow_cards = int(item.get("statistics", {}).get("yellowCards", 0))

        for item in tp.get("rating", []):
            p = _get_or_create(item)
            if p and item.get("statistics", {}).get("rating"):
                p.rating = round(float(item["statistics"]["rating"]), 2)

        # Integrar titulares de la alineación que tengan métricas Opta en seasonRatingBreakdown
        for pid, lp in lineup_map.items():
            if lp.position == "G":
                continue
            if pid not in players_map and (lp.season_sot_per_game or lp.season_shots_per_game):
                players_map[pid] = PlayerPropSchema(
                    player_id=pid,
                    name=lp.name,
                    position=lp.position,
                    team_name=team_name,
                    appearances=10,
                    rating=lp.avg_season_rating,
                    sot_per_match=lp.season_sot_per_game or 0.0,
                    shots_per_match=lp.season_shots_per_game or 0.0,
                    xg_per_match=lp.season_xg or 0.0,
                    xgot_per_match=lp.season_xgot or 0.0,
                    key_passes_per_match=lp.season_key_passes_per_game or 0.0,
                    touches_per_match=lp.season_touches_per_game or 0.0,
                    is_missing=lp.name.lower() in missing_set,
                    is_starter=not lp.substitute,
                )

        # Si Sofascore no tenía estadísticas de jugadores para este torneo/temporada, extraer plantel y estadísticas desde ESPN API
        if not players_map:
            self._populate_players_from_espn_roster(
                team_id=team_id,
                team_name=team_name,
                unique_tournament_id=unique_tournament_id,
                players_map=players_map,
                missing_set=missing_set,
                starter_ids=starter_ids,
                sub_ids=sub_ids,
            )

        # Factor de permisividad defensiva del rival (promedio estándar = 4.2 SoT concedidos/P)
        opp_factor = 1.0
        if opp_sot_conceded_avg is not None and opp_sot_conceded_avg > 0:
            opp_factor = max(0.82, min(1.22, opp_sot_conceded_avg / 4.20))

        result_list: List[PlayerPropSchema] = []
        for pid, p in players_map.items():
            apps = max(1, p.appearances)
            lp_info = lineup_map.get(pid)

            if p.shots_on_target > 0 or p.total_shots > 0:
                effective_total_shots = max(p.total_shots, int(p.shots_on_target * 2.2))
                p.total_shots = effective_total_shots
                p.sot_per_match = round(p.shots_on_target / apps, 2)
                p.shots_per_match = round(p.total_shots / apps, 2)
                effective_xg = p.expected_goals if p.expected_goals > 0 else (p.goals * 0.82)
                p.xg_per_match = round(effective_xg / apps, 2)
                p.cards_per_match = round(p.yellow_cards / apps, 2)

            if lp_info:
                if lp_info.season_xgot is not None:
                    p.xgot_per_match = lp_info.season_xgot
                if lp_info.season_key_passes_per_game is not None and p.key_passes_per_match == 0:
                    p.key_passes_per_match = lp_info.season_key_passes_per_game
                if lp_info.season_touches_per_game is not None:
                    p.touches_per_match = lp_info.season_touches_per_game
                if lp_info.avg_season_rating and not p.rating:
                    p.rating = round(lp_info.avg_season_rating, 2)
                # Mezclar con el promedio Opta por partido de seasonRatingBreakdown si existe
                if lp_info.season_sot_per_game is not None and p.sot_per_match > 0:
                    p.sot_per_match = round((0.65 * p.sot_per_match) + (0.35 * lp_info.season_sot_per_game), 2)
                elif lp_info.season_sot_per_game is not None and p.sot_per_match == 0:
                    p.sot_per_match = lp_info.season_sot_per_game
                if lp_info.season_shots_per_game is not None and p.shots_per_match == 0:
                    p.shots_per_match = lp_info.season_shots_per_game

            if p.xgot_per_match == 0 and p.xg_per_match > 0:
                p.xgot_per_match = round(p.xg_per_match * 0.95, 2)

            # Ajuste por minutos esperados según si hoy es Titular (🟢) o Suplente (🟡)
            role_mult = 1.08 if p.is_starter is True else (0.52 if p.is_starter is False else 1.0)
            adj_sot_lam = max(0.01, p.sot_per_match * role_mult * opp_factor)
            adj_shots_lam = max(0.02, p.shots_per_match * role_mult * opp_factor)
            adj_xg_lam = max(0.01, ((0.60 * p.xg_per_match) + (0.40 * p.xgot_per_match)) * role_mult * opp_factor)

            p.prob_1plus_sot = round((1.0 - math.exp(-adj_sot_lam)) * 100, 1)
            p.prob_2plus_sot = round(
                (1.0 - math.exp(-adj_sot_lam) * (1.0 + adj_sot_lam)) * 100, 1
            )
            p.prob_2plus_shots = round(
                (1.0 - math.exp(-adj_shots_lam) * (1.0 + adj_shots_lam)) * 100, 1
            )
            p.prob_anytime_goal = round((1.0 - math.exp(-adj_xg_lam)) * 100, 1)
            result_list.append(p)

        result_list.sort(
            key=lambda x: (
                not x.is_missing,
                x.is_starter is not False,
                x.prob_1plus_sot,
                x.sot_per_match,
                x.xg_per_match,
            ),
            reverse=True,
        )
        return result_list
