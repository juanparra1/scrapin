"""
Worker CLI de Sincronización Precomputada hacia Supabase (Turbo Cache Feeder).

Permite poblar en segundos la tabla `precomputed_matches` de Supabase desde cualquier entorno
con acceso directo a Sofascore Opta (ej. máquina local o cron job), garantizando que Render Cloud
sirva todos los partidos, cuotas y predicciones en < 0.05 segundos sin sufrir bloqueos HTTP 403.

Uso:
    python -m src.services.sync_worker --days 7 --include-yesterday
"""

import argparse
import concurrent.futures
import logging
from datetime import datetime, timedelta, timezone
from typing import List

from src.analytics.predictor import compute_team_form, predict_match_poisson
from src.models.database import init_db, save_precomputed_analysis
from src.models.schemas import MatchDetailSchema
from src.scrapers.sofascore import SofascoreScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLOMBIA_TZ = timezone(timedelta(hours=-5))


def sync_single_match(scraper: SofascoreScraper, match: MatchDetailSchema, date_str: str) -> int:
    """Enriquece, calcula el modelo Poisson/Dixon-Coles y persiste un partido en Supabase."""
    try:
        is_finished = match.status_type == "finished"
        scraper.enrich_match(
            match,
            include_stats=is_finished,
            include_odds=(not is_finished),
            include_h2h=False,
            include_lineups=False,
            include_streaks=False,
            include_referee=False,
        )
        home_recent = scraper.get_team_recent_matches(match.home_team.id, limit=6, with_stats=False)
        away_recent = scraper.get_team_recent_matches(match.away_team.id, limit=6, with_stats=False)

        home_form = compute_team_form(
            match.home_team.id,
            match.home_team.name,
            home_recent,
            target_venue="home",
            reference_timestamp=match.start_timestamp,
        )
        away_form = compute_team_form(
            match.away_team.id,
            match.away_team.name,
            away_recent,
            target_venue="away",
            reference_timestamp=match.start_timestamp,
        )
        prediction = predict_match_poisson(match, home_form, away_form)

        ok = save_precomputed_analysis(
            event_id=match.event_id,
            match_date=date_str,
            tournament_name=match.tournament.name if match.tournament else "Liga",
            home_team_name=match.home_team.name,
            away_team_name=match.away_team.name,
            status_type=match.status_type,
            match_detail=match.model_dump(),
            prediction=prediction.model_dump(),
            deep_analysis={
                "match": match.model_dump(),
                "prediction": prediction.model_dump(),
                "home_form": home_form.model_dump(),
                "away_form": away_form.model_dump(),
                "home_recent": [x.model_dump() for x in home_recent],
                "away_recent": [x.model_dump() for x in away_recent],
                "home_players": [],
                "away_players": [],
                "standings_map": {},
                "flashscore_id": "N/A",
            },
        )
        return 1 if ok else 0
    except Exception as exc:
        logger.warning("Error sincronizando partido %s (%s vs %s): %s", match.event_id, match.home_team.name, match.away_team.name, exc)
        return 0


def sync_date_range(
    start_date_str: str,
    days_ahead: int = 6,
    include_yesterday: bool = True,
    max_workers: int = 5,
) -> int:
    """Sincroniza y precalcula en paralelo un rango de fechas hacia Supabase."""
    init_db()
    scraper = SofascoreScraper(rate_limit_seconds=0.18, use_cache=True)
    base_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()

    target_dates: List[str] = []
    if include_yesterday:
        target_dates.append((base_dt - timedelta(days=1)).strftime("%Y-%m-%d"))
    for offset in range(days_ahead + 1):
        target_dates.append((base_dt + timedelta(days=offset)).strftime("%Y-%m-%d"))

    total_saved = 0
    for d_str in target_dates:
        matches = scraper.get_scheduled_events(d_str, only_top_tournaments=True)
        logger.info("📅 Fecha %s: %d partidos Élite detectados", d_str, len(matches))
        if not matches:
            continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(sync_single_match, scraper, m, d_str) for m in matches]
            saved_day = sum(f.result() for f in concurrent.futures.as_completed(futures))
        total_saved += saved_day
        logger.info("✅ Fecha %s: %d/%d partidos precalculados en Supabase", d_str, saved_day, len(matches))

    return total_saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincronizador Precomputado de Partidos hacia Supabase")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(COLOMBIA_TZ).strftime("%Y-%m-%d"),
        help="Fecha base en formato YYYY-MM-DD (por defecto: hoy en Hora Colombia)",
    )
    parser.add_argument("--days", type=int, default=6, help="Cantidad de días hacia adelante a sincronizar (default: 6)")
    parser.add_argument("--include-yesterday", action="store_true", default=True, help="Incluir el día de ayer para auditoría")
    parser.add_argument("--workers", type=int, default=5, help="Hilos paralelos de procesamiento")
    args = parser.parse_args()

    total = sync_date_range(
        start_date_str=args.date,
        days_ahead=args.days,
        include_yesterday=args.include_yesterday,
        max_workers=args.workers,
    )
    logger.info("🚀 Sincronización completada: %d partidos almacenados en Supabase Turbo Cache.", total)


if __name__ == "__main__":
    main()
