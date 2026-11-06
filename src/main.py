"""Punto de entrada CLI para ejecutar Sofascore + Flashscore + EntityResolver + Motor xG/Elo."""

import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.analytics.entity_resolver import EntityResolver
from src.analytics.predictor import compute_team_form, predict_match_poisson
from src.models.database import init_db, save_team_alias, upsert_match
from src.scrapers.flashscore import FlashscoreScraper
from src.scrapers.sofascore import SofascoreScraper

console = Console(width=165)


def run_pipeline(date_str: str, limit: int = 8, all_leagues: bool = False) -> None:
    """Ejecuta extracción multi-fuente (Sofascore + Flashscore), resolución de entidades y predicción con xG real + Elo."""
    init_db()
    scraper = SofascoreScraper(rate_limit_seconds=0.6)
    fs_scraper = FlashscoreScraper(rate_limit_seconds=0.8)
    resolver = EntityResolver(confidence_threshold=78.0)

    console.print(
        Panel.fit(
            f"[bold cyan]Sistema Multi-Fuente de Scraping & Predicción Deportiva (Sofascore + Flashscore + xG/Elo)[/bold cyan]\n"
            f"Fecha objetivo: [yellow]{date_str}[/yellow] | Filtro Top Ligas: [green]{not all_leagues}[/green]",
            border_style="cyan",
        )
    )

    with console.status("[bold green]1/3 Consultando calendario de Sofascore..."):
        matches = scraper.get_scheduled_events(
            date_str=date_str, only_top_tournaments=not all_leagues
        )

    if not matches and not all_leagues:
        console.print(
            "[yellow]No se encontraron partidos en Top Ligas para esta fecha. Ampliando a todas las ligas...[/yellow]"
        )
        matches = scraper.get_scheduled_events(date_str=date_str, only_top_tournaments=False)

    if not matches:
        console.print("[bold red]No se encontraron partidos para la fecha indicada.[/bold red]")
        return

    # Registrar todos los equipos de la jornada en el EntityResolver
    canonical_teams = []
    for m in matches:
        canonical_teams.append((m.home_team.id, m.home_team.name))
        canonical_teams.append((m.away_team.id, m.away_team.name))
    resolver.register_canonical_teams(canonical_teams)

    # Extraer feed de Flashscore para cruzar entidades
    with console.status("[bold green]2/3 Extrayendo feed en vivo de Flashscore (~AA÷...¬) y unificando equipos..."):
        try:
            target_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            today_dt = datetime.now().date()
            day_offset = max(-7, min(7, (target_dt - today_dt).days))
            fs_matches = fs_scraper.get_daily_matches(day_offset=day_offset)
        except Exception:
            fs_matches = []

    fs_matched_map: Dict[tuple[int, int], str] = {}
    resolved_count = 0
    for fsm in fs_matches:
        res_home = resolver.resolve(fsm.home_team_name)
        res_away = resolver.resolve(fsm.away_team_name)
        if res_home and res_home.is_confident:
            save_team_alias(
                source="flashscore",
                external_id=fsm.home_team_id,
                external_name=fsm.home_team_name,
                canonical_team_id=res_home.canonical_id,
                canonical_name=res_home.canonical_name,
                similarity_score=res_home.similarity_score,
            )
        if res_away and res_away.is_confident:
            save_team_alias(
                source="flashscore",
                external_id=fsm.away_team_id,
                external_name=fsm.away_team_name,
                canonical_team_id=res_away.canonical_id,
                canonical_name=res_away.canonical_name,
                similarity_score=res_away.similarity_score,
            )
        if res_home and res_away and res_home.is_confident and res_away.is_confident:
            fs_matched_map[(res_home.canonical_id, res_away.canonical_id)] = fsm.flashscore_id
            resolved_count += 1

    console.print(
        f"✔ Sofascore: [bold green]{len(matches)}[/bold green] partidos | "
        f"Flashscore Feed: [bold cyan]{len(fs_matches)}[/bold cyan] partidos | "
        f"Unificados por EntityResolver: [bold yellow]{resolved_count}[/bold yellow] cruces exactos.\n"
    )

    selected_matches = matches[:limit]
    standings_cache: Dict[tuple[int, int], Dict[int, dict]] = {}

    summary_table = Table(
        title=f"Predicciones Híbridas: xG Histórico Real + Rating Elo + Dixon-Coles ({date_str})",
        show_lines=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Torneo / Cruce", style="cyan")
    summary_table.add_column("Partido (Local vs Visitante)", style="bold white")
    summary_table.add_column("Estado", justify="center")
    summary_table.add_column("Elo & Forma (5p)", justify="center")
    summary_table.add_column("xG Real (5p)\nLoc / Vis", justify="center")
    summary_table.add_column("xG Proy.\n& Córners", justify="center")
    summary_table.add_column("Prob. 1 / X / 2", justify="center")
    summary_table.add_column("Cuotas 1 / X / 2", justify="center")
    summary_table.add_column("O2.5 / BTTS", justify="center")
    summary_table.add_column("Pronóstico & Value Bet", style="bold green")

    for idx, match in enumerate(selected_matches, start=1):
        console.print(
            f"[dim]({idx}/{len(selected_matches)}) Calculando xG histórico, Elo y modelo para:[/dim] "
            f"[bold]{match.home_team.name} vs {match.away_team.name}[/bold]"
        )

        scraper.enrich_match(
            match,
            include_stats=True,
            include_odds=True,
            include_h2h=True,
            include_lineups=True,
            include_streaks=True,
        )

        # Tabla de posiciones para bonificación de Elo
        ut_id = match.tournament.unique_id or match.tournament.id
        s_id = match.season_id or 0
        if ut_id and s_id and (ut_id, s_id) not in standings_cache:
            standings_cache[(ut_id, s_id)] = scraper.get_tournament_standings_map(ut_id, s_id)
        table_map = standings_cache.get((ut_id, s_id), {})

        # Últimos partidos enriquecidos con estadísticas de xG y córners reales
        home_recent = scraper.get_team_recent_matches(
            match.home_team.id, limit=6, with_stats=True, max_stat_matches=4
        )
        away_recent = scraper.get_team_recent_matches(
            match.away_team.id, limit=6, with_stats=True, max_stat_matches=4
        )

        home_form = compute_team_form(
            match.home_team.id,
            match.home_team.name,
            home_recent,
            standing_info=table_map.get(match.home_team.id),
            target_venue="home",
            reference_timestamp=match.start_timestamp,
            standings_map=table_map,
        )
        away_form = compute_team_form(
            match.away_team.id,
            match.away_team.name,
            away_recent,
            standing_info=table_map.get(match.away_team.id),
            target_venue="away",
            reference_timestamp=match.start_timestamp,
            standings_map=table_map,
        )

        prediction = predict_match_poisson(match, home_form, away_form)
        upsert_match(match)

        if match.status_type == "finished":
            status_str = f"[bold green]FIN ({match.home_score}-{match.away_score})[/bold green]"
        elif match.status_type == "inprogress":
            status_str = f"[bold yellow]VIVO ({match.home_score}-{match.away_score})[/bold yellow]"
        else:
            tz_col = timezone(timedelta(hours=-5))
            col_time = datetime.fromtimestamp(match.start_timestamp, tz=tz_col).strftime("%H:%M")
            status_str = f"[bold cyan]{col_time} COL[/bold cyan]"

        odds_str = (
            f"{match.odds.home_win:.2f} / {match.odds.draw:.2f} / {match.odds.away_win:.2f}"
            if (match.odds and match.odds.home_win and match.odds.draw and match.odds.away_win)
            else "Sin cuotas"
        )

        fs_code = fs_matched_map.get((match.home_team.id, match.away_team.id))
        source_badge = f"SF:{match.event_id}\nFS:{fs_code}" if fs_code else f"SF:{match.event_id}"

        rec_text = (
            f"{prediction.recommended_pick}\n"
            f"[dim]Marcador top: {prediction.most_likely_score}[/dim]"
        )
        if prediction.value_edge_home is not None:
            max_edge = max(
                [
                    ("Local", prediction.value_edge_home),
                    ("Empate", prediction.value_edge_draw or -99),
                    ("Visita", prediction.value_edge_away or -99),
                ],
                key=lambda x: x[1],
            )
            if max_edge[1] >= 4.0:
                rec_text += f"\n[bold yellow]Value Bet: {max_edge[0]} (+{max_edge[1]:.1f}%)[/bold yellow]"

        summary_table.add_row(
            f"{match.tournament.name}\n[dim]{source_badge}[/dim]",
            f"{match.home_team.name}\nvs\n{match.away_team.name}",
            status_str,
            f"L: {home_form.elo_rating:.0f} ({home_form.form_string})\nV: {away_form.elo_rating:.0f} ({away_form.form_string})",
            f"L: {home_form.xg_for_avg:.2f} xGF\nV: {away_form.xg_for_avg:.2f} xGF",
            f"{prediction.expected_goals_home:.2f} - {prediction.expected_goals_away:.2f}\n[dim]Crn: {prediction.expected_corners_total:.1f}[/dim]",
            f"{prediction.prob_home_win:.0f}% / {prediction.prob_draw:.0f}% / {prediction.prob_away_win:.0f}%",
            odds_str,
            f"O2.5: {prediction.prob_over_2_5:.0f}%\nBTTS: {prediction.prob_btts:.0f}%",
            rec_text,
        )

    console.print("\n")
    console.print(summary_table)
    console.print(
        "\n[bold cyan]✔ Partidos, estadísticas xG y alias multi-fuente guardados en:[/bold cyan] [underline]data/sports_analytics.db[/underline]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scraper Multi-Fuente (Sofascore + Flashscore) y Predictor xG/Elo"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Fecha en formato YYYY-MM-DD (por defecto hoy)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Número máximo de partidos a enriquecer y analizar en detalle",
    )
    parser.add_argument(
        "--all-leagues",
        action="store_true",
        help="Incluir todas las ligas disponibles",
    )
    args = parser.parse_args()
    run_pipeline(date_str=args.date, limit=args.limit, all_leagues=args.all_leagues)


if __name__ == "__main__":
    main()
