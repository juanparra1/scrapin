"""Script de pre-cálculo masivo en segundo plano.
Extrae los partidos curados del día, ejecuta el análisis profundo de IA en paralelo
y almacena los pronósticos en Supabase para que el dashboard cargue al instante (0.2s).
"""

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dashboard import analyze_matches_batch_parallel, load_day_catalog
from src.models.database import init_db, save_precomputed_analysis


def run_precomputation(target_date: str) -> None:
    print(f"🚀 Iniciando pre-cálculo de alta velocidad para la jornada: {target_date}")
    t0 = time.time()

    # Asegurar que las tablas existan en Supabase
    init_db()

    # 1. Cargar catálogo curado (sin forzar todas las ligas para enfocarse en la élite)
    catalog = load_day_catalog(target_date, include_all_leagues=False)
    matches = catalog.get("matches", [])
    fs_lookup = catalog.get("fs_lookup", {})
    print(f"📅 Partidos élite identificados: {len(matches)} (Fuente: {catalog.get('source')})")

    if not matches:
        print("⚠️ No hay partidos para precalcular en esta fecha.")
        return

    # 2. Emparejar con Flashscore para enriquecimiento
    pairs = [
        (m, fs_lookup.get((m.home_team.id, m.away_team.id), "N/A"))
        for m in matches
    ]

    # 3. Ejecutar análisis profundo en paralelo
    print("⚡ Ejecutando análisis Opta, H2H y predicción ML en paralelo...")
    batch_results = analyze_matches_batch_parallel(
        pairs,
        status_text_prefix="Precalculando",
        max_workers=6,
    )

    # 4. Guardar en Supabase con todo el detalle de deep_analysis para carga instantánea
    saved_count = 0
    for res in batch_results:
        m = res["match"]
        pred = res["prediction"]
        save_precomputed_analysis(
            event_id=m.event_id,
            match_date=target_date,
            tournament_name=m.tournament.name,
            home_team_name=m.home_team.name,
            away_team_name=m.away_team.name,
            status_type=m.status_type,
            match_detail=m.model_dump(),
            prediction=pred.model_dump() if hasattr(pred, "model_dump") else pred,
            deep_analysis={
                "home_form": res["home_form"].model_dump() if hasattr(res.get("home_form"), "model_dump") else res.get("home_form"),
                "away_form": res["away_form"].model_dump() if hasattr(res.get("away_form"), "model_dump") else res.get("away_form"),
                "home_recent": [rm.model_dump() if hasattr(rm, "model_dump") else rm for rm in res.get("home_recent", [])],
                "away_recent": [rm.model_dump() if hasattr(rm, "model_dump") else rm for rm in res.get("away_recent", [])],
                "home_players": [hp.model_dump() if hasattr(hp, "model_dump") else hp for hp in res.get("home_players", [])],
                "away_players": [ap.model_dump() if hasattr(ap, "model_dump") else ap for ap in res.get("away_players", [])],
                "standings_map": res.get("standings_map", {}),
                "flashscore_id": res.get("flashscore_id", ""),
            },
        )
        saved_count += 1

    t1 = time.time()
    print(f"✅ ¡Éxito! Se precalcularon y persistieron {saved_count} partidos en Supabase en {t1 - t0:.2f} segundos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precalculador diario de pronósticos para Scrapin / ProSports AI")
    parser.add_argument("--date", type=str, default=date.today().strftime("%Y-%m-%d"), help="Fecha YYYY-MM-DD")
    args = parser.parse_args()
    run_precomputation(args.date)
