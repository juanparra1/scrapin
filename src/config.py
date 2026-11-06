"""Configuración central del proyecto de scraping deportivo."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "sports_analytics.db"

# Base de datos en la nube (Supabase / PostgreSQL) o local (SQLite)
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Concurrencia para enriquecimiento acelerado en paralelo
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "6"))

# Crear directorios necesarios
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ligas principales en Sofascore (IDs de uniqueTournament oficiales de Sofascore)
TOP_TOURNAMENTS = {
    8: {"name": "LaLiga", "country": "Spain"},
    17: {"name": "Premier League", "country": "England"},
    23: {"name": "Serie A", "country": "Italy"},
    35: {"name": "Bundesliga", "country": "Germany"},
    34: {"name": "Ligue 1", "country": "France"},
    7: {"name": "UEFA Champions League", "country": "Europe"},
    679: {"name": "UEFA Europa League", "country": "Europe"},
    17015: {"name": "UEFA Conference League", "country": "Europe"},
    384: {"name": "Copa Libertadores", "country": "South America"},
    325: {"name": "Brasileirão Série A", "country": "Brazil"},
    155: {"name": "Liga Profesional", "country": "Argentina"},
    11539: {"name": "Liga BetPlay", "country": "Colombia"},
    11621: {"name": "Liga MX", "country": "Mexico"},
}

# Configuración de peticiones y Rate Limiting
SOFASCORE_API_BASE = "https://www.sofascore.com/api/v1"
DEFAULT_RATE_LIMIT_SECONDS = 1.5
CACHE_TTL_HOURS = 12
