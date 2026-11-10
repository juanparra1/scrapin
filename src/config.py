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
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# Concurrencia para enriquecimiento acelerado en paralelo
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "6"))

# API Externa de Cuotas (The Odds API) para Betsson, Pinnacle, etc.
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "05b314e29b0cbb1f38344df9dc92f76b")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Crear directorios necesarios
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Competiciones de Élite Seleccionadas (Ligas Top, Copas Nacionales, Torneos Continentales y Selecciones)
TOP_TOURNAMENTS = {
    # --- 1. LIGAS NACIONALES ---
    17: {"name": "Premier League", "country": "England"},
    8: {"name": "LaLiga", "country": "Spain"},
    23: {"name": "Serie A", "country": "Italy"},
    35: {"name": "Bundesliga", "country": "Germany"},
    34: {"name": "Ligue 1", "country": "France"},
    11539: {"name": "Primera A (Apertura)", "country": "Colombia"},
    11542: {"name": "Primera A (Clausura)", "country": "Colombia"},
    242: {"name": "Major League Soccer (MLS)", "country": "USA"},
    155: {"name": "Liga Profesional de Fútbol", "country": "Argentina"},
    955: {"name": "Saudi Pro League", "country": "Saudi Arabia"},
    52: {"name": "Trendyol Süper Lig", "country": "Turkey"},
    325: {"name": "Brasileirão Série A", "country": "Brazil"},
    238: {"name": "Liga Portugal Betclic", "country": "Portugal"},
    11621: {"name": "Liga MX", "country": "Mexico"},
    18: {"name": "Championship", "country": "England"},

    # --- 2. COPAS NACIONALES Y SUPERTROFEOS ---
    19: {"name": "FA Cup", "country": "England"},
    21: {"name": "EFL Carabao Cup", "country": "England"},
    401: {"name": "FA Community Shield", "country": "England"},
    329: {"name": "Copa del Rey", "country": "Spain"},
    400: {"name": "Supercopa de España", "country": "Spain"},
    328: {"name": "Coppa Italia", "country": "Italy"},
    402: {"name": "Supercoppa Italiana", "country": "Italy"},
    217: {"name": "DFB-Pokal", "country": "Germany"},
    404: {"name": "DFL-Supercup", "country": "Germany"},
    335: {"name": "Coupe de France", "country": "France"},
    405: {"name": "Trophée des Champions", "country": "France"},
    1335: {"name": "Copa Colombia (BetPlay)", "country": "Colombia"},
    10825: {"name": "Superliga de Colombia", "country": "Colombia"},
    495: {"name": "US Open Cup", "country": "USA"},
    1024: {"name": "Copa Argentina", "country": "Argentina"},
    11707: {"name": "Copa de la Liga Argentina", "country": "Argentina"},
    2058: {"name": "King's Cup", "country": "Saudi Arabia"},
    96: {"name": "Türkiye Kupası", "country": "Turkey"},
    373: {"name": "Copa do Brasil", "country": "Brazil"},
    10459: {"name": "Supercopa do Brasil", "country": "Brazil"},
    336: {"name": "Taça de Portugal", "country": "Portugal"},
    713: {"name": "Taça da Liga", "country": "Portugal"},

    # --- 3. TORNEOS INTERNACIONALES DE CLUBES ---
    7: {"name": "UEFA Champions League", "country": "Europe"},
    692: {"name": "UEFA Women's Champions League", "country": "Europe"},
    679: {"name": "UEFA Europa League", "country": "Europe"},
    17015: {"name": "UEFA Conference League", "country": "Europe"},
    465: {"name": "UEFA Super Cup", "country": "Europe"},
    384: {"name": "CONMEBOL Libertadores", "country": "South America"},
    480: {"name": "CONMEBOL Sudamericana", "country": "South America"},
    490: {"name": "CONMEBOL Recopa", "country": "South America"},
    357: {"name": "FIFA Club World Cup", "country": "World"},
    13783: {"name": "Leagues Cup (MLS vs Liga MX)", "country": "North & Central America"},
    498: {"name": "CONCACAF Champions Cup", "country": "North & Central America"},

    # --- 4. SELECCIONES NACIONALES & AMISTOSOS ---
    16: {"name": "FIFA World Cup", "country": "World"},
    377: {"name": "Eliminatorias CONMEBOL (Mundial)", "country": "South America"},
    110: {"name": "Eliminatorias UEFA (Mundial)", "country": "Europe"},
    120: {"name": "Eliminatorias CONCACAF (Mundial)", "country": "North & Central America"},
    1: {"name": "UEFA EURO", "country": "Europe"},
    112: {"name": "Clasificación UEFA EURO", "country": "Europe"},
    133: {"name": "Copa América", "country": "South America"},
    10783: {"name": "UEFA Nations League", "country": "Europe"},
    14100: {"name": "CONCACAF Nations League", "country": "North & Central America"},
    130: {"name": "CONCACAF Gold Cup", "country": "North & Central America"},
    1077: {"name": "Amistosos Internacionales (Selecciones)", "country": "World"},
    851: {"name": "Amistosos Internacionales FIFA", "country": "World"},
}

# Torneos clave para rescate rápido (sin disparar tormenta de 50 peticiones)
PRIORITY_FALLBACK_TOURNAMENTS = [
    692,    # UEFA Women's Champions League
    7,      # UEFA Champions League
    10783,  # UEFA Nations League
    14100,  # CONCACAF Nations League
    851,    # International Friendly Games
    377,    # Eliminatorias CONMEBOL
    11539,  # Primera A Apertura (Colombia)
    11542,  # Primera A Clausura (Colombia)
    155,    # Liga Profesional Argentina
    325,    # Brasileirão Série A
]

# Mapeo de unique_tournament_id (Sofascore) -> slug de liga en ESPN Public API (fallback 100% libre de bloqueos 403)
TOURNAMENT_ESPN_SLUGS = {
    692: "uefa.wchampions",
    7: "uefa.champions",
    679: "uefa.europa",
    17015: "uefa.europa.conf",
    10783: "uefa.nations",
    14100: "concacaf.nations.league",
    851: "fifa.friendly",
    1077: "fifa.friendly",
    17: "eng.1",
    18: "eng.2",
    19: "eng.fa",
    21: "eng.league_cup",
    8: "esp.1",
    329: "esp.copa_del_rey",
    23: "ita.1",
    328: "ita.coppa_italia",
    35: "ger.1",
    217: "ger.dfb_pokal",
    34: "fra.1",
    335: "fra.coupe_de_france",
    11539: "col.1",
    11542: "col.1",
    1335: "col.copa",
    242: "usa.1",
    155: "arg.1",
    1024: "arg.copa",
    955: "ksa.1",
    52: "tur.1",
    325: "bra.1",
    373: "bra.copa_do_brazil",
    238: "por.1",
    11621: "mex.1",
    384: "conmebol.libertadores",
    480: "conmebol.sudamericana",
    377: "fifa.worldq.conmebol",
    110: "fifa.worldq.uefa",
    120: "fifa.worldq.concacaf",
    1: "uefa.euro",
    133: "conmebol.america",
    130: "concacaf.gold",
}

# Configuración de peticiones y Rate Limiting
SOFASCORE_API_BASE = "https://www.sofascore.com/api/v1"
DEFAULT_RATE_LIMIT_SECONDS = 1.5
CACHE_TTL_HOURS = 12
