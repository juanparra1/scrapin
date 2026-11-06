"""Extractor de feeds delimitados (~AA÷...¬) de Flashscore y cruce con EntityResolver."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel
from src.scrapers.base import BaseScraperClient

FLASHSCORE_FEED_BASE = "https://local-global.flashscore.ninja/2/x/feed"
FLASHSCORE_SIGN_HEADER = "SW9D1eZo"


class FlashscoreMatchSchema(BaseModel):
    flashscore_id: str
    tournament_title: str
    country: str
    start_timestamp: int
    start_time_utc: datetime
    status_code: str  # 1=Scheduled, 2=Live, 3=Finished
    home_team_id: Optional[str] = None
    home_team_name: str
    home_team_short: Optional[str] = None
    away_team_id: Optional[str] = None
    away_team_name: str
    away_team_short: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None


def parse_flashscore_chunk(chunk: str) -> Dict[str, str]:
    """Decodifica un bloque de texto delimitado por '¬' y pares 'CLAVE÷VALOR' de Flashscore."""
    fields: Dict[str, str] = {}
    for pair in chunk.split("¬"):
        if "÷" in pair:
            key, val = pair.split("÷", 1)
            fields[key] = val
    return fields


class FlashscoreScraper(BaseScraperClient):
    """Cliente para consumir y decodificar el protocolo propietario (~AA÷...¬) de Flashscore."""

    def __init__(self, rate_limit_seconds: float = 1.0, use_cache: bool = True) -> None:
        super().__init__(rate_limit_seconds=rate_limit_seconds, use_cache=use_cache)
        self.session.headers.update(
            {
                "x-fsign": FLASHSCORE_SIGN_HEADER,
                "Origin": "https://www.flashscore.es",
                "Referer": "https://www.flashscore.es/",
            }
        )

    def get_daily_matches(self, day_offset: int = 0) -> List[FlashscoreMatchSchema]:
        """
        Extrae todos los partidos de fútbol del feed diario de Flashscore.
        day_offset: 0 = hoy, 1 = mañana, -1 = ayer (entre -7 y +7).
        """
        url = f"{FLASHSCORE_FEED_BASE}/f_1_{day_offset}_2_es_1"
        self._wait_rate_limit()
        response = self.session.get(url, timeout=20)
        if response.status_code != 200 or not response.text:
            return []

        raw_text = response.text
        chunks = raw_text.split("~")

        current_tournament = "Unknown Tournament"
        current_country = "World"
        matches: List[FlashscoreMatchSchema] = []

        for chunk in chunks:
            if not chunk:
                continue
            fields = parse_flashscore_chunk(chunk)

            # Cabecera de Torneo/Liga
            if "ZA" in fields:
                current_tournament = fields.get("ZA", "Unknown Tournament")
                current_country = fields.get("ZY", "World")
                continue

            # Bloque de Partido
            if "AA" in fields and "AE" in fields and "AF" in fields:
                try:
                    fs_id = fields["AA"]
                    start_ts = int(fields.get("AD", 0))
                    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
                    status_map = {"1": "notstarted", "2": "inprogress", "3": "finished"}
                    status_str = status_map.get(fields.get("AB", "1"), "notstarted")

                    h_score = int(fields["AG"]) if "AG" in fields and fields["AG"].isdigit() else None
                    a_score = int(fields["AH"]) if "AH" in fields and fields["AH"].isdigit() else None

                    matches.append(
                        FlashscoreMatchSchema(
                            flashscore_id=fs_id,
                            tournament_title=current_tournament,
                            country=current_country,
                            start_timestamp=start_ts,
                            start_time_utc=start_dt,
                            status_code=status_str,
                            home_team_id=fields.get("PX"),
                            home_team_name=fields["AE"],
                            home_team_short=fields.get("WM"),
                            away_team_id=fields.get("PY"),
                            away_team_name=fields["AF"],
                            away_team_short=fields.get("WN"),
                            home_score=h_score,
                            away_score=a_score,
                        )
                    )
                except (ValueError, KeyError):
                    continue

        return matches
