"""Resolución de Entidades (Entity Resolver) para unificar nombres de equipos entre fuentes."""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process
from pydantic import BaseModel

# Tokens genéricos de clubes que suelen variar entre Sofascore, FBref, Flashscore y BeSoccer
STOP_TOKENS = {
    "fc",
    "cf",
    "cd",
    "ud",
    "sd",
    "rcd",
    "rc",
    "ac",
    "as",
    "sc",
    "sk",
    "fk",
    "nk",
    "bsc",
    "vfb",
    "vfl",
    "tsg",
    "fsv",
    "sv",
    "bv",
    "rb",
    "losc",
    "ogc",
    "aj",
    "afc",
    "bk",
    "if",
    "ik",
    "ff",
    "ca",
    "cs",
    "cn",
    "se",
    "ec",
    "cr",
    "club",
    "deportivo",
    "foot",
    "football",
    "calcio",
    "1899",
    "1904",
    "1909",
    "04",
    "05",
    "96",
}

# Diccionario de sinónimos canónicos frecuentes entre casas de datos
CANONICAL_SYNONYMS: Dict[str, str] = {
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "nottm forest": "nottingham forest",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "leeds": "leeds united",
    "brighton": "brighton hove albion",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "stuttgart": "stuttgart",
    "bayern munich": "bayern munchen",
    "fc bayern": "bayern munchen",
    "bayern": "bayern munchen",
    "leverkusen": "bayer leverkusen",
    "dortmund": "borussia dortmund",
    "mgladbach": "borussia monchengladbach",
    "gladbach": "borussia monchengladbach",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "ac milan": "milan",
    "atletico": "atletico madrid",
    "atl madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "athletic bilbao": "athletic club",
    "betis": "real betis",
    "real sociedad": "real sociedad",
    "celta": "celta vigo",
    "alaves": "deportivo alaves",
    "sporting lisbon": "sporting cp",
    "sporting": "sporting cp",
    "bodo glimt": "bodo glimt",
    "lille": "lille",
    "barcelona": "barcelona",
    "barca": "barcelona",
    "shakhtar": "shakhtar donetsk",
}


class ResolvedEntityMatch(BaseModel):
    source_name: str
    canonical_id: int
    canonical_name: str
    normalized_source: str
    normalized_target: str
    similarity_score: float
    is_confident: bool


def normalize_team_name(raw_name: str) -> str:
    """
    Limpia acentos, puntuación y prefijos/sufijos de club ('FC', 'VfB', 'LOSC', etc.)
    para maximizar la precisión del emparejamiento entre plataformas.
    """
    if not raw_name:
        return ""

    # Eliminar diacríticos y caracteres especiales nórdicos/europeos
    text = (
        raw_name.replace("ø", "o")
        .replace("Ø", "O")
        .replace("ß", "ss")
        .replace("ł", "l")
        .replace("Ł", "L")
        .replace("đ", "d")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = [t for t in text.split() if t and t not in STOP_TOKENS]
    cleaned = " ".join(tokens) if tokens else text.strip()

    if cleaned in CANONICAL_SYNONYMS:
        return CANONICAL_SYNONYMS[cleaned]

    for syn_key, syn_val in CANONICAL_SYNONYMS.items():
        if cleaned == syn_key:
            return syn_val

    return cleaned


class EntityResolver:
    """Emparejador difuso de equipos entre Sofascore, Flashscore, FBref y BeSoccer."""

    def __init__(self, confidence_threshold: float = 78.0) -> None:
        self.confidence_threshold = confidence_threshold
        self._catalog: Dict[int, str] = {}
        self._normalized_catalog: Dict[int, str] = {}

    def register_canonical_teams(self, teams: List[Tuple[int, str]]) -> None:
        """Registra equipos base (ej. desde Sofascore) con su (id, nombre)."""
        for tid, tname in teams:
            self._catalog[tid] = tname
            self._normalized_catalog[tid] = normalize_team_name(tname)

    def resolve(self, external_name: str) -> Optional[ResolvedEntityMatch]:
        """Busca el equipo equivalente más cercano dentro del catálogo canónico."""
        if not self._normalized_catalog or not external_name:
            return None

        norm_query = normalize_team_name(external_name)

        # 1. Coincidencia exacta tras normalización
        for tid, norm_target in self._normalized_catalog.items():
            if norm_query == norm_target:
                return ResolvedEntityMatch(
                    source_name=external_name,
                    canonical_id=tid,
                    canonical_name=self._catalog[tid],
                    normalized_source=norm_query,
                    normalized_target=norm_target,
                    similarity_score=100.0,
                    is_confident=True,
                )

        # 2. Búsqueda difusa ponderada (WRatio + token_set_ratio)
        best_match = process.extractOne(
            norm_query,
            self._normalized_catalog,
            scorer=fuzz.WRatio,
        )
        if not best_match:
            return None

        matched_norm, score, matched_tid = best_match
        token_score = fuzz.token_set_ratio(norm_query, matched_norm)
        combined_score = round(max(score, token_score * 0.96), 1)

        return ResolvedEntityMatch(
            source_name=external_name,
            canonical_id=matched_tid,
            canonical_name=self._catalog[matched_tid],
            normalized_source=norm_query,
            normalized_target=matched_norm,
            similarity_score=combined_score,
            is_confident=combined_score >= self.confidence_threshold,
        )
