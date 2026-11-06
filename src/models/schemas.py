"""Esquemas de validación de datos con Pydantic para partidos, árbitros, cuotas, Opta stats, porteros, H2H y jugadores."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TeamSchema(BaseModel):
    id: int
    name: str
    short_name: Optional[str] = None
    slug: Optional[str] = None
    country: Optional[str] = None


class TournamentSchema(BaseModel):
    id: int
    unique_id: Optional[int] = None
    name: str
    country: Optional[str] = None


class RefereeSchema(BaseModel):
    id: Optional[int] = None
    name: str
    country: Optional[str] = None
    games: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    yellow_red_cards: int = 0
    yellow_per_game: float = 4.2
    red_per_game: float = 0.18
    strictness_label: str = "Promedio"


class OddsSchema(BaseModel):
    home_win: Optional[float] = None
    draw: Optional[float] = None
    away_win: Optional[float] = None
    initial_home_win: Optional[float] = None
    initial_draw: Optional[float] = None
    initial_away_win: Optional[float] = None
    drop_pct_home: float = 0.0
    drop_pct_draw: float = 0.0
    drop_pct_away: float = 0.0
    smart_money_side: Optional[str] = None
    implied_prob_home: Optional[float] = None
    implied_prob_draw: Optional[float] = None
    implied_prob_away: Optional[float] = None
    over_2_5: Optional[float] = None
    under_2_5: Optional[float] = None
    initial_over_2_5: Optional[float] = None
    drop_pct_over_2_5: float = 0.0
    over_1_5: Optional[float] = None
    under_1_5: Optional[float] = None
    over_3_5: Optional[float] = None
    under_3_5: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None
    dc_1x: Optional[float] = None
    dc_x2: Optional[float] = None
    dc_12: Optional[float] = None
    is_live: bool = False
    bookmaker_name: str = "Betano / Mercado Oficial"
    raw_markets: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class H2HMatchItemSchema(BaseModel):
    event_id: int
    date_str: str
    timestamp: int
    tournament_name: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_score: int
    away_score: int
    xg_home: Optional[float] = None
    xg_away: Optional[float] = None
    corners_total: Optional[int] = None
    yellow_cards_total: Optional[int] = None
    sot_total: Optional[int] = None
    saves_total: Optional[int] = None


class H2HSummarySchema(BaseModel):
    team_wins_home: int = 0
    draws: int = 0
    team_wins_away: int = 0
    total_matches: int = 0
    # Métricas profundas de enfrentamientos directos recientes (últimos 3-4 años)
    recent_matches_analyzed: int = 0
    h2h_avg_goals: Optional[float] = None
    h2h_home_goals_avg: Optional[float] = None
    h2h_away_goals_avg: Optional[float] = None
    h2h_btts_pct: Optional[float] = None
    h2h_over_2_5_pct: Optional[float] = None
    h2h_corners_avg: Optional[float] = None
    h2h_cards_avg: Optional[float] = None
    h2h_sot_avg: Optional[float] = None
    h2h_saves_avg: Optional[float] = None
    recent_h2h_events: List[H2HMatchItemSchema] = Field(default_factory=list)


class MatchStatsSchema(BaseModel):
    # Métricas principales (ALL)
    expected_goals_home: Optional[float] = None
    expected_goals_away: Optional[float] = None
    xg_on_target_home: Optional[float] = None
    xg_on_target_away: Optional[float] = None
    goals_prevented_home: Optional[float] = None
    goals_prevented_away: Optional[float] = None
    ball_possession_home: Optional[float] = None
    ball_possession_away: Optional[float] = None
    total_shots_home: Optional[int] = None
    total_shots_away: Optional[int] = None
    shots_on_target_home: Optional[int] = None
    shots_on_target_away: Optional[int] = None
    shots_inside_box_home: Optional[int] = None
    shots_inside_box_away: Optional[int] = None
    shots_outside_box_home: Optional[int] = None
    shots_outside_box_away: Optional[int] = None
    touches_in_box_home: Optional[int] = None
    touches_in_box_away: Optional[int] = None
    big_chances_home: Optional[int] = None
    big_chances_away: Optional[int] = None
    big_chances_missed_home: Optional[int] = None
    big_chances_missed_away: Optional[int] = None
    final_third_entries_home: Optional[int] = None
    final_third_entries_away: Optional[int] = None
    ball_recoveries_home: Optional[int] = None
    ball_recoveries_away: Optional[int] = None
    interceptions_home: Optional[int] = None
    interceptions_away: Optional[int] = None
    corner_kicks_home: Optional[int] = None
    corner_kicks_away: Optional[int] = None
    fouls_home: Optional[int] = None
    fouls_away: Optional[int] = None
    yellow_cards_home: Optional[int] = None
    yellow_cards_away: Optional[int] = None
    red_cards_home: Optional[int] = None
    red_cards_away: Optional[int] = None
    saves_home: Optional[int] = None
    saves_away: Optional[int] = None
    dive_saves_home: Optional[int] = None
    dive_saves_away: Optional[int] = None
    offsides_home: Optional[int] = None
    offsides_away: Optional[int] = None
    # Métricas reales de 1er Tiempo (1ST Period Opta)
    ht_xg_home: Optional[float] = None
    ht_xg_away: Optional[float] = None
    ht_corners_home: Optional[int] = None
    ht_corners_away: Optional[int] = None
    ht_sot_home: Optional[int] = None
    ht_sot_away: Optional[int] = None
    ht_saves_home: Optional[int] = None
    ht_saves_away: Optional[int] = None
    # Métricas avanzadas Opta estilo StatsBomb (PPDA, npxG, Recuperaciones)
    ppda_home: Optional[float] = None
    ppda_away: Optional[float] = None
    npxg_home: Optional[float] = None
    npxg_away: Optional[float] = None
    clearances_home: Optional[int] = None
    clearances_away: Optional[int] = None
    tackles_won_home: Optional[int] = None
    tackles_won_away: Optional[int] = None
    raw_metrics: Dict[str, Dict[str, Optional[float]]] = Field(default_factory=dict)


class PlayerPropSchema(BaseModel):
    player_id: int
    name: str
    position: Optional[str] = None
    team_name: str
    appearances: int = 1
    goals: int = 0
    expected_goals: float = 0.0
    xgot_per_match: float = 0.0
    key_passes_per_match: float = 0.0
    touches_per_match: float = 0.0
    total_shots: int = 0
    shots_on_target: int = 0
    yellow_cards: int = 0
    rating: Optional[float] = None
    shots_per_match: float = 0.0
    sot_per_match: float = 0.0
    xg_per_match: float = 0.0
    cards_per_match: float = 0.0
    prob_1plus_sot: float = 0.0
    prob_2plus_sot: float = 0.0
    prob_2plus_shots: float = 0.0
    prob_anytime_goal: float = 0.0
    is_missing: bool = False
    is_starter: Optional[bool] = None


class PlayerLineupSchema(BaseModel):
    player_id: int
    name: str
    position: Optional[str] = None
    jersey_number: Optional[str] = None
    substitute: bool = False
    rating: Optional[float] = None
    avg_season_rating: Optional[float] = None
    # Métricas Opta de Portero en la alineación (seasonRatingBreakdown.goalkeeping)
    gk_saves_per_game: Optional[float] = None
    gk_goals_prevented: Optional[float] = None
    match_saves: Optional[int] = None
    # Métricas Opta de Jugador de Campo en la alineación (seasonRatingBreakdown.shot / pass)
    season_shots_per_game: Optional[float] = None
    season_sot_per_game: Optional[float] = None
    season_xg: Optional[float] = None
    season_xgot: Optional[float] = None
    season_key_passes_per_game: Optional[float] = None
    season_touches_per_game: Optional[float] = None


class TeamLineupSchema(BaseModel):
    formation: Optional[str] = None
    confirmed: bool = False
    players: List[PlayerLineupSchema] = Field(default_factory=list)
    missing_players: List[str] = Field(default_factory=list)


class MomentumPointSchema(BaseModel):
    minute: float
    value: float


class LiveMomentumSummarySchema(BaseModel):
    home_pressure_avg: float = 0.0
    away_pressure_avg: float = 0.0
    recent_10m_net: float = 0.0
    recent_10m_abs_intensity: float = 0.0
    dominant_team: str = "Equilibrado"
    goal_alert_level: str = "Normal"  # "Normal", "Media 🟡", "ALTA 🔥"
    goal_alert_reason: str = "Sin asedio extremo en los últimos 10 min"
    points: List[MomentumPointSchema] = Field(default_factory=list)


class MatchDetailSchema(BaseModel):
    event_id: int
    custom_id: Optional[str] = None
    slug: str
    start_timestamp: int
    start_time_utc: datetime
    status_type: str  # notstarted, inprogress, finished, postponed, canceled
    status_description: Optional[str] = None
    tournament: TournamentSchema
    season_id: Optional[int] = None
    round_number: Optional[int] = None
    home_team: TeamSchema
    away_team: TeamSchema
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_score_ht: Optional[int] = None
    away_score_ht: Optional[int] = None
    referee: Optional[RefereeSchema] = None
    stats: Optional[MatchStatsSchema] = None
    odds: Optional[OddsSchema] = None
    h2h: Optional[H2HSummarySchema] = None
    home_lineup: Optional[TeamLineupSchema] = None
    away_lineup: Optional[TeamLineupSchema] = None
    momentum: Optional[LiveMomentumSummarySchema] = None
    streaks: List[Dict[str, str]] = Field(default_factory=list)
