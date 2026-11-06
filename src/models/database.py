"""Persistencia en SQLite mediante SQLAlchemy para almacenar equipos, partidos y estadísticas."""

import json
from datetime import datetime
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import DATABASE_URL, DB_PATH
from src.models.schemas import MatchDetailSchema

Base = declarative_base()


class TeamRecord(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=False)  # ID de Sofascore
    name = Column(String(120), nullable=False)
    short_name = Column(String(60), nullable=True)
    slug = Column(String(120), nullable=True)
    country = Column(String(80), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TeamAliasRecord(Base):
    __tablename__ = "team_aliases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(40), nullable=False, index=True)  # flashscore, fbref, besoccer
    external_id = Column(String(80), nullable=True)
    external_name = Column(String(140), nullable=False, index=True)
    canonical_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    canonical_name = Column(String(120), nullable=False)
    similarity_score = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MatchRecord(Base):
    __tablename__ = "matches"

    event_id = Column(Integer, primary_key=True, autoincrement=False)
    slug = Column(String(180), nullable=False)
    start_timestamp = Column(Integer, nullable=False, index=True)
    start_time_utc = Column(DateTime, nullable=False, index=True)
    status_type = Column(String(40), nullable=False, index=True)
    tournament_id = Column(Integer, nullable=True, index=True)
    tournament_name = Column(String(120), nullable=False)
    country = Column(String(80), nullable=True)
    season_id = Column(Integer, nullable=True)
    round_number = Column(Integer, nullable=True)

    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)

    # Estadísticas clave desnormalizadas para consultas rápidas de ML
    xg_home = Column(Float, nullable=True)
    xg_away = Column(Float, nullable=True)
    possession_home = Column(Float, nullable=True)
    possession_away = Column(Float, nullable=True)
    shots_home = Column(Integer, nullable=True)
    shots_away = Column(Integer, nullable=True)
    shots_on_target_home = Column(Integer, nullable=True)
    shots_on_target_away = Column(Integer, nullable=True)
    corners_home = Column(Integer, nullable=True)
    corners_away = Column(Integer, nullable=True)

    # Cuotas 1X2
    odds_home = Column(Float, nullable=True)
    odds_draw = Column(Float, nullable=True)
    odds_away = Column(Float, nullable=True)

    # H2H
    h2h_home_wins = Column(Integer, nullable=True)
    h2h_draws = Column(Integer, nullable=True)
    h2h_away_wins = Column(Integer, nullable=True)

    # JSON extendido (rachas, alineaciones y métricas adicionales)
    extra_data_json = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


if DATABASE_URL:
    # Conexión optimizada a PostgreSQL en la nube (Supabase) con connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False,
    )
else:
    # Base de datos local SQLite con soporte multihilo
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        echo=False,
    )

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Inicializa las tablas en la base de datos (PostgreSQL en Supabase o SQLite local)."""
    Base.metadata.create_all(bind=engine)


def upsert_match(match: MatchDetailSchema) -> None:
    """Inserta o actualiza un partido y sus equipos en la base de datos."""
    session = SessionLocal()
    try:
        for team in (match.home_team, match.away_team):
            existing_team = session.get(TeamRecord, team.id)
            if not existing_team:
                session.add(
                    TeamRecord(
                        id=team.id,
                        name=team.name,
                        short_name=team.short_name,
                        slug=team.slug,
                        country=team.country,
                    )
                )
            else:
                existing_team.name = team.name
                existing_team.short_name = team.short_name or existing_team.short_name
                existing_team.country = team.country or existing_team.country

        extra_payload = {
            "streaks": match.streaks,
            "home_lineup": match.home_lineup.model_dump() if match.home_lineup else None,
            "away_lineup": match.away_lineup.model_dump() if match.away_lineup else None,
            "raw_metrics": match.stats.raw_metrics if match.stats else {},
        }

        record = session.get(MatchRecord, match.event_id)
        if not record:
            record = MatchRecord(event_id=match.event_id, slug=match.slug)
            session.add(record)

        record.slug = match.slug
        record.start_timestamp = match.start_timestamp
        record.start_time_utc = match.start_time_utc
        record.status_type = match.status_type
        record.tournament_id = match.tournament.unique_id or match.tournament.id
        record.tournament_name = match.tournament.name
        record.country = match.tournament.country
        record.season_id = match.season_id
        record.round_number = match.round_number
        record.home_team_id = match.home_team.id
        record.away_team_id = match.away_team.id
        record.home_score = match.home_score
        record.away_score = match.away_score

        if match.stats:
            record.xg_home = match.stats.expected_goals_home
            record.xg_away = match.stats.expected_goals_away
            record.possession_home = match.stats.ball_possession_home
            record.possession_away = match.stats.ball_possession_away
            record.shots_home = match.stats.total_shots_home
            record.shots_away = match.stats.total_shots_away
            record.shots_on_target_home = match.stats.shots_on_target_home
            record.shots_on_target_away = match.stats.shots_on_target_away
            record.corners_home = match.stats.corner_kicks_home
            record.corners_away = match.stats.corner_kicks_away

        if match.odds:
            record.odds_home = match.odds.home_win
            record.odds_draw = match.odds.draw
            record.odds_away = match.odds.away_win

        if match.h2h:
            record.h2h_home_wins = match.h2h.team_wins_home
            record.h2h_draws = match.h2h.draws
            record.h2h_away_wins = match.h2h.team_wins_away

        record.extra_data_json = json.dumps(extra_payload, ensure_ascii=False)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_team_alias(
    source: str,
    external_id: str | None,
    external_name: str,
    canonical_team_id: int,
    canonical_name: str,
    similarity_score: float,
) -> None:
    """Registra o actualiza una equivalencia de equipo entre plataformas en SQLite."""
    session = SessionLocal()
    try:
        existing = (
            session.query(TeamAliasRecord)
            .filter_by(source=source, external_name=external_name)
            .first()
        )
        if not existing:
            session.add(
                TeamAliasRecord(
                    source=source,
                    external_id=external_id,
                    external_name=external_name,
                    canonical_team_id=canonical_team_id,
                    canonical_name=canonical_name,
                    similarity_score=similarity_score,
                )
            )
        else:
            existing.canonical_team_id = canonical_team_id
            existing.canonical_name = canonical_name
            existing.similarity_score = similarity_score
            existing.external_id = external_id or existing.external_id
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

