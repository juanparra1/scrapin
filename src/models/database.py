"""Persistencia en SQLite mediante SQLAlchemy para almacenar equipos, partidos y estadísticas."""

import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
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


class PrecomputedMatchRecord(Base):
    __tablename__ = "precomputed_matches"

    event_id = Column(Integer, primary_key=True, autoincrement=False)
    match_date = Column(String(10), nullable=False, index=True)
    tournament_name = Column(String(120), nullable=False, index=True)
    home_team_name = Column(String(120), nullable=False)
    away_team_name = Column(String(120), nullable=False)
    status_type = Column(String(40), nullable=False)
    match_detail_json = Column(Text, nullable=False)
    prediction_json = Column(Text, nullable=False)
    deep_analysis_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PortfolioBetORM(Base):
    __tablename__ = "portfolio_bets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_hash = Column(String(64), unique=True, nullable=False, index=True)
    event_id = Column(Integer, nullable=False, index=True)
    event_name = Column(String(150), nullable=False)
    tournament = Column(String(100), nullable=True)
    match_date = Column(String(10), nullable=False, index=True)
    strategy = Column(String(30), nullable=False)  # SUREBET, VALUE_BET, DUTCHING
    market_code = Column(String(40), nullable=False, index=True)
    pick_name = Column(String(80), nullable=False)
    odds_taken = Column(Float, nullable=False)
    fair_odds = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)
    stake_amount = Column(Float, nullable=False)
    stake_percentage = Column(Float, nullable=False)
    potential_payout = Column(Float, nullable=False)
    status = Column(String(20), nullable=False, default="PENDING", index=True)  # PENDING, WON, LOST, VOID_PUSH, CANCELLED
    pnl = Column(Float, default=0.0)
    closing_odds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime, nullable=True)

    @staticmethod
    def generate_hash(event_id: int, market_code: str, pick_name: str, match_date: str) -> str:
        raw = f"{event_id}_{market_code}_{pick_name}_{match_date}".lower().strip()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class BankrollLedgerORM(Base):
    __tablename__ = "bankroll_ledger"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    transaction_type = Column(String(30), nullable=False)  # INITIAL, BET_PLACED, BET_SETTLED, DEPOSIT, WITHDRAWAL
    amount = Column(Float, nullable=False)
    resulting_balance = Column(Float, nullable=False)
    description = Column(String(255), nullable=True)
    bet_id = Column(Integer, nullable=True)


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


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Activa modo WAL y configuraciones de alta concurrencia en SQLite."""
    try:
        driver = getattr(engine.url, "drivername", "")
        if "sqlite" in driver:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA busy_timeout=10000;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.close()
    except Exception as e:
        logger.debug(f"Aviso configurando pragmas SQLite: {e}")


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_db_session():
    """Gestor de contexto de sesión seguro con commit/rollback transaccional atómico."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


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

        # Forzar flush para que los equipos existan y respeten la clave foránea antes de insertar MatchRecord
        session.flush()

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
    except Exception as e:
        session.rollback()
        logger.warning(f"Error upserting match {match.event_id}: {e}")
    finally:
        session.close()


_SAVED_ALIASES_CACHE: set[tuple[str, str]] = set()


def save_team_alias(
    source: str,
    external_id: str | None,
    external_name: str,
    canonical_team_id: int,
    canonical_name: str,
    similarity_score: float,
) -> None:
    """Registra o actualiza una equivalencia de equipo entre plataformas en la base de datos (con caché en memoria ultra-rápida)."""
    cache_key = (source.lower().strip(), external_name.lower().strip())
    if cache_key in _SAVED_ALIASES_CACHE:
        return

    session = SessionLocal()
    try:
        # Asegurar que el equipo canónico exista primero en teams para respetar la ForeignKey en PostgreSQL
        existing_team = session.get(TeamRecord, canonical_team_id)
        if not existing_team:
            session.add(
                TeamRecord(
                    id=canonical_team_id,
                    name=canonical_name,
                    short_name=canonical_name[:60] if canonical_name else None,
                )
            )
            session.flush()

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
        _SAVED_ALIASES_CACHE.add(cache_key)
    except Exception as e:
        session.rollback()
        logger.warning(f"No se pudo guardar alias {external_name} -> {canonical_name}: {e}")
    finally:
        session.close()


def save_precomputed_analysis(
    event_id: int,
    match_date: str,
    tournament_name: str,
    home_team_name: str,
    away_team_name: str,
    status_type: str,
    match_detail: dict,
    prediction: dict,
    deep_analysis: dict,
) -> None:
    """Almacena o actualiza el análisis pre-calculado de un partido en Supabase / SQLite."""
    session = SessionLocal()
    try:
        rec = session.get(PrecomputedMatchRecord, event_id)
        if not rec:
            rec = PrecomputedMatchRecord(event_id=event_id, match_date=match_date)
            session.add(rec)
        rec.match_date = match_date
        rec.tournament_name = tournament_name
        rec.home_team_name = home_team_name
        rec.away_team_name = away_team_name
        rec.status_type = status_type
        rec.match_detail_json = json.dumps(match_detail, default=str, ensure_ascii=False)
        rec.prediction_json = json.dumps(prediction, default=str, ensure_ascii=False)
        rec.deep_analysis_json = json.dumps(deep_analysis, default=str, ensure_ascii=False)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.warning(f"Error guardando análisis precalculado para evento {event_id}: {e}")
    finally:
        session.close()


def get_precomputed_matches_for_date(match_date: str) -> List[Dict[str, Any]]:
    """Recupera todos los partidos pre-calculados de una fecha directamente desde la base de datos."""
    session = SessionLocal()
    try:
        records = (
            session.query(PrecomputedMatchRecord)
            .filter_by(match_date=match_date)
            .all()
        )
        results = []
        for r in records:
            results.append({
                "event_id": r.event_id,
                "match_date": r.match_date,
                "tournament_name": r.tournament_name,
                "home_team_name": r.home_team_name,
                "away_team_name": r.away_team_name,
                "status_type": r.status_type,
                "match_detail": json.loads(r.match_detail_json) if r.match_detail_json else {},
                "prediction": json.loads(r.prediction_json) if r.prediction_json else {},
                "deep_analysis": json.loads(r.deep_analysis_json) if r.deep_analysis_json else {},
            })
        return results
    except Exception as e:
        logger.warning(f"Error consultando partidos precalculados para {match_date}: {e}")
        return []
    finally:
        session.close()


# ============================================================================
# GESTIÓN CONTABLE DE BANCA Y REGISTRO DE APUESTAS DE PORTAFOLIO
# ============================================================================


def get_or_create_initial_bankroll(initial_amount: float = 1000.0) -> float:
    """Obtiene la banca actual o inicializa el primer registro si el libro diario está vacío."""
    with get_db_session() as session:
        last_tx = session.query(BankrollLedgerORM).order_by(BankrollLedgerORM.id.desc()).first()
        if last_tx:
            return round(last_tx.resulting_balance, 2)

        init_tx = BankrollLedgerORM(
            transaction_type="INITIAL",
            amount=round(initial_amount, 2),
            resulting_balance=round(initial_amount, 2),
            description="Fondos iniciales de asignación de capital",
        )
        session.add(init_tx)
        session.flush()
        return round(init_tx.resulting_balance, 2)


def get_current_bankroll() -> float:
    """Retorna el saldo neto disponible en la banca."""
    return get_or_create_initial_bankroll(1000.0)


def record_bankroll_transaction(
    tx_type: str,
    amount: float,
    description: str = "",
    bet_id: Optional[int] = None,
) -> float:
    """Registra un movimiento en el libro contable de la banca de manera atómica."""
    with get_db_session() as session:
        last_tx = session.query(BankrollLedgerORM).order_by(BankrollLedgerORM.id.desc()).first()
        prev_balance = last_tx.resulting_balance if last_tx else 1000.0
        new_balance = round(prev_balance + amount, 2)

        tx = BankrollLedgerORM(
            transaction_type=tx_type,
            amount=round(amount, 2),
            resulting_balance=new_balance,
            description=description,
            bet_id=bet_id,
        )
        session.add(tx)
        session.flush()
        return new_balance


def save_portfolio_bets(bets: List[Dict[str, Any]]) -> tuple[int, int]:
    """
    Inserta una lista de apuestas de portafolio con verificación idempotente anti-rebote (SHA-256).
    Deduce el stake del libro diario (BET_PLACED) si la apuesta no existía previamente.
    Retorna (guardadas_exitosas, duplicadas_ignoradas).
    """
    saved_count = 0
    skipped_count = 0

    with get_db_session() as session:
        for b in bets:
            b_hash = b.get("bet_hash") or PortfolioBetORM.generate_hash(
                event_id=int(b["event_id"]),
                market_code=str(b["market_code"]),
                pick_name=str(b["pick_name"]),
                match_date=str(b["match_date"]),
            )

            existing = session.query(PortfolioBetORM).filter_by(bet_hash=b_hash).first()
            if existing:
                skipped_count += 1
                continue

            stake = round(float(b["stake_amount"]), 2)
            odds = round(float(b["odds_taken"]), 2)
            payout = round(stake * odds, 2)

            bet_orm = PortfolioBetORM(
                bet_hash=b_hash,
                event_id=int(b["event_id"]),
                event_name=str(b["event_name"]),
                tournament=b.get("tournament"),
                match_date=str(b["match_date"]),
                strategy=str(b.get("strategy", "VALUE_BET")),
                market_code=str(b["market_code"]),
                pick_name=str(b["pick_name"]),
                odds_taken=odds,
                fair_odds=round(float(b.get("fair_odds", odds)), 2),
                model_probability=round(float(b.get("model_probability", 0.5)), 4),
                stake_amount=stake,
                stake_percentage=round(float(b.get("stake_percentage", 0.0)), 2),
                potential_payout=payout,
                status="PENDING",
                pnl=0.0,
                closing_odds=b.get("closing_odds"),
            )
            session.add(bet_orm)
            session.flush()

            # Registrar deducción contable del stake en el libro diario
            last_tx = session.query(BankrollLedgerORM).order_by(BankrollLedgerORM.id.desc()).first()
            prev_balance = last_tx.resulting_balance if last_tx else 1000.0
            new_balance = round(prev_balance - stake, 2)

            ledger_entry = BankrollLedgerORM(
                transaction_type="BET_PLACED",
                amount=-stake,
                resulting_balance=new_balance,
                description=f"Apuesta colocada: {bet_orm.pick_name} @ {bet_orm.odds_taken:.2f} ({bet_orm.event_name})",
                bet_id=bet_orm.id,
            )
            session.add(ledger_entry)
            saved_count += 1

    return saved_count, skipped_count


def get_portfolio_bets(status: Optional[str] = None, limit: int = 150) -> List[Dict[str, Any]]:
    """Recupera el historial de apuestas del portafolio con filtro opcional por estado."""
    session = SessionLocal()
    try:
        query = session.query(PortfolioBetORM)
        if status and status != "TODAS":
            query = query.filter_by(status=status)
        records = query.order_by(PortfolioBetORM.id.desc()).limit(limit).all()

        results = []
        for r in records:
            results.append({
                "id": r.id,
                "bet_hash": r.bet_hash,
                "event_id": r.event_id,
                "event_name": r.event_name,
                "tournament": r.tournament,
                "match_date": r.match_date,
                "strategy": r.strategy,
                "market_code": r.market_code,
                "pick_name": r.pick_name,
                "odds_taken": r.odds_taken,
                "fair_odds": r.fair_odds,
                "model_probability": r.model_probability,
                "stake_amount": r.stake_amount,
                "stake_percentage": r.stake_percentage,
                "potential_payout": r.potential_payout,
                "status": r.status,
                "pnl": r.pnl,
                "closing_odds": r.closing_odds,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
                "settled_at": r.settled_at.strftime("%Y-%m-%d %H:%M") if r.settled_at else "",
            })
        return results
    finally:
        session.close()


def get_bankroll_ledger_history(limit: int = 300) -> List[Dict[str, Any]]:
    """Recupera la secuencia de transacciones del libro diario para graficar la curva de saldo."""
    session = SessionLocal()
    try:
        records = (
            session.query(BankrollLedgerORM)
            .order_by(BankrollLedgerORM.id.asc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
                "transaction_type": r.transaction_type,
                "amount": r.amount,
                "resulting_balance": r.resulting_balance,
                "description": r.description or "",
            }
            for r in records
        ]
    finally:
        session.close()


def get_portfolio_kpis() -> Dict[str, Any]:
    """Calcula las métricas de rendimiento financiero y riesgo del portafolio."""
    session = SessionLocal()
    try:
        last_tx = session.query(BankrollLedgerORM).order_by(BankrollLedgerORM.id.desc()).first()
        current_balance = round(last_tx.resulting_balance, 2) if last_tx else 1000.0

        all_bets = session.query(PortfolioBetORM).all()
        total_bets = len(all_bets)
        won_bets = sum(1 for b in all_bets if b.status == "WON")
        lost_bets = sum(1 for b in all_bets if b.status == "LOST")
        void_bets = sum(1 for b in all_bets if b.status == "VOID_PUSH")
        pending_bets = sum(1 for b in all_bets if b.status == "PENDING")

        settled_bets = [b for b in all_bets if b.status in ("WON", "LOST", "VOID_PUSH")]
        total_staked = sum(b.stake_amount for b in settled_bets)
        total_pnl = sum(b.pnl for b in settled_bets)

        roi_pct = round((total_pnl / total_staked * 100.0), 2) if total_staked > 0 else 0.0
        win_rate = round((won_bets / max(1, won_bets + lost_bets) * 100.0), 1) if (won_bets + lost_bets) > 0 else 0.0

        return {
            "current_bankroll": current_balance,
            "total_pnl": round(total_pnl, 2),
            "total_staked": round(total_staked, 2),
            "roi_pct": roi_pct,
            "win_rate_pct": win_rate,
            "total_bets": total_bets,
            "won_count": won_bets,
            "lost_count": lost_bets,
            "void_count": void_bets,
            "pending_count": pending_bets,
        }
    finally:
        session.close()


