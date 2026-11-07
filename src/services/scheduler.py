"""
Servicio en segundo plano (Heartbeat & Scheduler 24/7):
- Ejecuta ciclos cada 14 minutos para evitar la suspensión de Render (15 min) y Supabase (7 días).
- Purgado automático de partidos viejos (> 3 días) para mantener la base de datos limpia (< 25 MB).
- Liquidación automática de apuestas del portafolio al terminar los partidos.
- Precomputación de partidos del día para que el dashboard cargue en menos de 1 segundo.
"""

import concurrent.futures
import json
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.models.database import (
    MatchRecord,
    PortfolioBetORM,
    PrecomputedMatchRecord,
    SessionLocal,
    ping_database,
    purge_old_records,
    save_precomputed_analysis,
)
from src.scrapers.sofascore import SofascoreScraper

logger = logging.getLogger(__name__)

COLOMBIA_TZ = timezone(timedelta(hours=-5))
DEFAULT_INTERVAL_SECONDS = 840  # 14 minutos (justo antes del sleep de 15 min de Render)


class HeartbeatScheduler:
    _instance: Optional["HeartbeatScheduler"] = None
    _lock = threading.Lock()

    def __init__(self, interval_seconds: int = DEFAULT_INTERVAL_SECONDS):
        self.interval_seconds = interval_seconds
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.last_status: str = "Iniciando..."
        self.cycles_count: int = 0
        self.last_purged: Dict[str, int] = {"purged_precomputed": 0, "purged_matches": 0}
        self.last_settled_count: int = 0
        self.last_precomputed_count: int = 0
        self._scraper = SofascoreScraper(rate_limit_seconds=0.6)

    @classmethod
    def get_instance(cls) -> "HeartbeatScheduler":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(self) -> None:
        """Inicia el hilo demonio de fondo si aún no está corriendo."""
        with self._lock:
            if self.is_running and self.thread and self.thread.is_alive():
                return
            self.is_running = True
            self.thread = threading.Thread(
                target=self._run_loop, name="HeartbeatSchedulerThread", daemon=True
            )
            self.thread.start()
            logger.info("HeartbeatScheduler iniciado exitosamente (intervalo: %ss)", self.interval_seconds)

    def trigger_now(self) -> Dict[str, Any]:
        """Dispara un ciclo de actualización de inmediato."""
        logger.info("Disparando ciclo manual inmediato de Heartbeat...")
        return self._execute_cycle()

    def get_status(self) -> Dict[str, Any]:
        """Retorna el estado de salud y métricas del scheduler para la UI."""
        now_col = datetime.now(COLOMBIA_TZ)
        last_str = self.last_run.astimezone(COLOMBIA_TZ).strftime("%H:%M:%S") if self.last_run else "Pendiente"
        next_str = self.next_run.astimezone(COLOMBIA_TZ).strftime("%H:%M:%S") if self.next_run else "Pendiente"
        mins_left = max(0, int((self.next_run - datetime.now(timezone.utc)).total_seconds() / 60)) if self.next_run else 0

        return {
            "is_running": self.is_running,
            "cycles_count": self.cycles_count,
            "last_run_str": last_str,
            "next_run_str": next_str,
            "mins_until_next": mins_left,
            "last_status": self.last_status,
            "last_purged": self.last_purged,
            "last_settled": self.last_settled_count,
            "last_precomputed": self.last_precomputed_count,
        }

    def _run_loop(self) -> None:
        """Bucle continuo en segundo plano cada 14 minutos."""
        # Ejecutar primer ciclo rápido al inicio
        time.sleep(5)
        while self.is_running:
            try:
                self._execute_cycle()
            except Exception as e:
                logger.error("Error inesperado en ciclo de HeartbeatScheduler: %s", e)
                self.last_status = f"Error: {e}"

            self.next_run = datetime.now(timezone.utc) + timedelta(seconds=self.interval_seconds)
            # Esperar el intervalo respetando la bandera de apagado
            slept = 0
            while slept < self.interval_seconds and self.is_running:
                time.sleep(2)
                slept += 2

    def _execute_cycle(self) -> Dict[str, Any]:
        """Ejecuta un ciclo completo de mantenimiento."""
        start_t = time.time()
        self.last_run = datetime.now(timezone.utc)
        self.last_status = "Ejecutando pulso..."

        # 1. PING A SUPABASE (Evita suspensión de 7 días)
        db_alive = ping_database()
        if not db_alive:
            logger.warning("Fallo en ping_database de Supabase.")

        # 2. PURGA AUTOMÁTICA ZERO-WASTE (Partidos > 3 días)
        try:
            self.last_purged = purge_old_records(days_to_keep=3)
        except Exception as e:
            logger.warning("Error en purga de BD: %s", e)

        # 3. LIQUIDACIÓN DE APUESTAS PENDIENTES DEL PORTAFOLIO
        settled_count = 0
        try:
            settled_count = self._settle_pending_portfolio_bets()
            self.last_settled_count = settled_count
        except Exception as e:
            logger.warning("Error liquidando apuestas pendientes: %s", e)

        # 4. PRECOMPUTACIÓN DE PARTIDOS DE HOY (Turbo Cache)
        precomputed_count = 0
        try:
            precomputed_count = self._precompute_today_top_matches()
            self.last_precomputed_count = precomputed_count
        except Exception as e:
            logger.warning("Error precomputando partidos de hoy: %s", e)

        elapsed = round(time.time() - start_t, 2)
        self.cycles_count += 1
        self.last_status = (
            f"Completado en {elapsed}s • Supabase: {'Activo' if db_alive else 'Error'} • "
            f"Precomputados: {precomputed_count} • Apuestas cerradas: {settled_count}"
        )
        logger.info("Ciclo #%s finalizado: %s", self.cycles_count, self.last_status)

        return self.get_status()

    def _settle_pending_portfolio_bets(self) -> int:
        """Consulta marcadores de apuestas pendientes y liquida su P&L."""
        from src.analytics.settlement_engine import audit_and_settle_open_bets

        session = SessionLocal()
        try:
            pending_bets = session.query(PortfolioBetORM).filter_by(status="PENDING").all()
            if not pending_bets:
                return 0

            # Coleccionar IDs únicos de eventos pendientes
            event_ids = {b.event_id for b in pending_bets}
            scores_map: Dict[int, Dict[str, Any]] = {}

            # Buscar primero en MatchRecord local
            for eid in event_ids:
                m_rec = session.query(MatchRecord).filter_by(event_id=eid).first()
                if m_rec and m_rec.home_score is not None:
                    scores_map[eid] = {
                        "home_score": m_rec.home_score,
                        "away_score": m_rec.away_score,
                        "status": m_rec.status_type,
                    }

            # Si faltan eventos, consultar rápidamente a Sofascore
            missing_eids = event_ids - set(scores_map.keys())
            for eid in list(missing_eids)[:6]:
                try:
                    ev_data = self._scraper.get_json(f"https://www.sofascore.com/api/v1/event/{eid}")
                    ev_info = ev_data.get("event", {})
                    st_type = ev_info.get("status", {}).get("type")
                    if st_type == "finished":
                        hs = ev_info.get("homeScore", {}).get("current")
                        as_ = ev_info.get("awayScore", {}).get("current")
                        if hs is not None and as_ is not None:
                            scores_map[eid] = {
                                "home_score": int(hs),
                                "away_score": int(as_),
                                "status": "finished",
                            }
                except Exception:
                    pass

            def _scores_provider(event_id: int, match_date: str) -> Optional[Dict[str, Any]]:
                return scores_map.get(event_id)

            report = audit_and_settle_open_bets(session, _scores_provider)
            return report.settled_count
        finally:
            session.close()

    def _precompute_single_match(self, m_item, today_str: str) -> int:
        from src.analytics.predictor import compute_team_form, predict_match_poisson
        try:
            if m_item.odds is None:
                m_item.odds = self._scraper.get_match_odds(
                    m_item.event_id,
                    is_live=m_item.status_type in ("inprogress", "live")
                )

            home_recent = self._scraper.get_team_recent_matches(m_item.home_team.id)
            away_recent = self._scraper.get_team_recent_matches(m_item.away_team.id)

            hf = compute_team_form(m_item.home_team.id, m_item.home_team.name, home_recent)
            af = compute_team_form(m_item.away_team.id, m_item.away_team.name, away_recent)

            prediction = predict_match_poisson(m_item, hf, af)

            save_precomputed_analysis(
                event_id=m_item.event_id,
                match_date=today_str,
                tournament_name=m_item.tournament.name if m_item.tournament else "Liga",
                home_team_name=m_item.home_team.name,
                away_team_name=m_item.away_team.name,
                status_type=m_item.status_type,
                match_detail=m_item.model_dump(),
                prediction=prediction.model_dump(),
                deep_analysis={
                    "match": m_item.model_dump(),
                    "prediction": prediction.model_dump(),
                    "home_form": hf.model_dump(),
                    "away_form": af.model_dump(),
                },
            )
            return 1
        except Exception as e:
            logger.debug("Error precomputando evento %s: %s", m_item.event_id, e)
            return 0

    def _precompute_today_top_matches(self) -> int:
        """Precomputa en paralelo con el modelo Poisson y cuotas los principales partidos de hoy."""
        today_str = date.today().strftime("%Y-%m-%d")

        try:
            matches = self._scraper.get_scheduled_events(today_str, only_top_tournaments=True)
        except Exception:
            matches = []

        if not matches:
            return 0

        pending = [
            m for m in matches
            if m.status_type in ("notstarted", "scheduled", "inprogress", "live")
        ]
        if not pending:
            return 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._precompute_single_match, m_item, today_str)
                for m_item in pending[:8]
            ]
            saved = sum(f.result() for f in concurrent.futures.as_completed(futures))

        return saved


def start_heartbeat_scheduler(interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> HeartbeatScheduler:
    """Función de conveniencia para arrancar el scheduler."""
    scheduler = HeartbeatScheduler.get_instance()
    scheduler.interval_seconds = interval_seconds
    scheduler.start()
    return scheduler


def get_heartbeat_scheduler() -> HeartbeatScheduler:
    """Obtiene la instancia del scheduler."""
    return HeartbeatScheduler.get_instance()
