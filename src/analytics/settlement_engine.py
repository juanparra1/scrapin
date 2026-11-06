"""Motor determinista de liquidación contable y auditoría de apuestas deportivas.

Biblioteca de lógica pura (SIN dependencias de Streamlit) para:
- Evaluación matemática formal de mercados (1X2, Over/Under, BTTS, Doble Oportunidad, Marcador Exacto, Hándicaps).
- Liquidación atómica de apuestas PENDING con actualización contable en BankrollLedgerORM.
- Prevención de fallos en cascada: captura selectiva por partido sin abortar la auditoría general.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.database import BankrollLedgerORM, PortfolioBetORM

logger = logging.getLogger("ProSportsAI.SettlementEngine")


@dataclass
class SettledBetResult:
    """Detalle de una apuesta individual auditada y liquidada."""
    bet_id: int
    event_id: int
    event_name: str
    pick_name: str
    market_code: str
    odds_taken: float
    stake_amount: float
    status: str  # WON, LOST, VOID_PUSH
    pnl: float
    final_score: str


@dataclass
class SettlementReport:
    """Informe consolidado de una ronda de liquidación."""
    total_audited: int = 0
    settled_count: int = 0
    won_count: int = 0
    lost_count: int = 0
    void_count: int = 0
    still_pending_count: int = 0
    total_pnl: float = 0.0
    bankroll_delta: float = 0.0
    new_bankroll: float = 0.0
    settled_bets: List[SettledBetResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# EVALUADOR DETERMINISTA DE MERCADOS DEPORTIVOS
# ============================================================================


def evaluate_bet(
    market_code: str,
    home_score: Optional[int],
    away_score: Optional[int],
    match_status: str,
    extra_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Evalúa matemáticamente una apuesta contra el resultado final del encuentro.

    Retorna: (status, reason)
    - status: "WON", "LOST", "VOID_PUSH", "PENDING"
    - reason: Descripción humana del motivo de la resolución
    """
    status_lower = str(match_status).lower().strip()

    # 1. Partidos suspendidos, cancelados o aplazados -> Anulación contable con reintegro total
    if any(term in status_lower for term in ("cancel", "postpon", "abandon", "suspend", "interrup", "walkover")):
        return "VOID_PUSH", f"Partido {match_status} - Stake reembolsado"

    # Si aún no ha concluido el partido, mantener en espera
    if status_lower not in ("finished", "ft", "ended", "3", "finalizado"):
        return "PENDING", f"Partido aún no finalizado (Estado: {match_status})"

    if home_score is None or away_score is None:
        return "PENDING", "Marcador final aún no disponible"

    code = market_code.upper().strip()
    total_goals = home_score + away_score

    # 2. Mercado 1X2 (Resultado Final)
    if code in ("1X2_HOME", "HOME_WIN", "1"):
        if home_score > away_score:
            return "WON", f"Victoria Local ({home_score}-{away_score})"
        return "LOST", f"No ganó el Local ({home_score}-{away_score})"

    if code in ("1X2_DRAW", "DRAW", "X"):
        if home_score == away_score:
            return "WON", f"Empate ({home_score}-{away_score})"
        return "LOST", f"No hubo empate ({home_score}-{away_score})"

    if code in ("1X2_AWAY", "AWAY_WIN", "2"):
        if away_score > home_score:
            return "WON", f"Victoria Visitante ({home_score}-{away_score})"
        return "LOST", f"No ganó el Visitante ({home_score}-{away_score})"

    # 3. Mercado Over / Under Goles (Totales)
    if "OU_" in code or "OVER_" in code or "UNDER_" in code:
        parts = code.split("_")
        try:
            # Soportar formatos: OU_2.5_OVER, OU_1.5_UNDER, OVER_2.5, UNDER_1.5
            if "OVER" in code:
                line_str = [p for p in parts if p.replace(".", "", 1).isdigit()][0]
                line = float(line_str)
                if total_goals > line:
                    return "WON", f"Total goles ({total_goals}) > {line}"
                elif total_goals == line:
                    return "VOID_PUSH", f"Línea entera igualada ({total_goals} == {line})"
                return "LOST", f"Total goles ({total_goals}) <= {line}"
            elif "UNDER" in code:
                line_str = [p for p in parts if p.replace(".", "", 1).isdigit()][0]
                line = float(line_str)
                if total_goals < line:
                    return "WON", f"Total goles ({total_goals}) < {line}"
                elif total_goals == line:
                    return "VOID_PUSH", f"Línea entera igualada ({total_goals} == {line})"
                return "LOST", f"Total goles ({total_goals}) >= {line}"
        except Exception as e:
            logger.warning(f"Error evaluando línea Over/Under {code}: {e}")

    # 4. Mercado Ambos Equipos Anotan (BTTS)
    if code in ("BTTS_YES", "AMBOS_SI", "GG"):
        if home_score > 0 and away_score > 0:
            return "WON", f"Ambos marcaron ({home_score}-{away_score})"
        return "LOST", f"Uno o ambos no marcaron ({home_score}-{away_score})"

    if code in ("BTTS_NO", "AMBOS_NO", "NG"):
        if home_score == 0 or away_score == 0:
            return "WON", f"Al menos un equipo no marcó ({home_score}-{away_score})"
        return "LOST", f"Ambos anotaron ({home_score}-{away_score})"

    # 5. Mercado Doble Oportunidad
    if code in ("DC_1X", "DOUBLE_CHANCE_1X", "1X"):
        if home_score >= away_score:
            return "WON", f"1X cumplido ({home_score}-{away_score})"
        return "LOST", f"Visitante ganó ({home_score}-{away_score})"

    if code in ("DC_X2", "DOUBLE_CHANCE_X2", "X2"):
        if away_score >= home_score:
            return "WON", f"X2 cumplido ({home_score}-{away_score})"
        return "LOST", f"Local ganó ({home_score}-{away_score})"

    if code in ("DC_12", "DOUBLE_CHANCE_12", "12"):
        if home_score != away_score:
            return "WON", f"12 cumplido sin empate ({home_score}-{away_score})"
        return "LOST", f"Hubo empate ({home_score}-{away_score})"

    # 6. Mercado Marcador Exacto
    if "SCORE_" in code or "EXACT_" in code:
        # Formato esperado: SCORE_2-1, EXACT_0-0
        for part in code.split("_"):
            if "-" in part:
                h_target, a_target = part.split("-")
                if h_target.isdigit() and a_target.isdigit():
                    if home_score == int(h_target) and away_score == int(a_target):
                        return "WON", f"Marcador exacto {home_score}-{away_score} acertado"
                    return "LOST", f"Marcador real {home_score}-{away_score} != {part}"

    # 7. Mercado Hándicap Asiático Simple (AH)
    if "AH_" in code:
        try:
            # Ejemplo: AH_HOME_-0.5, AH_AWAY_+1.0
            parts = code.split("_")
            side = "HOME" if "HOME" in parts else "AWAY"
            line_str = parts[-1].replace("+", "")
            ah_line = float(line_str)

            diff = (home_score - away_score) if side == "HOME" else (away_score - home_score)
            effective_diff = diff + ah_line

            if effective_diff > 0:
                return "WON", f"Hándicap {side} {ah_line:+} cubierto (Diff: {diff:+})"
            elif effective_diff == 0:
                return "VOID_PUSH", f"Hándicap {side} {ah_line:+} empatado (Push exacto)"
            else:
                return "LOST", f"Hándicap {side} {ah_line:+} fallido (Diff: {diff:+})"
        except Exception as e:
            logger.warning(f"Error evaluando handicap asiático {code}: {e}")

    # Fallback genérico por nombre de selección textual
    return "PENDING", f"Mercado {market_code} requiere resolución manual o datos extendidos"


# ============================================================================
# AUDITORÍA Y SINCRONIZACIÓN CONTABLE MASIVA
# ============================================================================


def audit_and_settle_open_bets(
    session,
    scores_provider_fn: Callable[[int, str], Optional[Dict[str, Any]]],
) -> SettlementReport:
    """
    Audita y liquida atómicamente todas las apuestas en estado 'PENDING'.

    Parámetros:
    - session: Sesión activa de SQLAlchemy (SQLite o PostgreSQL en Supabase).
    - scores_provider_fn: Función (event_id, custom_id) -> Dict con:
        {"home_score": int, "away_score": int, "status": str}

    Regla de resiliencia:
    Ningún error individual detiene la liquidación de las demás apuestas.
    """
    report = SettlementReport()

    pending_bets: List[PortfolioBetORM] = (
        session.query(PortfolioBetORM)
        .filter(PortfolioBetORM.status == "PENDING")
        .order_by(PortfolioBetORM.id.asc())
        .all()
    )

    report.total_audited = len(pending_bets)
    if not pending_bets:
        # Obtener banca actual
        last_tx = session.query(BankrollLedgerORM).order_by(BankrollLedgerORM.id.desc()).first()
        report.new_bankroll = last_tx.resulting_balance if last_tx else 1000.0
        return report

    # Obtener el saldo contable previo más reciente
    last_tx = session.query(BankrollLedgerORM).order_by(BankrollLedgerORM.id.desc()).first()
    running_balance = last_tx.resulting_balance if last_tx else 1000.0

    now_utc = datetime.utcnow()

    for bet in pending_bets:
        try:
            match_data = scores_provider_fn(bet.event_id, bet.match_date)
            if not match_data:
                report.still_pending_count += 1
                continue

            h_score = match_data.get("home_score")
            a_score = match_data.get("away_score")
            match_status = match_data.get("status", "notstarted")

            status, reason = evaluate_bet(
                market_code=bet.market_code,
                home_score=h_score,
                away_score=a_score,
                match_status=match_status,
                extra_stats=match_data.get("extra_stats"),
            )

            if status == "PENDING":
                report.still_pending_count += 1
                continue

            # Calcular PnL y movimiento contable exacto
            stake = bet.stake_amount
            pnl = 0.0
            credit_amount = 0.0
            desc = ""

            if status == "WON":
                # Ganancia neta: stake * (odds - 1)
                pnl = round(stake * (bet.odds_taken - 1.0), 2)
                # Retorno bruto que regresa a la banca: stake + pnl
                credit_amount = round(stake + pnl, 2)
                desc = f"Apuesta GANADA (+${pnl:.2f}): {bet.pick_name} ({reason})"
                report.won_count += 1

            elif status == "LOST":
                pnl = round(-stake, 2)
                # En apuesta perdida, el stake ya se había deducido al colocarla; no hay crédito de vuelta
                credit_amount = 0.0
                desc = f"Apuesta PERDIDA (-${stake:.2f}): {bet.pick_name} ({reason})"
                report.lost_count += 1

            elif status == "VOID_PUSH":
                pnl = 0.0
                # En anulación/push, se reintegra el 100% del stake original
                credit_amount = round(stake, 2)
                desc = f"Apuesta ANULADA (Reintegro ${stake:.2f}): {bet.pick_name} ({reason})"
                report.void_count += 1

            # Actualizar registro ORM de la apuesta
            bet.status = status
            bet.pnl = pnl
            bet.settled_at = now_utc
            session.add(bet)

            # Si hay retorno de capital (ganancia o reintegro), asentar en el libro diario
            if credit_amount > 0.0:
                running_balance = round(running_balance + credit_amount, 2)
                ledger_entry = BankrollLedgerORM(
                    timestamp=now_utc,
                    transaction_type="BET_SETTLED",
                    amount=credit_amount,
                    resulting_balance=running_balance,
                    description=desc,
                    bet_id=bet.id,
                )
                session.add(ledger_entry)

            report.settled_count += 1
            report.total_pnl = round(report.total_pnl + pnl, 2)
            report.bankroll_delta = round(report.bankroll_delta + (credit_amount - stake if status == "LOST" else pnl), 2)

            score_label = f"{h_score}-{a_score}" if h_score is not None and a_score is not None else "N/A"
            report.settled_bets.append(
                SettledBetResult(
                    bet_id=bet.id,
                    event_id=bet.event_id,
                    event_name=bet.event_name,
                    pick_name=bet.pick_name,
                    market_code=bet.market_code,
                    odds_taken=bet.odds_taken,
                    stake_amount=bet.stake_amount,
                    status=status,
                    pnl=pnl,
                    final_score=score_label,
                )
            )

        except Exception as bet_err:
            err_msg = f"Error auditando apuesta ID={bet.id} ({bet.event_name}): {bet_err}"
            logger.error(err_msg, exc_info=True)
            report.errors.append(err_msg)

    # Persistir todos los cambios transaccionales
    session.flush()
    report.new_bankroll = running_balance
    return report
