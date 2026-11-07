"""Servicio de Notificaciones y Alertas por Telegram.

Permite enviar alertas automáticas en tiempo real al canal o chat privado del usuario:
- Apuestas de Alto Valor (+EV >= 8.0%) detectadas por el motor analítico.
- Parlays / Combinadas óptimas del día con probabilidad matemática favorable.
- Resúmenes periódicos de los ciclos de monitoreo.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set
import urllib.parse
import urllib.request
import json

logger = logging.getLogger(__name__)

# Caché en memoria de claves ya notificadas (para evitar spam)
# Clave: f"{event_id}_{market_code}" -> timestamp de envío
_NOTIFIED_BETS: Dict[str, float] = {}
_COOLDOWN_SECONDS = 6 * 3600  # 6 horas de cooldown por apuesta


def get_telegram_credentials() -> Tuple[Optional[str], Optional[str]]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None
    return token, chat_id


def is_telegram_configured() -> bool:
    token, chat_id = get_telegram_credentials()
    return bool(token and chat_id)


def send_telegram_message(text: str, parse_mode: str = "HTML") -> bool:
    """Envía un mensaje formateado a Telegram mediante la API oficial sin librerías externas."""
    token, chat_id = get_telegram_credentials()
    if not token or not chat_id:
        logger.debug("Telegram Bot no configurado (falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID).")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "ScrapinPro/3.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                logger.info("Notificación enviada a Telegram con éxito.")
                return True
            else:
                logger.warning(f"Telegram API respondió con código {response.status}")
                return False
    except Exception as e:
        logger.warning(f"Error enviando mensaje a Telegram: {e}")
        return False


def notify_value_bet(
    match_name: str,
    tournament: str,
    market: str,
    ev_pct: float,
    odds: float,
    confidence_pct: float,
    start_time: str = "",
    smart_money: bool = False,
) -> bool:
    """Envía una alerta elegante de Value Bet (+EV)."""
    smart_badge = "\n🔥 <b>ALERTA SMART MONEY (Caída de cuota detectada)</b>" if smart_money else ""
    time_str = f"\n🕒 <b>Hora:</b> {start_time}" if start_time else ""

    text = (
        f"🚨 <b>¡ALERTA DE VALOR MATEMÁTICO (+EV)!</b>{smart_badge}\n\n"
        f"⚽ <b>Partido:</b> {match_name}\n"
        f"🏆 <b>Torneo:</b> {tournament}{time_str}\n\n"
        f"🎯 <b>Selección:</b> <code>{market}</code>\n"
        f"📈 <b>Cuota Casa:</b> <code>{odds:.2f}</code>\n"
        f"💎 <b>Valor Esperado (+EV):</b> <b>+{ev_pct:.1f}%</b>\n"
        f"📊 <b>Probabilidad Modelo:</b> {confidence_pct:.1f}%\n\n"
        f"🤖 <i>Scrapin Pro v3 • Motor Autónomo Cuántico</i>"
    )
    return send_telegram_message(text)


def notify_parlay_ready(
    parlay_title: str,
    total_odds: float,
    win_prob_pct: float,
    legs: List[str],
) -> bool:
    """Envía una alerta con la cartera o parlay óptimo generado."""
    legs_formatted = "\n".join([f"  ▫️ {leg}" for leg in legs])

    text = (
        f"📦 <b>CARTERA ÓPTIMA / PARLAY DEL DÍA</b>\n\n"
        f"🏆 <b>Estrategia:</b> {parlay_title}\n"
        f"💰 <b>Cuota Total:</b> <code>{total_odds:.2f}</code>\n"
        f"📈 <b>Probabilidad Combinada:</b> {win_prob_pct:.1f}%\n\n"
        f"📝 <b>Selecciones:</b>\n{legs_formatted}\n\n"
        f"💡 <i>Gestión sugerida: Criterio de Kelly Conservador (1-2% del bankroll)</i>\n"
        f"🤖 <i>Scrapin Pro v3</i>"
    )
    return send_telegram_message(text)


def check_and_alert_value_bets(candidates: List[Dict[str, Any]]) -> int:
    """Revisa una lista de candidatos calculados y dispara alertas si superan el umbral EV."""
    if not is_telegram_configured():
        return 0

    now = time.time()
    alerts_sent = 0

    for c in candidates:
        ev = float(c.get("ev", 0.0) or 0.0)
        # Disparar para apuestas con EV >= 8.0%
        if ev < 8.0:
            continue

        event_id = c.get("event_id") or c.get("match_name", "")
        market = c.get("market_name") or c.get("selection") or ""
        dedup_key = f"{event_id}_{market}"

        # Evitar re-enviar la misma apuesta dentro del periodo de enfriamiento
        if dedup_key in _NOTIFIED_BETS:
            if now - _NOTIFIED_BETS[dedup_key] < _COOLDOWN_SECONDS:
                continue

        odds = float(c.get("odds", 1.0) or 1.0)
        prob = float(c.get("model_prob", 0.0) or c.get("confidence", 0.0) or 0.0)
        match_name = c.get("match_name", "Partido")
        tournament = c.get("tournament", "Fútbol")
        start_time = c.get("start_time", "")
        smart_money = bool(c.get("smart_money", False))

        ok = notify_value_bet(
            match_name=match_name,
            tournament=tournament,
            market=market,
            ev_pct=ev,
            odds=odds,
            confidence_pct=prob * 100 if prob <= 1.0 else prob,
            start_time=start_time,
            smart_money=smart_money,
        )
        if ok:
            _NOTIFIED_BETS[dedup_key] = now
            alerts_sent += 1
            # Pausa breve entre mensajes para respetar el rate limit de Telegram
            time.sleep(1.0)

    return alerts_sent
