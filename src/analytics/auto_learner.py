"""Motor de Auto-Aprendizaje y Calibración Continua (Closed-Loop Autonomous Learning).

Aprende de aciertos y errores en partidos finalizados:
1. Ajuste Bayesiano de Pesos del Ensamble (Dixon-Coles vs ML Gradient Boosting según Brier Score).
2. Calibración y Atenuación de Probabilidades por Línea (Shrinkage empírico para evitar sobreconfianza).
3. Algoritmo de Bandido Multibrazo (Thompson Sampling) para priorizar selecciones de alta confianza.
4. Reentrenamiento autónomo del clasificador Gradient Boosting con partidos guardados en SQLite.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import DATA_DIR
from src.models.schemas import MatchDetailSchema

LEARNED_STATE_PATH = Path(DATA_DIR) / "learned_weights.json"

DEFAULT_CANONICAL_LINES = [
    "⚽ Over 1.5 Goles FT (+1.5)",
    "⚽ Over 2.5 Goles FT (+2.5)",
    "⚽ Under 2.5 Goles FT (-2.5)",
    "⚽ Under 3.5 Goles FT (-3.5)",
    "🔥 Ambos Equipos Anotan: SÍ (BTTS)",
    "⛔ Ambos Equipos Anotan: NO",
    "⏱️ Over 0.5 Goles 1H (+0.5 1er Tiempo)",
    "⏱️ Under 1.5 Goles 1H (-1.5 1er Tiempo)",
    "🚩 Over 6.5 Córners (+6.5 Tiros de Esquina)",
    "🚩 Over 7.5 Córners (+7.5 Tiros de Esquina)",
    "🚩 Over 8.5 Córners (+8.5 Tiros de Esquina)",
    "🚩 Over 9.5 Córners (+9.5 Tiros de Esquina)",
    "🚩 Under 10.5 Córners (-10.5 Tiros de Esquina)",
    "🎯 Over 6.5 Remates al Arco (+6.5 SoT)",
    "🎯 Over 7.5 Remates al Arco (+7.5 SoT)",
    "🎯 Over 8.5 Remates al Arco (+8.5 SoT)",
    "🎯 Under 8.5 Remates al Arco",
    "💥 Over 20.5 Remates Totales",
    "💥 Over 22.5 Remates Totales",
    "💥 Over 24.5 Remates Totales",
    "🟨 Over 3.5 Tarjetas Amarillas",
    "🟨 Over 4.5 Tarjetas Amarillas",
    "🟨 Under 4.5 Tarjetas Amarillas",
    "🧤 Portero: 2+ Atajadas (+1.5 Saves)",
    "🧤 Portero: 3+ Atajadas (+2.5 Saves)",
    "🛡️ Doble Oportunidad 1X (Local o Empate)",
    "🛡️ Doble Oportunidad X2 (Visitante o Empate)",
    "⚽ Equipo Individual: Over 1.5 Goles",
]


class AutoLearner:
    """Gestiona el aprendizaje continuo, calibración dinámica y bandidos de confianza."""

    def __init__(self, state_path: Path = LEARNED_STATE_PATH) -> None:
        self.state_path = state_path
        self.state: Dict[str, Any] = self._load_or_init_state()

    def _default_state(self) -> Dict[str, Any]:
        lines_calib: Dict[str, Any] = {}
        bandit_priors: Dict[str, Any] = {}
        for line in DEFAULT_CANONICAL_LINES:
            # Priors no informativos equilibrados (alpha=3, beta=2 ~ 60% inicial)
            lines_calib[line] = {
                "hits": 0,
                "trials": 0,
                "observed_win_rate": 0.60,
                "avg_predicted_prob": 0.60,
                "multiplier": 1.0,
                "status": "Neutral (Sin suficientes datos)",
            }
            bandit_priors[line] = {
                "alpha": 4.0,
                "beta": 2.5,
                "mean_reward": 0.615,
                "status": "Estándar",
            }

        return {
            "version": 1,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "total_matches_learned": 0,
            "learned_sessions_count": 0,
            "ensemble_weights": {
                "weight_dixon_coles": 0.65,
                "weight_ml": 0.35,
            },
            "recent_brier_scores": {
                "dixon_coles_1x2": 0.205,
                "ml_1x2": 0.210,
            },
            "lines_calibration": lines_calib,
            "bandit_priors": bandit_priors,
            "training_history": [],
        }

    def _load_or_init_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Asegurar claves mínimas
                    if "ensemble_weights" in data and "lines_calibration" in data:
                        return data
            except Exception:
                pass
        st = self._default_state()
        self._save_state(st)
        return st

    def _save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        if state is not None:
            self.state = state
        self.state["last_updated"] = datetime.utcnow().isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def reset_learning(self) -> None:
        """Restablece el cerebro a su configuración de fábrica."""
        self.state = self._default_state()
        self._save_state()

    def get_ensemble_weights(self) -> Tuple[float, float]:
        """Retorna (peso_dixon_coles, peso_ml) aprendidos y equilibrados."""
        ew = self.state.get("ensemble_weights", {})
        w_d = float(ew.get("weight_dixon_coles", 0.65))
        w_m = float(ew.get("weight_ml", 0.35))
        tot = max(0.01, w_d + w_m)
        return round(w_d / tot, 3), round(w_m / tot, 3)

    def calibrate_line_probability(self, line_name: str, raw_prob: float) -> float:
        """
        Ajusta la probabilidad predicha usando el multiplicador bayesiano aprendido.
        Evita sobreconfianza si la línea suele fallar, y refuerza líneas estrella.
        """
        lc = self.state.get("lines_calibration", {}).get(line_name)
        if not lc or lc.get("trials", 0) < 5:
            return raw_prob

        mult = float(lc.get("multiplier", 1.0))
        # Suavizar el multiplicador entre 0.75 y 1.25 para no distorsionar en extremo
        bounded_mult = max(0.75, min(1.25, mult))
        calibrated = raw_prob * bounded_mult
        return round(float(np.clip(calibrated, 4.0, 96.0)), 1)

    def get_bandit_line_score(self, line_name: str) -> float:
        """
        Retorna la prioridad de selección de la línea según Thompson Sampling / Recompensa.
        Valor entre 0.0 y 1.0.
        """
        bp = self.state.get("bandit_priors", {}).get(line_name)
        if not bp:
            return 0.50
        alpha = float(bp.get("alpha", 4.0))
        beta_val = float(bp.get("beta", 2.5))
        # Media posterior de la distribución Beta
        return round(alpha / (alpha + beta_val), 3)

    def learn_from_audited_matches(
        self,
        audited_pairs: List[Tuple[MatchDetailSchema, Any]],
        systematic_lines_evaluator_fn: Any,
    ) -> Dict[str, Any]:
        """
        Ejecuta el ciclo de aprendizaje completo sobre un conjunto de partidos auditados:
        1. Evalúa Brier Scores y ajusta pesos del ensamble (Dixon-Coles vs ML).
        2. Actualiza la calibración y multiplicadores de las 28 líneas.
        3. Actualiza los priors de Thompson Sampling para el ranking de líneas.
        4. Reentrena el clasificador Gradient Boosting con los nuevos datos de SQLite.
        """
        if not audited_pairs:
            return {"status": "warning", "message": "No hay partidos auditados para aprender."}

        # 1. EVALUAR BRIER SCORES DE 1X2 (DIXON-COLES VS ML)
        dixon_briers: List[float] = []
        ml_briers: List[float] = []
        line_evals: Dict[str, List[Tuple[float, bool]]] = {}

        for match, pred in audited_pairs:
            hs = match.home_score
            as_ = match.away_score
            if hs is None or as_ is None:
                continue

            # Vector One-Hot del resultado real [Home, Draw, Away]
            y_real = [1.0 if hs > as_ else 0.0, 1.0 if hs == as_ else 0.0, 1.0 if hs < as_ else 0.0]

            # Probabilidades predichas por Dixon-Coles puro
            p_dixon = [
                pred.stat_prob_home / 100.0,
                pred.stat_prob_draw / 100.0,
                pred.stat_prob_away / 100.0,
            ]
            tot_d = max(0.001, sum(p_dixon))
            p_dixon = [v / tot_d for v in p_dixon]

            # Probabilidades predichas por ML puro
            p_ml = [
                pred.ml_prob_home / 100.0,
                pred.ml_prob_draw / 100.0,
                pred.ml_prob_away / 100.0,
            ]
            tot_m = max(0.001, sum(p_ml))
            p_ml = [v / tot_m for v in p_ml]

            brier_d = sum((p - y) ** 2 for p, y in zip(p_dixon, y_real)) / 3.0
            brier_m = sum((p - y) ** 2 for p, y in zip(p_ml, y_real)) / 3.0

            dixon_briers.append(brier_d)
            ml_briers.append(brier_m)

            # Evaluar líneas sistemáticas de este partido
            lines_in_match = systematic_lines_evaluator_fn(match, pred)
            for item in lines_in_match:
                lname = item["line_name"]
                lprob = float(item["prob"]) / 100.0
                lhit = bool(item["hit"])
                line_evals.setdefault(lname, []).append((lprob, lhit))

        # 2. ADAPTAR PESOS DEL ENSAMBLE (ONLINE BAYESIAN UPDATE)
        old_w_dixon, old_w_ml = self.get_ensemble_weights()
        avg_brier_d = float(np.mean(dixon_briers)) if dixon_briers else 0.20
        avg_brier_m = float(np.mean(ml_briers)) if ml_briers else 0.21

        # Si ML tiene menor Brier score (menor error), premiar a ML; de lo contrario, premiar a Dixon
        delta_brier = avg_brier_d - avg_brier_m
        learning_rate = 0.05
        adjustment = float(np.clip(delta_brier * learning_rate * 5.0, -0.06, 0.06))

        new_w_ml = float(np.clip(old_w_ml + adjustment, 0.20, 0.75))
        new_w_dixon = round(1.0 - new_w_ml, 3)
        new_w_ml = round(new_w_ml, 3)

        self.state["ensemble_weights"]["weight_dixon_coles"] = new_w_dixon
        self.state["ensemble_weights"]["weight_ml"] = new_w_ml
        self.state["recent_brier_scores"]["dixon_coles_1x2"] = round(avg_brier_d, 4)
        self.state["recent_brier_scores"]["ml_1x2"] = round(avg_brier_m, 4)

        # 3. ACTUALIZAR CALIBRACIÓN Y BANDIDO PARA CADA LÍNEA
        lines_updated_summary: List[Dict[str, Any]] = []
        for lname, outcomes in line_evals.items():
            if lname not in self.state["lines_calibration"]:
                self.state["lines_calibration"][lname] = {
                    "hits": 0,
                    "trials": 0,
                    "observed_win_rate": 0.60,
                    "avg_predicted_prob": 0.60,
                    "multiplier": 1.0,
                    "status": "Inicial",
                }
            if lname not in self.state["bandit_priors"]:
                self.state["bandit_priors"][lname] = {
                    "alpha": 4.0,
                    "beta": 2.5,
                    "mean_reward": 0.615,
                    "status": "Estándar",
                }

            cur_calib = self.state["lines_calibration"][lname]
            cur_bandit = self.state["bandit_priors"][lname]

            batch_hits = sum(1 for _, h in outcomes if h)
            batch_trials = len(outcomes)
            batch_avg_pred = float(np.mean([p for p, _ in outcomes])) if outcomes else 0.60

            # Actualizar historial acumulado
            tot_h = cur_calib.get("hits", 0) + batch_hits
            tot_t = cur_calib.get("trials", 0) + batch_trials
            obs_rate = (tot_h / tot_t) if tot_t > 0 else 0.60

            # Bayesian shrinkage hacia la predicción teórica para calibrar
            # Multiplicador = (Tasa real observada / Probabilidad predicha promedio)
            # Con regularización para evitar saltos bruscos con muestras pequeñas
            prior_trials = 8.0
            smooth_obs = (tot_h + (prior_trials * 0.60)) / (tot_t + prior_trials)
            multiplier = smooth_obs / max(0.20, batch_avg_pred)
            multiplier = round(float(np.clip(multiplier, 0.78, 1.22)), 3)

            # Clasificar estatus de la línea
            if obs_rate >= 0.75 and tot_t >= 4:
                status_label = "🌟 Línea Estrella (Máxima Confianza Estadística)"
            elif obs_rate >= 0.55:
                status_label = "🟢 Línea Estable / Confiable"
            elif obs_rate >= 0.42:
                status_label = "🟡 Línea Moderada (Neutro)"
            else:
                status_label = "⚠️ Línea Penalizada por Alta Tasa de Fallo"

            cur_calib["hits"] = tot_h
            cur_calib["trials"] = tot_t
            cur_calib["observed_win_rate"] = round(obs_rate * 100.0, 1)
            cur_calib["avg_predicted_prob"] = round(batch_avg_pred * 100.0, 1)
            cur_calib["multiplier"] = multiplier
            cur_calib["status"] = status_label

            # Actualizar prior de Thompson Sampling
            cur_bandit["alpha"] = round(cur_bandit.get("alpha", 4.0) + (batch_hits * 0.7), 2)
            cur_bandit["beta"] = round(cur_bandit.get("beta", 2.5) + ((batch_trials - batch_hits) * 0.7), 2)
            cur_bandit["mean_reward"] = round(cur_bandit["alpha"] / (cur_bandit["alpha"] + cur_bandit["beta"]), 3)
            cur_bandit["status"] = status_label

            lines_updated_summary.append({
                "line_name": lname,
                "batch_hits": batch_hits,
                "batch_trials": batch_trials,
                "cum_win_rate": round(obs_rate * 100.0, 1),
                "multiplier": multiplier,
                "status": status_label,
            })

        # 4. REENTRENAR EL MODELO GRADIENT BOOSTING CON DATOS RECIENTES DE SQLITE
        from src.analytics.ml_ensemble import get_ml_predictor
        ml_p = get_ml_predictor()
        ml_p._fit_model()

        # Guardar en el historial de sesiones
        self.state["total_matches_learned"] += len(audited_pairs)
        self.state["learned_sessions_count"] += 1

        history_entry = {
            "session_id": self.state["learned_sessions_count"],
            "timestamp": datetime.utcnow().isoformat(),
            "matches_processed": len(audited_pairs),
            "old_weights": {"dixon": old_w_dixon, "ml": old_w_ml},
            "new_weights": {"dixon": new_w_dixon, "ml": new_w_ml},
            "brier_dixon": round(avg_brier_d, 4),
            "brier_ml": round(avg_brier_m, 4),
            "db_records_in_ml": ml_p.db_matches_count,
        }
        self.state.setdefault("training_history", []).append(history_entry)
        if len(self.state["training_history"]) > 25:
            self.state["training_history"] = self.state["training_history"][-25:]

        self._save_state()

        # Ordenar líneas actualizadas para el reporte
        lines_updated_summary.sort(key=lambda x: x["cum_win_rate"], reverse=True)

        return {
            "status": "success",
            "matches_processed": len(audited_pairs),
            "total_matches_learned": self.state["total_matches_learned"],
            "old_weights": {"dixon": old_w_dixon, "ml": old_w_ml},
            "new_weights": {"dixon": new_w_dixon, "ml": new_w_ml},
            "delta_ml_pct": round((new_w_ml - old_w_ml) * 100.0, 1),
            "brier_dixon": round(avg_brier_d, 4),
            "brier_ml": round(avg_brier_m, 4),
            "top_promoted_lines": lines_updated_summary[:4],
            "top_penalized_lines": [item for item in lines_updated_summary if "Penalizada" in item["status"]][:4],
            "db_records_in_ml": ml_p.db_matches_count,
        }

    def learn_from_settled_portfolio_bets(
        self,
        settled_bets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Retroalimenta el motor de aprendizaje a partir de las apuestas reales liquidadas del portafolio.
        
        Actualiza:
        1. Calibración empírica de líneas (penaliza si acierto real < probabilidad modelo).
        2. Priors beta de Thompson Sampling (premia líneas rentables, penaliza líneas con drawdown).
        """
        if not settled_bets:
            return {"status": "neutral", "message": "No hay apuestas liquidadas para calibrar."}

        # Mapeo de códigos de mercado a líneas canónicas
        market_mapping = {
            "1X2_HOME": "🛡️ Doble Oportunidad 1X (Local o Empate)",
            "1X2_AWAY": "🛡️ Doble Oportunidad X2 (Visitante o Empate)",
            "OU_1.5_OVER": "⚽ Over 1.5 Goles FT (+1.5)",
            "OU_2.5_OVER": "⚽ Over 2.5 Goles FT (+2.5)",
            "OU_2.5_UNDER": "⚽ Under 2.5 Goles FT (-2.5)",
            "OU_3.5_UNDER": "⚽ Under 3.5 Goles FT (-3.5)",
            "BTTS_YES": "🔥 Ambos Equipos Anotan: SÍ (BTTS)",
            "BTTS_NO": "⛔ Ambos Equipos Anotan: NO",
            "DC_1X": "🛡️ Doble Oportunidad 1X (Local o Empate)",
            "DC_X2": "🛡️ Doble Oportunidad X2 (Visitante o Empate)",
        }

        updated_lines: List[str] = []
        for bet in settled_bets:
            status = bet.get("status")
            if status not in ("WON", "LOST"):
                continue

            m_code = bet.get("market_code", "")
            target_line = market_mapping.get(m_code)
            if not target_line:
                # Buscar por coincidencia parcial en líneas por defecto
                for default_l in DEFAULT_CANONICAL_LINES:
                    if m_code.lower() in default_l.lower():
                        target_line = default_l
                        break
            if not target_line:
                target_line = f"Mercado: {m_code}"

            if target_line not in self.state["lines_calibration"]:
                self.state["lines_calibration"][target_line] = {
                    "hits": 0,
                    "trials": 0,
                    "observed_win_rate": 0.60,
                    "avg_predicted_prob": 0.60,
                    "multiplier": 1.0,
                    "status": "Inicial",
                }
            if target_line not in self.state["bandit_priors"]:
                self.state["bandit_priors"][target_line] = {
                    "alpha": 4.0,
                    "beta": 2.5,
                    "mean_reward": 0.615,
                    "status": "Estándar",
                }

            calib = self.state["lines_calibration"][target_line]
            bandit = self.state["bandit_priors"][target_line]

            is_hit = 1 if status == "WON" else 0
            model_prob = float(bet.get("model_probability", 0.60))

            calib["hits"] = calib.get("hits", 0) + is_hit
            calib["trials"] = calib.get("trials", 0) + 1
            obs_rate = calib["hits"] / max(1, calib["trials"])
            calib["observed_win_rate"] = round(obs_rate * 100.0, 1)

            # Calibración: penaliza si acierto real < prob modelo
            prior_n = 6.0
            smooth_rate = (calib["hits"] + (prior_n * 0.60)) / (calib["trials"] + prior_n)
            mult = smooth_rate / max(0.20, model_prob)
            calib["multiplier"] = round(float(np.clip(mult, 0.75, 1.25)), 3)

            # Thompson Sampling update: +alpha si gana, +beta si pierde
            if status == "WON":
                bandit["alpha"] = round(bandit.get("alpha", 4.0) + 1.0, 2)
            else:
                bandit["beta"] = round(bandit.get("beta", 2.5) + 1.2, 2)
            bandit["mean_reward"] = round(bandit["alpha"] / (bandit["alpha"] + bandit["beta"]), 3)

            if obs_rate >= 0.70 and calib["trials"] >= 3:
                status_str = "🌟 Línea Estrella (Rentable en Portafolio)"
            elif obs_rate >= 0.52:
                status_str = "🟢 Línea Estable / Confiable"
            elif obs_rate >= 0.40:
                status_str = "🟡 Línea Neutra"
            else:
                status_str = "⚠️ Línea Penalizada por Drawdown"

            calib["status"] = status_str
            bandit["status"] = status_str
            updated_lines.append(target_line)

        self._save_state()
        return {
            "status": "success",
            "bets_learned": len(settled_bets),
            "updated_lines_count": len(set(updated_lines)),
        }



_GLOBAL_AUTO_LEARNER: Optional[AutoLearner] = None


def get_auto_learner() -> AutoLearner:
    """Singleton del motor de auto-aprendizaje."""
    global _GLOBAL_AUTO_LEARNER
    if _GLOBAL_AUTO_LEARNER is None:
        _GLOBAL_AUTO_LEARNER = AutoLearner()
    return _GLOBAL_AUTO_LEARNER
