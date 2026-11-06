"""Módulo de Machine Learning (Gradient Boosting Classifier) entrenado con SQLite + Prior Calibrado."""

from typing import Any, Dict, List
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from src.models.database import MatchRecord, SessionLocal

FEATURE_NAMES = [
    "Diferencial Rating Elo",
    "Diferencial xG Proyectado",
    "Total xG Esperado",
    "Diferencial Remates al Arco",
    "Diferencial Córners",
    "Ventaja Días de Descanso",
    "Prob. Implícita Cuota Local",
    "Prob. Implícita Cuota Empate",
    "Prob. Implícita Cuota Visita",
]


class MLEnsemblePredictor:
    """Clasificador Gradient Boosting multi-clase (1, X, 2) que aprende de SQLite y calibra probabilidades."""

    def __init__(self) -> None:
        self.model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=42,
        )
        self.db_matches_count: int = 0
        self.is_trained: bool = False
        self._fit_model()

    def _generate_calibrated_prior(self, n_samples: int = 450) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Genera muestras base sintéticas calibradas con las leyes empíricas de Poisson/Elo en fútbol."""
        rng = np.random.default_rng(42)
        elo_diff = rng.normal(35.0, 115.0, size=n_samples)
        xg_home = np.clip(1.42 + (elo_diff / 340.0) + rng.normal(0, 0.32, size=n_samples), 0.35, 3.8)
        xg_away = np.clip(1.15 - (elo_diff / 380.0) + rng.normal(0, 0.28, size=n_samples), 0.25, 3.4)
        xg_diff = xg_home - xg_away
        xg_tot = xg_home + xg_away
        sot_diff = xg_diff * 2.9 + rng.normal(0, 1.2, size=n_samples)
        crn_diff = xg_diff * 1.8 + rng.normal(0, 1.5, size=n_samples)
        rest_diff = rng.normal(0.0, 2.2, size=n_samples)

        # Simular resultados reales Poisson con influencia de descanso
        adj_xg_h = np.clip(xg_home * (1.0 + np.clip(rest_diff * 0.015, -0.08, 0.08)), 0.2, 4.5)
        adj_xg_a = np.clip(xg_away * (1.0 - np.clip(rest_diff * 0.015, -0.08, 0.08)), 0.2, 4.5)
        goals_h = rng.poisson(adj_xg_h)
        goals_a = rng.poisson(adj_xg_a)

        # Etiquetas: 0 = Local (1), 1 = Empate (X), 2 = Visita (2)
        y = np.where(goals_h > goals_a, 0, np.where(goals_h == goals_a, 1, 2))

        # Probabilidades implícitas de mercado aproximadas
        raw_p_h = 1.0 / (1.0 + np.exp(-0.95 * xg_diff))
        imp_d = np.clip(0.27 - 0.04 * np.abs(xg_diff), 0.14, 0.33)
        imp_h = (1.0 - imp_d) * raw_p_h
        imp_a = np.clip(1.0 - imp_d - imp_h, 0.05, 0.85)

        X = np.column_stack(
            [elo_diff, xg_diff, xg_tot, sot_diff, crn_diff, rest_diff, imp_h, imp_d, imp_a]
        )
        weights = np.ones(n_samples, dtype=float)
        return X, y, weights

    def _fit_model(self) -> None:
        X_prior, y_prior, w_prior = self._generate_calibrated_prior(450)
        db_rows_X: List[List[float]] = []
        db_rows_y: List[int] = []

        session = SessionLocal()
        try:
            finished_records = (
                session.query(MatchRecord)
                .filter(
                    MatchRecord.status_type == "finished",
                    MatchRecord.home_score.isnot(None),
                    MatchRecord.away_score.isnot(None),
                )
                .all()
            )
            for r in finished_records:
                xgh = r.xg_home if r.xg_home is not None else 1.40
                xga = r.xg_away if r.xg_away is not None else 1.15
                xg_d = xgh - xga
                xg_t = xgh + xga
                elo_d = xg_d * 145.0
                sot_d = (
                    (r.shots_on_target_home - r.shots_on_target_away)
                    if (r.shots_on_target_home is not None and r.shots_on_target_away is not None)
                    else (xg_d * 2.8)
                )
                crn_d = (
                    (r.corners_home - r.corners_away)
                    if (r.corners_home is not None and r.corners_away is not None)
                    else (xg_d * 1.7)
                )
                if r.odds_home and r.odds_draw and r.odds_away:
                    inv = (1.0 / r.odds_home) + (1.0 / r.odds_draw) + (1.0 / r.odds_away)
                    ih = (1.0 / r.odds_home) / inv
                    id_ = (1.0 / r.odds_draw) / inv
                    ia = (1.0 / r.odds_away) / inv
                else:
                    ih, id_, ia = 0.45, 0.27, 0.28

                label = (
                    0
                    if r.home_score > r.away_score
                    else (1 if r.home_score == r.away_score else 2)
                )
                db_rows_X.append([elo_d, xg_d, xg_t, float(sot_d), float(crn_d), 0.0, ih, id_, ia])
                db_rows_y.append(label)
        except Exception:
            pass
        finally:
            session.close()

        self.db_matches_count = len(db_rows_y)
        if db_rows_X:
            X_db = np.array(db_rows_X, dtype=float)
            y_db = np.array(db_rows_y, dtype=int)
            # Los partidos reales en SQLite tienen triple peso de aprendizaje
            w_db = np.full(len(db_rows_y), 3.0, dtype=float)
            X_all = np.vstack([X_prior, X_db])
            y_all = np.concatenate([y_prior, y_db])
            w_all = np.concatenate([w_prior, w_db])
        else:
            X_all, y_all, w_all = X_prior, y_prior, w_prior

        self.model.fit(X_all, y_all, sample_weight=w_all)
        self.is_trained = True

    def predict_proba_match(
        self,
        elo_diff: float,
        xg_home: float,
        xg_away: float,
        sot_diff: float,
        corners_diff: float,
        rest_days_diff: float,
        implied_home: float | None,
        implied_draw: float | None,
        implied_away: float | None,
    ) -> Dict[str, Any]:
        """Retorna las probabilidades (1, X, 2) del modelo Gradient Boosting y la importancia de variables."""
        xg_d = xg_home - xg_away
        xg_t = xg_home + xg_away

        if implied_home and implied_draw and implied_away:
            ih, id_, ia = implied_home, implied_draw, implied_away
        else:
            raw_p_h = 1.0 / (1.0 + float(np.exp(-0.90 * xg_d)))
            id_ = max(0.15, min(0.32, 0.27 - 0.035 * abs(xg_d)))
            ih = (1.0 - id_) * raw_p_h
            ia = max(0.05, 1.0 - id_ - ih)

        x_vec = np.array(
            [[elo_diff, xg_d, xg_t, sot_diff, corners_diff, rest_days_diff, ih, id_, ia]],
            dtype=float,
        )
        probs = self.model.predict_proba(x_vec)[0]
        importances = self.model.feature_importances_

        feat_imp_list = [
            {"Variable": name, "Importancia (%)": round(float(imp) * 100.0, 1)}
            for name, imp in zip(FEATURE_NAMES, importances)
        ]
        feat_imp_list.sort(key=lambda item: item["Importancia (%)"], reverse=True)

        return {
            "ml_prob_home": round(float(probs[0]) * 100.0, 1),
            "ml_prob_draw": round(float(probs[1]) * 100.0, 1),
            "ml_prob_away": round(float(probs[2]) * 100.0, 1),
            "db_samples_used": self.db_matches_count,
            "feature_importances": feat_imp_list,
        }


_GLOBAL_ML_PREDICTOR: MLEnsemblePredictor | None = None


def get_ml_predictor() -> MLEnsemblePredictor:
    global _GLOBAL_ML_PREDICTOR
    if _GLOBAL_ML_PREDICTOR is None:
        _GLOBAL_ML_PREDICTOR = MLEnsemblePredictor()
    return _GLOBAL_ML_PREDICTOR
