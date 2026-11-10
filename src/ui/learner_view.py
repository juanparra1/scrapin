"""Vista modularizada para el Centro de Auto-Aprendizaje e IA Adaptativa."""

from datetime import date, timedelta
from typing import Any, Callable, Dict, List
import pandas as pd
import streamlit as st

from src.analytics.auto_learner import get_auto_learner
from src.config import MAX_PARALLEL_WORKERS
from src.models.schemas import MatchDetailSchema
from src.ui.audit_view import evaluate_all_systematic_lines_for_match


def render_learner_view(
    selected_date: date,
    date_str: str,
    catalog: Dict[str, Any],
    load_day_catalog_fn: Callable[[str, bool], Dict[str, Any]],
    analyze_matches_batch_parallel_fn: Callable[..., List[Dict[str, Any]]],
) -> None:
    """Renderiza bajo demanda el centro de auto-aprendizaje y calibración bayesiana."""
    st.subheader("🧠 Centro de Auto-Aprendizaje y Calibración Continua de la IA")
    st.caption(
        "Este motor permite que el sistema **aprenda autónomamente de sus errores y aciertos en partidos finalizados**: "
        "ajusta los pesos del ensamble (Dixon-Coles vs Machine Learning), calibra las 28 líneas de apuesta para eliminar la sobreconfianza, "
        "optimiza las prioridades del Bandido Thompson Sampling para selecciones de alta confianza y reentrena el modelo Gradient Boosting con nuevos registros en la base de datos."
    )

    learner = get_auto_learner()
    state = learner.state
    w_dixon, w_ml = learner.get_ensemble_weights()
    b_d = state.get("recent_brier_scores", {}).get("dixon_coles_1x2", 0.20)
    b_m = state.get("recent_brier_scores", {}).get("ml_1x2", 0.21)
    tot_learned = state.get("total_matches_learned", 0)
    sessions_cnt = state.get("learned_sessions_count", 0)

    # 1. TARJETAS KPI DE ESTADO DEL CEREBRO
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "⚖️ Peso Dixon-Coles (Poisson)",
        f"{w_dixon * 100:.1f}%",
        f"{(w_dixon - 0.65) * 100:+.1f}% vs base",
    )
    k2.metric(
        "🤖 Peso Machine Learning (ML)",
        f"{w_ml * 100:.1f}%",
        f"{(w_ml - 0.35) * 100:+.1f}% vs base",
    )
    better_engine = "🏆 ML Menor Error" if b_m <= b_d else "🏆 Dixon Menor Error"
    k3.metric(
        "🎯 Brier Score (Error 1X2)",
        f"{b_m:.4f} (ML)",
        f"Dixon: {b_d:.4f} • {better_engine}",
    )
    k4.metric(
        "🏟️ Partidos en Memoria de IA",
        str(tot_learned),
        f"{sessions_cnt} ciclos de aprendizaje",
    )

    st.markdown(
        f"""
        <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px 16px; margin:10px 0 18px 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:700; margin-bottom:6px;">
                <span style="color:#38bdf8;">🔵 Dixon-Coles / Opta: {w_dixon * 100:.1f}%</span>
                <span style="color:#94a3b8;">⚖️ Balance Dinámico del Ensamble Híbrido</span>
                <span style="color:#10b981;">🟢 Machine Learning (Gradient Boosting): {w_ml * 100:.1f}%</span>
            </div>
            <div style="width:100%; height:10px; background:#1e293b; border-radius:99px; display:flex; overflow:hidden;">
                <div style="width:{w_dixon * 100:.1f}%; background:linear-gradient(90deg, #0284c7, #38bdf8); height:100%;"></div>
                <div style="width:{w_ml * 100:.1f}%; background:linear-gradient(90deg, #10b981, #34d399); height:100%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. CONTROL PARA EJECUTAR CICLO DE AUTO-APRENDIZAJE
    st.markdown("### 🚀 Ejecutar Nuevo Ciclo de Aprendizaje")
    l_col1, l_col2, l_col3 = st.columns([1.6, 1.8, 1.0])
    with l_col1:
        learn_scope = st.radio(
            "Selecciona los partidos finalizados a asimilar:",
            [
                "📅 Partidos Finalizados de Hoy",
                "📅 Últimos 3 Días (Hoy, Ayer y Antier)",
                "📅 Última Semana (7 Días)",
            ],
            horizontal=False,
        )
    with l_col2:
        st.markdown(
            "La IA comparará cada pronóstico contra el marcador y estadísticas oficiales reales, "
            "actualizará los multiplicadores de calibración de las 28 líneas, ajustará los pesos de ensamble según Brier Score "
            "y reentrenará el modelo ML en SQLite."
        )
        btn_learn_now = st.button("🧠 Iniciar Ciclo de Auto-Aprendizaje", type="primary", use_container_width=True)
    with l_col3:
        st.write("")
        btn_reset_learner = st.button("🔄 Restablecer", use_container_width=True, help="Restablece los pesos a su valor de fábrica inicial")

    if btn_reset_learner:
        learner.reset_learning()
        st.success("✅ El cerebro de la IA ha sido restablecido a la configuración inicial.")
        st.rerun()

    if btn_learn_now:
        days_back = 1
        if "3 Días" in learn_scope:
            days_back = 3
        elif "Semana" in learn_scope:
            days_back = 7

        base_dt = selected_date
        scope_matches_to_learn: List[tuple[MatchDetailSchema, str]] = []
        seen_ids = set()

        for d_off in range(days_back):
            target_dt = base_dt - timedelta(days=d_off)
            target_str = target_dt.strftime("%Y-%m-%d")
            cat_d = catalog if (target_str == date_str) else load_day_catalog_fn(target_str, False)
            for m_item in cat_d["matches"]:
                if (
                    m_item.status_type == "finished"
                    and m_item.home_score is not None
                    and m_item.event_id not in seen_ids
                ):
                    seen_ids.add(m_item.event_id)
                    fs_code = cat_d["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A")
                    scope_matches_to_learn.append((m_item, fs_code))

        if not scope_matches_to_learn:
            st.warning("No se encontraron partidos finalizados en este período para entrenar.")
            return

        prog_learn = st.progress(0, text="Extrayendo estadísticas y analizando partidos en paralelo...")
        audited_pairs_for_learning = []

        batch_res = analyze_matches_batch_parallel_fn(
            scope_matches_to_learn,
            progress_bar=prog_learn,
            status_text_prefix="Analizando para IA",
            max_workers=MAX_PARALLEL_WORKERS,
        )
        for res_deep in batch_res:
            m_val = res_deep["match"]
            st.session_state["audited_predictions_cache"][m_val.event_id] = res_deep["prediction"]
            st.session_state["audited_matches_cache"][m_val.event_id] = m_val
            audited_pairs_for_learning.append((m_val, res_deep["prediction"]))

        prog_learn.progress(1.0, text="Calibrando probabilidades y reentrenando Gradient Boosting...")
        report = learner.learn_from_audited_matches(
            audited_pairs_for_learning, evaluate_all_systematic_lines_for_match
        )
        prog_learn.empty()

        st.session_state["last_learning_report"] = report
        st.success(
            f"🎉 ¡Ciclo de aprendizaje completado con éxito! Se procesaron {report['matches_processed']} partidos. "
            f"Pesos actualizados: Dixon-Coles {report['new_weights']['dixon']*100:.1f}% | ML {report['new_weights']['ml']*100:.1f}% "
            f"(Variación ML: {report['delta_ml_pct']:+.1f}%)."
        )
        st.rerun()

    last_rep = st.session_state.get("last_learning_report")
    if last_rep:
        st.markdown("#### 📋 Reporte del Último Ciclo de Auto-Entrenamiento")
        rep_c1, rep_c2 = st.columns(2)
        with rep_c1:
            st.markdown("##### 🟢 Líneas Promovidas a 'Estrella' por la IA (Alta Efectividad Real)")
            for item in last_rep.get("top_promoted_lines", []):
                st.markdown(
                    f"<div style='background:#064e3b; border-left:4px solid #10b981; padding:8px 12px; border-radius:6px; margin-bottom:6px; font-size:0.85rem;'>"
                    f"<b>{item['line_name']}</b><br>"
                    f"<span style='color:#34d399;'>Acierto acumulado: <b>{item['cum_win_rate']:.1f}%</b></span> • Multiplicador: <b>x{item['multiplier']:.3f}</b> • {item['status']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with rep_c2:
            st.markdown("##### 🔴 Líneas Atenuadas / Penalizadas por Varianza")
            pen_list = last_rep.get("top_penalized_lines", [])
            if not pen_list:
                st.info("No se registraron líneas penalizadas en este ciclo.")
            for item in pen_list:
                st.markdown(
                    f"<div style='background:#4c0519; border-left:4px solid #f43f5e; padding:8px 12px; border-radius:6px; margin-bottom:6px; font-size:0.85rem;'>"
                    f"<b>{item['line_name']}</b><br>"
                    f"<span style='color:#fda4af;'>Acierto acumulado: <b>{item['cum_win_rate']:.1f}%</b></span> • Multiplicador de contención: <b>x{item['multiplier']:.3f}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    tab_bandit, tab_calib, tab_history = st.tabs(
        [
            "🎲 Bandido Thompson Sampling (Ranking de Confianza y Calibración Adaptativa)",
            "🎯 Calibración y Atenuación de las 28 Líneas",
            "📜 Historial de Sesiones y Evolución",
        ]
    )

    with tab_bandit:
        st.markdown("#### 🎲 Algoritmo de Bandido Multibrazo (Thompson Sampling)")
        st.caption(
            "El modelo consulta este ranking en tiempo real: las líneas que acumulan aciertos reciben recompensa positiva (+1) "
            "y ganan mayor prioridad estadística; las líneas con rachas de fallo son atenuadas para proteger tu banca."
        )

        bandit_rows = []
        for lname, bdata in state.get("bandit_priors", {}).items():
            alpha = bdata.get("alpha", 4.0)
            beta_val = bdata.get("beta", 2.5)
            score = round(alpha / (alpha + beta_val), 3)
            bandit_rows.append({
                "Línea de Apuesta": lname,
                "Puntuación de Recompensa (0-1)": score,
                "Éxitos (Alpha)": alpha,
                "Fallos (Beta)": beta_val,
                "Nivel de Confianza": bdata.get("status", "Estándar"),
            })

        bandit_df = pd.DataFrame(bandit_rows).sort_values("Puntuación de Recompensa (0-1)", ascending=False)
        st.dataframe(bandit_df, use_container_width=True, hide_index=True)

    with tab_calib:
        st.markdown("#### 🎯 Calibración Empírica y Multiplicadores Bayesianos")
        st.caption(
            "Muestra el ajuste aplicado a cada mercado. Si el modelo teóricamente proyecta 75% pero la realidad histórica es 55%, "
            "el multiplicador bayesiano atempera la probabilidad hacia abajo para eliminar el sesgo de sobreestimación."
        )

        calib_rows = []
        for lname, cdata in state.get("lines_calibration", {}).items():
            calib_rows.append({
                "Línea": lname,
                "Evaluados": cdata.get("trials", 0),
                "Aciertos": cdata.get("hits", 0),
                "% Acierto Observado": f"{cdata.get('observed_win_rate', 0):.1f}%",
                "Prob. Predicha Promedio": f"{cdata.get('avg_predicted_prob', 0):.1f}%",
                "Multiplicador Bayesiano": f"x{cdata.get('multiplier', 1.0):.3f}",
                "Veredicto IA": cdata.get("status", "-"),
            })

        calib_df = pd.DataFrame(calib_rows).sort_values("Evaluados", ascending=False)
        st.dataframe(calib_df, use_container_width=True, hide_index=True)

    with tab_history:
        st.markdown("#### 📜 Registro de Ciclos de Aprendizaje Realizados")
        hist_list = state.get("training_history", [])
        if not hist_list:
            st.info("Aún no se han ejecutado ciclos de entrenamiento registrados en el historial.")
        else:
            hist_rows = []
            for entry in reversed(hist_list):
                hist_rows.append({
                    "Ciclo #": entry.get("session_id"),
                    "Fecha / Hora (UTC)": entry.get("timestamp", "").replace("T", " ")[:19],
                    "Partidos Asimilados": entry.get("matches_processed"),
                    "Peso Dixon": f"{entry.get('new_weights', {}).get('dixon', 0.65)*100:.1f}%",
                    "Peso ML": f"{entry.get('new_weights', {}).get('ml', 0.35)*100:.1f}%",
                    "Brier Dixon": entry.get("brier_dixon"),
                    "Brier ML": entry.get("brier_ml"),
                    "Muestras ML SQLite": entry.get("db_records_in_ml"),
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
