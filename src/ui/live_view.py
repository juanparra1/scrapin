"""Vista modularizada para el Escáner de Partidos EN VIVO y Gráfico de Presión (Attack Momentum)."""

from typing import List
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.schemas import LiveMomentumSummarySchema, MatchDetailSchema
from src.scrapers.sofascore import SofascoreScraper


def plot_momentum_chart(
    momentum: LiveMomentumSummarySchema, home_name: str, away_name: str
) -> go.Figure:
    """Construye el gráfico interactivo de Presión Minuto a Minuto (Attack Momentum)."""
    mins = [pt.minute for pt in momentum.points]
    vals = [pt.value for pt in momentum.points]
    colors = ["#38bdf8" if v >= 0 else "#f43f5e" for v in vals]

    fig = go.Figure(
        data=[
            go.Bar(
                x=mins,
                y=vals,
                marker_color=colors,
                name="Presión",
                hovertemplate="Min %{x}: %{y:.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"⚡ Gráfico de Presión Minuto a Minuto (Arriba Azul: {home_name} | Abajo Rojo: {away_name})",
        xaxis_title="Minuto de Partido",
        yaxis_title="Índice de Presión Ofensiva (Momentum)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        bargap=0.05,
    )
    return fig


def render_live_view(sf_scraper: SofascoreScraper) -> None:
    """Renderiza el escáner en vivo y gráficos de presión minuto a minuto."""
    st.subheader("🔴 Escáner de Partidos EN VIVO (Attack Momentum & Alertas de Gol Inminente)")
    live_events = sf_scraper.get_live_events()

    if not live_events:
        st.info("No hay partidos de fútbol en juego en este exacto segundo.")
        return

    st.markdown(
        f"Se detectaron **`{len(live_events)}` partidos en juego ahora mismo**. Analizando presión ofensiva (*Momentum*) en los principales..."
    )
    live_rows = []
    live_with_graphs: List[tuple[MatchDetailSchema, LiveMomentumSummarySchema]] = []

    for lm in live_events[:15]:
        mom = sf_scraper.get_match_momentum(lm.event_id, lm.home_team.name, lm.away_team.name)
        st_stats = sf_scraper.get_match_statistics(lm.event_id)
        xgh = st_stats.expected_goals_home if st_stats else None
        xga = st_stats.expected_goals_away if st_stats else None
        sot_h = st_stats.shots_on_target_home if st_stats else None
        sot_a = st_stats.shots_on_target_away if st_stats else None

        if mom:
            live_with_graphs.append((lm, mom))

        live_rows.append(
            {
                "Torneo": f"{lm.tournament.name} ({lm.tournament.country or ''})",
                "Partido en Juego": f"{lm.home_team.name} vs {lm.away_team.name}",
                "Marcador Vivo": f"{lm.home_score or 0} - {lm.away_score or 0} ({lm.status_description or 'Vivo'})",
                "xG en Vivo": f"{xgh:.2f} - {xga:.2f}" if (xgh is not None and xga is not None) else "S/D",
                "Rem. Arco Vivo": f"{sot_h} - {sot_a}" if (sot_h is not None and sot_a is not None) else "S/D",
                "Dominador Reciente": mom.dominant_team if mom else "S/D",
                "Nivel Alerta de Gol": mom.goal_alert_level if mom else "Normal",
                "Diagnóstico Momentum (Últ. 10m)": mom.goal_alert_reason if mom else "Sin gráfico de presión",
            }
        )

    st.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True)

    if live_with_graphs:
        st.markdown("#### 📈 Inspeccionar Gráfico de Presión Minuto a Minuto (Attack Momentum)")
        chosen_live_idx = st.selectbox(
            "Selecciona un partido en vivo con gráfico de presión:",
            options=range(len(live_with_graphs)),
            format_func=lambda idx: f"{live_with_graphs[idx][0].home_team.name} {live_with_graphs[idx][0].home_score}-{live_with_graphs[idx][0].away_score} {live_with_graphs[idx][0].away_team.name} ({live_with_graphs[idx][1].goal_alert_level})",
        )
        lm_sel, mom_sel = live_with_graphs[chosen_live_idx]
        st.plotly_chart(
            plot_momentum_chart(mom_sel, lm_sel.home_team.name, lm_sel.away_team.name),
            use_container_width=True,
        )
