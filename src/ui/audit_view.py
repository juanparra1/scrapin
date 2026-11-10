"""Vista modularizada para Auditoría de Pronósticos, Evaluación de Líneas y Radar Top Picks."""

from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.auto_learner import get_auto_learner
from src.analytics.predictor import MatchPrediction, _poisson_prob_over
from src.config import MAX_PARALLEL_WORKERS
from src.models.schemas import MatchDetailSchema


def _normalize_market_category(cat: str) -> str:
    """Normaliza los nombres de los 10 mercados para agregación uniforme."""
    c = cat.lower()
    if "resultado" in c or "1x2" in c or "doble oportunidad" in c:
        return "Resultado (Ensamble Opta + H2H + ML)"
    if "tiempo completo" in c or "mercado de goles" in c or "over 2.5" in c:
        return "Mercado de Goles (Tiempo Completo)"
    if "ambos" in c or "btts" in c:
        return "Ambos Equipos Anotan (BTTS)"
    if "1er tiempo" in c or "1h" in c or "descanso" in c:
        return "Goles en el 1er Tiempo (Opta 1ST)"
    if "remates al arco" in c or "shots on target" in c or "sot" in c:
        return "Remates al Arco (Shots on Target)"
    if "atajadas" in c or "portero" in c or "saves" in c:
        return "Atajadas del Portero (GK Saves)"
    if "remates totales" in c or "total shots" in c:
        return "Remates Totales del Partido"
    if "córners" in c or "corners" in c or "saques de esquina" in c:
        return "Córners (Saques de Esquina)"
    if "tarjetas" in c or "árbitro" in c or "arbitro" in c or "cards" in c:
        return "Tarjetas Amarillas & Árbitro"
    if "hándicap" in c or "handicap" in c or "dnb" in c or "10k" in c:
        return "Hándicap Asiático & Equipos (10K Sims)"
    return cat


def _extract_specific_bet_line(pick_text: str, category: str) -> str:
    """Normaliza cada pronóstico exacto en su línea de apuesta específica y comprensible."""
    p = pick_text.lower()

    if "más de 1.5 goles" in p and "1er tiempo" not in p and "equ." not in p:
        return "⚽ Over 1.5 Goles FT (+1.5)"
    if "más de 2.5 goles" in p or "over 2.5" in p:
        return "⚽ Over 2.5 Goles FT (+2.5)"
    if "menos de 2.5 goles" in p or "under 2.5" in p:
        return "⚽ Under 2.5 Goles FT (-2.5)"
    if "ambos anotan: sí" in p or "ambos anotan: s" in p:
        return "🔥 Ambos Equipos Anotan: SÍ (BTTS)"
    if "ambos anotan: no" in p:
        return "⛔ Ambos Equipos Anotan: NO"

    if "+0.5 1h" in p or ("más de 0.5 goles" in p and "1er tiempo" in p):
        return "⏱️ Over 0.5 Goles 1H (+0.5 1er Tiempo)"
    if "-1.5 1h" in p or ("menos de 1.5 goles" in p and "1er tiempo" in p):
        return "⏱️ Under 1.5 Goles 1H (-1.5 1er Tiempo)"

    if "3+ atajadas" in p or ("+2.5" in p and "atajadas" in p):
        return "🧤 Portero: 3+ Atajadas (+2.5 Saves)"
    if "2+ atajadas" in p or ("+1.5" in p and "atajadas" in p):
        return "🧤 Portero: 2+ Atajadas (+1.5 Saves)"
    if "5.5 atajadas" in p:
        return "🧤 Más de 5.5 Atajadas Totales"

    if "córner" in p or "corner" in p or "crner" in p or "saque" in p:
        if "9.5" in p and ("más" in p or "over" in p or "+" in p):
            return "🚩 Over 9.5 Córners (+9.5 Tiros de Esquina)"
        if "8.5" in p and ("más" in p or "over" in p or "+" in p):
            return "🚩 Over 8.5 Córners (+8.5 Tiros de Esquina)"
        if "7.5" in p and ("más" in p or "over" in p or "+" in p):
            return "🚩 Over 7.5 Córners (+7.5 Tiros de Esquina)"
        if "6.5" in p and ("más" in p or "over" in p or "+" in p):
            return "🚩 Over 6.5 Córners (+6.5 Tiros de Esquina)"
        if "menos" in p or "under" in p:
            return "🚩 Under Córners (-9.5 Tiros de Esquina)"
        return "🚩 Córners / Saques de Esquina"

    if "remates al arco" in p or "al arco" in p or "shots on target" in p or "sot" in p:
        if "8.5" in p and "más" in p:
            return "🎯 Over 8.5 Remates al Arco (+8.5 SoT)"
        if "7.5" in p and "más" in p:
            return "🎯 Over 7.5 Remates al Arco (+7.5 SoT)"
        if "menos" in p:
            return "🎯 Under 8.5 Remates al Arco"
        return "🎯 Remates al Arco (SoT)"

    if "remates totales" in p or "total shots" in p:
        if "24.5" in p and "más" in p:
            return "💥 Over 24.5 Remates Totales"
        if "22.5" in p and "más" in p:
            return "💥 Over 22.5 Remates Totales"
        if "menos" in p:
            return "💥 Under Remates Totales (-24.5)"
        return "💥 Remates Totales del Partido"

    if "tarjeta" in p or "card" in p:
        if "4.5" in p and "más" in p:
            return "🟨 Over 4.5 Tarjetas Amarillas"
        if "3.5" in p and "más" in p:
            return "🟨 Over 3.5 Tarjetas Amarillas"
        if "menos" in p:
            return "🟨 Under Tarjetas Amarillas (-4.5)"
        return "🟨 Tarjetas Amarillas"

    if "doble oportunidad" in p:
        return "🛡️ Doble Oportunidad (1X / X2)"
    if "apuesta sin empate" in p or "dnb" in p:
        return "⚖️ Apuesta Sin Empate (DNB)"
    if "equ." in p or "equipo más de 1.5" in p:
        return "⚽ Equipo Individual: Over 1.5 Goles"
    if "hándicap" in p or "handicap" in p:
        return "📈 Hándicap Asiático (+1.5)"
    if "victoria" in p:
        return "🏆 Victoria Directa 1X2"

    return pick_text


def evaluate_all_systematic_lines_for_match(
    match: MatchDetailSchema, pred: MatchPrediction
) -> List[Dict[str, Any]]:
    """Evalúa sistemáticamente las 28 líneas estándar de apuestas para un partido finalizado."""
    hs = match.home_score
    as_ = match.away_score
    tot_g = (hs + as_) if (hs is not None and as_ is not None) else None
    ht_hs = match.home_score_ht
    ht_as = match.away_score_ht
    ht_g = (ht_hs + ht_as) if (ht_hs is not None and ht_as is not None) else None

    st_obj = match.stats
    crn = (
        (st_obj.corner_kicks_home + st_obj.corner_kicks_away)
        if (st_obj and st_obj.corner_kicks_home is not None and st_obj.corner_kicks_away is not None)
        else None
    )
    sot = (
        (st_obj.shots_on_target_home + st_obj.shots_on_target_away)
        if (st_obj and st_obj.shots_on_target_home is not None and st_obj.shots_on_target_away is not None)
        else None
    )
    sh = (
        (st_obj.total_shots_home + st_obj.total_shots_away)
        if (st_obj and st_obj.total_shots_home is not None and st_obj.total_shots_away is not None)
        else None
    )
    crd = (
        (st_obj.yellow_cards_home + st_obj.yellow_cards_away)
        if (st_obj and st_obj.yellow_cards_home is not None and st_obj.yellow_cards_away is not None)
        else None
    )
    sv_h = st_obj.saves_home if (st_obj and st_obj.saves_home is not None) else None
    sv_a = st_obj.saves_away if (st_obj and st_obj.saves_away is not None) else None
    max_sv = max(sv_h, sv_a) if (sv_h is not None and sv_a is not None) else None

    match_label = f"{match.home_team.name} vs {match.away_team.name}"
    lines: List[Dict[str, Any]] = []

    if tot_g is not None:
        lines.extend([
            {
                "line_name": "⚽ Over 1.5 Goles FT (+1.5)",
                "category": "Mercado de Goles (Tiempo Completo)",
                "prob": pred.prob_over_1_5,
                "hit": tot_g >= 2,
                "actual": f"Total: {tot_g} goles ({hs}-{as_})",
                "match": match_label,
            },
            {
                "line_name": "⚽ Over 2.5 Goles FT (+2.5)",
                "category": "Mercado de Goles (Tiempo Completo)",
                "prob": pred.prob_over_2_5,
                "hit": tot_g >= 3,
                "actual": f"Total: {tot_g} goles ({hs}-{as_})",
                "match": match_label,
            },
            {
                "line_name": "⚽ Under 2.5 Goles FT (-2.5)",
                "category": "Mercado de Goles (Tiempo Completo)",
                "prob": pred.prob_under_2_5,
                "hit": tot_g <= 2,
                "actual": f"Total: {tot_g} goles ({hs}-{as_})",
                "match": match_label,
            },
            {
                "line_name": "⚽ Under 3.5 Goles FT (-3.5)",
                "category": "Mercado de Goles (Tiempo Completo)",
                "prob": round(max(1.0, 100.0 - pred.prob_over_3_5), 1),
                "hit": tot_g <= 3,
                "actual": f"Total: {tot_g} goles ({hs}-{as_})",
                "match": match_label,
            },
            {
                "line_name": "🔥 Ambos Equipos Anotan: SÍ (BTTS)",
                "category": "Ambos Equipos Anotan (BTTS)",
                "prob": pred.prob_btts,
                "hit": (hs >= 1 and as_ >= 1),
                "actual": f"Marcador: {hs}-{as_}",
                "match": match_label,
            },
            {
                "line_name": "⛔ Ambos Equipos Anotan: NO",
                "category": "Ambos Equipos Anotan (BTTS)",
                "prob": pred.prob_btts_no,
                "hit": (hs == 0 or as_ == 0),
                "actual": f"Marcador: {hs}-{as_}",
                "match": match_label,
            },
        ])

    if ht_g is not None:
        lines.extend([
            {
                "line_name": "⏱️ Over 0.5 Goles 1H (+0.5 1er Tiempo)",
                "category": "Goles en el 1er Tiempo (Opta 1ST)",
                "prob": pred.prob_1h_over_0_5,
                "hit": ht_g >= 1,
                "actual": f"1H: {ht_g} goles ({ht_hs}-{ht_as})",
                "match": match_label,
            },
            {
                "line_name": "⏱️ Under 1.5 Goles 1H (-1.5 1er Tiempo)",
                "category": "Goles en el 1er Tiempo (Opta 1ST)",
                "prob": round(max(1.0, 100.0 - pred.prob_1h_over_1_5), 1),
                "hit": ht_g <= 1,
                "actual": f"1H: {ht_g} goles ({ht_hs}-{ht_as})",
                "match": match_label,
            },
        ])

    if crn is not None:
        exp_c = pred.expected_corners_total or 9.2
        lines.extend([
            {
                "line_name": "🚩 Over 6.5 Córners (+6.5 Tiros de Esquina)",
                "category": "Córners (Saques de Esquina)",
                "prob": round(_poisson_prob_over(6.5, exp_c) * 100.0, 1),
                "hit": crn >= 7,
                "actual": f"{crn} córners reales",
                "match": match_label,
            },
            {
                "line_name": "🚩 Over 7.5 Córners (+7.5 Tiros de Esquina)",
                "category": "Córners (Saques de Esquina)",
                "prob": round(_poisson_prob_over(7.5, exp_c) * 100.0, 1),
                "hit": crn >= 8,
                "actual": f"{crn} córners reales",
                "match": match_label,
            },
            {
                "line_name": "🚩 Over 8.5 Córners (+8.5 Tiros de Esquina)",
                "category": "Córners (Saques de Esquina)",
                "prob": pred.prob_over_8_5_corners,
                "hit": crn >= 9,
                "actual": f"{crn} córners reales",
                "match": match_label,
            },
            {
                "line_name": "🚩 Over 9.5 Córners (+9.5 Tiros de Esquina)",
                "category": "Córners (Saques de Esquina)",
                "prob": pred.prob_over_9_5_corners,
                "hit": crn >= 10,
                "actual": f"{crn} córners reales",
                "match": match_label,
            },
            {
                "line_name": "🚩 Under 10.5 Córners (-10.5 Tiros de Esquina)",
                "category": "Córners (Saques de Esquina)",
                "prob": round(max(1.0, 100.0 - pred.prob_over_10_5_corners), 1),
                "hit": crn <= 10,
                "actual": f"{crn} córners reales",
                "match": match_label,
            },
        ])

    if sot is not None:
        exp_sot = pred.expected_sot_total or 8.2
        lines.extend([
            {
                "line_name": "🎯 Over 6.5 Remates al Arco (+6.5 SoT)",
                "category": "Remates al Arco (Shots on Target)",
                "prob": round(_poisson_prob_over(6.5, exp_sot) * 100.0, 1),
                "hit": sot >= 7,
                "actual": f"{sot} remates al arco",
                "match": match_label,
            },
            {
                "line_name": "🎯 Over 7.5 Remates al Arco (+7.5 SoT)",
                "category": "Remates al Arco (Shots on Target)",
                "prob": pred.prob_over_7_5_sot,
                "hit": sot >= 8,
                "actual": f"{sot} remates al arco",
                "match": match_label,
            },
            {
                "line_name": "🎯 Over 8.5 Remates al Arco (+8.5 SoT)",
                "category": "Remates al Arco (Shots on Target)",
                "prob": pred.prob_over_8_5_sot,
                "hit": sot >= 9,
                "actual": f"{sot} remates al arco",
                "match": match_label,
            },
            {
                "line_name": "🎯 Under 8.5 Remates al Arco",
                "category": "Remates al Arco (Shots on Target)",
                "prob": round(max(1.0, 100.0 - pred.prob_over_8_5_sot), 1),
                "hit": sot <= 8,
                "actual": f"{sot} remates al arco",
                "match": match_label,
            },
        ])

    if sh is not None:
        exp_sh = pred.expected_shots_total or 23.0
        lines.extend([
            {
                "line_name": "💥 Over 20.5 Remates Totales",
                "category": "Remates Totales del Partido",
                "prob": round(_poisson_prob_over(20.5, exp_sh) * 100.0, 1),
                "hit": sh >= 21,
                "actual": f"{sh} remates totales",
                "match": match_label,
            },
            {
                "line_name": "💥 Over 22.5 Remates Totales",
                "category": "Remates Totales del Partido",
                "prob": pred.prob_over_22_5_shots,
                "hit": sh >= 23,
                "actual": f"{sh} remates totales",
                "match": match_label,
            },
            {
                "line_name": "💥 Over 24.5 Remates Totales",
                "category": "Remates Totales del Partido",
                "prob": pred.prob_over_24_5_shots,
                "hit": sh >= 25,
                "actual": f"{sh} remates totales",
                "match": match_label,
            },
        ])

    if crd is not None:
        lines.extend([
            {
                "line_name": "🟨 Over 3.5 Tarjetas Amarillas",
                "category": "Tarjetas Amarillas & Árbitro",
                "prob": pred.prob_over_3_5_cards,
                "hit": crd >= 4,
                "actual": f"{crd} tarjetas amarillas",
                "match": match_label,
            },
            {
                "line_name": "🟨 Over 4.5 Tarjetas Amarillas",
                "category": "Tarjetas Amarillas & Árbitro",
                "prob": pred.prob_over_4_5_cards,
                "hit": crd >= 5,
                "actual": f"{crd} tarjetas amarillas",
                "match": match_label,
            },
            {
                "line_name": "🟨 Under 4.5 Tarjetas Amarillas",
                "category": "Tarjetas Amarillas & Árbitro",
                "prob": round(max(1.0, 100.0 - pred.prob_over_4_5_cards), 1),
                "hit": crd <= 4,
                "actual": f"{crd} tarjetas amarillas",
                "match": match_label,
            },
        ])

    if max_sv is not None:
        lines.extend([
            {
                "line_name": "🧤 Portero: 2+ Atajadas (+1.5 Saves)",
                "category": "Atajadas del Portero (GK Saves)",
                "prob": max(pred.prob_home_gk_2plus_saves, pred.prob_away_gk_2plus_saves),
                "hit": max_sv >= 2,
                "actual": f"Máx portero: {max_sv} atajadas",
                "match": match_label,
            },
            {
                "line_name": "🧤 Portero: 3+ Atajadas (+2.5 Saves)",
                "category": "Atajadas del Portero (GK Saves)",
                "prob": max(pred.prob_home_gk_3plus_saves, pred.prob_away_gk_3plus_saves),
                "hit": max_sv >= 3,
                "actual": f"Máx portero: {max_sv} atajadas",
                "match": match_label,
            },
        ])

    if hs is not None and as_ is not None:
        lines.extend([
            {
                "line_name": "🛡️ Doble Oportunidad 1X (Local o Empate)",
                "category": "Resultado (Ensamble Opta + H2H + ML)",
                "prob": pred.prob_1x,
                "hit": hs >= as_,
                "actual": f"Marcador: {hs}-{as_}",
                "match": match_label,
            },
            {
                "line_name": "🛡️ Doble Oportunidad X2 (Visitante o Empate)",
                "category": "Resultado (Ensamble Opta + H2H + ML)",
                "prob": pred.prob_x2,
                "hit": as_ >= hs,
                "actual": f"Marcador: {hs}-{as_}",
                "match": match_label,
            },
            {
                "line_name": "⚽ Equipo Individual: Over 1.5 Goles",
                "category": "Hándicap Asiático & Equipos (10K Sims)",
                "prob": max(pred.prob_home_over_1_5, pred.prob_away_over_1_5),
                "hit": (hs >= 2 or as_ >= 2),
                "actual": f"Marcador: {hs}-{as_}",
                "match": match_label,
            },
        ])

    return lines


def compute_market_accuracy_breakdown(
    predictions: List[MatchPrediction],
    matches_map: Optional[Dict[int, MatchDetailSchema]] = None,
) -> Dict[str, Any]:
    """Analiza el rendimiento y porcentaje de aciertos por categorías y líneas específicas."""
    canonical_categories = [
        "Resultado (Ensamble Opta + H2H + ML)",
        "Mercado de Goles (Tiempo Completo)",
        "Ambos Equipos Anotan (BTTS)",
        "Goles en el 1er Tiempo (Opta 1ST)",
        "Remates al Arco (Shots on Target)",
        "Atajadas del Portero (GK Saves)",
        "Remates Totales del Partido",
        "Córners (Saques de Esquina)",
        "Tarjetas Amarillas & Árbitro",
        "Hándicap Asiático & Equipos (10K Sims)",
    ]

    category_map: Dict[str, Dict[str, Any]] = {
        cat: {
            "category": cat,
            "total": 0,
            "hits": 0,
            "misses": 0,
            "win_rate": 0.0,
            "probs": [],
            "picks": [],
        }
        for cat in canonical_categories
    }

    specific_lines_map: Dict[str, Dict[str, Any]] = {}
    exhaustive_lines_map: Dict[str, Dict[str, Any]] = {}

    overall_total = 0
    overall_hits = 0
    matches_audited = 0

    ex_overall_total = 0
    ex_overall_hits = 0

    for p in predictions:
        if not p.all_market_picks:
            continue
        has_finished = any(pk.get("hit") is not None for pk in p.all_market_picks)
        if has_finished:
            matches_audited += 1

        for pk in p.all_market_picks:
            raw_cat = pk.get("category", "")
            pick_label = pk.get("pick", "")
            hit = pk.get("hit")
            if hit is None:
                continue

            norm_cat = _normalize_market_category(raw_cat)
            if norm_cat not in category_map:
                category_map[norm_cat] = {
                    "category": norm_cat,
                    "total": 0,
                    "hits": 0,
                    "misses": 0,
                    "win_rate": 0.0,
                    "probs": [],
                    "picks": [],
                }

            entry = category_map[norm_cat]
            entry["total"] += 1
            overall_total += 1
            if hit is True:
                entry["hits"] += 1
                overall_hits += 1
            else:
                entry["misses"] += 1

            prob_val = pk.get("prob", 50.0)
            entry["probs"].append(prob_val)
            entry["picks"].append({
                "match": f"{p.home_team} vs {p.away_team}",
                "pick": pick_label,
                "prob": prob_val,
                "hit": hit,
                "actual": pk.get("actual", ""),
            })

            spec_line = _extract_specific_bet_line(pick_label, norm_cat)
            if spec_line not in specific_lines_map:
                specific_lines_map[spec_line] = {
                    "line_name": spec_line,
                    "category": norm_cat,
                    "total": 0,
                    "hits": 0,
                    "misses": 0,
                    "win_rate": 0.0,
                    "probs": [],
                    "picks": [],
                }
            l_entry = specific_lines_map[spec_line]
            l_entry["total"] += 1
            if hit is True:
                l_entry["hits"] += 1
            else:
                l_entry["misses"] += 1
            l_entry["probs"].append(prob_val)
            l_entry["picks"].append({
                "match": f"{p.home_team} vs {p.away_team}",
                "pick": pick_label,
                "prob": prob_val,
                "hit": hit,
                "actual": pk.get("actual", ""),
            })

        if matches_map and p.event_id in matches_map:
            m_obj = matches_map[p.event_id]
            if m_obj.status_type == "finished" and m_obj.home_score is not None:
                sys_lines = evaluate_all_systematic_lines_for_match(m_obj, p)
                for sl_item in sys_lines:
                    lname = sl_item["line_name"]
                    if lname not in exhaustive_lines_map:
                        exhaustive_lines_map[lname] = {
                            "line_name": lname,
                            "category": sl_item["category"],
                            "total": 0,
                            "hits": 0,
                            "misses": 0,
                            "win_rate": 0.0,
                            "probs": [],
                            "picks": [],
                        }
                    ex_entry = exhaustive_lines_map[lname]
                    ex_entry["total"] += 1
                    ex_overall_total += 1
                    if sl_item["hit"] is True:
                        ex_entry["hits"] += 1
                        ex_overall_hits += 1
                    else:
                        ex_entry["misses"] += 1
                    ex_entry["probs"].append(sl_item["prob"])
                    ex_entry["picks"].append({
                        "match": sl_item["match"],
                        "pick": lname,
                        "prob": sl_item["prob"],
                        "hit": sl_item["hit"],
                        "actual": sl_item["actual"],
                    })

    stats_list = []
    for cat, data in category_map.items():
        tot = data["total"]
        h = data["hits"]
        wr = round((h / tot) * 100.0, 1) if tot > 0 else 0.0
        data["win_rate"] = wr
        data["avg_prob"] = round(sum(data["probs"]) / len(data["probs"]), 1) if data["probs"] else 0.0

        if tot == 0:
            status_tag = "Sin datos en partidos analizados"
            badge = "⚪ Pendiente"
        elif wr >= 72.0:
            status_tag = "Alta Efectividad (Top Acierto)"
            badge = "🟢 Muy Confiable"
        elif wr >= 55.0:
            status_tag = "Rendimiento Estable"
            badge = "🔵 Estable"
        elif wr >= 45.0:
            status_tag = "Rendimiento Moderado (Cautela)"
            badge = "🟡 Cautela"
        else:
            status_tag = "Alta Volatilidad (Fallo Frecuente)"
            badge = "🔴 Riesgoso"

        data["status_tag"] = status_tag
        data["badge"] = badge
        stats_list.append(data)

    stats_list.sort(key=lambda x: (x["total"] > 0, x["win_rate"], x["total"]), reverse=True)

    lines_list = []
    for line_name, data in specific_lines_map.items():
        tot = data["total"]
        h = data["hits"]
        wr = round((h / tot) * 100.0, 1) if tot > 0 else 0.0
        data["win_rate"] = wr
        data["avg_prob"] = round(sum(data["probs"]) / len(data["probs"]), 1) if data["probs"] else 0.0

        if tot == 0:
            status_tag = "Sin datos"
            badge = "⚪ Pendiente"
        elif wr >= 72.0:
            status_tag = "Línea Estrella (Muy Alta Efectividad)"
            badge = "🟢 Estrella"
        elif wr >= 55.0:
            status_tag = "Línea Sólida"
            badge = "🔵 Sólida"
        elif wr >= 45.0:
            status_tag = "Línea Moderada"
            badge = "🟡 Cautela"
        else:
            status_tag = "Línea de Alto Fallo (Evitar / Revisar)"
            badge = "🔴 Evitar"

        data["status_tag"] = status_tag
        data["badge"] = badge
        lines_list.append(data)

    lines_list.sort(key=lambda x: (x["total"] > 0, x["win_rate"], x["total"]), reverse=True)

    exhaustive_list = []
    for line_name, data in exhaustive_lines_map.items():
        tot = data["total"]
        h = data["hits"]
        wr = round((h / tot) * 100.0, 1) if tot > 0 else 0.0
        data["win_rate"] = wr
        data["avg_prob"] = round(sum(data["probs"]) / len(data["probs"]), 1) if data["probs"] else 0.0

        if tot == 0:
            status_tag = "Sin datos"
            badge = "⚪ Pendiente"
        elif wr >= 75.0:
            status_tag = "Línea Estrella (Muy Alta Frecuencia de Acierto)"
            badge = "🟢 Estrella"
        elif wr >= 55.0:
            status_tag = "Línea Sólida / Frecuente"
            badge = "🔵 Sólida"
        elif wr >= 45.0:
            status_tag = "Línea Moderada (50/50)"
            badge = "🟡 Cautela"
        else:
            status_tag = "Línea de Alto Fallo / Riesgo Elevado"
            badge = "🔴 Riesgo Alto"

        data["status_tag"] = status_tag
        data["badge"] = badge
        exhaustive_list.append(data)

    exhaustive_list.sort(key=lambda x: (x["total"] > 0, x["win_rate"], x["total"]), reverse=True)

    active_cats = [s for s in stats_list if s["total"] > 0]
    best_cat = active_cats[0] if active_cats else None
    worst_cat = active_cats[-1] if active_cats else None

    active_lines = [s for s in lines_list if s["total"] > 0]
    best_line = active_lines[0] if active_lines else None
    worst_line = active_lines[-1] if active_lines else None

    active_ex = [s for s in exhaustive_list if s["total"] > 0]
    best_ex_line = active_ex[0] if active_ex else None
    worst_ex_line = active_ex[-1] if active_ex else None

    overall_win_rate = round((overall_hits / overall_total) * 100.0, 1) if overall_total > 0 else 0.0
    ex_win_rate = round((ex_overall_hits / ex_overall_total) * 100.0, 1) if ex_overall_total > 0 else 0.0

    return {
        "matches_audited": matches_audited,
        "overall_total": overall_total,
        "overall_hits": overall_hits,
        "overall_misses": overall_total - overall_hits,
        "overall_win_rate": overall_win_rate,
        "category_stats": stats_list,
        "specific_lines_stats": lines_list,
        "exhaustive_lines_stats": exhaustive_list,
        "exhaustive_total": ex_overall_total,
        "exhaustive_hits": ex_overall_hits,
        "exhaustive_win_rate": ex_win_rate,
        "best_category": best_cat,
        "worst_category": worst_cat,
        "best_line": best_line,
        "worst_line": worst_line,
        "best_exhaustive_line": best_ex_line,
        "worst_exhaustive_line": worst_ex_line,
    }


def plot_specific_lines_chart(lines_list: List[Dict[str, Any]], overall_win_rate: float, title: str) -> go.Figure:
    """Construye un gráfico horizontal de barras de todas las líneas específicas ordenadas por % de acierto."""
    active = [s for s in lines_list if s["total"] > 0]
    if not active:
        active = lines_list

    active_rev = list(reversed(active))
    labels = [s["line_name"] for s in active_rev]
    win_rates = [s["win_rate"] for s in active_rev]

    colors = []
    text_labels = []
    for s in active_rev:
        wr = s["win_rate"]
        tot = s["total"]
        h = s["hits"]
        if tot == 0:
            colors.append("#64748b")
            text_labels.append("Sin datos")
        elif wr >= 72.0:
            colors.append("#10b981")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos) 🏆")
        elif wr >= 55.0:
            colors.append("#38bdf8")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos)")
        elif wr >= 45.0:
            colors.append("#f59e0b")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos)")
        else:
            colors.append("#f43f5e")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos) ⚠️")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=win_rates,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.25)", width=1),
            ),
            text=text_labels,
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="#ffffff", size=12, family="sans-serif"),
            hovertemplate="<b>%{y}</b><br>Efectividad Real: %{x:.1f}%<extra></extra>",
        )
    )

    fig.add_vline(
        x=50.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="50% Umbral Neutro",
        annotation_position="top right",
        annotation_font=dict(color="#94a3b8", size=11),
    )

    if overall_win_rate > 0:
        fig.add_vline(
            x=overall_win_rate,
            line_width=2,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"Promedio: {overall_win_rate:.1f}%",
            annotation_position="bottom right",
            annotation_font=dict(color="#38bdf8", size=11),
        )

    chart_height = max(420, len(active) * 36)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f8fafc")),
        xaxis=dict(title="% de Acierto Real", range=[0, 105], ticksuffix="%", gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#1e293b", color="#f8fafc", tickfont=dict(size=12, weight="bold")),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        height=chart_height,
        margin=dict(l=20, r=30, t=50, b=30),
    )
    return fig


def plot_market_accuracy_chart(breakdown: Dict[str, Any]) -> go.Figure:
    """Construye un gráfico horizontal de barras de los 10 mercados coloreado por efectividad."""
    stats = [s for s in breakdown["category_stats"] if s["total"] > 0]
    if not stats:
        stats = breakdown["category_stats"]

    stats_rev = list(reversed(stats))
    categories = [s["category"] for s in stats_rev]
    win_rates = [s["win_rate"] for s in stats_rev]

    colors = []
    text_labels = []
    for s in stats_rev:
        wr = s["win_rate"]
        tot = s["total"]
        h = s["hits"]
        if tot == 0:
            colors.append("#64748b")
            text_labels.append("Sin datos evaluados")
        elif wr >= 72.0:
            colors.append("#10b981")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos) 🏆")
        elif wr >= 55.0:
            colors.append("#38bdf8")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos)")
        elif wr >= 45.0:
            colors.append("#f59e0b")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos)")
        else:
            colors.append("#f43f5e")
            text_labels.append(f"  {wr:.1f}% ({h}/{tot} Aciertos) ⚠️")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=categories,
            x=win_rates,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.25)", width=1)),
            text=text_labels,
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="#ffffff", size=13, family="sans-serif"),
            hovertemplate="<b>%{y}</b><br>Efectividad Real: %{x:.1f}%<extra></extra>",
        )
    )

    fig.add_vline(
        x=50.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="50% Umbral Neutro",
        annotation_position="top right",
        annotation_font=dict(color="#94a3b8", size=11),
    )

    if breakdown["overall_total"] > 0:
        fig.add_vline(
            x=breakdown["overall_win_rate"],
            line_width=2,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"Promedio Global: {breakdown['overall_win_rate']:.1f}%",
            annotation_position="bottom right",
            annotation_font=dict(color="#38bdf8", size=12),
        )

    fig.update_layout(
        title=dict(
            text=f"📊 Efectividad y Tasa de Acierto por Mercado ({breakdown['matches_audited']} Partidos Auditados • {breakdown['overall_hits']}/{breakdown['overall_total']} Aciertos Totales)",
            font=dict(size=15, color="#f8fafc"),
        ),
        xaxis=dict(title="% de Acierto Real", range=[0, 105], ticksuffix="%", gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(gridcolor="#1e293b", color="#f8fafc", tickfont=dict(size=12, weight="bold")),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        height=450,
        margin=dict(l=20, r=30, t=50, b=30),
    )
    return fig


def render_audit_view(
    selected_date: date,
    date_str: str,
    catalog: Dict[str, Any],
    load_day_catalog_fn: Callable[[str, bool], Dict[str, Any]],
    analyze_matches_batch_parallel_fn: Callable[..., List[Dict[str, Any]]],
) -> None:
    """Renderiza bajo demanda la pestaña de Auditoría de Pronósticos."""
    st.subheader("📊 Auditoría de Pronósticos: Promedios de Acierto, Gráfica y Ranking de Mercados y Líneas")
    st.caption(
        "Este módulo audita automáticamente los pronósticos generados por el modelo contra los marcadores y estadísticas oficiales "
        "de partidos finalizados para responder exactamente **cuántos aciertos tuvo, cuál es el promedio general y cuáles son las líneas (+1.5, +2.5 goles, córners, atajadas) que más acierta y las que más falla**."
    )

    st.markdown("### 🎛️ Alcance y Selección de Partidos a Auditar")
    scope_col1, scope_col2 = st.columns([1.6, 2.4])
    with scope_col1:
        scope_mode = st.radio(
            "📅 Selecciona el rango temporal de partidos finalizados:",
            [
                "📅 Partidos Finalizados de Hoy",
                "📅 Últimos 3 Días (Hoy, Ayer y Antier)",
                "📅 Última Semana Completa (7 Días)",
            ],
            horizontal=False,
        )

    days_to_check = 1
    if "3 Días" in scope_mode:
        days_to_check = 3
    elif "Semana" in scope_mode:
        days_to_check = 7

    base_date = selected_date
    scope_finished_matches: List[tuple[MatchDetailSchema, str]] = []
    seen_events = set()

    for d_offset in range(days_to_check):
        d_target = base_date - timedelta(days=d_offset)
        d_target_str = d_target.strftime("%Y-%m-%d")
        day_cat = catalog if (d_target_str == date_str) else load_day_catalog_fn(d_target_str, False)
        for m_item in day_cat["matches"]:
            if (
                m_item.status_type == "finished"
                and m_item.home_score is not None
                and m_item.event_id not in seen_events
            ):
                seen_events.add(m_item.event_id)
                fs_code = day_cat["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A")
                scope_finished_matches.append((m_item, fs_code))

    if not scope_finished_matches:
        st.warning("No se encontraron partidos finalizados en el período seleccionado. Cambia la fecha o el rango temporal.")
        return

    with scope_col2:
        st.markdown(f"**Partidos finalizados detectados en este período:** `{len(scope_finished_matches)}`")
        audit_slider_val = st.slider(
            "O selecciona una cantidad específica a auditar:",
            min_value=1,
            max_value=max(1, len(scope_finished_matches)),
            value=min(20, len(scope_finished_matches)),
        )

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1.2, 0.9, 1.6])
    with btn_col1:
        btn_audit_all = st.button(
            f"🚀 Auditar TODOS los Partidos ({len(scope_finished_matches)} partidos)",
            type="primary",
            use_container_width=True,
        )
    with btn_col2:
        btn_audit_slider = st.button(
            f"⚡ Auditar Selección ({audit_slider_val} partidos)",
            use_container_width=True,
        )
    with btn_col3:
        btn_clear_audit = st.button("🗑️ Limpiar", use_container_width=True, help="Reinicia la memoria de auditoría")
    with btn_col4:
        btn_feed_brain = st.button(
            "🧠 Entrenar IA con esto",
            use_container_width=True,
            help="Asimila estos partidos auditados en el cerebro de la IA para actualizar pesos y calibración",
        )

    audited_cache = st.session_state.get("audited_predictions_cache", {})
    matches_cache = st.session_state.get("audited_matches_cache", {})

    if btn_feed_brain:
        audited_pairs_to_feed = [
            (matches_cache[m_id], pred_item)
            for m_id, pred_item in audited_cache.items()
            if m_id in matches_cache
        ]
        if audited_pairs_to_feed:
            learner_inst = get_auto_learner()
            rep = learner_inst.learn_from_audited_matches(
                audited_pairs_to_feed, evaluate_all_systematic_lines_for_match
            )
            st.success(
                f"🧠 ¡Cerebro de la IA actualizado! Asimiló {len(audited_pairs_to_feed)} partidos. "
                f"Nuevos pesos: Dixon-Coles {rep['new_weights']['dixon']*100:.1f}% | ML {rep['new_weights']['ml']*100:.1f}%."
            )
            st.rerun()
        else:
            st.warning("Primero audita partidos finalizados para poder alimentar a la IA.")

    if btn_clear_audit:
        st.session_state["audited_predictions_cache"] = {}
        st.session_state["audited_matches_cache"] = {}
        st.rerun()

    if len(audited_cache) == 0 and scope_finished_matches:
        auto_sample = scope_finished_matches[: min(10, len(scope_finished_matches))]
        with st.spinner(f"⚡ Procesando auditoría inicial de {len(auto_sample)} partidos en paralelo..."):
            batch_res = analyze_matches_batch_parallel_fn(auto_sample, max_workers=MAX_PARALLEL_WORKERS)
            for res_auto in batch_res:
                m_val = res_auto["match"]
                st.session_state["audited_predictions_cache"][m_val.event_id] = res_auto["prediction"]
                st.session_state["audited_matches_cache"][m_val.event_id] = m_val
        audited_cache = st.session_state["audited_predictions_cache"]
        matches_cache = st.session_state["audited_matches_cache"]

    if btn_audit_all or btn_audit_slider:
        target_list = scope_finished_matches if btn_audit_all else scope_finished_matches[:audit_slider_val]
        prog = st.progress(0, text="Iniciando auditoría paralela de alta velocidad...")
        batch_res = analyze_matches_batch_parallel_fn(
            target_list,
            progress_bar=prog,
            status_text_prefix="Auditando en paralelo",
            max_workers=MAX_PARALLEL_WORKERS,
        )
        for res_deep in batch_res:
            m_val = res_deep["match"]
            st.session_state["audited_predictions_cache"][m_val.event_id] = res_deep["prediction"]
            st.session_state["audited_matches_cache"][m_val.event_id] = m_val
        prog.empty()
        st.success(f"✅ Se auditaron exitosamente {len(batch_res)} partidos finalizados a máxima velocidad.")
        st.rerun()

    all_audited_preds = list(st.session_state.get("audited_predictions_cache", {}).values())
    all_audited_matches = st.session_state.get("audited_matches_cache", {})

    if not all_audited_preds:
        st.warning("No hay partidos auditados aún. Pulsa 'Auditar TODOS' para comenzar.")
        return

    breakdown = compute_market_accuracy_breakdown(all_audited_preds, matches_map=all_audited_matches)

    tab_spec_lines, tab_general_cats = st.tabs(
        [
            "🎯 Líneas Específicas (+1.5, +2.5 Goles, Córners, Atajadas, etc.)",
            "📁 Resumen por 10 Categorías Generales",
        ]
    )

    with tab_spec_lines:
        st.markdown("#### 🎯 Tasa de Acierto y Rendimiento por Línea Específica")
        audit_mode_view = st.radio(
            "📐 Modalidad de Auditoría de Líneas:",
            [
                "🌐 Auditoría Exhaustiva de TODAS las Líneas en CADA Partido (Recomendada: Muestra Completa de +1.5, +2.5, Córners, SoT, etc.)",
                "🎯 Solo Pronósticos Recomendados por el Modelo (Las 10 Tarjetas Seleccionadas)",
            ],
            horizontal=False,
        )

        is_exhaustive = audit_mode_view.startswith("🌐")
        if is_exhaustive and breakdown.get("exhaustive_lines_stats"):
            active_lines_dataset = breakdown["exhaustive_lines_stats"]
            active_win_rate = breakdown["exhaustive_win_rate"]
            active_title = f"🌐 Efectividad Exhaustiva de TODAS las Líneas ({breakdown['matches_audited']} Partidos Auditados • {breakdown['exhaustive_hits']}/{breakdown['exhaustive_total']} Aciertos Totales)"
            top_line_disp = breakdown.get("best_exhaustive_line")
            worst_line_disp = breakdown.get("worst_exhaustive_line")
            disp_total_evals = breakdown["exhaustive_total"]
        else:
            active_lines_dataset = breakdown["specific_lines_stats"]
            active_win_rate = breakdown["overall_win_rate"]
            active_title = f"🎯 Efectividad de Líneas Recomendadas por el Modelo ({breakdown['matches_audited']} Partidos Auditados • {breakdown['overall_hits']}/{breakdown['overall_total']} Aciertos Totales)"
            top_line_disp = breakdown.get("best_line")
            worst_line_disp = breakdown.get("worst_line")
            disp_total_evals = breakdown["overall_total"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "🎯 % Acierto Promedio",
            f"{active_win_rate:.1f}%",
            f"{breakdown['exhaustive_hits'] if is_exhaustive else breakdown['overall_hits']} aciertos de {disp_total_evals}",
        )
        k2.metric(
            "🏆 Línea Estrella Más Certera",
            f"{top_line_disp['win_rate']:.1f}%" if top_line_disp else "-",
            top_line_disp["line_name"] if top_line_disp else "N/A",
        )
        k3.metric(
            "⚠️ Línea de Mayor Riesgo",
            f"{worst_line_disp['win_rate']:.1f}%" if worst_line_disp else "-",
            worst_line_disp["line_name"] if worst_line_disp else "N/A",
        )
        k4.metric("🏟️ Partidos Auditados", str(breakdown["matches_audited"]), f"{disp_total_evals} líneas evaluadas")

        st.markdown("---")
        st.plotly_chart(
            plot_specific_lines_chart(active_lines_dataset, active_win_rate, active_title),
            use_container_width=True,
        )

        ld_col1, ld_col2 = st.columns(2)
        active_evaluated = [s for s in active_lines_dataset if s["total"] > 0]
        top_spec = active_evaluated[:5] if len(active_evaluated) >= 5 else active_evaluated
        bottom_spec = list(reversed(active_evaluated[-5:])) if len(active_evaluated) >= 5 else []

        with ld_col1:
            st.markdown("#### 🟢 Líneas Específicas con Mayor Tasa de Acierto (Alta Confianza)")
            for sl in top_spec:
                st.markdown(
                    f"<div style='background:#064e3b; border-left:4px solid #10b981; padding:10px 14px; border-radius:6px; margin-bottom:8px;'>"
                    f"<div style='font-weight:700; color:#ffffff;'>{sl['line_name']}</div>"
                    f"<div style='color:#34d399; font-weight:800; font-size:1.05rem;'>{sl['win_rate']:.1f}% de Acierto ({sl['hits']}/{sl['total']} Aciertos)</div>"
                    f"<div style='font-size:0.80rem; color:#a7f3d0;'>{sl['status_tag']} • Prob. promedio: {sl['avg_prob']:.1f}% • Cuota justa: {round(100.0/max(1.0, sl['avg_prob']), 2):.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with ld_col2:
            st.markdown("#### 🔴 Líneas Específicas que MÁS FALLA (Alto Riesgo)")
            for bl in bottom_spec:
                st.markdown(
                    f"<div style='background:#4c0519; border-left:4px solid #f43f5e; padding:10px 14px; border-radius:6px; margin-bottom:8px;'>"
                    f"<div style='font-weight:700; color:#ffffff;'>{bl['line_name']}</div>"
                    f"<div style='color:#fda4af; font-weight:800; font-size:1.05rem;'>{bl['win_rate']:.1f}% de Acierto ({bl['misses']} fallos de {bl['total']})</div>"
                    f"<div style='font-size:0.80rem; color:#fecdd3;'>{bl['status_tag']} • Alta volatilidad</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### 📋 Tabla Resumen de Todas las Líneas Evaluadas")
        line_table_rows = [
            {
                "Línea de Apuesta": sl["line_name"],
                "Categoría": sl["category"],
                "Partidos Evaluados": sl["total"],
                "Aciertos (✅)": sl["hits"],
                "Fallos (❌)": sl["misses"],
                "% Acierto Real": f"{sl['win_rate']:.1f}%",
                "Prob. Modelo": f"{sl['avg_prob']:.1f}%",
                "Cuota Justa": f"{round(100.0 / max(1.0, sl['avg_prob']), 2):.2f}",
                "Veredicto": sl["badge"],
            }
            for sl in active_lines_dataset
            if sl["total"] > 0
        ]
        if line_table_rows:
            st.dataframe(pd.DataFrame(line_table_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🔍 Desglose de Partidos por Línea")
        for sl in active_lines_dataset:
            if sl["total"] == 0:
                continue
            with st.expander(f"Ver partidos de: {sl['line_name']} ({sl['hits']}/{sl['total']} aciertos — {sl['win_rate']:.1f}%)"):
                pick_rows = [
                    {
                        "Partido": pk_item["match"],
                        "Pronóstico / Línea": pk_item["pick"],
                        "Probabilidad Modelo": f"{pk_item['prob']:.1f}%",
                        "Dato Real": pk_item.get("actual") or "-",
                        "Dictamen": "✅ ACIERTO" if pk_item["hit"] is True else "❌ FALLO",
                    }
                    for pk_item in sl["picks"]
                ]
                st.dataframe(pd.DataFrame(pick_rows), use_container_width=True, hide_index=True)

    with tab_general_cats:
        st.markdown("#### 📁 Resumen por 10 Categorías Principales")
        st.plotly_chart(plot_market_accuracy_chart(breakdown), use_container_width=True)

        d_col1, d_col2 = st.columns(2)
        active_stats = [s for s in breakdown["category_stats"] if s["total"] > 0]
        top_cats = active_stats[:4] if len(active_stats) >= 4 else active_stats
        bottom_cats = list(reversed(active_stats[-4:])) if len(active_stats) >= 4 else []

        with d_col1:
            st.markdown("#### 🟢 Categorías que MÁS ACIERTA")
            for s in top_cats:
                st.markdown(
                    f"<div style='background:#064e3b; border-left:4px solid #10b981; padding:10px 14px; border-radius:6px; margin-bottom:8px;'>"
                    f"<div style='font-weight:700; color:#ffffff;'>{s['category']}</div>"
                    f"<div style='color:#34d399; font-weight:800; font-size:1.05rem;'>{s['win_rate']:.1f}% de Acierto ({s['hits']}/{s['total']})</div>"
                    f"<div style='font-size:0.80rem; color:#a7f3d0;'>{s['status_tag']} • Probabilidad promedio: {s['avg_prob']:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with d_col2:
            st.markdown("#### 🔴 Categorías que MÁS FALLA")
            for s in bottom_cats:
                st.markdown(
                    f"<div style='background:#4c0519; border-left:4px solid #f43f5e; padding:10px 14px; border-radius:6px; margin-bottom:8px;'>"
                    f"<div style='font-weight:700; color:#ffffff;'>{s['category']}</div>"
                    f"<div style='color:#fda4af; font-weight:800; font-size:1.05rem;'>{s['win_rate']:.1f}% de Acierto ({s['misses']} fallos de {s['total']})</div>"
                    f"<div style='font-size:0.80rem; color:#fecdd3;'>{s['status_tag']} • Volatilidad táctica</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        table_rows = [
            {
                "Mercado": s["category"],
                "Evaluados": s["total"],
                "Aciertos (✅)": s["hits"],
                "Fallos (❌)": s["misses"],
                "% Acierto": f"{s['win_rate']:.1f}%",
                "Prob. Media": f"{s['avg_prob']:.1f}%",
                "Cuota Justa": f"{round(100.0 / max(1.0, s['avg_prob']), 2):.2f}",
                "Veredicto": s["badge"],
            }
            for s in breakdown["category_stats"]
            if s["total"] > 0
        ]
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def render_radar_top_picks_view(
    filtered_matches: List[MatchDetailSchema],
    catalog: Dict[str, Any],
    analyze_matches_batch_parallel_fn: Callable[..., List[Dict[str, Any]]],
) -> None:
    """Renderiza bajo demanda el Radar Top Picks y Backtesting rápido."""
    st.subheader("🔥 Escáner Inteligente: Auditoría de Aciertos/Fallos en los 10 Mercados & Top Picks")

    finished_in_filter = [m for m in filtered_matches if m.status_type == "finished"]

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.3])
    with ctrl_col1:
        scan_count = st.slider(
            "Cantidad de partidos a procesar:",
            min_value=1,
            max_value=max(1, min(25, len(filtered_matches))),
            value=min(8, max(1, len(filtered_matches))),
        )
    with ctrl_col2:
        min_prob_filter = st.slider("Probabilidad mínima para Top Picks (%):", 55, 85, 65)
    with ctrl_col3:
        st.markdown(f"**Partidos finalizados disponibles hoy:** `{len(finished_in_filter)}`")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_audit_btn = st.button(
            f"📊 Auditar Partidos Finalizados ({min(scan_count, len(finished_in_filter))} P)",
            type="primary",
            use_container_width=True,
        )
    with btn_col2:
        run_scan_btn = st.button(
            f"🚀 Escanear Mejores Picks de la Jornada ({scan_count} P)",
            use_container_width=True,
        )

    if run_audit_btn:
        matches_to_audit = finished_in_filter[:scan_count] if finished_in_filter else filtered_matches[:scan_count]
        if not matches_to_audit:
            st.warning("No se encontraron partidos finalizados en la selección actual para auditar.")
        else:
            prog = st.progress(0, text="Auditando aciertos y fallos en paralelo...")
            audit_targets = [
                (m_item, catalog["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A"))
                for m_item in matches_to_audit
            ]
            batch_res = analyze_matches_batch_parallel_fn(
                audit_targets,
                progress_bar=prog,
                status_text_prefix="Auditando",
                max_workers=MAX_PARALLEL_WORKERS,
            )
            for res_deep in batch_res:
                m_val = res_deep["match"]
                st.session_state["audited_predictions_cache"][m_val.event_id] = res_deep["prediction"]
                st.session_state.setdefault("audited_matches_cache", {})[m_val.event_id] = m_val
            prog.empty()
            st.success(f"✅ Se auditaron exitosamente {len(batch_res)} partidos finalizados a máxima velocidad.")

    if run_scan_btn:
        scan_targets = [
            (m_item, catalog["fs_lookup"].get((m_item.home_team.id, m_item.away_team.id), "N/A"))
            for m_item in filtered_matches[:scan_count]
        ]
        prog = st.progress(0, text="Escaneando partidos en paralelo...")
        scanned_results = analyze_matches_batch_parallel_fn(
            scan_targets,
            progress_bar=prog,
            status_text_prefix="Escaneando Radar",
            max_workers=MAX_PARALLEL_WORKERS,
        )
        for res_deep in scanned_results:
            m_val = res_deep["match"]
            if m_val.status_type == "finished" and m_val.home_score is not None:
                st.session_state["audited_predictions_cache"][m_val.event_id] = res_deep["prediction"]
        prog.empty()
        st.session_state["radar_results"] = scanned_results

    all_audited_preds: List[MatchPrediction] = list(st.session_state.get("audited_predictions_cache", {}).values())
    radar_data = st.session_state.get("radar_results", [])
    for entry in radar_data:
        p_obj: MatchPrediction = entry["prediction"]
        if p_obj.verified_total > 0 and p_obj.event_id not in [ap.event_id for ap in all_audited_preds]:
            all_audited_preds.append(p_obj)

    audit_tab, picks_tab = st.tabs(
        [
            "📊 Auditoría de los 10 Mercados (Promedio, Ranking & Gráfica)",
            "🏆 Ranking Top Picks & Oportunidades de Valor",
        ]
    )

    with audit_tab:
        if not all_audited_preds:
            st.info(
                "💡 Aún no se han auditado partidos finalizados en esta sesión. "
                "Haz clic en el botón **'📊 Auditar Partidos Finalizados'** para procesar los marcadores."
            )
        else:
            breakdown = compute_market_accuracy_breakdown(all_audited_preds)
            ak1, ak2, ak3, ak4 = st.columns(4)
            ak1.metric("🎯 Promedio Global de Acierto", f"{breakdown['overall_win_rate']:.1f}%", f"{breakdown['overall_hits']} aciertos de {breakdown['overall_total']}")
            best_name = breakdown["best_category"]["category"] if breakdown["best_category"] else "N/A"
            best_wr = f"{breakdown['best_category']['win_rate']:.1f}%" if breakdown["best_category"] else "-"
            ak2.metric("🏆 Mercado que Más Acierta", best_wr, best_name[:26] + "..." if len(best_name) > 26 else best_name)
            worst_name = breakdown["worst_category"]["category"] if breakdown["worst_category"] else "N/A"
            worst_wr = f"{breakdown['worst_category']['win_rate']:.1f}%" if breakdown["worst_category"] else "-"
            ak3.metric("⚠️ Mercado que Más Falla", worst_wr, worst_name[:26] + "..." if len(worst_name) > 26 else worst_name)
            ak4.metric("🏟️ Partidos Finalizados Auditados", str(breakdown["matches_audited"]), f"{breakdown['overall_total']} pronósticos evaluados")

            st.markdown("---")
            st.plotly_chart(plot_market_accuracy_chart(breakdown), use_container_width=True)

            table_rows = [
                {
                    "Mercado Pronosticado": s["category"],
                    "Evaluados": s["total"],
                    "Aciertos (✅)": s["hits"],
                    "Fallos (❌)": s["misses"],
                    "Tasa de Acierto (%)": f"{s['win_rate']:.1f}%",
                    "Prob. Media": f"{s['avg_prob']:.1f}%",
                    "Cuota Justa": f"{round(100.0 / max(1.0, s['avg_prob']), 2):.2f}",
                    "Veredicto / Fiabilidad": s["badge"],
                }
                for s in breakdown["category_stats"]
                if s["total"] > 0
            ]
            if table_rows:
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with picks_tab:
        all_top_picks = []
        value_and_smart = []
        total_verified = 0
        total_hits = 0

        for entry in radar_data:
            m_obj = entry["match"]
            pred_obj = entry["prediction"]
            match_title = f"{m_obj.home_team.name} vs {m_obj.away_team.name}"

            for pk in pred_obj.all_market_picks:
                if pk["hit"] is not None:
                    total_verified += 1
                    if pk["hit"] is True:
                        total_hits += 1

                if pk["prob"] >= min_prob_filter:
                    hit_str = (
                        "✅ ACIERTO"
                        if pk["hit"] is True
                        else ("❌ FALLO" if pk["hit"] is False else "⏳ Pendiente")
                    )
                    all_top_picks.append(
                        {
                            "Torneo": m_obj.tournament.name,
                            "Partido": match_title,
                            "Mercado": pk["category"],
                            "Pronóstico": pk["pick"],
                            "Probabilidad (%)": pk["prob"],
                            "Cuota Justa": round(100.0 / max(1.0, pk["prob"]), 2),
                            "Verificación Real": f"{hit_str} ({pk['actual']})" if pk.get("actual") else hit_str,
                        }
                    )

            max_edge = max(
                [
                    ("Local (1)", pred_obj.value_edge_home or -99.0),
                    ("Empate (X)", pred_obj.value_edge_draw or -99.0),
                    ("Visita (2)", pred_obj.value_edge_away or -99.0),
                ],
                key=lambda x: x[1],
            )
            smart_alert = m_obj.odds.smart_money_side if m_obj.odds else None
            if max_edge[1] >= 3.5 or smart_alert:
                value_and_smart.append(
                    {
                        "Torneo": m_obj.tournament.name,
                        "Partido": match_title,
                        "Mejor Value Bet": f"{max_edge[0]} ({max_edge[1]:+.1f}%)" if max_edge[1] >= 3.5 else "Sin ventaja >3.5%",
                        "Alerta Smart Money (Caída de Cuota)": smart_alert or "Estable",
                        "Descanso (Loc vs Vis)": f"{pred_obj.rest_days_home}d vs {pred_obj.rest_days_away}d",
                        "Pronóstico Modelo": pred_obj.recommended_pick,
                    }
                )

        win_rate = round((total_hits / total_verified) * 100.0, 1) if total_verified > 0 else 0.0
        k1, k2, k3 = st.columns(3)
        k1.metric("Picks de Alta Confianza Detectados", len(all_top_picks))
        k2.metric("Alertas Value Bet / Smart Money", len(value_and_smart))
        k3.metric(
            "Efectividad Verificada (Backtesting)",
            f"{win_rate:.1f}% ({total_hits}/{total_verified})" if total_verified > 0 else "Partidos por jugar",
        )

        all_top_picks.sort(key=lambda x: x["Probabilidad (%)"], reverse=True)
        st.markdown("#### 🏆 Ranking de Picks Más Seguros de la Jornada")
        if all_top_picks:
            st.dataframe(pd.DataFrame(all_top_picks), use_container_width=True, hide_index=True)

        if value_and_smart:
            st.markdown("#### 💎 Oportunidades de Valor (Value Bets) & Caídas de Cuota (Smart Money)")
            st.dataframe(pd.DataFrame(value_and_smart), use_container_width=True, hide_index=True)
