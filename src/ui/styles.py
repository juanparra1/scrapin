"""Estilos CSS institucionales y scripts de protección DOM para ProSports AI v3."""

import streamlit as st
import streamlit.components.v1 as components


def inject_custom_styles_and_scripts() -> None:
    """Inyecta protección contra Google Translate y la hoja de estilos Dark Sports Analytics Pro v3."""
    components.html(
        """
        <script>
        try {
            const doc = window.parent.document;
            doc.documentElement.setAttribute('translate', 'no');
            doc.documentElement.classList.add('notranslate');
            if (!doc.querySelector('meta[name="google"][content="notranslate"]')) {
                const meta = doc.createElement('meta');
                meta.name = 'google';
                meta.content = 'notranslate';
                doc.head.appendChild(meta);
            }
        } catch (e) {}
        </script>
        """,
        height=0,
    )

    st.markdown(
        """
        <style>
        .scoreboard-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 22px 26px;
            color: #f8fafc;
            margin-bottom: 18px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
        }
        .league-pill {
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.80rem;
            font-weight: 700;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }
        .ref-pill {
            display: inline-block;
            background: rgba(245, 158, 11, 0.16);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.4);
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .team-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: #ffffff;
            margin: 6px 0 4px 0;
        }
        .elo-tag {
            font-size: 0.83rem;
            color: #94a3b8;
            font-weight: 600;
        }
        .badge-w {
            display: inline-block; background: #16a34a; color: white;
            font-weight: 700; font-size: 0.74rem; padding: 2px 7px;
            border-radius: 5px; margin: 0 2px;
        }
        .badge-d {
            display: inline-block; background: #ca8a04; color: white;
            font-weight: 700; font-size: 0.74rem; padding: 2px 7px;
            border-radius: 5px; margin: 0 2px;
        }
        .badge-l {
            display: inline-block; background: #dc2626; color: white;
            font-weight: 700; font-size: 0.74rem; padding: 2px 7px;
            border-radius: 5px; margin: 0 2px;
        }
        .market-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 15px 17px;
            height: 100%;
            margin-bottom: 14px;
        }
        .market-header {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .market-pick {
            font-size: 1.10rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 6px;
        }
        .conf-high {
            background: rgba(34, 197, 94, 0.18); color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.4);
            padding: 2px 9px; border-radius: 999px; font-size: 0.73rem; font-weight: 700;
        }
        .conf-med {
            background: rgba(250, 204, 21, 0.18); color: #facc15;
            border: 1px solid rgba(250, 204, 21, 0.4);
            padding: 2px 9px; border-radius: 999px; font-size: 0.73rem; font-weight: 700;
        }
        .conf-low {
            background: rgba(148, 163, 184, 0.18); color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.4);
            padding: 2px 9px; border-radius: 999px; font-size: 0.73rem; font-weight: 700;
        }
        .hit-yes {
            background: rgba(22, 163, 74, 0.25); color: #4ade80;
            border: 1px solid #22c55e;
            padding: 2px 9px; border-radius: 6px; font-size: 0.74rem; font-weight: 800;
        }
        .hit-no {
            background: rgba(220, 38, 38, 0.25); color: #f87171;
            border: 1px solid #ef4444;
            padding: 2px 9px; border-radius: 6px; font-size: 0.74rem; font-weight: 800;
        }
        .prob-bar-bg {
            width: 100%; background: #0f172a; height: 8px;
            border-radius: 99px; overflow: hidden; margin: 9px 0 6px 0;
        }
        .prob-bar-fill {
            height: 100%; border-radius: 99px;
            background: linear-gradient(90deg, #38bdf8, #22c55e);
        }
        .stat-compare-box {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }
        .smart-money-badge {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(245, 158, 11, 0.25) 100%);
            border: 1px solid #f59e0b;
            border-radius: 10px;
            padding: 8px 14px;
            color: #fef08a;
            font-size: 0.86rem;
            font-weight: 700;
            margin: 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
        }
        .telegram-btn {
            display: inline-flex;
            align-items: center;
            background: #0088cc;
            color: white !important;
            border-radius: 8px;
            padding: 6px 14px;
            font-weight: 700;
            text-decoration: none;
            font-size: 0.85rem;
            margin: 4px 0;
        }
        @media (max-width: 768px) {
            .scoreboard-card {
                padding: 14px 14px !important;
                border-radius: 12px !important;
            }
            .team-title {
                font-size: 1.15rem !important;
            }
            .market-card {
                padding: 12px 12px !important;
                margin-bottom: 10px !important;
            }
            .market-pick {
                font-size: 0.98rem !important;
            }
            .league-pill {
                font-size: 0.72rem !important;
                padding: 2px 10px !important;
            }
            div[data-testid="column"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
                margin-bottom: 6px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
