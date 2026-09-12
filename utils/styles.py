"""
NSE MOMENTUM 5™ — Professional Institutional UI Styling Engine
================================================================================
Injected globally into Streamlit to hide raw file docstrings, enforce clean header
typography, compact metric cards, and establish a professional trading terminal aesthetic.
================================================================================
"""

import streamlit as st


def apply_custom_styling() -> None:
    """Injects professional institutional CSS rules into the Streamlit app session."""
    st.markdown(
        """
        <style>
        /* 1. COMPLETELY HIDE RAW MODULE DOCSTRING HEADERS AT TOP OF PAGES */
        .main .block-container > div:first-child h1:first-of-type {
            display: none !important;
        }
        .main .block-container > div:first-child p:first-of-type {
            display: none !important;
        }

        /* 2. LAYOUT & PADDING */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }

        /* 3. PROFESSIONAL CARD CONTAINERS & METRICS */
        [data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(150, 150, 150, 0.18);
            padding: 16px 18px !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #555555 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            font-weight: 700;
        }

        /* 4. EXPANDERS & FORM CONTROLS */
        [data-testid="stExpander"] {
            border: 1px solid rgba(150, 150, 150, 0.2);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }

        /* 5. TABLES & DATAFRAMES */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(150, 150, 150, 0.15);
        }

        /* 6. BUTTONS */
        .stButton button {
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
        }

        /* 7. MOBILE RESPONSIVE TWEAKS */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
            }
            [data-testid="stMetric"] {
                padding: 12px 14px !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.2rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
