"""
NSE MOMENTUM 5™ — Professional Institutional UI Styling Engine
================================================================================
Injected globally to ensure clean typography, balanced margins, and elite
fintech terminal aesthetics.
================================================================================
"""

import streamlit as st


def apply_custom_styling() -> None:
    """Injects professional institutional CSS rules into the Streamlit app session."""
    st.markdown(
        """
        <style>
        /* 1. AGGRESSIVELY HIDE ANY LEAKED DOCSTRING HEADERS */
        .main .block-container > div:first-child h1:first-of-type,
        .main .block-container > div:first-child p:first-of-type,
        .main .block-container > div:first-child span:first-of-type {
            display: none !important;
        }

        /* 2. LAYOUT & PADDING */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }

        /* 3. TYPOGRAPHY HIERARCHY */
        h1 {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem !important;
        }
        h2 {
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
        }

        /* 4. PROFESSIONAL CARD CONTAINERS & METRICS */
        [data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(150, 150, 150, 0.15);
            padding: 14px 16px !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #666666 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700;
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
            padding: 0.4rem 1rem !important;
        }

        /* 7. MOBILE RESPONSIVE TWEAKS */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.5rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
            }
            [data-testid="stMetric"] {
                padding: 10px 12px !important;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
