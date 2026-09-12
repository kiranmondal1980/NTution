"""
NSE MOMENTUM 5™ — Professional Mobile & Desktop CSS Styling Engine
================================================================================
Injects institutional CSS to guarantee a polished, responsive mobile UI
with fluid tables, compact metric cards, and clean touch targets.
================================================================================
"""

import streamlit as st


def apply_custom_styling() -> None:
    """Injects responsive CSS rules into the Streamlit app session."""
    st.markdown(
        """
        <style>
        /* Hide redundant module docstring headers at the top of pages */
        .main .block-container > div:first-child h1:first-of-type {
            display: none !important;
        }

        /* Global Mobile & Desktop Layout Adjustments */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }

        /* Streamlit Header / Toolbar Cleanup */
        header[data-testid="stHeader"] {
            background-color: transparent;
        }

        /* Responsive Metric Cards */
        [data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(120, 120, 120, 0.15);
            padding: 12px 14px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            margin-bottom: 8px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            font-weight: 600;
            color: #555555;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
            font-weight: 700;
        }

        /* Responsive DataFrames / Tables */
        [data-testid="stDataFrame"] {
            width: 100%;
            overflow-x: auto;
            border-radius: 8px;
        }

        /* Professional Buttons & Inputs */
        .stButton button {
            width: 100%;
            border-radius: 6px;
            font-weight: 600;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            background-color: #00897b;
            color: white;
        }
        .stButton button:hover {
            background-color: #00695c;
            color: white;
        }

        /* Mobile Specific Overrides */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.5rem;
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }
            [data-testid="stMetric"] {
                padding: 8px 10px;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.1rem !important;
            }
            h1 { font-size: 1.4rem !important; }
            h2 { font-size: 1.2rem !important; }
            h3 { font-size: 1.0rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
