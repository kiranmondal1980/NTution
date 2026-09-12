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
        /* Global Mobile & Desktop Layout Adjustments */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            max-width: 100%;
        }

        /* Responsive Metric Cards */
        [data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(120, 120, 120, 0.15);
            padding: 14px 16px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            margin-bottom: 10px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            font-weight: 600;
            color: #666666;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
            font-weight: 700;
        }

        /* Responsive DataFrames / Tables (Horizontal scroll wrapper on mobile) */
        [data-testid="stDataFrame"] {
            width: 100%;
            overflow-x: auto;
        }

        /* Professional Buttons & Inputs */
        .stButton button {
            width: 100%;
            border-radius: 6px;
            font-weight: 600;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
        }
        
        /* Mobile Specific Overrides */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            [data-testid="stMetric"] {
                padding: 10px 12px;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.15rem !important;
            }
            h1 { font-size: 1.6rem !important; }
            h2 { font-size: 1.35rem !important; }
            h3 { font-size: 1.1rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
