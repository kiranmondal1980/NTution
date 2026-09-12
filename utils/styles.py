"""
NSE MOMENTUM 5™
============================================================
Institutional Trading Terminal UI Styling Engine

Provides the global visual system for the Streamlit
application.

Design objectives:
    • Professional quantitative-trading appearance
    • Excellent information hierarchy
    • Compact trading-terminal layout
    • Strong readability
    • Responsive desktop/mobile behaviour
    • Consistent cards, tables, badges and controls
    • Dark/light Streamlit theme compatibility
    • Minimal dependency on fragile Streamlit DOM selectors
============================================================
"""

from __future__ import annotations

import streamlit as st


# ============================================================
# MAIN STYLE ENGINE
# ============================================================

def apply_custom_styling() -> None:
    """
    Inject the complete NSE MOMENTUM 5™ visual design system.
    """

    st.markdown(
        """
        <style>

        /* ====================================================
           ROOT / GLOBAL
           ==================================================== */

        :root {
            --nm5-radius-sm: 6px;
            --nm5-radius-md: 10px;
            --nm5-radius-lg: 14px;

            --nm5-border:
                rgba(128, 128, 128, 0.20);

            --nm5-border-strong:
                rgba(128, 128, 128, 0.32);

            --nm5-muted:
                rgba(128, 128, 128, 0.78);

            --nm5-success:
                #16a34a;

            --nm5-warning:
                #d97706;

            --nm5-danger:
                #dc2626;

            --nm5-info:
                #2563eb;

            --nm5-brand:
                #2563eb;

            --nm5-brand-dark:
                #1d4ed8;
        }


        /* ====================================================
           STREAMLIT MAIN CONTAINER
           ==================================================== */

        .block-container {

            max-width: 100% !important;

            padding-top:
                0.85rem !important;

            padding-bottom:
                3.5rem !important;

            padding-left:
                1.75rem !important;

            padding-right:
                1.75rem !important;
        }


        /* ====================================================
           REMOVE EXCESS TOP SPACE
           ==================================================== */

        header[data-testid="stHeader"] {

            background:
                transparent !important;

            height:
                2.25rem !important;
        }


        /* ====================================================
           GLOBAL TYPOGRAPHY
           ==================================================== */

        html,
        body,
        [class*="css"] {

            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Roboto,
                Helvetica,
                Arial,
                sans-serif;
        }


        h1 {

            font-size:
                1.72rem !important;

            font-weight:
                750 !important;

            line-height:
                1.15 !important;

            letter-spacing:
                -0.025em !important;

            margin-top:
                0.15rem !important;

            margin-bottom:
                0.35rem !important;
        }


        h2 {

            font-size:
                1.32rem !important;

            font-weight:
                700 !important;

            letter-spacing:
                -0.018em !important;

            margin-top:
                1rem !important;

            margin-bottom:
                0.45rem !important;
        }


        h3 {

            font-size:
                1.04rem !important;

            font-weight:
                680 !important;

            margin-top:
                0.7rem !important;

            margin-bottom:
                0.35rem !important;
        }


        p {

            line-height:
                1.48 !important;
        }


        /* ====================================================
           TOP APPLICATION BAR
           ==================================================== */

        .nm5-topbar {

            display:
                flex;

            align-items:
                center;

            justify-content:
                space-between;

            gap:
                1.25rem;

            width:
                100%;

            min-height:
                72px;

            padding:
                12px 16px;

            margin-bottom:
                10px;

            border:
                1px solid var(--nm5-border);

            border-radius:
                var(--nm5-radius-md);

            background:
                var(--secondary-background-color);

            box-shadow:
                0 2px 10px
                rgba(0, 0, 0, 0.04);
        }


        .nm5-brand {

            display:
                flex;

            align-items:
                center;

            gap:
                12px;

            min-width:
                240px;
        }


        .nm5-logo {

            display:
                flex;

            align-items:
                center;

            justify-content:
                center;

            width:
                42px;

            height:
                42px;

            border-radius:
                10px;

            background:
                var(--nm5-brand);

            color:
                white;

            font-size:
                1.25rem;

            font-weight:
                800;

            box-shadow:
                0 4px 12px
                rgba(37, 99, 235, 0.25);
        }


        .nm5-brand-name {

            font-size:
                1.05rem;

            font-weight:
                800;

            letter-spacing:
                -0.025em;
        }


        .nm5-tm {

            font-size:
                0.52rem;

            vertical-align:
                top;

            margin-left:
                2px;

            opacity:
                0.65;
        }


        .nm5-brand-subtitle {

            margin-top:
                2px;

            font-size:
                0.61rem;

            font-weight:
                650;

            letter-spacing:
                0.085em;

            color:
                var(--nm5-muted);
        }


        /* ====================================================
           HEADER STATUS
           ==================================================== */

        .nm5-header-status {

            display:
                flex;

            align-items:
                stretch;

            justify-content:
                flex-end;

            gap:
                8px;

            flex-wrap:
                wrap;
        }


        .nm5-status-item {

            min-width:
                92px;

            padding:
                7px 10px;

            border:
                1px solid var(--nm5-border);

            border-radius:
                var(--nm5-radius-sm);

            background:
                rgba(128, 128, 128, 0.035);
        }


        .nm5-status-label {

            display:
                block;

            margin-bottom:
                2px;

            font-size:
                0.58rem;

            font-weight:
                700;

            letter-spacing:
                0.08em;

            color:
                var(--nm5-muted);
        }


        .nm5-status-value {

            display:
                block;

            font-size:
                0.78rem;

            font-weight:
                750;
        }


        .status-live {

            display:
                block;

            color:
                var(--nm5-success);

            font-size:
                0.72rem;

            font-weight:
                750;
        }


        .status-warning {

            display:
                block;

            color:
                var(--nm5-warning);

            font-size:
                0.72rem;

            font-weight:
                750;
        }


        /* ====================================================
           DATA STATUS BAR
           ==================================================== */

        .nm5-data-bar {

            display:
                flex;

            align-items:
                center;

            flex-wrap:
                wrap;

            gap:
                7px;

            padding:
                7px 11px;

            margin:
                0 0 14px 0;

            border-radius:
                var(--nm5-radius-sm);

            border:
                1px solid var(--nm5-border);

            font-size:
                0.72rem;

            line-height:
                1.2;
        }


        .nm5-data-ok {

            color:
                var(--nm5-success);

            background:
                rgba(22, 163, 74, 0.055);

            border-color:
                rgba(22, 163, 74, 0.20);
        }


        .nm5-data-warning {

            color:
                var(--nm5-warning);

            background:
                rgba(217, 119, 6, 0.055);

            border-color:
                rgba(217, 119, 6, 0.20);
        }


        .nm5-data-separator {

            opacity:
                0.35;
        }


        /* ====================================================
           PAGE HEADING
           ==================================================== */

        .nm5-page-heading {

            display:
                flex;

            align-items:
                flex-end;

            justify-content:
                space-between;

            margin:
                4px 0 14px 0;

            padding:
                0 2px;
        }


        .nm5-page-kicker {

            margin-bottom:
                4px;

            font-size:
                0.61rem;

            font-weight:
                750;

            letter-spacing:
                0.12em;

            color:
                var(--nm5-brand);

            text-transform:
                uppercase;
        }


        .nm5-page-heading h1 {

            margin:
                0 !important;
        }


        .nm5-page-description {

            margin-top:
                5px;

            font-size:
                0.78rem;

            color:
                var(--nm5-muted);
        }


        /* ====================================================
           SIDEBAR
           ==================================================== */

        section[data-testid="stSidebar"] {

            border-right:
                1px solid var(--nm5-border);
        }


        section[data-testid="stSidebar"] > div {

            padding-top:
                0.75rem;
        }


        .nm5-sidebar-brand {

            display:
                flex;

            align-items:
                center;

            gap:
                10px;

            padding:
                4px 4px 14px 4px;

            margin-bottom:
                4px;

            border-bottom:
                1px solid var(--nm5-border);
        }


        .nm5-sidebar-mark {

            display:
                flex;

            align-items:
                center;

            justify-content:
                center;

            width:
                34px;

            height:
                34px;

            border-radius:
                8px;

            background:
                var(--nm5-brand);

            color:
                white;

            font-size:
                1rem;

            font-weight:
                800;
        }


        .nm5-sidebar-title {

            font-size:
                0.88rem;

            font-weight:
                800;

            letter-spacing:
                -0.015em;
        }


        .nm5-sidebar-subtitle {

            margin-top:
                1px;

            font-size:
                0.58rem;

            font-weight:
                700;

            letter-spacing:
                0.1em;

            color:
                var(--nm5-muted);
        }


        .nm5-sidebar-section-title {

            margin:
                13px 4px 5px 4px;

            font-size:
                0.58rem;

            font-weight:
                800;

            letter-spacing:
                0.11em;

            color:
                var(--nm5-muted);
        }


        /* Sidebar buttons */

        section[data-testid="stSidebar"]
        .stButton > button {

            min-height:
                38px;

            padding:
                0.25rem 0.65rem;

            margin:
                1px 0;

            border-radius:
                7px;

            border:
                1px solid transparent;

            font-size:
                0.78rem;

            font-weight:
                620;

            text-align:
                left;

            transition:
                all 0.15s ease;
        }


        section[data-testid="stSidebar"]
        .stButton > button:hover {

            border-color:
                var(--nm5-border-strong);

            transform:
                translateX(1px);
        }


        section[data-testid="stSidebar"]
        .stButton > button[kind="primary"] {

            background:
                rgba(37, 99, 235, 0.12);

            color:
                var(--nm5-brand);

            border-color:
                rgba(37, 99, 235, 0.22);
        }


        .nm5-sidebar-footer {

            margin-top:
                10px;

            padding:
                10px;

            border:
                1px solid var(--nm5-border);

            border-radius:
                var(--nm5-radius-md);

            background:
                rgba(128, 128, 128, 0.035);
        }


        .nm5-sidebar-risk {

            font-size:
                0.59rem;

            font-weight:
                750;

            color:
                var(--nm5-warning);
        }


        .nm5-sidebar-disclaimer {

            margin-top:
                5px;

            font-size:
                0.59rem;

            line-height:
                1.4;

            color:
                var(--nm5-muted);
        }


        /* ====================================================
           METRIC CARDS
           ==================================================== */

        [data-testid="stMetric"] {

            min-height:
                92px;

            padding:
                13px 15px !important;

            border:
                1px solid var(--nm5-border);

            border-radius:
                var(--nm5-radius-md);

            background:
                var(--secondary-background-color);

            box-shadow:
                0 1px 5px
                rgba(0, 0, 0, 0.035);
        }


        [data-testid="stMetricLabel"] {

            font-size:
                0.62rem !important;

            font-weight:
                750 !important;

            letter-spacing:
                0.075em;

            text-transform:
                uppercase;
        }


        [data-testid="stMetricValue"] {

            margin-top:
                3px;

            font-size:
                1.36rem !important;

            font-weight:
                760 !important;

            letter-spacing:
                -0.025em;
        }


        [data-testid="stMetricDelta"] {

            font-size:
                0.68rem !important;

            font-weight:
                650 !important;
        }


        /* ====================================================
           DATAFRAMES / TABLES
           ==================================================== */

        [data-testid="stDataFrame"] {

            width:
                100%;

            border:
                1px solid var(--nm5-border);

            border-radius:
                var(--nm5-radius-md);

            overflow:
                hidden;
        }


        /* ====================================================
           BUTTONS
           ==================================================== */

        .stButton > button {

            min-height:
                38px;

            border-radius:
                7px !important;

            font-weight:
                650 !important;

            transition:
                all 0.15s ease !important;
        }


        .stButton > button:hover {

            transform:
                translateY(-1px);
        }


        /* ====================================================
           SELECTBOX / INPUTS
           ==================================================== */

        div[data-baseweb="select"] > div {

            min-height:
                38px;

            border-radius:
                7px;
        }


        .stTextInput input,
        .stNumberInput input,
        .stDateInput input {

            border-radius:
                7px !important;
        }


        /* ====================================================
           TABS
           ==================================================== */

        .stTabs [data-baseweb="tab-list"] {

            gap:
                5px;

            border-bottom:
                1px solid var(--nm5-border);
        }


        .stTabs [data-baseweb="tab"] {

            padding:
                8px 12px;

            font-size:
                0.75rem;

            font-weight:
                650;
        }


        /* ====================================================
           EXPANDERS
           ==================================================== */

        [data-testid="stExpander"] {

            border:
                1px solid var(--nm5-border) !important;

            border-radius:
                var(--nm5-radius-md) !important;

            overflow:
                hidden;
        }


        /* ====================================================
           ALERTS
           ==================================================== */

        [data-testid="stAlert"] {

            border-radius:
                var(--nm5-radius-md);

            font-size:
                0.78rem;
        }


        /* ====================================================
           CUSTOM SELECTED STOCK
           ==================================================== */

        .nm5-selected-stock {

            display:
                flex;

            align-items:
                center;

            gap:
                10px;

            margin:
                3px 0 12px 0;

            padding:
                8px 11px;

            border-left:
                3px solid var(--nm5-brand);

            border-top:
                1px solid var(--nm5-border);

            border-right:
                1px solid var(--nm5-border);

            border-bottom:
                1px solid var(--nm5-border);

            border-radius:
                6px;

            background:
                rgba(37, 99, 235, 0.045);
        }


        .nm5-selected-label {

            font-size:
                0.57rem;

            font-weight:
                750;

            letter-spacing:
                0.09em;

            color:
                var(--nm5-muted);
        }


        .nm5-selected-symbol {

            font-size:
                0.9rem;

            font-weight:
                800;

            color:
                var(--nm5-brand);
        }


        /* ====================================================
           FOOTER
           ==================================================== */

        .nm5-footer {

            display:
                flex;

            align-items:
                center;

            justify-content:
                space-between;

            gap:
                12px;

            margin-top:
                32px;

            padding:
                12px 2px 4px 2px;

            border-top:
                1px solid var(--nm5-border);

            color:
                var(--nm5-muted);

            font-size:
                0.62rem;

            line-height:
                1.4;
        }


        /* ====================================================
           SCROLLBAR
           ==================================================== */

        ::-webkit-scrollbar {

            width:
                7px;

            height:
                7px;
        }


        ::-webkit-scrollbar-track {

            background:
                transparent;
        }


        ::-webkit-scrollbar-thumb {

            background:
                rgba(128, 128, 128, 0.28);

            border-radius:
                10px;
        }


        ::-webkit-scrollbar-thumb:hover {

            background:
                rgba(128, 128, 128, 0.42);
        }


        /* ====================================================
           MOBILE
           ==================================================== */

        @media (max-width: 1100px) {

            .nm5-topbar {

                align-items:
                    flex-start;

                flex-direction:
                    column;
            }


            .nm5-header-status {

                width:
                    100%;

                justify-content:
                    flex-start;
            }


            .nm5-status-item {

                flex:
                    1 1 auto;
            }
        }


        @media (max-width: 768px) {

            .block-container {

                padding-top:
                    0.45rem !important;

                padding-left:
                    0.65rem !important;

                padding-right:
                    0.65rem !important;

                padding-bottom:
                    2.5rem !important;
            }


            .nm5-topbar {

                padding:
                    10px;

                min-height:
                    auto;
            }


            .nm5-brand {

                width:
                    100%;
            }


            .nm5-header-status {

                display:
                    grid;

                grid-template-columns:
                    repeat(2, 1fr);

                width:
                    100%;
            }


            .nm5-status-item {

                min-width:
                    0;
            }


            h1 {

                font-size:
                    1.38rem !important;
            }


            h2 {

                font-size:
                    1.15rem !important;
            }


            .nm5-page-description {

                font-size:
                    0.7rem;
            }


            [data-testid="stMetric"] {

                min-height:
                    78px;

                padding:
                    10px 11px !important;
            }


            [data-testid="stMetricValue"] {

                font-size:
                    1.05rem !important;
            }


            .nm5-footer {

                flex-direction:
                    column;

                align-items:
                    flex-start;
            }
        }


        /* ====================================================
           ACCESSIBILITY / REDUCED MOTION
           ==================================================== */

        @media (prefers-reduced-motion: reduce) {

            * {

                scroll-behavior:
                    auto !important;

                transition:
                    none !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
