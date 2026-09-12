"""
NSE MOMENTUM 5™ — CLI Tool: Event-Driven Backtesting & Simulation Runner
================================================================================
Command-line utility to execute rigorous chronological event-driven backtests
directly from the terminal, recording the run and trades in the database.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --capital 1000000 --score 75.0 --horizon 5
    python scripts/run_backtest.py --export-csv backtest_trades_output.csv
================================================================================
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.initialize import init_database
from database.db import get_db_session
from database.models import BacktestRunRecord, BacktestTradeRecord
from data.downloader import MarketDataDownloader
from features import build_feature_pipeline
from backtest.engine import EventDrivenBacktester
from config import CONFIG
from utils.logger import setup_logger

logger = setup_logger("CLI_BACKTEST_RUNNER")


def main() -> int:
    parser = argparse.ArgumentParser(description="NSE MOMENTUM 5™ CLI Backtester")
    parser.add_argument(
        "--capital",
        type=float,
        default=CONFIG.risk.default_capital_inr,
        help="Initial testing capital in INR (default: 1,000,000 INR)."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=CONFIG.backtest.default_holding_horizon_days,
        help="Maximum holding horizon in sessions (default: 5 sessions)."
    )
    parser.add_argument(
        "--score",
        type=float,
        default=CONFIG.signal.momentum,
        help="Minimum Momentum Score trigger threshold (default: 75.0)."
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default="",
        help="Optional CSV file path to export executed trade records."
    )
    args = parser.parse_args()

    # 1. Ensure database is initialized
    init_database()

    # 2. Load universe data and compute feature pipelines
    logger.info("Loading market data from SQLite database...")
    downloader = MarketDataDownloader()
    bench_df = downloader.load_ohlcv_from_db(CONFIG.universe.benchmark_symbol)

    if bench_df.empty:
        logger.error(
            "Benchmark data (^NSEI) is missing. Please run `python scripts/download_data.py` first."
        )
        return 1

    symbols = CONFIG.universe.default_symbols
    universe_features = {}

    logger.info(f"Computing quantitative feature pipelines across {len(symbols)} tickers...")
    for sym in symbols:
        df = downloader.load_ohlcv_from_db(sym)
        if not df.empty and len(df) >= 30:
            pipeline_df = build_feature_pipeline(df, benchmark_df=bench_df)
            universe_features[sym] = pipeline_df

    if not universe_features:
        logger.error("No valid universe price history loaded. Please download market data first.")
        return 1

    # 3. Execute chronological backtest simulation
    logger.info(
        f"Executing Event Backtest (Capital: INR {args.capital:,.0f}, "
        f"Score Threshold: {args.score}, Max Horizon: {args.horizon} sessions)..."
    )

    tester = EventDrivenBacktester(
        initial_capital=args.capital,
        max_holding_sessions=args.horizon,
        min_momentum_score=args.score
    )
    results = tester.run(universe_features, benchmark_df=bench_df)

    summary = results["summary"]
    trades = results["trades"]

    if not summary or summary.total_trades == 0:
        logger.warning("No trades were generated with the given configuration.")
        return 0

    # 4. Display Institutional Summary Report
    print("\n" + "=" * 70)
    print("           NSE MOMENTUM 5™ — BACKTEST SIMULATION REPORT")
    print("=" * 70)
    print(f"  • Strategy Status       : {'ROBUST (PASSED)' if summary.is_robust else 'NON-ROBUST (REJECTED)'}")
    print(f"  • Total Trades Executed : {summary.total_trades} ({summary.winning_trades}W / {summary.losing_trades}L)")
    print(f"  • Win Rate              : {summary.win_rate_pct:.1f}%")
    print(f"  • Profit Factor         : {summary.profit_factor:.2f}")
    print(f"  • Net Expectancy / Trade: {summary.expectancy_pct:+.2f}% (INR {summary.expectancy_inr:+,.2f})")
    print(f"  • Cumulative Return     : {summary.cumulative_return_pct:+.2f}%")
    print(f"  • Compounded Annual (CAGR): {summary.cagr_pct:.2f}%")
    print(f"  • Maximum Drawdown      : {summary.max_drawdown_pct:.2f}%")
    print(f"  • Sharpe Ratio          : {summary.sharpe_ratio:.2f}")
    print(f"  • Sortino Ratio         : {summary.sortino_ratio:.2f}")
    print(f"  • Calmar Ratio          : {summary.calmar_ratio:.2f}")
    print(f"  • Avg Holding Sessions  : {summary.avg_holding_sessions:.1f} sessions")
    print("-" * 70)
    print("  RESEARCH TARGET HIT RATES:")
    print(f"  • +5% within 3 Sessions : {summary.target_hit_rate_5pct_3d:.1f}%")
    print(f"  • +10% within 5 Sessions: {summary.target_hit_rate_10pct_5d:.1f}%")
    print(f"  • +15% within 5 Sessions: {summary.target_hit_rate_15pct_5d:.1f}%")
    print(f"  • +20% within 5 Sessions: {summary.target_hit_rate_20pct_5d:.1f}%")
    print("-" * 70)
    print(f"  • Total Statutory Fees  : INR {summary.total_costs_inr:,.2f}")
    print(f"  • Slippage Impact       : INR {summary.total_slippage_inr:,.2f}")
    print("=" * 70 + "\n")

    # 5. Persist Backtest Run in Relational Database
    try:
        with get_db_session() as session:
            run_rec = BacktestRunRecord(
                strategy_name="NSE_MOMENTUM_5_MULTI_FACTOR",
                start_date=trades[0]["entry_date"],
                end_date=trades[-1]["exit_date"],
                total_trades=summary.total_trades,
                win_rate=summary.win_rate_pct,
                profit_factor=summary.profit_factor,
                expectancy=summary.expectancy_pct,
                cumulative_return=summary.cumulative_return_pct,
                cagr=summary.cagr_pct,
                max_drawdown=summary.max_drawdown_pct,
                sharpe_ratio=summary.sharpe_ratio,
                sortino_ratio=summary.sortino_ratio,
                is_robust=summary.is_robust,
                parameters_json=json.dumps({
                    "capital": args.capital,
                    "score_threshold": args.score,
                    "max_holding_sessions": args.horizon
                })
            )
            session.add(run_rec)
            session.flush()

            # Persist individual trades
            for t in trades:
                session.add(BacktestTradeRecord(
                    run_id=run_rec.id,
                    symbol=t["symbol"],
                    signal_date=t["entry_date"],
                    entry_date=t["entry_date"],
                    entry_price=t["entry_price"],
                    exit_date=t["exit_date"],
                    exit_price=t["exit_price"],
                    quantity=t["quantity"],
                    net_pnl=t["net_pnl"],
                    net_pnl_pct=t["net_pnl_pct"],
                    holding_days=t["holding_sessions"],
                    exit_reason=t["exit_reason"]
                ))
            session.commit()
            logger.info(f"Persisted Backtest Run #{run_rec.id} and {len(trades)} trades to database.")
    except Exception as exc:
        logger.warning(f"Could not persist backtest records to database: {exc}")

    # 6. Optional CSV Export
    if args.export_csv.strip():
        csv_path = Path(args.export_csv)
        df_trades = pd.DataFrame(trades)
        df_trades.to_csv(csv_path, index=False)
        logger.info(f"Exported {len(trades)} executed trade records to {csv_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
