"""
Backtest Runner
Main orchestrator for running complete backtests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import json

from backtest.data_provider import DataProvider
from backtest.signal_replayer import SignalReplayer
from backtest.trade_simulator import TradeSimulator, SimulatorConfig
from backtest.metrics import MetricsCalculator, BacktestMetrics


class BacktestRunner:
    """
    Main orchestrator for running backtests.
    Coordinates data fetching, signal generation, trade simulation, and metrics calculation.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 23,
        confidence_threshold: float = 65.0,
        position_size_pct: float = 10.0
    ):
        self.data_provider = DataProvider()
        self.signal_replayer = SignalReplayer(confidence_threshold=confidence_threshold)
        
        config = SimulatorConfig(
            initial_capital=initial_capital,
            leverage=leverage,
            position_size_pct=position_size_pct
        )
        self.trade_simulator = TradeSimulator(config)
        self.metrics_calculator = MetricsCalculator(initial_capital=initial_capital)
        
        self.confidence_threshold = confidence_threshold
        
    def run(
        self,
        symbol: str = "BTC/USDT:USDT",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "1h"
    ) -> BacktestMetrics:
        """
        Run a complete backtest.
        
        Args:
            symbol: Trading pair
            start_date: Start date (YYYY-MM-DD), defaults to 6 months ago
            end_date: End date (YYYY-MM-DD), defaults to now
            timeframe: Main timeframe for analysis
            
        Returns:
            BacktestMetrics object with all results
        """
        print("\n" + "=" * 65)
        print("               BTC SNIPER BOT - BACKTEST ENGINE")
        print("=" * 65)
        
        # Default dates
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
        
        print(f"\n📅 Period: {start_date} → {end_date}")
        print(f"📊 Symbol: {symbol}")
        print(f"⏱️  Timeframe: {timeframe}")
        print(f"💰 Initial Capital: ${self.trade_simulator.config.initial_capital:,.2f}")
        print(f"⚡ Leverage: {self.trade_simulator.config.leverage}x")
        print(f"🎯 Confidence Threshold: {self.confidence_threshold}%")
        
        # Step 1: Fetch historical data
        print("\n" + "-" * 40)
        print("STEP 1: Fetching Historical Data")
        print("-" * 40)
        
        ohlcv = self.data_provider.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not ohlcv:
            print("❌ Failed to fetch data")
            return BacktestMetrics(initial_capital=self.trade_simulator.config.initial_capital)
        
        print(f"   Total candles: {len(ohlcv)}")
        
        # Step 2: Generate signals
        print("\n" + "-" * 40)
        print("STEP 2: Generating Signals")
        print("-" * 40)
        
        signals = self.signal_replayer.generate_signals_from_candles(ohlcv)
        
        if not signals:
            print("❌ No signals generated")
            return BacktestMetrics(initial_capital=self.trade_simulator.config.initial_capital)
        
        print(f"   Generated: {len(signals)} signals")
        
        # Count by type
        by_type = {}
        for s in signals:
            t = s['signal_type']
            by_type[t] = by_type.get(t, 0) + 1
        for t, count in sorted(by_type.items()):
            print(f"   - {t}: {count}")
        
        # Step 3: Simulate trades
        print("\n" + "-" * 40)
        print("STEP 3: Simulating Trades")
        print("-" * 40)
        
        self.trade_simulator.reset()
        
        for i, signal in enumerate(signals):
            # Open trade
            trade = self.trade_simulator.open_trade(
                signal_timestamp=signal['timestamp'],
                entry_price=signal['price'],
                direction=signal['direction'],
                signal_type=signal['signal_type'],
                confidence=signal['confidence'],
                tp1=signal['tp1'],
                tp2=signal['tp2'],
                sl=signal['sl'],
                volatility_pct=signal.get('atr', 500) / signal['price'] * 100  # ATR as volatility
            )
            
            # Check exit on subsequent candles
            trade = self.trade_simulator.check_trade_exit(trade, ohlcv)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(signals)} signals...")
        
        closed_trades = self.trade_simulator.get_closed_trades()
        print(f"   Closed trades: {len(closed_trades)}")
        
        # Step 4: Calculate metrics
        print("\n" + "-" * 40)
        print("STEP 4: Calculating Metrics")
        print("-" * 40)
        
        metrics = self.metrics_calculator.calculate(closed_trades)
        
        # Print report
        report = self.metrics_calculator.format_report(metrics)
        print("\n" + report)
        
        # Save results
        self._save_results(metrics, signals, closed_trades, start_date, end_date)
        
        return metrics
    
    def _save_results(
        self,
        metrics: BacktestMetrics,
        signals: List[Dict],
        trades: List,
        start_date: str,
        end_date: str
    ):
        """Save backtest results to file"""
        results_dir = "backtest/results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Convert trades to dicts
        trades_data = []
        for t in trades:
            entry_dt = datetime.fromtimestamp(t.entry_timestamp / 1000, tz=timezone.utc)
            exit_dt = datetime.fromtimestamp(t.exit_timestamp / 1000, tz=timezone.utc) if t.exit_timestamp else None
            
            trades_data.append({
                "id": t.id,
                "signal_type": t.signal_type,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "entry_time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_price": t.exit_price,
                "exit_time": exit_dt.strftime("%Y-%m-%d %H:%M:%S") if exit_dt else None,
                "pnl_usdt": t.pnl_usdt,
                "pnl_pct": t.pnl_pct,
                "status": t.status.value,
                "exit_reason": t.exit_reason,
                "duration_hours": t.time_to_exit_hours
            })
        
        results = {
            "meta": {
                "timestamp": timestamp,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": metrics.initial_capital,
                "leverage": self.trade_simulator.config.leverage
            },
            "summary": {
                "total_trades": metrics.total_trades,
                "winrate": metrics.winrate_pct,
                "total_pnl_usdt": metrics.total_pnl_usdt,
                "total_pnl_pct": metrics.total_pnl_pct,
                "profit_factor": metrics.profit_factor,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "expectancy": metrics.expectancy
            },
            "by_signal_type": metrics.by_signal_type,
            "trades": trades_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")


def main():
    """CLI entry point for backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BTC Sniper Bot Backtest Engine")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital (USDT)")
    parser.add_argument("--leverage", type=int, default=23, help="Leverage")
    parser.add_argument("--confidence", type=float, default=65, help="Minimum confidence threshold")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (5m, 1h, 4h)")
    
    args = parser.parse_args()
    
    runner = BacktestRunner(
        initial_capital=args.capital,
        leverage=args.leverage,
        confidence_threshold=args.confidence
    )
    
    metrics = runner.run(
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe
    )
    
    return metrics


if __name__ == "__main__":
    main()
