"""
Metrics Calculator for Backtesting
Calculates professional trading metrics: Sharpe, Drawdown, Profit Factor, etc.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from backtest.trade_simulator import Trade, TradeStatus


@dataclass
class BacktestMetrics:
    """Container for all backtest metrics"""
    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    expired_trades: int = 0
    winrate_pct: float = 0.0
    
    # P&L
    total_pnl_usdt: float = 0.0
    total_pnl_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    
    # Averages
    avg_win_usdt: float = 0.0
    avg_loss_usdt: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_trade_duration_hours: float = 0.0
    
    # Risk metrics
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usdt: float = 0.0
    
    # Win/Loss analysis
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_win_loss_ratio: float = 0.0
    
    # Time analysis
    avg_win_duration_hours: float = 0.0
    avg_loss_duration_hours: float = 0.0
    
    # Capital
    initial_capital: float = 0.0
    final_capital: float = 0.0
    peak_capital: float = 0.0
    return_pct: float = 0.0
    
    # By signal type
    by_signal_type: Dict = None
    
    def __post_init__(self):
        if self.by_signal_type is None:
            self.by_signal_type = {}


class MetricsCalculator:
    """
    Calculates comprehensive trading metrics from backtest trades.
    """
    
    def __init__(self, initial_capital: float = 10000.0, risk_free_rate: float = 0.0):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate (for Sharpe)
    
    def calculate(self, trades: List[Trade]) -> BacktestMetrics:
        """
        Calculate all metrics from a list of trades.
        
        Args:
            trades: List of completed Trade objects
            
        Returns:
            BacktestMetrics object with all calculated metrics
        """
        if not trades:
            return BacktestMetrics(initial_capital=self.initial_capital)
        
        # Filter closed trades
        closed_trades = [t for t in trades if t.status != TradeStatus.OPEN and t.status != TradeStatus.PENDING]
        
        if not closed_trades:
            return BacktestMetrics(initial_capital=self.initial_capital)
        
        # Categorize trades
        wins = [t for t in closed_trades if t.status == TradeStatus.WIN]
        losses = [t for t in closed_trades if t.status == TradeStatus.LOSS]
        expired = [t for t in closed_trades if t.status == TradeStatus.EXPIRED]
        
        # Basic stats
        total = len(closed_trades)
        n_wins = len(wins)
        n_losses = len(losses)
        n_expired = len(expired)
        
        # P&L calculations
        pnl_list = [t.pnl_usdt for t in closed_trades]
        total_pnl = sum(pnl_list)
        gross_profit = sum(t.pnl_usdt for t in closed_trades if t.pnl_usdt > 0)
        gross_loss = abs(sum(t.pnl_usdt for t in closed_trades if t.pnl_usdt < 0))
        
        # Averages
        avg_win = np.mean([t.pnl_usdt for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_usdt for t in losses]) if losses else 0
        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
        
        # Duration
        avg_duration = np.mean([t.time_to_exit_hours for t in closed_trades])
        avg_win_duration = np.mean([t.time_to_exit_hours for t in wins]) if wins else 0
        avg_loss_duration = np.mean([t.time_to_exit_hours for t in losses]) if losses else 0
        
        # Profit Factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy (expected value per trade)
        winrate = n_wins / total if total > 0 else 0
        expectancy = (winrate * avg_win) - ((1 - winrate) * abs(avg_loss)) if total > 0 else 0
        
        # Calculate equity curve for drawdown and Sharpe
        equity_curve = self._calculate_equity_curve(closed_trades)
        max_dd_pct, max_dd_usdt = self._calculate_max_drawdown(equity_curve)
        
        # Sharpe Ratio (annualized)
        sharpe = self._calculate_sharpe_ratio(pnl_list)
        
        # Sortino Ratio (only downside volatility)
        sortino = self._calculate_sortino_ratio(pnl_list)
        
        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = self._calculate_consecutive_streaks(closed_trades)
        
        # Win/Loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Capital tracking
        final_capital = self.initial_capital + total_pnl
        peak_capital = max(equity_curve) if equity_curve else self.initial_capital
        return_pct = (total_pnl / self.initial_capital) * 100
        
        # By signal type breakdown
        by_type = self._calculate_by_signal_type(closed_trades)
        
        return BacktestMetrics(
            total_trades=total,
            winning_trades=n_wins,
            losing_trades=n_losses,
            expired_trades=n_expired,
            winrate_pct=winrate * 100,
            total_pnl_usdt=total_pnl,
            total_pnl_pct=(total_pnl / self.initial_capital) * 100,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            avg_win_usdt=avg_win,
            avg_loss_usdt=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            avg_trade_duration_hours=avg_duration,
            profit_factor=profit_factor,
            expectancy=expectancy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_usdt=max_dd_usdt,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            avg_win_loss_ratio=win_loss_ratio,
            avg_win_duration_hours=avg_win_duration,
            avg_loss_duration_hours=avg_loss_duration,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            peak_capital=peak_capital,
            return_pct=return_pct,
            by_signal_type=by_type
        )
    
    def _calculate_equity_curve(self, trades: List[Trade]) -> List[float]:
        """Calculate equity curve from trades"""
        equity = [self.initial_capital]
        current = self.initial_capital
        
        # Sort by exit timestamp
        sorted_trades = sorted(trades, key=lambda t: t.exit_timestamp or 0)
        
        for trade in sorted_trades:
            current += trade.pnl_usdt
            equity.append(current)
        
        return equity
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> tuple:
        """Calculate maximum drawdown from equity curve"""
        if not equity_curve:
            return 0.0, 0.0
        
        peak = equity_curve[0]
        max_dd_pct = 0.0
        max_dd_usdt = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            drawdown_usdt = peak - value
            drawdown_pct = (drawdown_usdt / peak) * 100 if peak > 0 else 0
            
            if drawdown_pct > max_dd_pct:
                max_dd_pct = drawdown_pct
                max_dd_usdt = drawdown_usdt
        
        return max_dd_pct, max_dd_usdt
    
    def _calculate_sharpe_ratio(self, pnl_list: List[float], periods_per_year: int = 365) -> float:
        """
        Calculate annualized Sharpe Ratio.
        Assumes daily returns, annualized.
        """
        if len(pnl_list) < 2:
            return 0.0
        
        returns = np.array(pnl_list)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        sharpe = (avg_return / std_return) * np.sqrt(periods_per_year)
        
        return sharpe
    
    def _calculate_sortino_ratio(self, pnl_list: List[float], periods_per_year: int = 365) -> float:
        """
        Calculate Sortino Ratio (uses only downside volatility).
        """
        if len(pnl_list) < 2:
            return 0.0
        
        returns = np.array(pnl_list)
        avg_return = np.mean(returns)
        
        # Only negative returns for downside deviation
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # No losses
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (avg_return / downside_std) * np.sqrt(periods_per_year)
        
        return sortino
    
    def _calculate_consecutive_streaks(self, trades: List[Trade]) -> tuple:
        """Calculate max consecutive wins and losses"""
        if not trades:
            return 0, 0
        
        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_timestamp or 0)
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in sorted_trades:
            if trade.status == TradeStatus.WIN:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.status == TradeStatus.LOSS:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                # Expired - reset both
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses
    
    def _calculate_by_signal_type(self, trades: List[Trade]) -> Dict:
        """Calculate metrics breakdown by signal type"""
        by_type = {}
        
        # Group trades by signal type
        types = set(t.signal_type for t in trades)
        
        for signal_type in types:
            type_trades = [t for t in trades if t.signal_type == signal_type]
            wins = [t for t in type_trades if t.status == TradeStatus.WIN]
            losses = [t for t in type_trades if t.status == TradeStatus.LOSS]
            
            by_type[signal_type] = {
                "total": len(type_trades),
                "wins": len(wins),
                "losses": len(losses),
                "winrate": (len(wins) / len(type_trades) * 100) if type_trades else 0,
                "total_pnl": sum(t.pnl_usdt for t in type_trades),
                "avg_pnl": np.mean([t.pnl_usdt for t in type_trades]) if type_trades else 0,
                "expectancy": self._calculate_type_expectancy(type_trades)
            }
        
        return by_type
    
    def _calculate_type_expectancy(self, trades: List[Trade]) -> float:
        """Calculate expectancy for a specific signal type"""
        if not trades:
            return 0.0
        
        wins = [t for t in trades if t.status == TradeStatus.WIN]
        losses = [t for t in trades if t.status == TradeStatus.LOSS]
        
        if not wins and not losses:
            return 0.0
        
        winrate = len(wins) / len(trades)
        avg_win = np.mean([t.pnl_usdt for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl_usdt for t in losses])) if losses else 0
        
        return (winrate * avg_win) - ((1 - winrate) * avg_loss)
    
    def format_report(self, metrics: BacktestMetrics) -> str:
        """Format metrics as a readable report"""
        lines = [
            "═" * 65,
            "                    BACKTEST RESULTS",
            "═" * 65,
            "",
            f"Total Trades: {metrics.total_trades}",
            f"├── Wins:    {metrics.winning_trades} ({metrics.winrate_pct:.1f}%)",
            f"├── Losses:  {metrics.losing_trades}",
            f"└── Expired: {metrics.expired_trades}",
            "",
            "PERFORMANCE",
            f"├── Total P&L:      ${metrics.total_pnl_usdt:+,.2f} ({metrics.total_pnl_pct:+.1f}%)",
            f"├── Gross Profit:   ${metrics.gross_profit:,.2f}",
            f"├── Gross Loss:     ${metrics.gross_loss:,.2f}",
            f"├── Profit Factor:  {metrics.profit_factor:.2f}",
            f"└── Expectancy:     ${metrics.expectancy:+,.2f} per trade",
            "",
            "RISK METRICS",
            f"├── Sharpe Ratio:   {metrics.sharpe_ratio:.2f}",
            f"├── Sortino Ratio:  {metrics.sortino_ratio:.2f}",
            f"├── Max Drawdown:   {metrics.max_drawdown_pct:.1f}% (${metrics.max_drawdown_usdt:,.2f})",
            f"├── Max Consec Wins:   {metrics.max_consecutive_wins}",
            f"└── Max Consec Losses: {metrics.max_consecutive_losses}",
            "",
            "AVERAGES",
            f"├── Avg Win:      ${metrics.avg_win_usdt:+,.2f} ({metrics.avg_win_pct:+.1f}%)",
            f"├── Avg Loss:     ${metrics.avg_loss_usdt:,.2f} ({metrics.avg_loss_pct:.1f}%)",
            f"├── Win/Loss:     {metrics.avg_win_loss_ratio:.2f}x",
            f"└── Avg Duration: {metrics.avg_trade_duration_hours:.1f}h",
            "",
            "CAPITAL",
            f"├── Initial: ${metrics.initial_capital:,.2f}",
            f"├── Final:   ${metrics.final_capital:,.2f}",
            f"├── Peak:    ${metrics.peak_capital:,.2f}",
            f"└── Return:  {metrics.return_pct:+.1f}%",
            "",
        ]
        
        # Add by signal type
        if metrics.by_signal_type:
            lines.append("BY SIGNAL TYPE")
            sorted_types = sorted(
                metrics.by_signal_type.items(),
                key=lambda x: x[1]["winrate"],
                reverse=True
            )
            for signal_type, stats in sorted_types:
                wr = stats["winrate"]
                emoji = "🟢" if wr >= 60 else "🟡" if wr >= 45 else "🔴"
                lines.append(f"├── {emoji} {signal_type}: {wr:.1f}% WR ({stats['wins']}W/{stats['losses']}L) ${stats['total_pnl']:+,.0f}")
            lines.append("")
        
        lines.append("═" * 65)
        
        return "\n".join(lines)


# Test function
def test_metrics_calculator():
    print("=" * 60)
    print("Testing Metrics Calculator")
    print("=" * 60)
    
    # Create mock trades
    from backtest.trade_simulator import Trade, TradeStatus
    
    trades = [
        Trade(id="1", signal_timestamp=1, entry_timestamp=1, entry_price=95000, direction="LONG",
              signal_type="FADE_LOW", confidence=70, tp1=96000, tp2=97000, sl=94000,
              position_size=1000, leverage=23, exit_timestamp=2, exit_price=96000,
              status=TradeStatus.WIN, pnl_usdt=230, pnl_pct=23, time_to_exit_hours=2),
        Trade(id="2", signal_timestamp=3, entry_timestamp=3, entry_price=96000, direction="SHORT",
              signal_type="FADE_HIGH", confidence=65, tp1=95000, tp2=94000, sl=97000,
              position_size=1000, leverage=23, exit_timestamp=4, exit_price=97000,
              status=TradeStatus.LOSS, pnl_usdt=-230, pnl_pct=-23, time_to_exit_hours=1),
        Trade(id="3", signal_timestamp=5, entry_timestamp=5, entry_price=97000, direction="LONG",
              signal_type="FADE_LOW", confidence=72, tp1=98000, tp2=99000, sl=96000,
              position_size=1000, leverage=23, exit_timestamp=6, exit_price=98000,
              status=TradeStatus.WIN, pnl_usdt=237, pnl_pct=23.7, time_to_exit_hours=3),
    ]
    
    calc = MetricsCalculator(initial_capital=10000)
    metrics = calc.calculate(trades)
    
    report = calc.format_report(metrics)
    print(report)


if __name__ == "__main__":
    test_metrics_calculator()
