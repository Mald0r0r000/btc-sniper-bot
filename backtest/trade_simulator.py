"""
Trade Simulator for Backtesting
Simulates trade execution with slippage, fees, and P&L calculation
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json


class TradeStatus(Enum):
    PENDING = "PENDING"
    WIN = "WIN"
    LOSS = "LOSS"
    EXPIRED = "EXPIRED"
    OPEN = "OPEN"


@dataclass
class Trade:
    """Represents a simulated trade"""
    id: str
    signal_timestamp: int
    entry_timestamp: int
    entry_price: float
    direction: str  # LONG or SHORT
    signal_type: str
    confidence: float
    
    # Targets
    tp1: float
    tp2: float
    sl: float
    
    # Trade params
    position_size: float  # in USDT
    leverage: int
    
    # Execution
    slippage: float = 0.0
    fees: float = 0.0
    actual_entry: float = 0.0
    
    # Result
    exit_timestamp: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    status: TradeStatus = TradeStatus.PENDING
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    hit_target: str = ""
    time_to_exit_hours: float = 0.0


@dataclass
class SimulatorConfig:
    """Configuration for trade simulator"""
    initial_capital: float = 10000.0  # USDT
    leverage: int = 23
    position_size_pct: float = 10.0  # % of capital per trade
    
    # Fees (taker for market orders)
    maker_fee: float = 0.02  # 0.02%
    taker_fee: float = 0.05  # 0.05%
    
    # Slippage model
    base_slippage: float = 0.03  # 0.03% base slippage
    volatility_slippage_factor: float = 0.5  # multiplier based on volatility
    
    # Trade management
    max_hold_hours: int = 24 * 7  # 7 days max
    use_trailing_stop: bool = False
    
    # Risk management
    max_daily_loss_pct: float = 5.0  # Stop trading if -5% daily
    max_drawdown_pct: float = 15.0  # Stop trading if -15% total


class TradeSimulator:
    """
    Simulates trade execution for backtesting.
    Features:
    - Realistic slippage modeling
    - Fee calculation (maker/taker)
    - P&L tracking
    - Position sizing
    """
    
    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()
        self.trades: List[Trade] = []
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.daily_pnl: Dict[str, float] = {}
        self.trade_counter = 0
        
    def reset(self):
        """Reset simulator state"""
        self.trades = []
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        self.daily_pnl = {}
        self.trade_counter = 0
    
    def calculate_slippage(self, price: float, direction: str, volatility_pct: float = 1.0) -> float:
        """
        Calculate realistic slippage based on direction and volatility.
        
        Args:
            price: Current price
            direction: LONG or SHORT
            volatility_pct: Recent volatility percentage
            
        Returns:
            Actual execution price after slippage
        """
        # Base slippage + volatility component
        slippage_pct = self.config.base_slippage + (volatility_pct * self.config.volatility_slippage_factor * 0.01)
        slippage_pct = min(slippage_pct, 0.2)  # Cap at 0.2%
        
        slippage_amount = price * (slippage_pct / 100)
        
        if direction == "LONG":
            # Buying: price goes up
            return price + slippage_amount
        else:
            # Selling: price goes down
            return price - slippage_amount
    
    def calculate_fees(self, position_size: float, is_maker: bool = False) -> float:
        """Calculate trading fees"""
        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        return position_size * (fee_rate / 100)
    
    def open_trade(
        self,
        signal_timestamp: int,
        entry_price: float,
        direction: str,
        signal_type: str,
        confidence: float,
        tp1: float,
        tp2: float,
        sl: float,
        volatility_pct: float = 1.0
    ) -> Trade:
        """
        Open a new trade based on signal.
        
        Returns:
            Trade object
        """
        self.trade_counter += 1
        trade_id = f"BT_{self.trade_counter:05d}"
        
        # Calculate position size
        position_size = self.current_capital * (self.config.position_size_pct / 100)
        
        # Calculate slippage
        actual_entry = self.calculate_slippage(entry_price, direction, volatility_pct)
        slippage = abs(actual_entry - entry_price)
        
        # Calculate entry fees
        notional = position_size * self.config.leverage
        fees = self.calculate_fees(notional)
        
        trade = Trade(
            id=trade_id,
            signal_timestamp=signal_timestamp,
            entry_timestamp=signal_timestamp,
            entry_price=entry_price,
            direction=direction,
            signal_type=signal_type,
            confidence=confidence,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            position_size=position_size,
            leverage=self.config.leverage,
            slippage=slippage,
            fees=fees,
            actual_entry=actual_entry,
            status=TradeStatus.OPEN
        )
        
        self.trades.append(trade)
        return trade
    
    def check_trade_exit(
        self,
        trade: Trade,
        candles: List[Dict],
        max_candles: int = 500
    ) -> Trade:
        """
        Check if trade hits TP, SL, or expires.
        
        Args:
            trade: Open trade to check
            candles: OHLCV candles after entry
            max_candles: Maximum candles to check
            
        Returns:
            Updated trade with exit info
        """
        if trade.status != TradeStatus.OPEN:
            return trade
        
        entry_idx = None
        for i, candle in enumerate(candles):
            if candle["timestamp"] >= trade.entry_timestamp:
                entry_idx = i
                break
        
        if entry_idx is None:
            return trade
        
        # Check subsequent candles
        for candle in candles[entry_idx:entry_idx + max_candles]:
            high = candle["high"]
            low = candle["low"]
            ts = candle["timestamp"]
            
            if trade.direction == "LONG":
                # Check SL first (worst case)
                if low <= trade.sl:
                    trade = self._close_trade(trade, trade.sl, ts, "SL", TradeStatus.LOSS)
                    break
                # Check TP1
                elif high >= trade.tp1:
                    trade = self._close_trade(trade, trade.tp1, ts, "TP1", TradeStatus.WIN)
                    break
                # Check TP2
                elif high >= trade.tp2:
                    trade = self._close_trade(trade, trade.tp2, ts, "TP2", TradeStatus.WIN)
                    break
            else:  # SHORT
                # Check SL first
                if high >= trade.sl:
                    trade = self._close_trade(trade, trade.sl, ts, "SL", TradeStatus.LOSS)
                    break
                # Check TP1
                elif low <= trade.tp1:
                    trade = self._close_trade(trade, trade.tp1, ts, "TP1", TradeStatus.WIN)
                    break
                # Check TP2
                elif low <= trade.tp2:
                    trade = self._close_trade(trade, trade.tp2, ts, "TP2", TradeStatus.WIN)
                    break
        
        # Check for expiry
        if trade.status == TradeStatus.OPEN:
            last_candle = candles[min(entry_idx + max_candles - 1, len(candles) - 1)]
            hours_held = (last_candle["timestamp"] - trade.entry_timestamp) / (1000 * 3600)
            
            if hours_held >= self.config.max_hold_hours:
                trade = self._close_trade(
                    trade, 
                    last_candle["close"], 
                    last_candle["timestamp"], 
                    "EXPIRED", 
                    TradeStatus.EXPIRED
                )
        
        return trade
    
    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        exit_timestamp: int,
        exit_reason: str,
        status: TradeStatus
    ) -> Trade:
        """Close a trade and calculate P&L"""
        # Apply exit slippage (worse for us)
        if exit_reason == "SL":
            # SL tends to slip more
            if trade.direction == "LONG":
                actual_exit = exit_price * 0.9995  # Slightly worse
            else:
                actual_exit = exit_price * 1.0005
        else:
            actual_exit = exit_price
        
        # Calculate P&L
        if trade.direction == "LONG":
            price_change_pct = ((actual_exit - trade.actual_entry) / trade.actual_entry) * 100
        else:
            price_change_pct = ((trade.actual_entry - actual_exit) / trade.actual_entry) * 100
        
        # Apply leverage
        leveraged_pnl_pct = price_change_pct * trade.leverage
        
        # Calculate USDT P&L
        pnl_usdt = trade.position_size * (leveraged_pnl_pct / 100)
        
        # Subtract fees (entry + exit)
        notional = trade.position_size * trade.leverage
        exit_fees = self.calculate_fees(notional)
        total_fees = trade.fees + exit_fees
        pnl_usdt -= total_fees
        
        # Update trade
        trade.exit_timestamp = exit_timestamp
        trade.exit_price = actual_exit
        trade.exit_reason = exit_reason
        trade.status = status
        trade.pnl_usdt = pnl_usdt
        trade.pnl_pct = (pnl_usdt / trade.position_size) * 100
        trade.hit_target = exit_reason
        trade.time_to_exit_hours = (exit_timestamp - trade.entry_timestamp) / (1000 * 3600)
        
        # Update capital
        self.current_capital += pnl_usdt
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Track daily P&L
        exit_date = datetime.fromtimestamp(exit_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        self.daily_pnl[exit_date] = self.daily_pnl.get(exit_date, 0) + pnl_usdt
        
        return trade
    
    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades"""
        return [t for t in self.trades if t.status != TradeStatus.OPEN]
    
    def get_trade_summary(self) -> Dict:
        """Get summary of all trades"""
        closed = self.get_closed_trades()
        
        if not closed:
            return {"total": 0}
        
        wins = [t for t in closed if t.status == TradeStatus.WIN]
        losses = [t for t in closed if t.status == TradeStatus.LOSS]
        expired = [t for t in closed if t.status == TradeStatus.EXPIRED]
        
        total_pnl = sum(t.pnl_usdt for t in closed)
        
        return {
            "total": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "expired": len(expired),
            "winrate": (len(wins) / len(closed) * 100) if closed else 0,
            "total_pnl_usdt": total_pnl,
            "total_pnl_pct": (total_pnl / self.config.initial_capital) * 100,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "max_drawdown_pct": ((self.peak_capital - self.current_capital) / self.peak_capital) * 100 if self.peak_capital > 0 else 0,
            "avg_win_pnl": sum(t.pnl_usdt for t in wins) / len(wins) if wins else 0,
            "avg_loss_pnl": sum(t.pnl_usdt for t in losses) / len(losses) if losses else 0,
            "avg_trade_duration_hours": sum(t.time_to_exit_hours for t in closed) / len(closed),
        }


# Test function
def test_trade_simulator():
    print("=" * 60)
    print("Testing Trade Simulator")
    print("=" * 60)
    
    sim = TradeSimulator()
    
    # Simulate a winning LONG trade
    trade = sim.open_trade(
        signal_timestamp=1705000000000,
        entry_price=95000.0,
        direction="LONG",
        signal_type="FADE_LOW",
        confidence=72.0,
        tp1=96500.0,
        tp2=98000.0,
        sl=94000.0,
        volatility_pct=1.5
    )
    
    print(f"\n📊 Opened trade: {trade.id}")
    print(f"   Entry: ${trade.entry_price:,.2f} → Actual: ${trade.actual_entry:,.2f}")
    print(f"   Slippage: ${trade.slippage:,.2f}")
    print(f"   Fees: ${trade.fees:,.2f}")
    
    # Simulate candles that hit TP1
    mock_candles = [
        {"timestamp": 1705000000000, "open": 95000, "high": 95200, "low": 94800, "close": 95100},
        {"timestamp": 1705003600000, "open": 95100, "high": 95500, "low": 95000, "close": 95400},
        {"timestamp": 1705007200000, "open": 95400, "high": 96600, "low": 95300, "close": 96500},  # Hits TP1
    ]
    
    trade = sim.check_trade_exit(trade, mock_candles)
    
    print(f"\n📈 Trade result:")
    print(f"   Status: {trade.status.value}")
    print(f"   Exit: ${trade.exit_price:,.2f} ({trade.exit_reason})")
    print(f"   P&L: ${trade.pnl_usdt:,.2f} ({trade.pnl_pct:+.2f}%)")
    print(f"   Duration: {trade.time_to_exit_hours:.1f}h")
    
    summary = sim.get_trade_summary()
    print(f"\n💰 Portfolio:")
    print(f"   Capital: ${summary['current_capital']:,.2f}")


if __name__ == "__main__":
    test_trade_simulator()
