"""
Signal Replayer for Backtesting
Replays the decision engine on historical data to generate signals
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np


class SignalReplayer:
    """
    Replays signal generation on historical data.
    Uses a simplified version of the decision logic for backtesting.
    """
    
    def __init__(self, confidence_threshold: float = 65.0):
        self.confidence_threshold = confidence_threshold
        
        # Signal type filters (based on production performance)
        self.allowed_signal_types = [
            'FADE_HIGH', 'FADE_LOW',
            'LONG_SNIPER', 'SHORT_SNIPER',
            'LONG_BREAKOUT', 'SHORT_BREAKOUT'
        ]
    
    def generate_signals_from_candles(
        self,
        ohlcv_1h: List[Dict],
        ohlcv_5m: Optional[List[Dict]] = None,
        ohlcv_4h: Optional[List[Dict]] = None,
        lookback: int = 50
    ) -> List[Dict]:
        """
        Generate signals from historical candle data.
        
        This is a simplified signal generator for backtesting purposes.
        It uses technical patterns similar to the live decision engine.
        
        Args:
            ohlcv_1h: 1-hour candle data
            ohlcv_5m: Optional 5-minute data for fractals
            ohlcv_4h: Optional 4-hour data for trend
            lookback: Number of candles to use for indicators
            
        Returns:
            List of signal dicts with entry info
        """
        signals = []
        
        if len(ohlcv_1h) < lookback + 10:
            print(f"⚠️ Not enough data: {len(ohlcv_1h)} candles, need {lookback + 10}")
            return signals
        
        print(f"🔄 Generating signals from {len(ohlcv_1h)} candles...")
        
        # Process each candle (starting after lookback period)
        for i in range(lookback, len(ohlcv_1h) - 1):
            candle = ohlcv_1h[i]
            history = ohlcv_1h[max(0, i - lookback):i + 1]
            
            # Generate signal for this point in time
            signal = self._analyze_candle(candle, history, i)
            
            if signal:
                signals.append(signal)
        
        print(f"✅ Generated {len(signals)} signals")
        return signals
    
    def _analyze_candle(self, candle: Dict, history: List[Dict], idx: int) -> Optional[Dict]:
        """Analyze a single candle and generate signal if conditions met"""
        
        # Calculate indicators
        closes = np.array([c['close'] for c in history])
        highs = np.array([c['high'] for c in history])
        lows = np.array([c['low'] for c in history])
        volumes = np.array([c['volume'] for c in history])
        
        current_price = candle['close']
        current_time = candle['timestamp']
        
        # Skip if not enough data
        if len(closes) < 20:
            return None
        
        # Calculate technical indicators
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
        rsi = self._calculate_rsi(closes, 14)
        atr = self._calculate_atr(highs, lows, closes, 14)
        
        # Detect fractals
        fractal_high = max(highs[-10:])
        fractal_low = min(lows[-10:])
        
        # Volume analysis
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Trend detection
        trend = "BULLISH" if current_price > sma_20 > sma_50 else \
                "BEARISH" if current_price < sma_20 < sma_50 else "NEUTRAL"
        
        # Signal generation logic
        signal = None
        
        # FADE_HIGH: Price near resistance + overbought
        if rsi > 70 and current_price >= fractal_high * 0.995:
            signal = self._create_signal(
                signal_type="FADE_HIGH",
                direction="SHORT",
                price=current_price,
                timestamp=current_time,
                confidence=min(85, 60 + (rsi - 70) + volume_ratio * 5),
                atr=atr,
                fractal_high=fractal_high,
                fractal_low=fractal_low
            )
        
        # FADE_LOW: Price near support + oversold
        elif rsi < 30 and current_price <= fractal_low * 1.005:
            signal = self._create_signal(
                signal_type="FADE_LOW",
                direction="LONG",
                price=current_price,
                timestamp=current_time,
                confidence=min(85, 60 + (30 - rsi) + volume_ratio * 5),
                atr=atr,
                fractal_high=fractal_high,
                fractal_low=fractal_low
            )
        
        # LONG_SNIPER: Strong bullish momentum
        elif trend == "BULLISH" and rsi > 50 and rsi < 65 and volume_ratio > 1.5:
            signal = self._create_signal(
                signal_type="LONG_SNIPER",
                direction="LONG",
                price=current_price,
                timestamp=current_time,
                confidence=min(80, 55 + (rsi - 50) / 2 + volume_ratio * 3),
                atr=atr,
                fractal_high=fractal_high,
                fractal_low=fractal_low
            )
        
        # SHORT_SNIPER: Strong bearish momentum
        elif trend == "BEARISH" and rsi < 50 and rsi > 35 and volume_ratio > 1.5:
            signal = self._create_signal(
                signal_type="SHORT_SNIPER",
                direction="SHORT",
                price=current_price,
                timestamp=current_time,
                confidence=min(80, 55 + (50 - rsi) / 2 + volume_ratio * 3),
                atr=atr,
                fractal_high=fractal_high,
                fractal_low=fractal_low
            )
        
        # BREAKOUT: Price breaks through fractal
        elif current_price > fractal_high and volume_ratio > 2:
            signal = self._create_signal(
                signal_type="LONG_BREAKOUT",
                direction="LONG",
                price=current_price,
                timestamp=current_time,
                confidence=min(75, 50 + volume_ratio * 5),
                atr=atr,
                fractal_high=fractal_high,
                fractal_low=fractal_low
            )
        
        elif current_price < fractal_low and volume_ratio > 2:
            signal = self._create_signal(
                signal_type="SHORT_BREAKOUT",
                direction="SHORT",
                price=current_price,
                timestamp=current_time,
                confidence=min(75, 50 + volume_ratio * 5),
                atr=atr,
                fractal_high=fractal_high,
                fractal_low=fractal_low
            )
        
        # Filter by confidence threshold
        if signal and signal['confidence'] >= self.confidence_threshold:
            return signal
        
        return None
    
    def _create_signal(
        self,
        signal_type: str,
        direction: str,
        price: float,
        timestamp: int,
        confidence: float,
        atr: float,
        fractal_high: float,
        fractal_low: float
    ) -> Dict:
        """Create a signal dict with targets"""
        
        # Calculate targets based on direction and fractals
        if direction == "LONG":
            tp1 = price + atr * 1.5  # 1.5 ATR for TP1
            tp2 = min(fractal_high, price + atr * 3)  # 3 ATR or fractal
            sl = max(fractal_low, price - atr * 1)  # 1 ATR or fractal
        else:  # SHORT
            tp1 = price - atr * 1.5
            tp2 = max(fractal_low, price - atr * 3)
            sl = min(fractal_high, price + atr * 1)
        
        return {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat(),
            "price": price,
            "direction": direction,
            "signal_type": signal_type,
            "confidence": confidence,
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "atr": atr
        }
    
    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        period: int = 14
    ) -> float:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return (highs[-1] - lows[-1])
        
        tr_list = []
        for i in range(-period, 0):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr_list.append(max(high_low, high_close, low_close))
        
        return np.mean(tr_list)


# Test function
def test_signal_replayer():
    print("=" * 60)
    print("Testing Signal Replayer")
    print("=" * 60)
    
    # Generate mock candles
    import random
    random.seed(42)
    
    base_price = 95000
    candles = []
    current_time = 1700000000000
    
    for i in range(200):
        # Random walk with trend
        change = random.gauss(0, 200)
        base_price += change
        
        open_price = base_price
        high = base_price + random.uniform(50, 300)
        low = base_price - random.uniform(50, 300)
        close = base_price + random.uniform(-100, 100)
        volume = random.uniform(1000, 5000)
        
        candles.append({
            "timestamp": current_time,
            "datetime": datetime.fromtimestamp(current_time / 1000, tz=timezone.utc).isoformat(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })
        
        current_time += 3600000  # 1 hour
    
    replayer = SignalReplayer(confidence_threshold=60.0)
    signals = replayer.generate_signals_from_candles(candles)
    
    print(f"\n📊 Generated {len(signals)} signals")
    
    if signals:
        # Show first few signals
        for signal in signals[:5]:
            print(f"\n{signal['signal_type']} {signal['direction']}:")
            print(f"   Price: ${signal['price']:,.2f}")
            print(f"   Confidence: {signal['confidence']:.0f}%")
            print(f"   TP1: ${signal['tp1']:,.2f} | SL: ${signal['sl']:,.2f}")


if __name__ == "__main__":
    test_signal_replayer()
