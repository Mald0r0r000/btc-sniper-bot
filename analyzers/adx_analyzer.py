"""
ADX (Average Directional Index) Analyzer
Detects market regime: RANGING, TRANSITION, or TRENDING
Used to filter out signals during low-volatility/dead markets
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

class ADXAnalyzer:
    """
    Calculates ADX to determine market regime.
    
    ADX measures trend STRENGTH, not direction:
    - ADX < 20: RANGING (no trend, avoid trading)
    - ADX 20-25: TRANSITION (trend forming, be cautious)
    - ADX > 25: TRENDING (strong trend, signals are reliable)
    
    +DI vs -DI determines trend DIRECTION:
    - +DI > -DI: Bullish trend
    - -DI > +DI: Bearish trend
    """
    
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        self.high = high
        self.low = low
        self.close = close
        self.period = period
        
    def calculate_adx(self) -> pd.DataFrame:
        """
        Calculates ADX, +DI, and -DI
        """
        high = self.high
        low = self.low
        close = self.close
        period = self.period
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'atr': atr
        })
        
    def analyze(self) -> Dict[str, Any]:
        """
        Analyzes current ADX state and determines market regime
        """
        if len(self.close) < self.period + 5:
            return {
                'adx': 0,
                'plus_di': 0,
                'minus_di': 0,
                'regime': 'UNKNOWN',
                'trend_direction': 'NEUTRAL',
                'atr': 0
            }
            
        df = self.calculate_adx()
        
        adx = df['adx'].iloc[-1]
        plus_di = df['plus_di'].iloc[-1]
        minus_di = df['minus_di'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Determine regime
        if adx < 20:
            regime = 'RANGING'
        elif adx < 25:
            regime = 'TRANSITION'
        else:
            regime = 'TRENDING'
            
        # Determine trend direction
        if plus_di > minus_di:
            trend_direction = 'BULLISH'
        elif minus_di > plus_di:
            trend_direction = 'BEARISH'
        else:
            trend_direction = 'NEUTRAL'
            
        # ADX Slope (is trend strengthening?)
        adx_prev = df['adx'].iloc[-2]
        adx_slope = adx - adx_prev
        
        return {
            'adx': round(adx, 1),
            'plus_di': round(plus_di, 1),
            'minus_di': round(minus_di, 1),
            'regime': regime,
            'trend_direction': trend_direction,
            'adx_slope': round(adx_slope, 2),
            'atr': round(atr, 2)
        }
