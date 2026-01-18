"""
HTF (Higher Timeframe) Bias Analyzer
Determines the dominant trend direction from 4H/Daily timeframe
Used to filter signals that go against the major trend
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

class HTFBiasAnalyzer:
    """
    Analyzes Higher Timeframe (4H) to determine trend bias.
    
    Logic:
    - Price vs EMA50: Above = Bullish structure, Below = Bearish
    - KDJ Direction: J-Slope positive = Bullish momentum
    - Combine both for final bias
    
    Used to filter counter-trend trades or reduce their confidence.
    """
    
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series):
        self.high = high
        self.low = low
        self.close = close
        
    def calculate_ema(self, period: int) -> pd.Series:
        """Calculate EMA"""
        return self.close.ewm(span=period, adjust=False).mean()
        
    def calculate_kdj(self, period: int = 9) -> Dict[str, float]:
        """Calculate KDJ for HTF"""
        if len(self.close) < period + 5:
            return {'k': 50, 'd': 50, 'j': 50, 'j_slope': 0}
            
        # RSV
        lowest_low = self.low.rolling(window=period).min()
        highest_high = self.high.rolling(window=period).max()
        
        rsv = 100 * ((self.close - lowest_low) / (highest_high - lowest_low))
        rsv = rsv.fillna(50)
        
        # K, D, J
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return {
            'k': k.iloc[-1],
            'd': d.iloc[-1],
            'j': j.iloc[-1],
            'j_slope': j.iloc[-1] - j.iloc[-2] if len(j) > 1 else 0
        }
        
    def analyze(self) -> Dict[str, Any]:
        """
        Analyzes HTF bias based on EMA structure and KDJ momentum
        """
        if len(self.close) < 55:
            return {
                'bias': 'NEUTRAL',
                'structure': 'NEUTRAL',
                'momentum': 'NEUTRAL',
                'ema50': 0,
                'price_vs_ema': 0,
                'kdj': {'k': 50, 'd': 50, 'j': 50}
            }
            
        # EMA Analysis
        ema20 = self.calculate_ema(20)
        ema50 = self.calculate_ema(50)
        
        current_price = self.close.iloc[-1]
        current_ema50 = ema50.iloc[-1]
        current_ema20 = ema20.iloc[-1]
        
        price_vs_ema = ((current_price - current_ema50) / current_ema50) * 100
        
        # Structure analysis
        if current_price > current_ema50 and current_ema20 > current_ema50:
            structure = 'BULLISH'
        elif current_price < current_ema50 and current_ema20 < current_ema50:
            structure = 'BEARISH'
        else:
            structure = 'NEUTRAL'
            
        # KDJ Momentum
        kdj = self.calculate_kdj()
        
        if kdj['j_slope'] > 2 and kdj['j'] > 50:
            momentum = 'BULLISH'
        elif kdj['j_slope'] < -2 and kdj['j'] < 50:
            momentum = 'BEARISH'
        else:
            momentum = 'NEUTRAL'
            
        # Final Bias (require both structure and momentum to agree)
        if structure == 'BULLISH' and momentum in ['BULLISH', 'NEUTRAL']:
            bias = 'BULLISH'
        elif structure == 'BEARISH' and momentum in ['BEARISH', 'NEUTRAL']:
            bias = 'BEARISH'
        elif structure == momentum:
            bias = structure
        else:
            bias = 'NEUTRAL'
            
        return {
            'bias': bias,
            'structure': structure,
            'momentum': momentum,
            'ema20': round(current_ema20, 2),
            'ema50': round(current_ema50, 2),
            'price_vs_ema': round(price_vs_ema, 2),
            'kdj': {
                'k': round(kdj['k'], 1),
                'd': round(kdj['d'], 1),
                'j': round(kdj['j'], 1),
                'j_slope': round(kdj['j_slope'], 1)
            }
        }
