"""
Oscillator Analyzer
Calculates momentum oscillators (KDJ, etc.) to identify overbought/oversold conditions and reversals.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class OscillatorAnalyzer:
    """
    KDJ (Stochastic Oscillator) Analyzer
    
    KDJ is a derived form of the Stochastic Oscillator with an extra J line.
    - K & D are the same as in standard Stochastic.
    - J = 3K - 2D (more reactive, can go >100 or <0)
    
    Signals:
    - Overbought: J > 100 or K > 80
    - Oversold: J < 0 or K < 20
    - Crossover: J crossing K (Golden/Dead Cross)
    """
    
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series):
        self.high = high
        self.low = low
        self.close = close
        
    def calculate_kdj(self, period: int = 9, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
        """
        Calculates K, D, J values
        """
        # Calculate RSV (Raw Stochastic Value)
        lowest_low = self.low.rolling(window=period).min()
        highest_high = self.high.rolling(window=period).max()
        
        # Handle division by zero
        rsv = 100 * ((self.close - lowest_low) / (highest_high - lowest_low))
        rsv = rsv.fillna(50)  # Default neutral
        
        # Calculate K, D using EMA-like smoothing logic
        # Pandas ewm is close approximation to the iterative 2/3 + 1/3 formula
        # K = 2/3 * PrevK + 1/3 * RSV  => alpha=1/3
        
        k = rsv.ewm(alpha=1/k_period, adjust=False).mean()
        d = k.ewm(alpha=1/d_period, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return pd.DataFrame({'k': k, 'd': d, 'j': j})
        
    def analyze(self) -> Dict[str, Any]:
        """
        Analyzes current KDJ state and recent signals
        """
        if len(self.close) < 15:
            return {'score': 50, 'signal': 'NEUTRAL', 'kdj': {'k': 50, 'd': 50, 'j': 50}}
            
        kdj = self.calculate_kdj()
        
        # Current values
        k = kdj['k'].iloc[-1]
        d = kdj['d'].iloc[-1]
        j = kdj['j'].iloc[-1]
        
        # Previous values (for crossover detection)
        prev_j = kdj['j'].iloc[-2]
        prev_k = kdj['k'].iloc[-2]
        prev_d = kdj['d'].iloc[-2]
        
        # Determine State
        state = 'NEUTRAL'
        if j > 100 or k > 80:
            state = 'OVERBOUGHT'
        elif j < 0 or k < 20:
            state = 'OVERSOLD'
            
        # Detect Crossovers
        # Golden Cross: J crosses above K/D (and preferably in oversold)
        is_golden_cross = prev_j < prev_k and j > k
        # Dead Cross: J crosses below K/D (and preferably in overbought)
        is_dead_cross = prev_j > prev_k and j < k
        
        signal = 'NEUTRAL'
        score = 50.0  # Base score
        
        if is_golden_cross:
            signal = 'GOLDEN_CROSS'
            if state == 'OVERSOLD' or k < 30:
                score = 80  # Strong Buy signal
            else:
                score = 65  # Weak Buy signal
        elif is_dead_cross:
            signal = 'DEAD_CROSS'
            if state == 'OVERBOUGHT' or k > 70:
                score = 20  # Strong Sell signal
            else:
                score = 35  # Weak Sell signal
        else:
            # No cross, evaluate based on levels and slope
            if state == 'OVERBOUGHT':
                score = 30  # Bearish bias due to extension
                if j < prev_j: # Turning down
                    score = 25
            elif state == 'OVERSOLD':
                score = 70  # Bullish bias due to extension
                if j > prev_j: # Turning up
                    score = 75
            else:
                # Middleware - follow trend of J
                if j > prev_j:
                    score = 55
                else:
                    score = 45
        
        return {
            'score': round(score, 1),
            'signal': signal,
            'state': state,
            'values': {
                'k': round(k, 1),
                'd': round(d, 1),
                'j': round(j, 1)
            }
        }
