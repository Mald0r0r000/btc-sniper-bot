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
        
        # Calculate Slope (Velocity of J)
        j_slope = j - kdj['j'].iloc[-2]
        
        # Calculate Deviation (Extension from Mean/K)
        # Large deviation means price moved too fast relative to trend -> Parabolic
        deviation = j - k
        
        # Determine State
        state = 'NEUTRAL'
        if j > 90: # Higher threshold for 1H/4H
            state = 'OVERBOUGHT'
        elif j < 10: # Lower threshold for 1H/4H
            state = 'OVERSOLD'
            
        signal = 'NEUTRAL'
        score = 50.0
        
        # === PARABOLIC REVERSAL LOGIC (User Request) ===
        # Detect turning points when J is extended and slope reverses
        
        # BEARISH: J is high, extended above K, and slope turns negative (or drops significantly)
        if state == 'OVERBOUGHT' and deviation > 15: # Highly extended
            if j_slope < 0: # Turning down
                signal = 'PARABOLIC_BEAR'
                score = 25 # Strong Sell
            elif j_slope < 2: # Losing momentum
                score = 40 # Weak Sell / Warning
                
        # BULLISH: J is low, extended below K, and slope turns positive
        elif state == 'OVERSOLD' and deviation < -15: # Highly extended down
            if j_slope > 0: # Turning up
                signal = 'PARABOLIC_BULL'
                score = 75 # Strong Buy
            elif j_slope > -2: # Losing downward momentum
                score = 60 # Weak Buy / Watch
        
        # Trend Confirmation (High Slope)
        elif abs(j_slope) > 5:
            if j_slope > 0:
                score = 55 # Bullish Momentum
            else:
                score = 45 # Bearish Momentum
                
        return {
            'score': round(score, 1),
            'signal': signal,
            'state': state,
            'values': {
                'k': round(k, 1),
                'd': round(d, 1),
                'j': round(j, 1),
                'j_slope': round(j_slope, 1),
                'deviation': round(deviation, 1)
            }
        }
