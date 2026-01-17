"""
Momentum Analyzer
Calculates momentum score based on CVD, OI delta, and Volume to determine optimal TP timeframe
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class MomentumStrength(Enum):
    STRONG = "STRONG"      # Score >= 70
    MODERATE = "MODERATE"  # Score 50-70
    WEAK = "WEAK"          # Score < 50


@dataclass
class MomentumResult:
    """Result of momentum analysis"""
    score: float  # 0-100
    strength: MomentumStrength
    direction: str  # BULLISH, BEARISH, NEUTRAL
    
    # Components
    cvd_score: float
    oi_score: float
    volume_score: float
    trend_score: float
    
    # Recommended timeframe for targets
    recommended_tf: str  # 5m, 15m, 1h, 4h
    
    # Recommended leverage multiplier
    leverage_multiplier: float  # 0.5-2.0
    
    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "strength": self.strength.value,
            "direction": self.direction,
            "components": {
                "cvd": self.cvd_score,
                "oi": self.oi_score,
                "volume": self.volume_score,
                "trend": self.trend_score
            },
            "recommended_tf": self.recommended_tf,
            "leverage_multiplier": self.leverage_multiplier
        }


class MomentumAnalyzer:
    """
    Analyzes market momentum to determine optimal TP timeframe.
    
    Momentum Score Components:
    - CVD Ratio: Measures buyer vs seller aggression
    - OI Delta: New positions = conviction
    - Volume Ratio: Current vs average volume
    - Trend Alignment: Price vs moving averages
    """
    
    def __init__(self):
        # Weights for each component
        self.weights = {
            "cvd": 0.30,
            "oi": 0.25,
            "volume": 0.25,
            "trend": 0.20
        }
        
        # Timeframe thresholds
        self.tf_thresholds = {
            "STRONG": "4h",    # Strong momentum → use 4H fractals
            "MODERATE": "1h",  # Moderate → use 1H fractals
            "WEAK": "15m"      # Weak → use 15m fractals (closer targets)
        }
    
    def analyze(
        self,
        cvd_data: Optional[Dict] = None,
        oi_data: Optional[Dict] = None,
        candles: Optional[List[Dict]] = None,
        direction_hint: str = "NEUTRAL"
    ) -> MomentumResult:
        """
        Calculate comprehensive momentum score.
        
        Args:
            cvd_data: CVD analysis result with ratio, delta
            oi_data: Open Interest data with delta_1h
            candles: Recent OHLCV candles for volume/trend
            direction_hint: Expected direction (LONG/SHORT) for alignment
            
        Returns:
            MomentumResult with score, strength, and recommendations
        """
        # Calculate component scores (0-100 each)
        cvd_score = self._analyze_cvd(cvd_data, direction_hint)
        oi_score = self._analyze_oi(oi_data, direction_hint)
        volume_score = self._analyze_volume(candles)
        trend_score = self._analyze_trend(candles, direction_hint)
        
        # Calculate weighted composite score
        composite = (
            cvd_score * self.weights["cvd"] +
            oi_score * self.weights["oi"] +
            volume_score * self.weights["volume"] +
            trend_score * self.weights["trend"]
        )
        
        # Determine strength
        if composite >= 70:
            strength = MomentumStrength.STRONG
        elif composite >= 50:
            strength = MomentumStrength.MODERATE
        else:
            strength = MomentumStrength.WEAK
        
        # Determine direction
        direction = self._determine_direction(cvd_data, oi_data, direction_hint)
        
        # Recommended timeframe for targets
        recommended_tf = self.tf_thresholds[strength.value]
        
        # Leverage multiplier (higher momentum = more aggressive)
        if strength == MomentumStrength.STRONG:
            leverage_mult = 1.5
        elif strength == MomentumStrength.MODERATE:
            leverage_mult = 1.0
        else:
            leverage_mult = 0.7
        
        return MomentumResult(
            score=composite,
            strength=strength,
            direction=direction,
            cvd_score=cvd_score,
            oi_score=oi_score,
            volume_score=volume_score,
            trend_score=trend_score,
            recommended_tf=recommended_tf,
            leverage_multiplier=leverage_mult
        )
    
    def _analyze_cvd(self, cvd_data: Optional[Dict], direction: str) -> float:
        """Analyze CVD for momentum"""
        if not cvd_data:
            return 50.0
        
        ratio = cvd_data.get("ratio", 1.0)
        delta = cvd_data.get("delta_pct", 0)
        pressure = cvd_data.get("pressure", "NEUTRAL")
        
        score = 50.0
        
        # Ratio analysis
        if direction in ["LONG", "BULLISH"]:
            # Higher ratio = more buying pressure = good
            if ratio > 1.5:
                score += 25
            elif ratio > 1.2:
                score += 15
            elif ratio > 1.0:
                score += 5
            elif ratio < 0.8:
                score -= 15
        else:  # SHORT
            # Lower ratio = more selling pressure = good
            if ratio < 0.7:
                score += 25
            elif ratio < 0.8:
                score += 15
            elif ratio < 1.0:
                score += 5
            elif ratio > 1.2:
                score -= 15
        
        # Delta analysis
        if direction in ["LONG", "BULLISH"]:
            if delta > 5:
                score += 15
            elif delta > 0:
                score += 5
            elif delta < -5:
                score -= 10
        else:
            if delta < -5:
                score += 15
            elif delta < 0:
                score += 5
            elif delta > 5:
                score -= 10
        
        return max(0, min(100, score))
    
    def _analyze_oi(self, oi_data: Optional[Dict], direction: str) -> float:
        """Analyze Open Interest for conviction"""
        if not oi_data:
            return 50.0
        
        # Get OI delta (1h change)
        delta = oi_data.get("delta", {})
        delta_1h = delta.get("1h", {})
        delta_pct = delta_1h.get("delta_oi_pct", 0)
        
        score = 50.0
        
        # Rising OI = new positions = conviction
        # Falling OI = positions closing = less conviction
        if delta_pct > 5:
            score += 25
        elif delta_pct > 2:
            score += 15
        elif delta_pct > 0:
            score += 5
        elif delta_pct < -5:
            score -= 20
        elif delta_pct < -2:
            score -= 10
        
        return max(0, min(100, score))
    
    def _analyze_volume(self, candles: Optional[List[Dict]]) -> float:
        """Analyze volume for strength"""
        if not candles or len(candles) < 20:
            return 50.0
        
        recent_volumes = [c.get("volume", 0) for c in candles[-5:]]
        avg_volume = np.mean([c.get("volume", 0) for c in candles[-20:]])
        current_avg = np.mean(recent_volumes)
        
        if avg_volume == 0:
            return 50.0
        
        volume_ratio = current_avg / avg_volume
        
        score = 50.0
        if volume_ratio > 2.0:
            score += 30
        elif volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.2:
            score += 10
        elif volume_ratio < 0.7:
            score -= 15
        
        return max(0, min(100, score))
    
    def _analyze_trend(self, candles: Optional[List[Dict]], direction: str) -> float:
        """Analyze trend alignment"""
        if not candles or len(candles) < 50:
            return 50.0
        
        closes = [c.get("close", 0) for c in candles]
        current = closes[-1]
        
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        
        score = 50.0
        
        if direction in ["LONG", "BULLISH"]:
            # Price > SMA20 > SMA50 = strong uptrend
            if current > sma_20 > sma_50:
                score += 30
            elif current > sma_20:
                score += 15
            elif current < sma_20 < sma_50:
                score -= 20
        else:  # SHORT
            # Price < SMA20 < SMA50 = strong downtrend
            if current < sma_20 < sma_50:
                score += 30
            elif current < sma_20:
                score += 15
            elif current > sma_20 > sma_50:
                score -= 20
        
        return max(0, min(100, score))
    
    def _determine_direction(
        self, 
        cvd_data: Optional[Dict], 
        oi_data: Optional[Dict],
        hint: str
    ) -> str:
        """Determine overall momentum direction"""
        bullish_signals = 0
        bearish_signals = 0
        
        if cvd_data:
            ratio = cvd_data.get("ratio", 1.0)
            if ratio > 1.2:
                bullish_signals += 1
            elif ratio < 0.8:
                bearish_signals += 1
        
        if hint in ["LONG", "BULLISH"]:
            bullish_signals += 1
        elif hint in ["SHORT", "BEARISH"]:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return "BULLISH"
        elif bearish_signals > bullish_signals:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def get_fractal_targets(
        self,
        candles_5m: List[Dict],
        candles_1h: List[Dict],
        candles_4h: List[Dict],
        direction: str,
        momentum_strength: MomentumStrength,
        current_price: float
    ) -> Dict[str, float]:
        """
        Get target levels from fractals based on momentum strength.
        
        Args:
            candles_*: OHLCV data for each timeframe
            direction: LONG or SHORT
            momentum_strength: STRONG, MODERATE, or WEAK
            current_price: Current price
            
        Returns:
            Dict with tp1, tp2, sl based on appropriate timeframe fractals
        """
        # Select candles based on momentum
        if momentum_strength == MomentumStrength.STRONG:
            candles = candles_4h[-50:] if candles_4h else candles_1h[-50:]
            tf_label = "4H"
        elif momentum_strength == MomentumStrength.MODERATE:
            candles = candles_1h[-50:] if candles_1h else candles_5m[-200:]
            tf_label = "1H"
        else:
            candles = candles_5m[-200:] if candles_5m else candles_1h[-50:]
            tf_label = "15m"
        
        if not candles or len(candles) < 10:
            return {}
        
        # Find fractals
        highs = [c.get("high", 0) for c in candles]
        lows = [c.get("low", 0) for c in candles]
        
        # Recent swing high/low (last 20 candles)
        recent_high = max(highs[-20:])
        recent_low = min(lows[-20:])
        
        # Further swing high/low
        if len(highs) > 40:
            far_high = max(highs[-40:-20])
            far_low = min(lows[-40:-20])
        else:
            far_high = recent_high
            far_low = recent_low
        
        targets = {}
        
        if direction in ["LONG", "BULLISH"]:
            # TP = swing highs above current price
            if recent_high > current_price:
                targets["tp1"] = recent_high
            if far_high > current_price and far_high > recent_high:
                targets["tp2"] = far_high
            elif recent_high > current_price:
                targets["tp2"] = recent_high * 1.01  # 1% above
            
            # SL = swing low below current price
            if recent_low < current_price:
                targets["sl"] = recent_low * 0.998  # Slightly below
        
        else:  # SHORT
            # TP = swing lows below current price
            if recent_low < current_price:
                targets["tp1"] = recent_low
            if far_low < current_price and far_low < recent_low:
                targets["tp2"] = far_low
            elif recent_low < current_price:
                targets["tp2"] = recent_low * 0.99
            
            # SL = swing high above current price
            if recent_high > current_price:
                targets["sl"] = recent_high * 1.002
        
        # Add metadata
        targets["_timeframe"] = tf_label
        targets["_momentum"] = momentum_strength.value
        
        return targets


# Test function
def test_momentum_analyzer():
    print("=" * 60)
    print("Testing Momentum Analyzer")
    print("=" * 60)
    
    analyzer = MomentumAnalyzer()
    
    # Test with mock data
    cvd_data = {"ratio": 1.45, "delta_pct": 3.2, "pressure": "BUYING"}
    oi_data = {"delta": {"1h": {"delta_oi_pct": 2.5}}}
    
    result = analyzer.analyze(
        cvd_data=cvd_data,
        oi_data=oi_data,
        direction_hint="LONG"
    )
    
    print(f"\n📊 Momentum Analysis:")
    print(f"   Score: {result.score:.1f}/100")
    print(f"   Strength: {result.strength.value}")
    print(f"   Direction: {result.direction}")
    print(f"\n   Components:")
    print(f"   - CVD: {result.cvd_score:.1f}")
    print(f"   - OI: {result.oi_score:.1f}")
    print(f"   - Volume: {result.volume_score:.1f}")
    print(f"   - Trend: {result.trend_score:.1f}")
    print(f"\n   Recommended TF: {result.recommended_tf}")
    print(f"   Leverage Mult: {result.leverage_multiplier:.1f}x")


if __name__ == "__main__":
    test_momentum_analyzer()
