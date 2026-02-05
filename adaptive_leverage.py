"""
Adaptive Leverage Calculator
Calculates optimal leverage based on TP distance and momentum strength
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LeverageRecommendation:
    """Recommendation for leverage"""
    base_leverage: int
    recommended_leverage: int
    max_safe_leverage: int
    
    # Factors
    tp_distance_pct: float
    momentum_multiplier: float
    risk_score: float  # 0-100 (100=highest risk)
    
    # Warnings
    warnings: list
    
    def to_dict(self) -> Dict:
        return {
            "base_leverage": self.base_leverage,
            "recommended": self.recommended_leverage,
            "max_safe": self.max_safe_leverage,
            "tp_distance_pct": self.tp_distance_pct,
            "momentum_mult": self.momentum_multiplier,
            "risk_score": self.risk_score,
            "warnings": self.warnings
        }


class AdaptiveLeverageCalculator:
    """
    Calculates adaptive leverage based on multiple factors:
    
    1. TP Distance: Closer TP = can use higher leverage
       - <1% TP distance → x50-100
       - 1-3% TP distance → x23-50
       - >3% TP distance → x10-23
    
    2. Momentum: Higher momentum = more confidence = slightly higher leverage
    
    3. Volatility: Higher volatility = reduce leverage
    
    4. Risk Management: Never exceed position sizing that would lose >5% on SL hit
    """
    
    def __init__(self, max_position_loss_pct: float = 5.0):
        """
        Args:
            max_position_loss_pct: Maximum % of capital to lose on any trade (default 5%)
        """
        self.max_position_loss_pct = max_position_loss_pct
        
        # Leverage tiers by TP distance
        # REVISED (Conservative): Cap at 25x to avoid liquidation risks
        self.leverage_tiers = [
            (1.0, (15, 25)),    # <1% TP → 15-25x (Scalping)
            (2.0, (10, 20)),    # 1-2% TP → 10-20x (Intraday)
            (3.0, (5, 15)),     # 2-3% TP → 5-15x (Swing)
            (5.0, (3, 10)),     # 3-5% TP → 3-10x
            (float('inf'), (2, 5))   # >5% TP → 2-5x
        ]
    
    def calculate(
        self,
        entry_price: float,
        tp1_price: float,
        sl_price: float,
        direction: str,
        momentum_score: float = 50.0,
        volatility_pct: float = 1.0,
        capital: float = 10000.0,
        position_size_pct: float = 10.0
    ) -> LeverageRecommendation:
        """
        Calculate recommended leverage.
        
        Args:
            entry_price: Entry price
            tp1_price: First take profit target
            sl_price: Stop loss price
            direction: LONG or SHORT
            momentum_score: Momentum score 0-100
            volatility_pct: Recent volatility %
            capital: Total capital
            position_size_pct: % of capital to use
            
        Returns:
            LeverageRecommendation
        """
        warnings = []
        
        # Calculate TP and SL distances
        if direction == "LONG":
            tp_distance_pct = (tp1_price - entry_price) / entry_price * 100
            sl_distance_pct = (entry_price - sl_price) / entry_price * 100
        else:
            tp_distance_pct = (entry_price - tp1_price) / entry_price * 100
            sl_distance_pct = (sl_price - entry_price) / entry_price * 100
        
        # Sanity check
        if tp_distance_pct <= 0:
            warnings.append("⚠️ TP distance invalid")
            tp_distance_pct = 1.0
        if sl_distance_pct <= 0:
            warnings.append("⚠️ SL distance invalid")
            sl_distance_pct = 1.0
        
        # 1. Base leverage from TP distance
        base_min, base_max = self._get_leverage_tier(tp_distance_pct)
        base_leverage = (base_min + base_max) // 2
        
        # 2. Adjust for momentum
        # High momentum = can be more aggressive
        if momentum_score >= 70:
            momentum_mult = 1.2
        elif momentum_score >= 50:
            momentum_mult = 1.0
        else:
            momentum_mult = 0.8
            warnings.append("⚠️ Faible momentum → levier réduit")
        
        adjusted_leverage = int(base_leverage * momentum_mult)
        
        # 3. Cap based on volatility
        if volatility_pct > 2.0:
            adjusted_leverage = int(adjusted_leverage * 0.7)
            warnings.append("⚠️ Haute volatilité → levier réduit")
        elif volatility_pct > 1.5:
            adjusted_leverage = int(adjusted_leverage * 0.85)
        
        # 4. Risk management cap
        # Max leverage where SL hit = max_position_loss_pct of capital
        position_size = capital * (position_size_pct / 100)
        max_loss_allowed = capital * (self.max_position_loss_pct / 100)
        
        # Loss = position_size * sl_distance_pct * leverage / 100
        # max_loss_allowed = position_size * sl_distance_pct * max_leverage / 100
        # max_leverage = max_loss_allowed * 100 / (position_size * sl_distance_pct)
        max_safe = int((max_loss_allowed * 100) / (position_size * sl_distance_pct))
        max_safe = max(5, min(100, max_safe))  # Clamp 5-100
        
        if adjusted_leverage > max_safe:
            adjusted_leverage = max_safe
            warnings.append(f"⚠️ Levier limité à x{max_safe} (gestion risque)")
        
        # Ensure within tier bounds
        adjusted_leverage = max(base_min, min(base_max, adjusted_leverage))
        
        # Risk score (0-100, higher = riskier)
        risk_score = min(100, (adjusted_leverage / 100) * 100 + (sl_distance_pct * 10))
        
        return LeverageRecommendation(
            base_leverage=base_leverage,
            recommended_leverage=adjusted_leverage,
            max_safe_leverage=max_safe,
            tp_distance_pct=tp_distance_pct,
            momentum_multiplier=momentum_mult,
            risk_score=risk_score,
            warnings=warnings
        )
    
    def _get_leverage_tier(self, tp_distance_pct: float) -> Tuple[int, int]:
        """Get leverage range based on TP distance"""
        for threshold, (min_lev, max_lev) in self.leverage_tiers:
            if tp_distance_pct <= threshold:
                return min_lev, max_lev
        return 5, 10  # Default conservative
    
    def format_recommendation(self, rec: LeverageRecommendation) -> str:
        """Format recommendation for display"""
        lines = [
            f"⚡ Levier recommandé: <b>x{rec.recommended_leverage}</b>",
            f"   (distance TP: {rec.tp_distance_pct:.2f}%)"
        ]
        
        if rec.recommended_leverage != rec.base_leverage:
            lines.append(f"   Ajusté de x{rec.base_leverage} (momentum: {rec.momentum_multiplier:.1f}x)")
        
        for warning in rec.warnings:
            lines.append(f"   {warning}")
        
        return "\n".join(lines)


# Test function
def test_leverage_calculator():
    print("=" * 60)
    print("Testing Adaptive Leverage Calculator")
    print("=" * 60)
    
    calc = AdaptiveLeverageCalculator()
    
    # Test case 1: Close TP, high momentum
    print("\n📊 Test 1: Close TP (0.8%), High Momentum")
    rec = calc.calculate(
        entry_price=95000,
        tp1_price=95760,  # 0.8%
        sl_price=94500,   # 0.5%
        direction="LONG",
        momentum_score=75
    )
    print(f"   Recommended: x{rec.recommended_leverage}")
    print(f"   Max Safe: x{rec.max_safe_leverage}")
    print(f"   Risk: {rec.risk_score:.0f}/100")
    
    # Test case 2: Far TP, low momentum
    print("\n📊 Test 2: Far TP (3.5%), Low Momentum")
    rec = calc.calculate(
        entry_price=95000,
        tp1_price=98325,  # 3.5%
        sl_price=93500,   # 1.6%
        direction="LONG",
        momentum_score=35
    )
    print(f"   Recommended: x{rec.recommended_leverage}")
    print(f"   Max Safe: x{rec.max_safe_leverage}")
    for w in rec.warnings:
        print(f"   {w}")
    
    print("\n" + calc.format_recommendation(rec))


if __name__ == "__main__":
    test_leverage_calculator()
