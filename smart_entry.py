"""
Smart Entry Module
Calculates optimal entry points by analyzing liquidation zones to avoid stop hunts
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import MTF fractal zone detection
from smart_entry_mtf import FractalZoneDetector, MTFMACDZoneSelector


class EntryStrategy(Enum):
    IMMEDIATE = "IMMEDIATE"      # Enter at current price
    WAIT_FOR_DIP = "WAIT_DIP"    # Wait for a dip to liq zone before entering
    LIMIT_ORDER = "LIMIT_ORDER"  # Set limit order at optimal price
    SKIP = "SKIP"                # Too risky, don't enter


@dataclass
class SmartEntryResult:
    """Result of smart entry analysis"""
    strategy: EntryStrategy
    current_price: float
    optimal_entry: float
    
    # Liquidation zone info
    nearest_liq_zone: Optional[float]
    liq_zone_distance_pct: float
    
    # Targets adjusted for smart entry
    adjusted_tp1: float
    adjusted_tp2: float
    adjusted_sl: float
    
    # Risk metrics
    original_rr_ratio: float
    improved_rr_ratio: float
    potential_improvement_pct: float
    
    # Entry window
    entry_timeout_hours: int
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy.value,
            "current_price": self.current_price,
            "optimal_entry": self.optimal_entry,
            "nearest_liq_zone": self.nearest_liq_zone,
            "liq_zone_distance_pct": self.liq_zone_distance_pct,
            "adjusted_targets": {
                "tp1": self.adjusted_tp1,
                "tp2": self.adjusted_tp2,
                "sl": self.adjusted_sl
            },
            "rr_improvement": {
                "original": self.original_rr_ratio,
                "improved": self.improved_rr_ratio,
                "improvement_pct": self.potential_improvement_pct
            },
            "entry_timeout_hours": self.entry_timeout_hours
        }


class SmartEntryAnalyzer:
    """
    Analyzes liquidation zones to find optimal entry points.
    
    Strategy:
    1. For LONG: Look for nearby liquidation zones of longs (below current price)
       - These are "stop hunt" zones where price might dip before going up
       - Enter slightly above the liq zone after the stop hunt
       
    2. For SHORT: Look for nearby liquidation zones of shorts (above current price)
       - These are where price might spike before going down
       - Enter slightly below the liq zone after the spike
    """
    
    def __init__(self):
        # Configuration
        self.max_wait_distance_pct = 2.0  # Max % to wait for dip
        self.min_improvement_pct = 0.3    # Minimum R:R improvement to recommend waiting
        self.entry_buffer_pct = 0.05      # Buffer above/below liq zone for entry
        self.sl_buffer_pct = 0.2          # Buffer for SL beyond liq zone
        
        # Timeout for waiting
        self.default_timeout_hours = 4
    
    def analyze(
        self,
        direction: str,
        current_price: float,
        original_tp1: float,
        original_tp2: float,
        original_sl: float,
        liq_zones: Optional[Dict] = None,
        candles: Optional[List[Dict]] = None,
        candles_15m: Optional[List[Dict]] = None,
        mtf_macd_context: Optional[Dict] = None
    ) -> SmartEntryResult:
        """
        Analyze and recommend optimal entry strategy.
        
        Args:
            direction: LONG or SHORT
            current_price: Current market price
            original_tp1/tp2/sl: Original targets from signal
            liq_zones: Liquidation zones data from analyzer
            candles: Recent candles for support/resistance
            
        Returns:
            SmartEntryResult with entry recommendation
        """
        # Default: immediate entry
        strategy = EntryStrategy.IMMEDIATE
        optimal_entry = current_price
        nearest_liq = None
        liq_distance_pct = 0.0
        
        adjusted_tp1 = original_tp1
        adjusted_tp2 = original_tp2
        adjusted_sl = original_sl
        
        # Calculate original R:R
        original_rr = self._calculate_rr(current_price, original_tp1, original_sl, direction)
        improved_rr = original_rr
        improvement = 0.0
        
        # === MTF MACD-AWARE ZONE SELECTION ===
        if mtf_macd_context and (candles or candles_15m):
            try:
                # 1. Detect fractal zones
                fractal_zones = FractalZoneDetector.detect_all_zones(
                    direction=direction,
                    current_price=current_price,
                    candles_1h=candles,
                    candles_15m=candles_15m
                )
                
                # 2. Select optimal zone based on MTF context
                selected_zone, zone_desc, timeout = MTFMACDZoneSelector.select_zone(
                    direction=direction,
                    current_price=current_price,
                    zones=fractal_zones,
                    mtf_macd=mtf_macd_context
                )
                
                # 3. Apply zone selection
                if selected_zone is None:
                    # SKIP signal
                    strategy = EntryStrategy.SKIP
                    optimal_entry = current_price
                elif abs(selected_zone - current_price) < 0.01:
                    # IMMEDIATE
                    strategy = EntryStrategy.IMMEDIATE
                    optimal_entry = current_price
                else:
                    # WAIT_FOR_DIP or LIMIT_ORDER
                    optimal_entry = selected_zone
                    strategy = EntryStrategy.WAIT_FOR_DIP if direction == "LONG" else EntryStrategy.LIMIT_ORDER
                    
                    # Adjust SL based on zone
                    if direction == "LONG":
                        # SL slightly below zone
                        adjusted_sl = optimal_entry * (1 - self.sl_buffer_pct / 100)
                    else:
                        # SL slightly above zone
                        adjusted_sl = optimal_entry * (1 + self.sl_buffer_pct / 100)
                    
                    # Calculate improved R:R
                    improved_rr = self._calculate_rr(optimal_entry, original_tp1, adjusted_sl, direction)
                    improvement = ((improved_rr - original_rr) / original_rr * 100) if original_rr > 0 else 0
                
                # Store zone description and timeout
                if strategy != EntryStrategy.IMMEDIATE:
                    print(f"   🎯 Smart Entry: {zone_desc} @ ${optimal_entry:,.0f} (timeout {timeout}h)")
                
                return SmartEntryResult(
                    strategy=strategy,
                    current_price=current_price,
                    optimal_entry=optimal_entry,
                    nearest_liq_zone=selected_zone,
                    liq_zone_distance_pct=abs(selected_zone - current_price) / current_price * 100 if selected_zone else 0,
                    adjusted_tp1=original_tp1,
                    adjusted_tp2=original_tp2,
                    adjusted_sl=adjusted_sl,
                    original_rr_ratio=original_rr,
                    improved_rr_ratio=improved_rr,
                    potential_improvement_pct=improvement,
                    entry_timeout_hours=timeout
                )
                
            except Exception as e:
                print(f"   ⚠️ MTF Smart Entry failed: {e}, falling back to legacy logic")
                # Fall through to legacy logic
        
        # === LEGACY LOGIC (Fallback if no MTF context) ===
        # Find nearest relevant liquidation zone
        if liq_zones:
            nearest_liq = self._find_nearest_liq_zone(
                direction, current_price, liq_zones
            )
            
            if nearest_liq:
                liq_distance_pct = abs(nearest_liq - current_price) / current_price * 100
                
                # Check if zone is within acceptable wait distance
                if liq_distance_pct <= self.max_wait_distance_pct:
                    # Calculate optimal entry point
                    optimal_entry, adjusted_sl = self._calculate_smart_entry(
                        direction, current_price, nearest_liq
                    )
                    
                    # Calculate improved R:R
                    improved_rr = self._calculate_rr(
                        optimal_entry, original_tp1, adjusted_sl, direction
                    )
                    
                    improvement = ((improved_rr - original_rr) / original_rr * 100) if original_rr > 0 else 0
                    
                    # Recommend waiting if improvement is significant
                    if improvement >= self.min_improvement_pct * 100:
                        strategy = EntryStrategy.WAIT_FOR_DIP if direction == "LONG" else EntryStrategy.LIMIT_ORDER
                        adjusted_tp1 = original_tp1  # Keep original TPs
                        adjusted_tp2 = original_tp2
        
        # Alternative: use recent support/resistance from candles
        if strategy == EntryStrategy.IMMEDIATE and candles:
            smart_entry = self._find_entry_from_structure(
                direction, current_price, candles, original_tp1, original_sl
            )
            if smart_entry:
                optimal_entry, adjusted_sl = smart_entry
                improved_rr = self._calculate_rr(
                    optimal_entry, original_tp1, adjusted_sl, direction
                )
                improvement = ((improved_rr - original_rr) / original_rr * 100) if original_rr > 0 else 0
                
                if improvement >= self.min_improvement_pct * 100:
                    strategy = EntryStrategy.LIMIT_ORDER
        
        return SmartEntryResult(
            strategy=strategy,
            current_price=current_price,
            optimal_entry=optimal_entry,
            nearest_liq_zone=nearest_liq,
            liq_zone_distance_pct=liq_distance_pct,
            adjusted_tp1=adjusted_tp1,
            adjusted_tp2=adjusted_tp2,
            adjusted_sl=adjusted_sl,
            original_rr_ratio=original_rr,
            improved_rr_ratio=improved_rr,
            potential_improvement_pct=improvement,
            entry_timeout_hours=self.default_timeout_hours
        )
    
    def _find_nearest_liq_zone(
        self,
        direction: str,
        current_price: float,
        liq_zones: Dict
    ) -> Optional[float]:
        """Find the nearest relevant liquidation zone"""
        # For LONG: look for liq zones of longs BELOW current price (stop hunt zone)
        # For SHORT: look for liq zones of shorts ABOVE current price
        
        zones = liq_zones.get('zones', [])
        
        if direction == "LONG":
            # Find nearest liq zone below current price
            relevant_zones = [
                z.get('price', 0) for z in zones 
                if z.get('side') in ['LONG', 'long', 'buy'] 
                and z.get('price', float('inf')) < current_price
            ]
            return max(relevant_zones) if relevant_zones else None
        else:
            # Find nearest liq zone above current price
            relevant_zones = [
                z.get('price', 0) for z in zones 
                if z.get('side') in ['SHORT', 'short', 'sell'] 
                and z.get('price', 0) > current_price
            ]
            return min(relevant_zones) if relevant_zones else None
    
    def _calculate_smart_entry(
        self,
        direction: str,
        current_price: float,
        liq_zone: float
    ) -> Tuple[float, float]:
        """Calculate optimal entry and SL based on liq zone"""
        buffer = liq_zone * (self.entry_buffer_pct / 100)
        sl_buffer = liq_zone * (self.sl_buffer_pct / 100)
        
        if direction == "LONG":
            # Enter slightly above liq zone (after stop hunt)
            optimal_entry = liq_zone + buffer
            # SL below the liq zone
            adjusted_sl = liq_zone - sl_buffer
        else:
            # Enter slightly below liq zone (after spike)
            optimal_entry = liq_zone - buffer
            # SL above the liq zone
            adjusted_sl = liq_zone + sl_buffer
        
        return round(optimal_entry, 1), round(adjusted_sl, 1)
    
    def _find_entry_from_structure(
        self,
        direction: str,
        current_price: float,
        candles: List[Dict],
        tp1: float,
        sl: float
    ) -> Optional[Tuple[float, float]]:
        """Find entry from recent support/resistance structure"""
        if not candles or len(candles) < 20:
            return None
        
        highs = [c.get('high', 0) for c in candles[-20:]]
        lows = [c.get('low', 0) for c in candles[-20:]]
        
        if direction == "LONG":
            # Look for recent support (swing low)
            recent_support = min(lows[-10:])
            
            # If support is not too far
            distance = (current_price - recent_support) / current_price * 100
            if 0.3 <= distance <= 1.5:
                # Entry at support + small buffer
                entry = recent_support * 1.001
                # SL below support
                new_sl = recent_support * 0.998
                return round(entry, 1), round(new_sl, 1)
        else:
            # Look for recent resistance (swing high)
            recent_resistance = max(highs[-10:])
            
            distance = (recent_resistance - current_price) / current_price * 100
            if 0.3 <= distance <= 1.5:
                entry = recent_resistance * 0.999
                new_sl = recent_resistance * 1.002
                return round(entry, 1), round(new_sl, 1)
        
        return None
    
    def _calculate_rr(
        self,
        entry: float,
        tp: float,
        sl: float,
        direction: str
    ) -> float:
        """Calculate risk/reward ratio"""
        if direction == "LONG":
            reward = tp - entry
            risk = entry - sl
        else:
            reward = entry - tp
            risk = sl - entry
        
        if risk <= 0:
            return 0.0
        
        return reward / risk
    
    def format_recommendation(self, result: SmartEntryResult, direction: str) -> str:
        """Format entry recommendation for display/notification"""
        lines = []
        
        if result.strategy == EntryStrategy.IMMEDIATE:
            lines.append("📍 ENTRÉE: Immédiate")
            lines.append(f"   Prix: ${result.current_price:,.0f}")
        elif result.strategy == EntryStrategy.WAIT_FOR_DIP:
            lines.append("⏳ ENTRÉE: Attendre le dip")
            lines.append(f"   Zone cible: ${result.optimal_entry:,.0f}")
            lines.append(f"   Stop hunt zone: ${result.nearest_liq_zone:,.0f}")
            lines.append(f"   Timeout: {result.entry_timeout_hours}h")
        elif result.strategy == EntryStrategy.LIMIT_ORDER:
            lines.append("📝 ENTRÉE: Ordre limite")
            lines.append(f"   Prix limite: ${result.optimal_entry:,.0f}")
        
        if result.potential_improvement_pct > 0:
            lines.append(f"   📈 R:R amélioré: {result.original_rr_ratio:.2f} → {result.improved_rr_ratio:.2f} (+{result.potential_improvement_pct:.0f}%)")
        
        return "\n".join(lines)


# Test function
def test_smart_entry():
    print("=" * 60)
    print("Testing Smart Entry Analyzer")
    print("=" * 60)
    
    analyzer = SmartEntryAnalyzer()
    
    # Mock liquidation zones
    liq_zones = {
        'zones': [
            {'price': 94300, 'side': 'LONG', 'volume': 500},
            {'price': 93800, 'side': 'LONG', 'volume': 300},
            {'price': 96500, 'side': 'SHORT', 'volume': 400},
        ]
    }
    
    result = analyzer.analyze(
        direction="LONG",
        current_price=95000,
        original_tp1=96500,
        original_tp2=97500,
        original_sl=94000,
        liq_zones=liq_zones
    )
    
    print(f"\n📊 Smart Entry Analysis:")
    print(f"   Strategy: {result.strategy.value}")
    print(f"   Current: ${result.current_price:,.0f}")
    print(f"   Optimal Entry: ${result.optimal_entry:,.0f}")
    print(f"   Nearest Liq Zone: ${result.nearest_liq_zone:,.0f}" if result.nearest_liq_zone else "   No liq zone found")
    print(f"   Distance: {result.liq_zone_distance_pct:.2f}%")
    print(f"\n   R:R: {result.original_rr_ratio:.2f} → {result.improved_rr_ratio:.2f}")
    print(f"   Improvement: {result.potential_improvement_pct:.0f}%")
    
    print(f"\n{analyzer.format_recommendation(result, 'LONG')}")


if __name__ == "__main__":
    test_smart_entry()
