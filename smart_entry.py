"""
Smart Entry Module
Calculates optimal entry points by analyzing liquidation zones to avoid stop hunts
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


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
        
        # Stop Hunt Protection
        self.psych_level_threshold_pct = 0.6   # Distance to consider "too close" to psych level
        self.psych_major_buffer_pct = 0.3      # Extra buffer for major levels (10k)
        self.psych_intermediate_buffer_pct = 0.15  # For intermediate levels (5k)
        self.psych_minor_buffer_pct = 0.08     # For minor levels (1k)
        self.max_rr_degradation = 0.1          # Max 10% R:R loss acceptable
        
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
                    
                    # Adjust SL based on zone with Stop Hunt Protection
                    if direction == "LONG":
                        # SL slightly below zone
                        initial_sl = optimal_entry * (1 - self.sl_buffer_pct / 100)
                    else:
                        # SL slightly above zone
                        initial_sl = optimal_entry * (1 + self.sl_buffer_pct / 100)
                    
                    # Apply Stop Hunt Protection
                    adjusted_sl, sl_justification = self.calculate_safe_sl(
                        initial_sl=initial_sl,
                        direction=direction,
                        entry_price=optimal_entry,
                        liq_zones={'long': selected_zone} if direction == 'LONG' else {'short': selected_zone},
                        current_price=current_price,
                        candles=candles
                    )
                    if 'moved' in sl_justification:
                        print(f"   🛡️ Stop Hunt Protection: {sl_justification}")
                    
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
                        direction, current_price, nearest_liq, candles
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
        
        # === FINAL SAFETY CHECK: Liquidity Shadowing ===
        # Ensure we are protected by shadowing strategy regardless of which path was taken
        final_safe_sl, shadow_reason = self.calculate_safe_sl(
            initial_sl=adjusted_sl,
            direction=direction,
            entry_price=optimal_entry,
            liq_zones=liq_zones,
            current_price=current_price,
            candles=candles
        )
        # If the shadow SL is different/safer, use it
        if final_safe_sl != adjusted_sl:
            adjusted_sl = final_safe_sl
            if strategy != EntryStrategy.IMMEDIATE:
                print(f"   🛡️ Shadow SL Applied: {shadow_reason}")
                
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
        liq_zone: float,
        candles: Optional[List[Dict]] = None
    ) -> Tuple[float, float]:
        """Calculate optimal entry and SL based on liq zone"""
        buffer = liq_zone * (self.entry_buffer_pct / 100)
        sl_buffer = liq_zone * (self.sl_buffer_pct / 100)
        
        if direction == "LONG":
            # Enter slightly above liq zone (after stop hunt)
            optimal_entry = liq_zone + buffer
            # SL below the liq zone (initial)
            initial_sl = liq_zone - sl_buffer
        else:
            # Enter slightly below liq zone (after spike)
            optimal_entry = liq_zone - buffer
            # SL above the liq zone (initial)
            initial_sl = liq_zone + sl_buffer
        
        # Apply Stop Hunt Protection
        adjusted_sl, _ = self.calculate_safe_sl(
            initial_sl=initial_sl,
            direction=direction,
            entry_price=optimal_entry,
            liq_zones={'long': liq_zone} if direction == 'LONG' else {'short': liq_zone},
            current_price=optimal_entry,
            candles=candles
        )
        
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
    
    def _detect_psychological_levels(self, price: float, scan_range_pct: float = 1.0) -> List[Dict]:
        """
        Detect psychological levels (round numbers) near a given price.
        
        Args:
            price: Price to scan around
            scan_range_pct: % range to scan (default 1%)
        
        Returns:
            List of dicts with 'level', 'type', 'distance_pct'
        """
        scan_lower = price * (1 - scan_range_pct / 100)
        scan_upper = price * (1 + scan_range_pct / 100)
        
        levels = []
        
        # Major levels (every $10k)
        major_step = 10000
        major_start = int(scan_lower / major_step) * major_step
        for level in range(major_start, int(scan_upper) + major_step, major_step):
            if scan_lower <= level <= scan_upper:
                distance_pct = abs(level - price) / price * 100
                levels.append({
                    'level': float(level),
                    'type': 'major',
                    'distance_pct': distance_pct
                })
        
        # Intermediate levels (every $5k, excluding majors)
        intermediate_step = 5000
        intermediate_start = int(scan_lower / intermediate_step) * intermediate_step
        for level in range(intermediate_start, int(scan_upper) + intermediate_step, intermediate_step):
            if scan_lower <= level <= scan_upper and level % major_step != 0:
                distance_pct = abs(level - price) / price * 100
                levels.append({
                    'level': float(level),
                    'type': 'intermediate',
                    'distance_pct': distance_pct
                })
        
        # Note: Minor levels (every $1k) removed - too sensitive for institutional trading
        
        # Sort by distance
        levels.sort(key=lambda x: x['distance_pct'])
        return levels
    
    def _find_shadow_zone(
        self,
        direction: str,
        current_price: float,
        liq_zones: Dict,
        candles: Optional[List[Dict]] = None
    ) -> Tuple[Optional[float], str]:
        """
        Find a safe "shadow" zone for SL placement.
        """
        shadow_price = None
        reason = "Default"
        
        # 1. Analyze Liquidation Clusters (Fuel)
        # Note: liq_zones structure from LiquidationAnalyzer is:
        # { 'clusters_json': { 'longs': [...], 'shorts': [...] } }
        
        clusters = []
        clusters_data = liq_zones.get('clusters_json', {}) if liq_zones else {}
        
        if direction == "LONG":
            # For LONG, we fear the dip that grabs liquidity below
            # So we look for Long Liquidation Clusters
            clusters = clusters_data.get('longs', [])
        else:
            # For SHORT, we fear the pump that grabs liquidity above
            # So we look for Short Liquidation Clusters
            clusters = clusters_data.get('shorts', [])
            
        if clusters:
            # Sort by proximity to current price
            # clusters are usually sorted by intensity, but let's be sure we get the nearest relevant one
            sorted_clusters = sorted(clusters, key=lambda c: abs(c['price'] - current_price))
            
            # Filter for clusters that are effectively "fuel" (not too far)
            valid_clusters = [c for c in sorted_clusters if c.get('intensity', 0) >= 1] 
            
            if valid_clusters:
                nearest_cluster = valid_clusters[0]
                cluster_price = nearest_cluster['price']
                
                # Logic: The "Zone" is roughly price +/- 0.5%? 
                # The analyzer returns a center price. 
                # We assume the cluster spans a bit around the center.
                # Let's infer the edge safely.
                
                if direction == "LONG":
                    # Shadow is BELOW the cluster.
                    # Assuming cluster width of ~0.3% to be safe if not provided
                    shadow_price = cluster_price * 0.997 
                    # Ensure shadow is actually BELOW current price
                    if shadow_price >= current_price:
                         shadow_price = current_price * 0.995 # Fallback
                else:
                    # Shadow is ABOVE the cluster.
                    shadow_price = cluster_price * 1.003
                    if shadow_price <= current_price:
                         shadow_price = current_price * 1.005 # Fallback
                         
                reason = f"Shadowing Liq Cluster (${cluster_price:,.0f})"
                return shadow_price, reason

        # 2. Fallback: Structural Swings (Fractals)
        if candles and len(candles) >= 20:
            if direction == "LONG":
                # Find recent significant low in last 30 candles
                lows = [float(c.get('low', c.get('l', float('inf')))) for c in candles[-30:]]
                if lows:
                    swing_low = min(lows)
                    # Ensure it's not too close (noise)
                    if (current_price - swing_low) / current_price > 0.002: # 0.2% min distance for structure
                        shadow_price = swing_low
                        reason = "Shadowing Swing Low"
            else:
                # Find recent significant high
                highs = [float(c.get('high', c.get('h', 0))) for c in candles[-30:]]
                if highs:
                    swing_high = max(highs)
                    if (swing_high - current_price) / current_price > 0.002:
                        shadow_price = swing_high
                        reason = "Shadowing Swing High"
                    
        return shadow_price, reason

    def calculate_safe_sl(
        self,
        initial_sl: float,
        direction: str,
        entry_price: float,
        liq_zones: Optional[Dict] = None,
        current_price: float = None,
        candles: Optional[List[Dict]] = None
    ) -> Tuple[float, str]:
        """
        Calculate a safe SL using Liquidity Shadowing.
        Overrides any initial naive SL if a better shadow zone is found.
        """
        if current_price is None:
            current_price = entry_price

        # 1. Try to find a Shadow Zone
        shadow_price, reason = self._find_shadow_zone(direction, current_price, liq_zones, candles)
        
        adjusted_sl = initial_sl
        justification = "Standard SL"
        
        # Buffer config
        shadow_buffer = 0.0035 # 0.35% buffer behind the wall
        min_dist_pct = 0.006   # 0.6% absolute minimum distance
        max_dist_pct = 0.08    # 8% maximum SL distance (Equity protection)

        if shadow_price:
            if direction == "LONG":
                shadow_sl = shadow_price * (1 - shadow_buffer)
                # Use shadow SL, but check constraints
                adjusted_sl = shadow_sl
            else:
                shadow_sl = shadow_price * (1 + shadow_buffer)
                adjusted_sl = shadow_sl
                
            justification = f"Shadow Strategy: {reason}"
        else:
            # Fallback if no shadow zone found: Use Min Distance
            if direction == "LONG":
                adjusted_sl = entry_price * (1 - min_dist_pct)
            else:
                adjusted_sl = entry_price * (1 + min_dist_pct)
            justification = "Min Distance (No Shadow Found)"

        # === SAFETY ENFORCEMENT ===
        
        # 1. Enforce Minimum Distance (Spread/Noise protection)
        if direction == "LONG":
            min_sl_price = entry_price * (1 - min_dist_pct)
            if adjusted_sl > min_sl_price: # SL is too high (too close)
                adjusted_sl = min_sl_price
                justification += " (+Min Dist Enforced)"
        else:
            min_sl_price = entry_price * (1 + min_dist_pct)
            if adjusted_sl < min_sl_price: # SL is too low (too close)
                adjusted_sl = min_sl_price
                justification += " (+Min Dist Enforced)"

        # 2. Enforce Maximum Distance (Equity protection / R:R sanity)
        # If SL is too far, we might want to cap it or invalidate the trade.
        # Here we just cap it to maintain sanity.
        if direction == "LONG":
            max_sl_price = entry_price * (1 - max_dist_pct)
            if adjusted_sl < max_sl_price: # SL is too low (too far)
                adjusted_sl = max_sl_price
                justification += " (Max Dist Cap)"
        else:
            max_sl_price = entry_price * (1 + max_dist_pct)
            if adjusted_sl > max_sl_price: # SL is too high (too far)
                adjusted_sl = max_sl_price
                justification += " (Max Dist Cap)"

        return round(adjusted_sl, 1), justification


def test_smart_entry():
    print("=" * 60)
    print("Testing Smart Entry Analyzer (Liquidity Shadowing)")
    print("=" * 60)
    
    analyzer = SmartEntryAnalyzer()
    
    # Mock liquidation zones with clusters
    liq_zones = {
        'clusters_below': [
            {'center_price': 94500, 'min_price': 94400, 'max_price': 94600, 'strength': 5},
            {'center_price': 93000, 'min_price': 92800, 'max_price': 93200, 'strength': 3}
        ],
        'clusters_above': [
            {'center_price': 96500, 'min_price': 96400, 'max_price': 96600, 'strength': 4}
        ]
    }
    
    # Test LONG
    print("\n--- Test LONG Setup ---")
    result = analyzer.analyze(
        direction="LONG",
        current_price=95000,
        original_tp1=96500,
        original_tp2=97500,
        original_sl=94800, # Naive SL, too close
        liq_zones=liq_zones
    )
    
    print(f"Current: 95000, Initial SL: 94800")
    print(f"Adjusted SL: {result.adjusted_sl}")
    print(f"Nearest Fuel Cluster: 94400-94600")
    # Expect SL < 94400
    
    # Test SHORT
    print("\n--- Test SHORT Setup ---")
    result_s = analyzer.analyze(
        direction="SHORT",
        current_price=95000,
        original_tp1=93000,
        original_tp2=92000,
        original_sl=95200, # Naive SL, too close
        liq_zones=liq_zones
    )
    print(f"Current: 95000, Initial SL: 95200")
    print(f"Adjusted SL: {result_s.adjusted_sl}")
    print(f"Nearest Fuel Cluster: 96400-96600")
    # Expect SL > 96600


if __name__ == "__main__":
    test_smart_entry()
