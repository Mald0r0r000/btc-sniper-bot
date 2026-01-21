"""
Smart Entry MTF Module
Fractal zone detection and MTF MACD decision tree for zone priority selection
"""

from typing import Dict, List, Optional, Tuple


class FractalZoneDetector:
    """Detects fractal zones on 15m and 1H timeframes"""
    
    @staticmethod
    def detect_all_zones(
        direction: str,
        current_price: float,
        candles_1h: Optional[List[Dict]],
        candles_15m: Optional[List[Dict]]
    ) -> Dict[str, Optional[float]]:
        """
        Detect all fractal zones.
        
        Returns dict with keys:
        - support_1h, resistance_1h
        - fvg_15m, bear_fvg_15m
        - vp_val_15m, vp_vah_15m
        """
        zones = {
            'support_1h': None,
            'resistance_1h': None,
            'fvg_15m': None,
            'bear_fvg_15m': None,
            'vp_val_15m': None,
            'vp_vah_15m': None
        }
        
        # Detect 1H Support/Resistance
        if candles_1h and len(candles_1h) >= 24:
            sr_zones = FractalZoneDetector.detect_support_resistance_1h(candles_1h, current_price)
            zones['support_1h'] = sr_zones.get('support')
            zones['resistance_1h'] = sr_zones.get('resistance')
        
        # Detect 15m FVG
        if candles_15m and len(candles_15m) >= 50:
            fvg_zones = FractalZoneDetector.detect_fvg_15m(candles_15m, current_price, direction)
            zones['fvg_15m'] = fvg_zones.get('bull_fvg')
            zones['bear_fvg_15m'] = fvg_zones.get('bear_fvg')
        
        # Detect 15m VP VAL/VAH
        if candles_15m and len(candles_15m) >= 50:
            vp_zones = FractalZoneDetector.detect_vp_val_vah_15m(candles_15m)
            zones['vp_val_15m'] = vp_zones.get('val')
            zones['vp_vah_15m'] = vp_zones.get('vah')
        
        return zones
    
    @staticmethod
    def detect_support_resistance_1h(
        candles: List[Dict],
        current_price: float
    ) -> Dict[str, Optional[float]]:
        """Detect last swing low (support) and swing high (resistance) on 1H"""
        try:
            # Get last 24 candles (24h)
            recent = candles[-24:]
            highs = [float(c.get('high', c.get('h', 0))) for c in recent]
            lows = [float(c.get('low', c.get('l', 0))) for c in recent]
            
            # Find swing low (support) - lowest low below current price
            support_candidates = [l for l in lows if l < current_price and l > 0]
            support = min(support_candidates) if support_candidates else None
            
            # Find swing high (resistance) - highest high above current price
            resistance_candidates = [h for h in highs if h > current_price]
            resistance = max(resistance_candidates) if resistance_candidates else None
            
            return {'support': support, 'resistance': resistance}
        except Exception:
            return {'support': None, 'resistance': None}
    
    @staticmethod
    def detect_fvg_15m(
        candles: List[Dict],
        current_price: float,
        direction: str
    ) -> Dict[str, Optional[float]]:
        """Detect nearest Fair Value Gap on 15m"""
        try:
            bull_fvg = None
            bear_fvg = None
            min_bull_dist = float('inf')
            min_bear_dist = float('inf')
            
            # Look for FVGs in last 50 candles
            for i in range(len(candles) - 3):
                c1 = candles[i]
                c2 = candles[i + 1]
                c3 = candles[i + 2]
                
                h1 = float(c1.get('high', c1.get('h', 0)))
                l1 = float(c1.get('low', c1.get('l', 0)))
                h3 = float(c3.get('high', c3.get('h', 0)))
                l3 = float(c3.get('low', c3.get('l', 0)))
                
                # Bullish FVG: l3 > h1
                if l3 > h1:
                    fvg_level = (h1 + l3) / 2
                    if fvg_level < current_price:
                        dist = current_price - fvg_level
                        if dist < min_bull_dist:
                            min_bull_dist = dist
                            bull_fvg = fvg_level
                
                # Bearish FVG: h3 < l1
                if h3 < l1:
                    fvg_level = (l1 + h3) / 2
                    if fvg_level > current_price:
                        dist = fvg_level - current_price
                        if dist < min_bear_dist:
                            min_bear_dist = dist
                            bear_fvg = fvg_level
            
            return {'bull_fvg': bull_fvg, 'bear_fvg': bear_fvg}
        except Exception:
            return {'bull_fvg': None, 'bear_fvg': None}
    
    @staticmethod
    def detect_vp_val_vah_15m(candles: List[Dict]) -> Dict[str, Optional[float]]:
        """Calculate Volume Profile VAL and VAH on 15m"""
        try:
            prices = []
            volumes = []
            
            for c in candles[-100:]:
                high = float(c.get('high', c.get('h', 0)))
                low = float(c.get('low', c.get('l', 0)))
                volume = float(c.get('volume', c.get('v', 0)))
                
                mid = (high + low) / 2
                prices.append(mid)
                volumes.append(volume)
            
            if not prices:
                return {'val': None, 'vah': None}
            
            # Simple VP calculation
            price_min = min(prices)
            price_max = max(prices)
            bins = 20
            price_range = price_max - price_min
            bin_size = price_range / bins if price_range > 0 else 1
            
            volume_by_price = {}
            for i, price in enumerate(prices):
                bin_idx = int((price - price_min) / bin_size) if bin_size > 0 else 0
                bin_idx = min(bin_idx, bins - 1)
                bin_price = price_min + (bin_idx * bin_size)
                volume_by_price[bin_price] = volume_by_price.get(bin_price, 0) + volumes[i]
            
            if not volume_by_price:
                return {'val': None, 'vah': None}
            
            # Find POC
            poc = max(volume_by_price.items(), key=lambda x: x[1])[0]
            
            # Calculate 70% value area
            sorted_prices = sorted(volume_by_price.keys())
            total_volume = sum(volume_by_price.values())
            target_volume = total_volume * 0.7
            
            accumulated = 0
            val = poc
            vah = poc
            
            for price in sorted_prices:
                vol = volume_by_price[price]
                accumulated += vol
                if price < poc:
                    val = price
                if price > poc:
                    vah = price
                if accumulated >= target_volume:
                    break
            
            return {'val': val, 'vah': vah}
        except Exception:
            return {'val': None, 'vah': None}


class MTFMACDZoneSelector:
    """Selects optimal zone based on MTF MACD context"""
    
    @staticmethod
    def select_zone(
        direction: str,
        current_price: float,
        zones: Dict[str, Optional[float]],
        mtf_macd: Dict
    ) -> Tuple[Optional[float], str, int]:
        """
        Select optimal entry zone based on MTF MACD context.
        
        Returns:
            (zone_price, zone_description, timeout_hours)
        """
        # Extract MTF context
        divergence_type = mtf_macd.get('divergence', {}).get('type', 'UNKNOWN')
        confluence = mtf_macd.get('confluence', 'MIXED')
        
        # Get 1H and 4H slopes
        mtf_data = mtf_macd.get('mtf_data', {})
        slope_1h = mtf_data.get('1h', {}).get('slope', 0)
        slope_4h = mtf_data.get('4h', {}).get('slope', 0)
        
        # LONG Decision Tree
        if direction == "LONG":
            return MTFMACDZoneSelector._select_long_zone(
                current_price, zones, divergence_type, confluence, slope_1h, slope_4h
            )
        # SHORT Decision Tree
        else:
            return MTFMACDZoneSelector._select_short_zone(
                current_price, zones, divergence_type, confluence, slope_1h, slope_4h
            )
    
    @staticmethod
    def _select_long_zone(
        current_price: float,
        zones: Dict,
        d_type: str,
        conf: str,
        s1h: float,
        s4h: float
    ) -> Tuple[Optional[float], str, int]:
        """LONG zone selection logic"""
        
        # BEARISH_DIVERGENCE
        if d_type == 'BEARISH_DIVERGENCE':
            if s1h > 0 and s4h > 0:
                return zones.get('support_1h'), "Support 1H (rebond confirmé)", 6
            elif s1h < 0 and s4h < 0:
                val = zones.get('vp_val_15m')
                if val and (current_price - val) / current_price * 100 < 2.5:
                    return val, "VAL 15m (défensif)", 4
                else:
                    return None, "SKIP (div + momentum bearish)", 0
            else:
                return zones.get('fvg_15m'), "FVG 15m (structure)", 8
        
        # WEAK_BEARISH_DIV
        elif d_type == 'WEAK_BEARISH_DIV':
            if s1h > 0:
                return zones.get('support_1h'), "Support 1H", 10
            else:
                return zones.get('fvg_15m'), "FVG 15m strict", 6
        
        # CONFLUENCE
        elif conf == 'ALL_BULLISH':
            if s1h > 150 and s4h > 150:
                return current_price, "IMMEDIATE (momentum explosif)", 0
            elif s1h > 0 and s4h > 0:
                return zones.get('support_1h'), "Support 1H (confluence)", 12
            else:
                return zones.get('fvg_15m'), "FVG 15m (pullback)", 18
        
        # MOMENTUM_WARNING_BEAR
        elif d_type == 'MOMENTUM_WARNING_BEAR':
            if s1h > 0:
                return zones.get('support_1h'), "Support 1H (LTF reprend)", 8
            else:
                return zones.get('vp_val_15m'), "VAL 15m (prudence)", 4
        
        # Default
        else:
            fvg = zones.get('fvg_15m')
            support = zones.get('support_1h')
            if fvg:
                return fvg, "FVG 15m (default)", 8
            elif support:
                return support, "Support 1H (default)", 8
            else:
                return current_price, "IMMEDIATE (no zones)", 0
    
    @staticmethod
    def _select_short_zone(
        current_price: float,
        zones: Dict,
        d_type: str,
        conf: str,
        s1h: float,
        s4h: float
    ) -> Tuple[Optional[float], str, int]:
        """SHORT zone selection logic (mirror of LONG)"""
        
        # BULLISH_DIV ERGENCE
        if d_type == 'BULLISH_DIVERGENCE':
            if s1h < 0 and s4h < 0:
                return zones.get('resistance_1h'), "Resistance 1H (dump confirmé)", 6
            elif s1h > 0 and s4h > 0:
                vah = zones.get('vp_vah_15m')
                if vah and (vah - current_price) / current_price * 100 < 2.5:
                    return vah, "VAH 15m (défensif)", 4
                else:
                    return None, "SKIP (div + momentum bullish)", 0
            else:
                return zones.get('bear_fvg_15m'), "Bear FVG 15m", 8
        
        # CONFLUENCE
        elif conf == 'ALL_BEARISH':
            if s1h < -150 and s4h < -150:
                return current_price, "IMMEDIATE (momentum explosif)", 0
            elif s1h < 0 and s4h < 0:
                return zones.get('resistance_1h'), "Resistance 1H (confluence)", 12
            else:
                return zones.get('bear_fvg_15m'), "Bear FVG 15m", 18
        
        # Default
        else:
            bear_fvg = zones.get('bear_fvg_15m')
            resistance = zones.get('resistance_1h')
            if bear_fvg:
                return bear_fvg, "Bear FVG 15m (default)", 8
            elif resistance:
                return resistance, "Resistance 1H (default)", 8
            else:
                return current_price, "IMMEDIATE (no zones)", 0
