from typing import Dict, List, Any
import numpy as np

class SpotPerpDivergenceAnalyzer:
    """
    Analyzes divergences between Spot and Perpetual Futures Cumulative Volume Delta (CVD)
    across multiple exchanges to identify market manipulation, absorption, and genuine trends.
    """
    
    def __init__(self):
        pass

    def analyze(self, global_cvd_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            global_cvd_data: Result from MultiExchangeAggregator.fetch_global_cvd_candles()
                             Structure: { '1h': { 'spot': { 'binance': [...] }, 'swap': { ... } } }
        """
        results = {}
        
        # We process each timeframe available
        for tf, markets in global_cvd_data.items():
            results[tf] = self._analyze_timeframe(markets, tf)
            
        return results

    def _analyze_timeframe(self, markets: Dict[str, Dict], timeframe: str) -> Dict[str, Any]:
        spot_data = markets.get('spot', {})
        perp_data = markets.get('swap', {})
        
        if not spot_data or not perp_data:
            return self._empty_result(timeframe)
            
        # 1. Calculate Individual CVDs
        spot_cvds = self._calculate_group_cvd(spot_data)
        perp_cvds = self._calculate_group_cvd(perp_data)
        
        if not spot_cvds or not perp_cvds:
            return self._empty_result(timeframe)

        # 2. Aggregate to Global Series (Weighted Average or Sum)
        # We simply sum them as they are all normalized to BTC volume
        global_spot_cvd_series = np.sum([s['cvd_series'] for s in spot_cvds], axis=0)
        global_perp_cvd_series = np.sum([s['cvd_series'] for s in perp_cvds], axis=0)
        
        # Ensure lengths match (trim to shortest)
        min_len = min(len(global_spot_cvd_series), len(global_perp_cvd_series))
        global_spot_cvd_series = global_spot_cvd_series[-min_len:]
        global_perp_cvd_series = global_perp_cvd_series[-min_len:]
        
        # 3. Calculate Divergence (Perp - Spot)
        # Positive = Perp buying > Spot buying (Aggressive positioning)
        # Negative = Spot buying > Perp buying (Genuine demand)
        divergence_series = global_perp_cvd_series - global_spot_cvd_series
        
        # 4. Analyze Trends and Signals (Last point)
        current_spot_delta = global_spot_cvd_series[-1] - global_spot_cvd_series[-2] if min_len > 1 else 0
        current_perp_delta = global_perp_cvd_series[-1] - global_perp_cvd_series[-2] if min_len > 1 else 0
        current_divergence = divergence_series[-1]
        
        # Market regime detection
        regime = "NEUTRAL"
        signal = "NONE"
        
        # Spot Driven Rally: Spot Buying + Perp Selling/Neutral
        if current_spot_delta > 0 and current_perp_delta <= 0:
            regime = "SPOT_DRIVEN_RALLY"
            signal = "BULLISH_QUALITY"
            
        # Perp Driven Rally: Perp Buying + Spot Selling/Neutral (FOMO/Squeeze risk)
        elif current_perp_delta > 0 and current_spot_delta <= 0:
            regime = "PERP_DRIVEN_RALLY"
            signal = "BEARISH_DIVERGENCE" # Often a trap
            
        # Spot Selling + Perp Buying (Absorption/Manipulation top)
        elif current_spot_delta < 0 and current_perp_delta > 0:
            regime = "SPOT_SELLING_INTO_PERP_BUYING"
            signal = "BEARISH_HEAVY"
            
        # Panic Selling
        elif current_spot_delta < 0 and current_perp_delta < 0:
            regime = "BROAD_SELLING"
            signal = "BEARISH"
            
        # Broad Buying
        elif current_spot_delta > 0 and current_perp_delta > 0:
            regime = "BROAD_BUYING"
            signal = "BULLISH"

        return {
            'global_spot_cvd': round(global_spot_cvd_series[-1], 2),
            'global_perp_cvd': round(global_perp_cvd_series[-1], 2),
            'divergence': round(current_divergence, 2),
            'regime': regime,
            'signal': signal,
            'spot_delta_recent': round(current_spot_delta, 2),
            'perp_delta_recent': round(current_perp_delta, 2),
            'details': {
                'spot_exchanges': list(spot_data.keys()),
                'perp_exchanges': list(perp_data.keys())
            }
        }
    
    def _calculate_group_cvd(self, exchange_data: Dict[str, List]) -> List[Dict]:
        """
        Calculates CVD series for each exchange in the group.
        """
        processed = []
        for ex_id, candles in exchange_data.items():
            if not candles: continue
            
            # Extract candle components
            # data format: {'open': ..., 'close': ..., 'volume': ...}
            closes = np.array([c['close'] for c in candles])
            opens = np.array([c['open'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            volumes = np.array([c['volume'] for c in candles])
            
            # Calculate Delta Proxy
            # Delta = Volume * ((Close - Open) / (High - Low))
            ranges = highs - lows
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                 body_strength = (closes - opens) / ranges
                 body_strength[ranges == 0] = 0
            
            deltas = volumes * body_strength
            cvd_series = np.cumsum(np.nan_to_num(deltas))
            
            processed.append({
                'exchange': ex_id,
                'cvd_series': cvd_series
            })
            
        return processed

    def _empty_result(self, timeframe):
        return {
            'global_spot_cvd': 0, 'global_perp_cvd': 0, 'divergence': 0,
            'regime': 'NO_DATA', 'signal': 'NEUTRAL',
            'spot_delta_recent': 0, 'perp_delta_recent': 0,
            'details': {}
        }
