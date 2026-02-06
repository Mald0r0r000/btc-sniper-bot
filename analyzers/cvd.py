"""
Analyseur CVD (Cumulative Volume Delta)
- Analyse des trades récents
- Ratio d'agression Taker Buy vs Sell
- MTF Analysis (5m, 1h, 4h)
"""
from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta


class CVDAnalyzer:
    """Analyse le Cumulative Volume Delta basé sur les trades récents"""
    
    # MTF Weights (optimized for intraday trading)
    MTF_WEIGHTS = {
        '5m': 0.30,   # Timing (30%)
        '1h': 0.50,   # Direction (50%)
        '4h': 0.20    # Trend (20%)
    }
    
    def __init__(self, trades: List[Dict]):
        """
        Args:
            trades: Liste des trades avec 'side', 'amount', 'timestamp'
        """
        self.trades = trades
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse le CVD et le ratio d'agression
        
        Returns:
            Dict avec net_cvd, buy_volume, sell_volume, aggression_ratio, status
        """
        if not self.trades:
            return self._empty_result()
        
        buy_vol = 0.0
        sell_vol = 0.0
        
        for trade in self.trades:
            amount = float(trade.get('amount', 0))
            side = trade.get('side', '')
            
            if side == 'buy':
                buy_vol += amount
            else:
                sell_vol += amount
        
        # Net CVD
        net_cvd = buy_vol - sell_vol
        total_vol = buy_vol + sell_vol
        
        # Ratio d'agression (> 1.0 = acheteurs dominent)
        aggression_ratio = buy_vol / sell_vol if sell_vol > 0 else 10.0
        
        # Interprétation
        if aggression_ratio > 1.2:
            status = "AGRESSION_ACHETEUSE"
            emoji = "🟢"
        elif aggression_ratio < 0.8:
            status = "AGRESSION_VENDEUSE"
            emoji = "🔴"
        else:
            status = "NEUTRE"
            emoji = "⚪"
        
        return {
            'net_cvd': round(net_cvd, 4),
            'buy_volume': round(buy_vol, 4),
            'sell_volume': round(sell_vol, 4),
            'total_volume': round(total_vol, 4),
            'aggression_ratio': round(aggression_ratio, 2),
            'status': status,
            'emoji': emoji,
            'is_bullish': aggression_ratio > 1.2,
            'is_bearish': aggression_ratio < 0.8
        }
    
    def analyze_mtf(self, ohlcv_data: Dict[str, List[Dict]] = None) -> Dict[str, Any]:
        """
        Multi-Timeframe CVD Analysis (Hybrid Approach)
        - 5m: Uses Tick Data (High Precision) + Efficiency
        - 1h/4h/1d: Uses Candle Data (Volume Delta Proxy) + Efficiency
        
        Args:
            ohlcv_data: Dict containing '1h' and '1d' candle data
        """
        now = datetime.now(timezone.utc)
        mtf_data = {}
        
        # 1. 5m Analysis (Tick-based)
        # ---------------------------------------------
        cutoff_5m = (now - timedelta(minutes=5)).timestamp() * 1000
        # Ensure trades are sorted by timestamp
        trades_5m = sorted([t for t in self.trades if t.get('timestamp', 0) >= cutoff_5m], key=lambda x: x.get('timestamp', 0))
        
        if trades_5m:
            buy_vol = sum(float(t.get('amount', 0)) for t in trades_5m if t.get('side') == 'buy')
            sell_vol = sum(float(t.get('amount', 0)) for t in trades_5m if t.get('side') != 'buy')
            net_cvd = buy_vol - sell_vol
            agg_ratio = buy_vol / sell_vol if sell_vol > 0 else 10.0
            
            # CVD Efficiency (R&D Point 1): Abs(Price Change) / Abs(Net CVD)
            start_price = float(trades_5m[0].get('price', 0))
            end_price = float(trades_5m[-1].get('price', 0))
            price_delta = end_price - start_price
            
            # Normalize efficiency: (Price % Change) / (Net CVD / 1000 BTC)
            if abs(net_cvd) > 0.1 and end_price > 0:
                efficiency = (abs(price_delta) / end_price) / (abs(net_cvd) / 1000.0) 
            else:
                efficiency = 1.0
                
            # Absorption Detection: High volume but price isn't moving
            # Threshold: > 20 BTC net delta with < 5% expected efficiency
            is_absorption = False
            if abs(net_cvd) > 20 and efficiency < 0.05:
                is_absorption = True

            trend, score = self._calculate_trend_score(agg_ratio)
            
            # R&D: Absorption Override
            # If sellers are absorbing (High Sell Vol + Price Stable), neutralize bearishness
            if is_absorption and trend == 'BEARISH':
                trend = 'NEUTRAL'
                score = 55 # Slight bullish bias (short squeeze potential)
            
            mtf_data['5m'] = {
                'net_cvd': round(net_cvd, 4),
                'buy_volume': round(buy_vol, 4),
                'sell_volume': round(sell_vol, 4),
                'aggression_ratio': round(agg_ratio, 2),
                'efficiency': round(efficiency, 4),
                'is_absorption': is_absorption,
                'trend': trend,
                'score': round(score, 1),
                'trade_count': len(trades_5m),
                'available': True,
                'method': 'TICK'
            }
        else:
            mtf_data['5m'] = self._empty_tf_result()

        # 2. Higher Timeframes (Candle-based Proxy)
        # ---------------------------------------------
        if ohlcv_data and '1h' in ohlcv_data:
            mtf_data['1h'] = self._calculate_candle_cvd(ohlcv_data['1h'], lookback=1)
            
        if ohlcv_data and '1h' in ohlcv_data:
            mtf_data['4h'] = self._calculate_candle_cvd(ohlcv_data['1h'], lookback=4)
            
        if ohlcv_data and '1d' in ohlcv_data:
            mtf_data['1d'] = self._calculate_candle_cvd(ohlcv_data['1d'], lookback=1)
            
        # Fill missing with empty
        for tf in ['1h', '4h', '1d']:
            if tf not in mtf_data:
                mtf_data[tf] = self._empty_tf_result()

        # 3. Composite Score & Confluence
        # ---------------------------------------------
        weights = {'5m': 0.3, '1h': 0.4, '4h': 0.2, '1d': 0.1}
        composite_score = 0
        total_weight = 0
        
        for tf, weight in weights.items():
            if mtf_data[tf]['available']:
                composite_score += mtf_data[tf]['score'] * weight
                total_weight += weight
                
        if total_weight > 0:
            composite_score = composite_score / total_weight
        else:
            composite_score = 50
            
        trends = [mtf_data[tf]['trend'] for tf in ['5m', '1h', '4h', '1d'] if mtf_data[tf]['available']]
        confluence = self._calculate_confluence(trends)
        
        # Overall Trend
        if composite_score > 60:
            overall_trend = 'BULLISH'
            emoji = '🟢'
        elif composite_score < 40:
            overall_trend = 'BEARISH'
            emoji = '🔴'
        else:
            overall_trend = 'NEUTRAL'
            emoji = '⚪'
            
        # Global Absorption Risk
        absorption_risk = any(mtf_data[tf].get('is_absorption', False) for tf in mtf_data)
        
        # Aggression Detection (User Request)
        agg_types = []
        for tf in ['5m', '1h', '4h']: # Focus on intraday agression
            if mtf_data[tf]['available']:
                ratio = mtf_data[tf]['aggression_ratio']
                if ratio > 1.25: agg_types.append('BULLISH')
                elif ratio < 0.75: agg_types.append('BEARISH')
        
        if agg_types.count('BULLISH') > agg_types.count('BEARISH'):
            aggression_status = 'BULLISH_AGGRESSION'
        elif agg_types.count('BEARISH') > agg_types.count('BULLISH'):
            aggression_status = 'BEARISH_AGGRESSION'
        else:
            aggression_status = 'BALANCED'
            
        return {
            'mtf_data': mtf_data,
            'composite_score': round(composite_score, 1),
            'confluence': confluence,
            'trend': overall_trend,
            'emoji': emoji,
            'absorption_risk': absorption_risk,
            'aggression_status': aggression_status,
            'available': True
        }

    def _calculate_candle_cvd(self, candles: List[Dict], lookback: int) -> Dict[str, Any]:
        """ 
        Estimates CVD from candle data using Weighted Volume Proxy + Heikin Ashi Smoothing
        Formula: Delta = Volume * ((Close - Open) / (High - Low))
        Smoothing: Standard Heikin Ashi on the Cumulative Delta series
        """
        if not candles:
            return self._empty_tf_result()
            
        # 1. Calculate weighted delta for ALL available candles to build proper cumulative history
        deltas = []
        for candle in candles:
            close = float(candle.get('close', 0))
            open_p = float(candle.get('open', 0))
            high = float(candle.get('high', 0))
            low = float(candle.get('low', 0))
            vol = float(candle.get('volume', 0))
            
            range_len = high - low
            body_strength = (close - open_p) / range_len if range_len > 0 else 0.0
            deltas.append(vol * body_strength)
            
        # 2. Build Cumulative Delta series
        cum_deltas = []
        current_cum = 0.0
        for d in deltas:
            current_cum += d
            cum_deltas.append(current_cum)
            
        if len(cum_deltas) < 2:
            return self._empty_tf_result()

        # 3. Apply Heikin Ashi Smoothing on the Cumulative Delta series
        ha_open_series = []
        ha_close_series = []
        last_ha_open = (cum_deltas[0] + cum_deltas[1]) / 2 
        
        for i in range(1, len(cum_deltas)):
            o = cum_deltas[i-1]
            c = cum_deltas[i]
            h = max(o, c)
            l = min(o, c)
            haclose = (o + h + l + c) / 4
            haopen = (last_ha_open + (ha_close_series[-1] if ha_close_series else haclose)) / 2
            ha_open_series.append(haopen)
            ha_close_series.append(haclose)
            last_ha_open = haopen
            
        # 4. Extract lookback results
        latest_ha_close = ha_close_series[-1]
        latest_ha_open = ha_open_series[-1]
        window_deltas = deltas[-lookback:]
        net_cvd_accum = sum(window_deltas)
        
        # 5. Efficiency Calculation (Candle)
        start_candle = candles[-lookback]
        end_candle = candles[-1]
        price_delta = float(end_candle.get('close', 0)) - float(start_candle.get('open', 0))
        end_price = float(end_candle.get('close', 1))
        
        if abs(net_cvd_accum) > 0.1 and end_price > 0:
            efficiency = (abs(price_delta) / end_price) / (abs(net_cvd_accum) / 1000.0)
        else:
            efficiency = 1.0
            
        is_absorption = False
        # Stricter thresholds for higher timeframes as proxy deltas are smoother
        if abs(net_cvd_accum) > 100 and efficiency < 0.02: 
            is_absorption = True

        if latest_ha_close > latest_ha_open:
            agg_ratio = 1.5 
        elif latest_ha_close < latest_ha_open:
            agg_ratio = 0.5 
        else:
            agg_ratio = 1.0

        trend, score = self._calculate_trend_score(agg_ratio)

        # R&D: Absorption Override
        if is_absorption and trend == 'BEARISH':
            trend = 'NEUTRAL'
            score = 55
        
        return {
            'net_cvd': round(net_cvd_accum, 4),
            'buy_volume': round(sum(d for d in window_deltas if d > 0), 4),
            'sell_volume': round(abs(sum(d for d in window_deltas if d < 0)), 4),
            'aggression_ratio': round(agg_ratio, 2),
            'efficiency': round(efficiency, 4),
            'is_absorption': is_absorption,
            'trend': trend,
            'score': round(score, 1),
            'trade_count': len(window_deltas),
            'available': True,
            'method': 'CANDLE_HA_SMOOTHED'
        }

    def _calculate_trend_score(self, ratio: float):
        if ratio > 1.2:
            return 'BULLISH', min(100, 50 + (ratio - 1.0) * 50)
        elif ratio < 0.8:
            return 'BEARISH', max(0, 50 - (1.0 - ratio) * 50)
        else:
            return 'NEUTRAL', 50

    def _calculate_confluence(self, trends: List[str]) -> str:
        if not trends: return 'UNKNOWN'
        if all(t == 'BULLISH' for t in trends): return 'ALL_BULLISH'
        if all(t == 'BEARISH' for t in trends): return 'ALL_BEARISH'
        if trends.count('BULLISH') >= len(trends) / 2: return 'MOSTLY_BULLISH'
        if trends.count('BEARISH') >= len(trends) / 2: return 'MOSTLY_BEARISH'
        return 'MIXED'

    def _empty_tf_result(self) -> Dict[str, Any]:
        return {
            'net_cvd': 0, 'buy_volume': 0, 'sell_volume': 0,
            'aggression_ratio': 1.0, 'trend': 'NEUTRAL', 'score': 50,
            'trade_count': 0, 'available': False, 'method': 'NONE'
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Résultat vide en cas d'erreur"""
        return {
            'net_cvd': 0, 'buy_volume': 0, 'sell_volume': 0,
            'total_volume': 0, 'aggression_ratio': 1.0,
            'status': 'NEUTRE', 'emoji': '⚪',
            'is_bullish': False, 'is_bearish': False
        }
    
    def _empty_mtf_result(self) -> Dict[str, Any]:
        """Empty MTF result"""
        return {
            'mtf_data': {},
            'composite_score': 50,
            'confluence': 'MIXED',
            'trend': 'NEUTRAL',
            'emoji': '⚪',
            'available': False
        }

