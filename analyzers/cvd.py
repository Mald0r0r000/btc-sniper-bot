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
        - 5m: Uses Tick Data (High Precision)
        - 1h/4h/1d: Uses Candle Data (Volume Delta Proxy)
        
        Args:
            ohlcv_data: Dict containing '1h' and '1d' candle data
        """
        now = datetime.now(timezone.utc)
        mtf_data = {}
        
        # 1. 5m Analysis (Tick-based)
        # ---------------------------------------------
        cutoff_5m = (now - timedelta(minutes=5)).timestamp() * 1000
        trades_5m = [t for t in self.trades if t.get('timestamp', 0) >= cutoff_5m]
        
        if trades_5m:
            buy_vol = sum(float(t.get('amount', 0)) for t in trades_5m if t.get('side') == 'buy')
            sell_vol = sum(float(t.get('amount', 0)) for t in trades_5m if t.get('side') != 'buy')
            net_cvd = buy_vol - sell_vol
            agg_ratio = buy_vol / sell_vol if sell_vol > 0 else 10.0
            
            trend, score = self._calculate_trend_score(agg_ratio)
            
            mtf_data['5m'] = {
                'net_cvd': round(net_cvd, 4),
                'buy_volume': round(buy_vol, 4),
                'sell_volume': round(sell_vol, 4),
                'aggression_ratio': round(agg_ratio, 2),
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
        # 1h: Uses last 1h candle (or equivalent candles)
        if ohlcv_data and '1h' in ohlcv_data:
            mtf_data['1h'] = self._calculate_candle_cvd(ohlcv_data['1h'], lookback=1)
            
        # 4h: Uses last 4 x 1h candles
        if ohlcv_data and '1h' in ohlcv_data:
            mtf_data['4h'] = self._calculate_candle_cvd(ohlcv_data['1h'], lookback=4)
            
        # 1d: Uses last 1d candle
        if ohlcv_data and '1d' in ohlcv_data:
            mtf_data['1d'] = self._calculate_candle_cvd(ohlcv_data['1d'], lookback=1)
            
        # Fill missing with empty
        for tf in ['1h', '4h', '1d']:
            if tf not in mtf_data:
                mtf_data[tf] = self._empty_tf_result()

        # 3. Composite Score & Confluence
        # ---------------------------------------------
        # Re-normalize weights if some TFs are missing
        # Weights: 5m (30%), 1h (40%), 4h (20%), 1d (10%)
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
            
        # Confluence
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
            
        return {
            'mtf_data': mtf_data,
            'composite_score': round(composite_score, 1),
            'confluence': confluence,
            'trend': overall_trend,
            'emoji': emoji,
            'available': True
        }

    def _calculate_candle_cvd(self, candles: List[Dict], lookback: int) -> Dict[str, Any]:
        """ Estimates CVD from candle data (Volume Delta Proxy) """
        if not candles or len(candles) < lookback:
            return self._empty_tf_result()
            
        relevant_candles = candles[-lookback:]
        
        buy_vol_proxy = 0.0
        sell_vol_proxy = 0.0
        
    def _calculate_candle_cvd(self, candles: List[Dict], lookback: int) -> Dict[str, Any]:
        """ 
        Estimates CVD from candle data using Weighted Volume Proxy (PineScript Logic)
        Formula: Delta = Volume * ((Close - Open) / (High - Low))
        """
        if not candles or len(candles) < lookback:
            return self._empty_tf_result()
            
        relevant_candles = candles[-lookback:]
        
        buy_vol_proxy = 0.0
        sell_vol_proxy = 0.0
        net_cvd_accum = 0.0
        
        for candle in relevant_candles:
            close = float(candle.get('close', 0))
            open_price = float(candle.get('open', 0))
            high = float(candle.get('high', 0))
            low = float(candle.get('low', 0))
            vol = float(candle.get('volume', 0))
            
            # Weighted Volume Delta (Body Strength)
            range_len = high - low
            if range_len > 0:
                body_strength = (close - open_price) / range_len
            else:
                body_strength = 0.0 # Doji/Flat
            
            # Delta for this candle
            delta = vol * body_strength
            net_cvd_accum += delta
            
            # Split vol based on strength
            # If strength is +0.5, then 75% buy, 25% sell approximation?
            # Or simpler: accumulation logic
            if delta > 0:
                buy_vol_proxy += abs(delta)
            else:
                sell_vol_proxy += abs(delta)
                
        # Recalculate Aggression Ratio based on the weighted volumes
        # Note: sell_vol_proxy is sum of absolute negative deltas
        agg_ratio = buy_vol_proxy / sell_vol_proxy if sell_vol_proxy > 0 else 10.0
        
        trend, score = self._calculate_trend_score(agg_ratio)
        
        return {
            'net_cvd': round(net_cvd_accum, 4),
            'buy_volume': round(buy_vol_proxy, 4),
            'sell_volume': round(sell_vol_proxy, 4),
            'aggression_ratio': round(agg_ratio, 2),
            'trend': trend,
            'score': round(score, 1),
            'trade_count': len(relevant_candles), # Count of candles
            'available': True,
            'method': 'CANDLE_WEIGHTED'
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

