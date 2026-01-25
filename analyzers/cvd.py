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
    
    def analyze_mtf(self) -> Dict[str, Any]:
        """
        Multi-Timeframe CVD Analysis
        Calculates CVD over 5m, 1h, 4h windows and provides confluence score.
        
        Returns:
            Dict with per-timeframe data and weighted composite score
        """
        if not self.trades:
            return self._empty_mtf_result()
        
        now = datetime.now(timezone.utc)
        
        # Time windows
        windows = {
            '5m': timedelta(minutes=5),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4)
        }
        
        mtf_data = {}
        
        for tf, delta in windows.items():
            cutoff = now - delta
            cutoff_ts = cutoff.timestamp() * 1000  # Convert to milliseconds
            
            # Filter trades within window
            filtered_trades = []
            for trade in self.trades:
                trade_ts = trade.get('timestamp', 0)
                if trade_ts >= cutoff_ts:
                    filtered_trades.append(trade)
            
            # Calculate CVD for this window
            if filtered_trades:
                buy_vol = sum(float(t.get('amount', 0)) for t in filtered_trades if t.get('side') == 'buy')
                sell_vol = sum(float(t.get('amount', 0)) for t in filtered_trades if t.get('side') != 'buy')
                
                net_cvd = buy_vol - sell_vol
                agg_ratio = buy_vol / sell_vol if sell_vol > 0 else 10.0
                
                # Determine trend for this TF
                if agg_ratio > 1.2:
                    trend = 'BULLISH'
                    score = min(100, 50 + (agg_ratio - 1.0) * 50)  # 50-100
                elif agg_ratio < 0.8:
                    trend = 'BEARISH'
                    score = max(0, 50 - (1.0 - agg_ratio) * 50)  # 0-50
                else:
                    trend = 'NEUTRAL'
                    score = 50
                
                mtf_data[tf] = {
                    'net_cvd': round(net_cvd, 4),
                    'buy_volume': round(buy_vol, 4),
                    'sell_volume': round(sell_vol, 4),
                    'aggression_ratio': round(agg_ratio, 2),
                    'trend': trend,
                    'score': round(score, 1),
                    'trade_count': len(filtered_trades),
                    'available': True
                }
            else:
                mtf_data[tf] = {
                    'net_cvd': 0,
                    'buy_volume': 0,
                    'sell_volume': 0,
                    'aggression_ratio': 1.0,
                    'trend': 'NEUTRAL',
                    'score': 50,
                    'trade_count': 0,
                    'available': False
                }
        
        # Weighted Composite Score
        composite_score = sum(
            mtf_data[tf]['score'] * self.MTF_WEIGHTS[tf]
            for tf in windows.keys()
        )
        
        # Confluence Check
        trends = [mtf_data[tf]['trend'] for tf in windows.keys()]
        if all(t == 'BULLISH' for t in trends):
            confluence = 'ALL_BULLISH'
        elif all(t == 'BEARISH' for t in trends):
            confluence = 'ALL_BEARISH'
        elif trends.count('BULLISH') >= 2:
            confluence = 'MOSTLY_BULLISH'
        elif trends.count('BEARISH') >= 2:
            confluence = 'MOSTLY_BEARISH'
        else:
            confluence = 'MIXED'
        
        # Overall trend from composite
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
    
    def _empty_result(self) -> Dict[str, Any]:
        """Résultat vide en cas d'erreur"""
        return {
            'net_cvd': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'total_volume': 0,
            'aggression_ratio': 1.0,
            'status': 'NEUTRE',
            'emoji': '⚪',
            'is_bullish': False,
            'is_bearish': False
        }
    
    def _empty_mtf_result(self) -> Dict[str, Any]:
        """Empty MTF result"""
        return {
            'mtf_data': {
                '5m': {'trend': 'NEUTRAL', 'score': 50, 'available': False},
                '1h': {'trend': 'NEUTRAL', 'score': 50, 'available': False},
                '4h': {'trend': 'NEUTRAL', 'score': 50, 'available': False}
            },
            'composite_score': 50,
            'confluence': 'MIXED',
            'trend': 'NEUTRAL',
            'emoji': '⚪',
            'available': False
        }

