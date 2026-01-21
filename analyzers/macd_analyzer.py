
import pandas as pd
from typing import Dict, Any, List
from exchange import BitgetConnector

class MACDAnalyzer:
    """
    Multi-Timeframe MACD Analyzer (3D, 1D, 4H, 1H)
    
    Optimized for intraday trading:
    - 1H has 50% weight (dominant for intraday)
    - 4H has 30% weight (confirmation)
    - 1D has 15% weight (context)
    - 3D has 5% weight (background trend only)
    """

    def __init__(self):
        self.connector = BitgetConnector()
        # MACD Parameters (12, 26, 9)
        self.fast = 12
        self.slow = 26
        self.signal = 9
        
        # MTF Weights for Intraday (must sum to 100)
        self.weights = {
            '3d': 5,
            '1d': 15,
            '4h': 30,
            '1h': 50
        }

    def analyze(self) -> Dict[str, Any]:
        """
        Analyzes MACD across multiple timeframes and returns weighted composite score.
        
        Returns:
            Dict containing:
            - mtf_data: Dict of MACD values per timeframe
            - composite_score: Weighted score (-100 to +100)
            - trend: Overall trend (BULLISH/BEARISH/NEUTRAL)
            - confluence: Alignment level (ALL_BULLISH, ALL_BEARISH, MIXED)
            - available: bool
        """
        try:
            mtf_data = {}
            
            # Analyze each timeframe
            for tf in ['1h', '4h', '1d', '3d']:
                macd_result = self._analyze_timeframe(tf)
                if macd_result['available']:
                    mtf_data[tf] = macd_result
            
            # Check if we have enough data
            if not mtf_data or '1h' not in mtf_data:
                print(f"   ❌ MTF MACD: Missing critical timeframes (need at least 1H)")
                return {
                    'mtf_data': {},
                    'composite_score': 0,
                    'trend': 'NEUTRAL',
                    'confluence': 'UNKNOWN',
                    'available': False
                }
            
            # Calculate weighted composite score
            composite_score = self._calculate_composite_score(mtf_data)
            
            # Detect MTF divergences and apply dynamic adjustments
            divergence_signal = self._detect_mtf_divergence(mtf_data)
            composite_score += divergence_signal['score_adjustment']
            
            # Determine overall trend
            if composite_score > 20:
                trend = 'BULLISH'
            elif composite_score < -20:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'
            
            # Check confluence (all timeframes aligned)
            confluence = self._check_confluence(mtf_data)
            
            # Apply confluence boost (reduced if slopes diverge)
            confluence_boost = 0
            if confluence == 'ALL_BULLISH':
                # Check if all slopes are also positive
                all_slopes_positive = all(data.get('slope', 0) > 0 for data in mtf_data.values())
                confluence_boost = 20 if all_slopes_positive else 10
                composite_score = min(100, composite_score + confluence_boost)
            elif confluence == 'ALL_BEARISH':
                # Check if all slopes are also negative
                all_slopes_negative = all(data.get('slope', 0) < 0 for data in mtf_data.values())
                confluence_boost = -20 if all_slopes_negative else -10
                composite_score = max(-100, composite_score + confluence_boost)
            
            # Enhanced logging with slope info
            print(f"   📊 MTF MACD: {trend} (Score: {composite_score:.0f}) | {divergence_signal['type']}")
            for tf, data in mtf_data.items():
                slope_emoji = "📈" if data.get('slope', 0) > 0 else "📉" if data.get('slope', 0) < 0 else "➡️"
                print(f"      {tf.upper()}: {data['trend']} (Hist: {data['hist']:.2f} | Slope: {slope_emoji} {data.get('slope', 0):.2f})")
            if divergence_signal['type'] != 'CONFLUENCE':
                print(f"      ⚠️ {divergence_signal['description']}")
            
            return {
                'mtf_data': mtf_data,
                'composite_score': composite_score,
                'trend': trend,
                'confluence': confluence,
                'divergence': divergence_signal,
                'available': True
            }

        except Exception as e:
            print(f"   ❌ MTF MACD Analysis Error: {e}")
            return {
                'mtf_data': {},
                'composite_score': 0,
                'trend': 'ERROR',
                'confluence': 'UNKNOWN',
                'available': False
            }

    def _analyze_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """
        Analyzes MACD for a single timeframe with slope calculation.
        
        Args:
            timeframe: '1h', '4h', '1d', or '3d'
        
        Returns:
            Dict with MACD values, trend, and slope
        """
        try:
            # Fetch candles
            if timeframe == '3d':
                # Use history API for 3D (need more candles)
                df = self.connector.fetch_history_candles(timeframe, limit=100)
            else:
                # Use regular OHLCV for 1H, 4H, 1D
                limit_map = {'1h': 100, '4h': 100, '1d': 100}
                df = self.connector.fetch_ohlcv(timeframe, limit=limit_map.get(timeframe, 100))
            
            # Validate data
            if df is None or df.empty or len(df) < 30:
                return {
                    'macd': 0, 'signal': 0, 'hist': 0,
                    'trend': 'NEUTRAL', 'slope': 0, 'available': False
                }
            
            # Calculate MACD
            ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            # Get latest values
            macd_val = float(macd_line.iloc[-1])
            sig_val = float(signal_line.iloc[-1])
            hist_val = float(histogram.iloc[-1])
            
            # Calculate histogram slope (momentum strength)
            # Compare current hist to hist N periods ago
            lookback_map = {'1h': 5, '4h': 3, '1d': 2, '3d': 2}
            lookback = lookback_map.get(timeframe, 2)
            
            if len(histogram) > lookback:
                hist_previous = float(histogram.iloc[-(lookback + 1)])
                slope = hist_val - hist_previous
            else:
                slope = 0
            
            # Determine trend
            if hist_val > 0:
                trend = 'BULLISH'
            elif hist_val < 0:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'
            
            return {
                'macd': macd_val,
                'signal': sig_val,
                'hist': hist_val,
                'slope': slope,
                'trend': trend,
                'available': True
            }

        except Exception as e:
            print(f"   ⚠️ MACD {timeframe.upper()} failed: {e}")
            return {
                'macd': 0, 'signal': 0, 'hist': 0,
                'slope': 0, 'trend': 'NEUTRAL', 'available': False
            }

    def _calculate_composite_score(self, mtf_data: Dict[str, Dict]) -> float:
        """
        Calculates weighted composite score from multiple timeframes.
        
        Score range: -100 (very bearish) to +100 (very bullish)
        """
        total_score = 0.0
        total_weight = 0.0
        
        for tf, weight in self.weights.items():
            if tf in mtf_data:
                data = mtf_data[tf]
                # Normalize histogram to -100/+100 scale
                # Use tanh to cap extreme values
                import math
                normalized_hist = math.tanh(data['hist'] / 1000) * 100
                
                weighted_score = normalized_hist * (weight / 100)
                total_score += weighted_score
                total_weight += weight
        
        # Ensure we have valid data
        if total_weight == 0:
            return 0.0
        
        return total_score

    def _detect_mtf_divergence(self, mtf_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Detects divergences between higher and lower timeframes.
        
        Early reversal signals:
        - HTF (3D) bullish but slope negative + LTF (1D, 4H) bearish → Bearish Divergence
        - HTF (3D) bearish but slope positive + LTF (1D, 4H) bullish → Bullish Divergence
        
        Returns:
            Dict with divergence type, score adjustment, and description
        """
        # Get HTF and LTF data
        htf_3d = mtf_data.get('3d', {})
        htf_1d = mtf_data.get('1d', {})
        ltf_4h = mtf_data.get('4h', {})
        ltf_1h = mtf_data.get('1h', {})
        
        # Check if we have enough data
        if not htf_3d.get('available') or not ltf_1h.get('available'):
            return {
                'type': 'UNKNOWN',
                'score_adjustment': 0,
                'description': 'Insufficient data'
            }
        
        # Extract trends and slopes
        trend_3d = htf_3d.get('trend', 'NEUTRAL')
        slope_3d = htf_3d.get('slope', 0)
        trend_1d = htf_1d.get('trend', 'NEUTRAL')
        trend_4h = ltf_4h.get('trend', 'NEUTRAL')
        trend_1h = ltf_1h.get('trend', 'NEUTRAL')
        
        # Count LTF bearish/bullish
        ltf_trends = [trend_1d, trend_4h, trend_1h]
        ltf_bearish_count = ltf_trends.count('BEARISH')
        ltf_bullish_count = ltf_trends.count('BULLISH')
        
        # ========== BEARISH DIVERGENCE DETECTION ==========
        # HTF still bullish but losing momentum + LTF already bearish
        if trend_3d == 'BULLISH' and slope_3d < 0 and ltf_bearish_count >= 2:
            return {
                'type': 'BEARISH_DIVERGENCE',
                'score_adjustment': -15,
                'description': f'Early reversal: 3D bullish but weakening (slope {slope_3d:.0f}), LTF bearish ({ltf_bearish_count}/3)'
            }
        
        # HTF bullish but LTF bearish (without slope confirmation)
        if trend_3d == 'BULLISH' and ltf_bearish_count >= 2:
            return {
                'type': 'WEAK_BEARISH_DIV',
                'score_adjustment': -8,
                'description': f'HTF bullish vs LTF bearish ({ltf_bearish_count}/3), watch for reversal'
            }
        
        # ========== BULLISH DIVERGENCE DETECTION ==========
        # HTF still bearish but recovering + LTF already bullish
        if trend_3d == 'BEARISH' and slope_3d > 0 and ltf_bullish_count >= 2:
            return {
                'type': 'BULLISH_DIVERGENCE',
                'score_adjustment': +15,
                'description': f'Early reversal: 3D bearish but recovering (slope {slope_3d:.0f}), LTF bullish ({ltf_bullish_count}/3)'
            }
        
        # HTF bearish but LTF bullish (without slope confirmation)
        if trend_3d == 'BEARISH' and ltf_bullish_count >= 2:
            return {
                'type': 'WEAK_BULLISH_DIV',
                'score_adjustment': +8,
                'description': f'HTF bearish vs LTF bullish ({ltf_bullish_count}/3), watch for reversal'
            }
        
        # ========== MOMENTUM WARNINGS ==========
        # HTF bullish but slope negative (early warning)
        if trend_3d == 'BULLISH' and slope_3d < -100:
            return {
                'type': 'MOMENTUM_WARNING_BEAR',
                'score_adjustment': -5,
                'description': f'3D bullish but momentum weakening (slope {slope_3d:.0f})'
            }
        
        # HTF bearish but slope positive (early recovery)
        if trend_3d == 'BEARISH' and slope_3d > 100:
            return {
                'type': 'MOMENTUM_WARNING_BULL',
                'score_adjustment': +5,
                'description': f'3D bearish but momentum recovering (slope {slope_3d:.0f})'
            }
        
        # No divergence detected
        return {
            'type': 'CONFLUENCE',
            'score_adjustment': 0,
            'description': 'Timeframes aligned'
        }

    def _check_confluence(self, mtf_data: Dict[str, Dict]) -> str:

        """
        Checks if all timeframes are aligned in the same direction.
        
        Returns:
            'ALL_BULLISH', 'ALL_BEARISH', or 'MIXED'
        """
        trends = [data['trend'] for data in mtf_data.values() if data.get('available')]
        
        if not trends:
            return 'UNKNOWN'
        
        if all(t == 'BULLISH' for t in trends):
            return 'ALL_BULLISH'
        elif all(t == 'BEARISH' for t in trends):
            return 'ALL_BEARISH'
        else:
            return 'MIXED'
