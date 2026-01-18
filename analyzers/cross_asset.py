"""
Cross-Asset Analyzer
Tracks DXY (Dollar Index), SPX (S&P500), and M2 Money Supply
to understand macro environment and correlations with BTC
"""
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd

# Lazy imports to avoid crashes if deps missing
def _get_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        return None

def _get_fred():
    try:
        from fredapi import Fred
        api_key = os.getenv('FRED_API_KEY')
        if api_key:
            return Fred(api_key=api_key)
        return None
    except ImportError:
        return None


class CrossAssetAnalyzer:
    """
    Analyzes cross-asset correlations and macro environment.
    
    DXY (Dollar Index):
    - Inverse correlation with BTC
    - Strong dollar = Bearish BTC
    - Weak dollar = Bullish BTC
    
    SPX (S&P500):
    - Risk-on/Risk-off indicator
    - SPX up = Risk-on = Bullish BTC
    - SPX down = Risk-off = Bearish BTC
    
    M2 Money Supply:
    - Leading indicator with ~90 day offset
    - M2 expansion → Bullish BTC (liquidity incoming)
    - M2 contraction → Bearish BTC (liquidity draining)
    """
    
    # Tickers
    DXY_TICKER = "DX-Y.NYB"  # Dollar Index Futures
    SPX_TICKER = "^GSPC"     # S&P 500
    M2_SERIES = "M2SL"       # M2 Money Stock (FRED)
    
    # Offset for M2 correlation (days)
    M2_OFFSET_DAYS = 90
    
    def __init__(self):
        self.yf = _get_yfinance()
        self.fred = _get_fred()
        
    def _fetch_ticker_data(self, ticker: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        if not self.yf:
            return None
        try:
            data = self.yf.download(ticker, period=period, progress=False, auto_adjust=True)
            return data
        except Exception as e:
            print(f"   ⚠️ Failed to fetch {ticker}: {e}")
            return None
    
    def _analyze_dxy(self) -> Dict[str, Any]:
        """Analyze Dollar Index"""
        default = {
            'value': 0,
            'change_24h': 0,
            'trend': 'UNKNOWN',
            'btc_impact': 'NEUTRAL',
            'available': False
        }
        
        data = self._fetch_ticker_data(self.DXY_TICKER)
        if data is None or len(data) < 2:
            return default
            
        current = data['Close'].iloc[-1].item()
        prev = data['Close'].iloc[-2].item()
        change = ((current - prev) / prev) * 100
        
        # 5-day trend
        first = data['Close'].iloc[0].item()
        trend_pct = ((current - first) / first) * 100
        
        if trend_pct > 0.5:
            trend = 'BULLISH'
            btc_impact = 'BEARISH'  # Strong dollar = Bad for BTC
        elif trend_pct < -0.5:
            trend = 'BEARISH'
            btc_impact = 'BULLISH'  # Weak dollar = Good for BTC
        else:
            trend = 'NEUTRAL'
            btc_impact = 'NEUTRAL'
            
        return {
            'value': round(current, 2),
            'change_24h': round(change, 2),
            'trend': trend,
            'btc_impact': btc_impact,
            'available': True
        }
    
    def _analyze_spx(self) -> Dict[str, Any]:
        """Analyze S&P 500"""
        default = {
            'value': 0,
            'change_24h': 0,
            'trend': 'UNKNOWN',
            'btc_impact': 'NEUTRAL',
            'market_open': False,
            'available': False
        }
        
        data = self._fetch_ticker_data(self.SPX_TICKER)
        if data is None or len(data) < 2:
            return default
            
        current = data['Close'].iloc[-1].item()
        prev = data['Close'].iloc[-2].item()
        change = ((current - prev) / prev) * 100
        
        # Check if market is open (rough check based on last data timestamp)
        last_date = data.index[-1].date() if hasattr(data.index[-1], 'date') else None
        today = datetime.now().date()
        market_open = last_date == today if last_date else False
        
        # Weekend check
        if today.weekday() >= 5:  # Saturday or Sunday
            market_open = False
        
        # 5-day trend
        first = data['Close'].iloc[0].item()
        trend_pct = ((current - first) / first) * 100
        
        if trend_pct > 1.0:
            trend = 'BULLISH'
            btc_impact = 'BULLISH'  # Risk-on
        elif trend_pct < -1.0:
            trend = 'BEARISH'
            btc_impact = 'BEARISH'  # Risk-off
        else:
            trend = 'NEUTRAL'
            btc_impact = 'NEUTRAL'
            
        return {
            'value': round(float(current), 2),
            'change_24h': round(float(change), 2),
            'trend': trend,
            'btc_impact': btc_impact,
            'market_open': market_open,
            'available': True
        }
    
    def _analyze_m2(self) -> Dict[str, Any]:
        """
        Analyze M2 Money Supply with 90-day offset
        M2 data is released monthly, so we compare YoY changes
        """
        default = {
            'current': 0,
            'yoy_change': 0,
            'offset_90d_trend': 'UNKNOWN',
            'btc_impact': 'NEUTRAL',
            'available': False
        }
        
        if not self.fred:
            return default
            
        try:
            # Get M2 data (last 2 years for YoY comparison)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years
            
            m2_data = self.fred.get_series(
                self.M2_SERIES,
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )
            
            if m2_data is None or len(m2_data) < 13:
                return default
                
            # Current value (most recent)
            current = m2_data.iloc[-1]
            
            # Value 12 months ago
            year_ago = m2_data.iloc[-13] if len(m2_data) >= 13 else m2_data.iloc[0]
            yoy_change = ((current - year_ago) / year_ago) * 100
            
            # 60-95 day offset analysis: Compare M2 from 2-3 months ago to 3-4 months ago
            # This represents the "liquidity wave" hitting BTC now (60-95 day lag)
            if len(m2_data) >= 5:
                # M2 data is monthly, so:
                # -3 = ~2 months ago (~60 days)
                # -4 = ~3 months ago (~90 days)
                # -5 = ~4 months ago (~120 days)
                # We average the 2-3 month window and compare to 3-4 month window
                m2_recent = (m2_data.iloc[-3] + m2_data.iloc[-4]) / 2  # 60-90 day avg
                m2_previous = (m2_data.iloc[-4] + m2_data.iloc[-5]) / 2  # 90-120 day avg
                
                offset_change = ((m2_recent - m2_previous) / m2_previous) * 100
                
                if offset_change > 0.5:
                    offset_trend = 'EXPANDING'
                    btc_impact = 'BULLISH'  # Liquidity was growing, should help BTC now
                elif offset_change < -0.5:
                    offset_trend = 'CONTRACTING'
                    btc_impact = 'BEARISH'  # Liquidity was shrinking, headwind for BTC
                else:
                    offset_trend = 'STABLE'
                    btc_impact = 'NEUTRAL'
            else:
                offset_trend = 'UNKNOWN'
                btc_impact = 'NEUTRAL'
                
            return {
                'current': round(current, 1),
                'yoy_change': round(yoy_change, 2),
                'offset_90d_trend': offset_trend,
                'btc_impact': btc_impact,
                'available': True
            }
            
        except Exception as e:
            print(f"   ⚠️ Failed to fetch M2 data: {e}")
            return default
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run full cross-asset analysis
        """
        dxy = self._analyze_dxy()
        spx = self._analyze_spx()
        m2 = self._analyze_m2()
        
        # Determine overall regime
        bullish_count = sum([
            1 if dxy.get('btc_impact') == 'BULLISH' else 0,
            1 if spx.get('btc_impact') == 'BULLISH' else 0,
            1 if m2.get('btc_impact') == 'BULLISH' else 0
        ])
        
        bearish_count = sum([
            1 if dxy.get('btc_impact') == 'BEARISH' else 0,
            1 if spx.get('btc_impact') == 'BEARISH' else 0,
            1 if m2.get('btc_impact') == 'BEARISH' else 0
        ])
        
        if bullish_count >= 2:
            overall_regime = 'RISK_ON'
            regime_btc_impact = 'BULLISH'
        elif bearish_count >= 2:
            overall_regime = 'RISK_OFF'
            regime_btc_impact = 'BEARISH'
        else:
            overall_regime = 'MIXED'
            regime_btc_impact = 'NEUTRAL'
            
        return {
            'dxy': dxy,
            'spx': spx,
            'm2': m2,
            'overall_regime': overall_regime,
            'regime_btc_impact': regime_btc_impact,
            'bullish_indicators': bullish_count,
            'bearish_indicators': bearish_count
        }
