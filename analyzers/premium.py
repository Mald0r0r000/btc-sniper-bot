"""
Premium Analyzer (Spot vs Perp)
Analyzes the price gap between Coinbase Spot (Institutional Proxy) and Perpetual Futures.
"""
from typing import Dict, Any, List

class PremiumAnalyzer:
    """
    Analyzes the Coinbase Premium Index and other Spot/Perp price dislocations.
    """
    
    def __init__(self):
        pass
        
    def analyze(self, tickers: Dict[str, Dict], 
                spot_price_override: float = None,
                perp_price_override: float = None) -> Dict[str, Any]:
        """
        Calculs le Premium Gap.
        
        Args:
            tickers: Dict of tickers from ExchangeAggregator
            spot_price_override: Forced price for Spot (e.g. from OHLCV)
            perp_price_override: Forced price for Perp (e.g. from OHLCV)
        """
        # 1. Identify Spot Reference (Coinbase > Kraken > Bybit Spot)
        spot_ref = self._get_price(tickers, ['coinbase', 'kraken', 'bybit'])
        
        # Override if needed
        if spot_price_override:
            spot_ref = {"exchange": "OVERRIDE", "price": spot_price_override}
            
        # 2. Identify Perp Reference (Binance > Bitget > Bybit Perp)
        perp_ref = self._get_price(tickers, ['binance', 'bitget', 'bybit', 'okx'])
        
        # Override if needed
        if perp_price_override:
            perp_ref = {"exchange": "OVERRIDE", "price": perp_price_override}
            
        if not spot_ref['price'] or not perp_ref['price']:
            return self._empty_result()
            
        # 3. Calculate Gap
        # Premium = Spot - Perp
        # Positive = Spot Premium (Insti Buying)
        # Negative = Spot Discount (Insti Selling)
        gap_usd = spot_ref['price'] - perp_ref['price']
        gap_pct = (gap_usd / spot_ref['price']) * 100
        
        # 4. Signal Generation
        # Thresholds: +/- 0.05% (approx $50 on $100k) is significant intraday
        signal = "NEUTRAL"
        if gap_pct > 0.02:
            signal = "PREMIUM_BUY_PRESSURE"
        elif gap_pct < -0.02:
            signal = "DISCOUNT_SELL_PRESSURE"
            
        return {
            "gap_usd": round(gap_usd, 2),
            "gap_pct": round(gap_pct, 4),
            "signal": signal,
            "spot_ref": spot_ref,
            "perp_ref": perp_ref,
            "valid": True
        }
    
    def _get_price(self, tickers: Dict[str, Dict], priority_list: List[str]) -> Dict[str, Any]:
        """Tries to find the first available price from the priority list"""
        for ex in priority_list:
            if ex in tickers and tickers[ex].get('success') and tickers[ex].get('last', 0) > 0:
                return {
                    "exchange": ex,
                    "price": tickers[ex]['last']
                }
        return {"exchange": None, "price": 0}

    def _empty_result(self):
        return {
            "gap_usd": 0,
            "gap_pct": 0,
            "signal": "NO_DATA",
            "spot_ref": {"exchange": None, "price": 0},
            "perp_ref": {"exchange": None, "price": 0},
            "valid": False
        }
