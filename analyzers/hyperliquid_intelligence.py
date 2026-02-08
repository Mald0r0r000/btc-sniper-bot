"""
Hyperliquid Intelligence Analyzer
Provides real-time whale position tracking, liquidation risk analysis,
and smart money flow indicators from Hyperliquid.

Features:
- Position Near Liquidation tracking
- Smart Money Flow sentiment
- Liquidation Cascade prediction

All data from free Hyperliquid public API.
"""
import json
import os
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


@dataclass
class WhalePosition:
    """Single whale position with liquidation data"""
    address: str
    coin: str
    size: float
    direction: str  # LONG or SHORT
    entry_price: float
    liquidation_price: float
    leverage: float
    unrealized_pnl: float
    account_value: float
    distance_to_liq_pct: float


class HyperliquidIntelligence:
    """
    Aggregates whale intelligence from Hyperliquid.
    Tracks positions near liquidation and smart money sentiment.
    """
    
    BASE_URL = "https://api.hyperliquid.xyz/info"
    
    # High-winrate whale addresses - Verified Hyperliquid traders
    # Updated 2026-02: Expanded list with active high-volume traders
    WHALE_ADDRESSES = [
        # === Tier 1: Mega Whales ($10M+ volume/week) ===
        "0x418aa6bf98a2b2bc93779f810330d88cde488888",  # $33.9M vol/7d, $3.7M PnL
        "0xb83de012dba672c76a7dbbf3e459cb59d7d6e361",
        "0xb317d2bc2d3d2df5fa441b5bae0ab9d8b07283ae",
        "0xc2a30212a8ddac9e123944d6e29faddce994e5f2",
        "0x4f9a37bc2a4a2861682c0e9be1f9417df03cc27c",
        "0x2ea18c23f72a4b6172c55b411823cdc5339f23f4",
        "0x3e10864b0efa14994c350ed247c815966a8fd962",
        "0x952044eb3c860b00778ea41467d0a6c8c22f84c6",
        
        # === Tier 2: High Winrate Traders (>60% WR) ===
        "0xa5b0edf6b55128e0ddae8e51ac538c3188401d41",
        "0x3c363e96d22c056d748f199fb728fc80d70e461a",
        "0xc26cbb6483229e0d0f9a1cab675271eda535b8f4",
        "0x175e7023e8dc93d0c044852685ac33e856b577b4",
        "0x0e41eb80e9a39ae7b887a94f6a88f6c791e26359",
        "0x0ddf9bae2af4b874b96d287a5ad42eb47138a902",
        "0xf97ad6704baec104d00b88e0c157e2b7b3a1ddd1",
        "0xefffa330cbae8d916ad1d33f0eeb16cfa711fa91",
        "0x6f1d35664eab0efa5a796091c28d14f1472d3162",
        
        # === Tier 3: Verified Active Performers ===
        "0x2c76be702ee99922754a6df71580091a5e33f762",
        "0x3e5dacb70247b57aca1d62b927c398ff05b7e570",
        "0x0284bbd3646b59740a167ef78a306028343f3806",
        "0x92b585bdf2d67c0fe321108b863ca4617dd39fe9",
        "0xad572a7894c7b0ba4db67c2a7602dd3376d4f094",
        "0x11eee2e0a613af4f636e23ff295e2dac6a191d1d",
        
        # === Tier 4: Additional Known Traders (from whale-tracker) ===
        "0x96fe5b76e6796bf2d3e6d6d39743234959ae384b",
        "0x31ca8395cf837de08b24da3f660e77761dfb974b",
        "0xf9109ada2f73c62e9889b45453065f0d99260a2d",
        "0xd4c1f7e8d876c4749228d515473d36f919583d1d",
        "0xee772e29e31b9972e1b683b04944bd9937ac0304",
        "0xff4cd3826ecee12acd4329aada4a2d3419fc463c",
        
        # === Tier 5: Recent High-Volume Trades (discovered via API) ===
        "0x7ba05f5a774a08b27d1a0434ee2ce902051f0040",
        "0x24ff31986cf6716d26780434ee2dd10203aa007e",
    ]
    
    def __init__(self, gist_token: str = None, gist_id: str = None):
        self.gist_token = gist_token or os.getenv('GIST_TOKEN')
        self.gist_id = gist_id or os.getenv('GIST_ID')
        self._cache = {}
        self._cache_time = None
    
    def _post(self, data: Dict) -> Optional[Any]:
        """Execute POST request to Hyperliquid API"""
        try:
            req = urllib.request.Request(
                self.BASE_URL,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            print(f"⚠️ Hyperliquid API error: {e}")
            return None
    
    def get_btc_price(self) -> float:
        """Get current BTC price from Hyperliquid"""
        data = self._post({"type": "allMids"})
        if data and 'BTC' in data:
            return float(data['BTC'])
        return 0.0
    
    def get_whale_position(self, address: str, coin: str = "BTC") -> Optional[WhalePosition]:
        """
        Get a whale's position with liquidation data.
        Returns None if no position or on error.
        """
        data = self._post({
            'type': 'clearinghouseState',
            'user': address
        })
        
        if not data or 'assetPositions' not in data:
            return None
        
        account_value = float(data.get('marginSummary', {}).get('accountValue', 0))
        btc_price = self.get_btc_price()
        
        for pos in data['assetPositions']:
            position = pos.get('position', {})
            if position.get('coin') == coin:
                size = float(position.get('szi', 0) or 0)
                if size == 0:
                    continue
                
                # Handle None values safely
                liq_px_raw = position.get('liquidationPx')
                entry_px_raw = position.get('entryPx')
                liq_price = float(liq_px_raw) if liq_px_raw else 0.0
                entry_price = float(entry_px_raw) if entry_px_raw else 0.0
                
                # Calculate distance to liquidation
                if liq_price > 0 and btc_price > 0:
                    distance_pct = abs(btc_price - liq_price) / btc_price * 100
                else:
                    distance_pct = 100.0  # Safe default (no liq price = not at risk)
                
                return WhalePosition(
                    address=address,
                    coin=coin,
                    size=size,
                    direction='LONG' if size > 0 else 'SHORT',
                    entry_price=entry_price,
                    liquidation_price=liq_price,
                    leverage=float(position.get('leverage', {}).get('value', 1)),
                    unrealized_pnl=float(position.get('unrealizedPnl', 0)),
                    account_value=account_value,
                    distance_to_liq_pct=round(distance_pct, 2)
                )
        
        return None
    
    def get_all_whale_positions(self, coin: str = "BTC") -> List[WhalePosition]:
        """Fetch all whale positions for a given coin"""
        positions = []
        
        for address in self.WHALE_ADDRESSES:
            pos = self.get_whale_position(address, coin)
            if pos:
                positions.append(pos)
        
        return positions
    
    def get_positions_near_liquidation(
        self, 
        coin: str = "BTC", 
        max_distance_pct: float = 15.0
    ) -> List[WhalePosition]:
        """
        Get whale positions within X% of their liquidation price.
        These are the most at-risk positions that could trigger cascades.
        """
        all_positions = self.get_all_whale_positions(coin)
        
        at_risk = [
            pos for pos in all_positions 
            if pos.distance_to_liq_pct <= max_distance_pct
        ]
        
        # Sort by distance (closest to liq first)
        at_risk.sort(key=lambda x: x.distance_to_liq_pct)
        
        return at_risk
    
    def get_smart_money_sentiment(self, coin: str = "BTC") -> Dict[str, Any]:
        """
        Calculate aggregate whale sentiment.
        Returns long/short ratio and sentiment classification.
        """
        positions = self.get_all_whale_positions(coin)
        
        if not positions:
            return {
                'whale_count': 0,
                'sentiment': 'UNKNOWN',
                'long_ratio_pct': 50.0,
                'total_long': 0.0,
                'total_short': 0.0,
                'avg_leverage_long': 0.0,
                'avg_leverage_short': 0.0
            }
        
        total_long = sum(p.size for p in positions if p.direction == 'LONG')
        total_short = abs(sum(p.size for p in positions if p.direction == 'SHORT'))
        
        long_positions = [p for p in positions if p.direction == 'LONG']
        short_positions = [p for p in positions if p.direction == 'SHORT']
        
        avg_lev_long = sum(p.leverage for p in long_positions) / len(long_positions) if long_positions else 0
        avg_lev_short = sum(p.leverage for p in short_positions) / len(short_positions) if short_positions else 0
        
        total_size = total_long + total_short
        long_ratio = (total_long / total_size * 100) if total_size > 0 else 50.0
        
        # Classify sentiment
        if long_ratio >= 70:
            sentiment = 'STRONG_LONG'
        elif long_ratio >= 55:
            sentiment = 'SLIGHTLY_LONG'
        elif long_ratio <= 30:
            sentiment = 'STRONG_SHORT'
        elif long_ratio <= 45:
            sentiment = 'SLIGHTLY_SHORT'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'whale_count': len(positions),
            'sentiment': sentiment,
            'long_ratio_pct': round(long_ratio, 1),
            'total_long': round(total_long, 2),
            'total_short': round(total_short, 2),
            'avg_leverage_long': round(avg_lev_long, 1),
            'avg_leverage_short': round(avg_lev_short, 1),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_liquidation_clusters(
        self, 
        coin: str = "BTC",
        price_zone_pct: float = 1.0
    ) -> List[Dict]:
        """
        Identify clusters of liquidations at similar price levels.
        A cluster with high BTC volume indicates cascade risk.
        """
        positions = self.get_all_whale_positions(coin)
        btc_price = self.get_btc_price()
        
        if not positions or btc_price == 0:
            return []
        
        # Group by price zones
        clusters = {}
        
        for pos in positions:
            liq_price = pos.liquidation_price
            # Normalize to nearest zone (e.g., 1% bands)
            zone_key = round(liq_price / (btc_price * price_zone_pct / 100)) * (btc_price * price_zone_pct / 100)
            zone_key = round(zone_key, -2)  # Round to nearest 100
            
            if zone_key not in clusters:
                clusters[zone_key] = {
                    'price_level': zone_key,
                    'side': pos.direction,
                    'total_btc': 0.0,
                    'wallet_count': 0,
                    'positions': [],
                    'avg_distance_pct': 0.0
                }
            
            clusters[zone_key]['total_btc'] += abs(pos.size)
            clusters[zone_key]['wallet_count'] += 1
            clusters[zone_key]['positions'].append(pos)
        
        # Calculate averages and risk level
        result = []
        for zone_key, cluster in clusters.items():
            if cluster['total_btc'] < 1.0:  # Ignore tiny clusters
                continue
            
            avg_dist = sum(p.distance_to_liq_pct for p in cluster['positions']) / len(cluster['positions'])
            cluster['avg_distance_pct'] = round(avg_dist, 1)
            
            # Determine cascade risk
            if cluster['total_btc'] >= 50 and avg_dist <= 5:
                cluster['cascade_risk'] = 'CRITICAL'
            elif cluster['total_btc'] >= 20 and avg_dist <= 10:
                cluster['cascade_risk'] = 'HIGH'
            elif cluster['total_btc'] >= 10:
                cluster['cascade_risk'] = 'MEDIUM'
            else:
                cluster['cascade_risk'] = 'LOW'
            
            # Remove position objects for cleaner output
            del cluster['positions']
            cluster['total_btc'] = round(cluster['total_btc'], 2)
            
            result.append(cluster)
        
        # Sort by cascade risk severity
        risk_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        result.sort(key=lambda x: (risk_order.get(x['cascade_risk'], 4), -x['total_btc']))
        
        return result
    
    def get_full_intelligence_report(self, coin: str = "BTC") -> Dict[str, Any]:
        """
        Generate complete intelligence report combining all data sources.
        This is the main method to call from the decision engine.
        """
        print("🔍 Fetching Hyperliquid whale intelligence...")
        
        btc_price = self.get_btc_price()
        sentiment = self.get_smart_money_sentiment(coin)
        at_risk = self.get_positions_near_liquidation(coin, max_distance_pct=15.0)
        clusters = self.get_liquidation_clusters(coin)
        
        # Count positions by side near liquidation
        longs_at_risk = [p for p in at_risk if p.direction == 'LONG']
        shorts_at_risk = [p for p in at_risk if p.direction == 'SHORT']
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'btc_price': btc_price,
            'smart_money': sentiment,
            'liquidation_risk': {
                'positions_at_risk': len(at_risk),
                'longs_at_risk': len(longs_at_risk),
                'shorts_at_risk': len(shorts_at_risk),
                'total_btc_at_risk': round(sum(abs(p.size) for p in at_risk), 2),
                'closest_long_liq': longs_at_risk[0].liquidation_price if longs_at_risk else None,
                'closest_short_liq': shorts_at_risk[0].liquidation_price if shorts_at_risk else None,
            },
            'cascade_clusters': clusters[:5],  # Top 5 most dangerous clusters
            'signal_modifiers': self._calculate_signal_modifiers(sentiment, at_risk, clusters, btc_price)
        }
        
        print(f"   📊 Whales: {sentiment['whale_count']} | Sentiment: {sentiment['sentiment']} | At Risk: {len(at_risk)}")
        
        return report
    
    def _calculate_signal_modifiers(
        self, 
        sentiment: Dict, 
        at_risk: List[WhalePosition],
        clusters: List[Dict],
        btc_price: float
    ) -> Dict[str, Any]:
        """
        Calculate signal modifiers for Decision Engine integration.
        Returns score adjustments and potential vetoes.
        """
        modifiers = {
            'score_adjustment': 0,
            'confidence_boost': 0,
            'veto_long': False,
            'veto_short': False,
            'veto_reason': None
        }
        
        # 1. Smart money sentiment adjustment (-10 to +10)
        if sentiment['sentiment'] == 'STRONG_LONG':
            modifiers['score_adjustment'] += 8
        elif sentiment['sentiment'] == 'SLIGHTLY_LONG':
            modifiers['score_adjustment'] += 4
        elif sentiment['sentiment'] == 'STRONG_SHORT':
            modifiers['score_adjustment'] -= 8
        elif sentiment['sentiment'] == 'SLIGHTLY_SHORT':
            modifiers['score_adjustment'] -= 4
        
        # 2. Leverage imbalance (high leverage on one side = squeeze risk)
        if sentiment['avg_leverage_long'] > 20 and sentiment['avg_leverage_short'] < 10:
            modifiers['veto_long'] = True
            modifiers['veto_reason'] = "High long leverage imbalance - squeeze risk"
        elif sentiment['avg_leverage_short'] > 20 and sentiment['avg_leverage_long'] < 10:
            modifiers['veto_short'] = True
            modifiers['veto_reason'] = "High short leverage imbalance - squeeze risk"
        
        # 3. Critical cascade clusters near current price
        for cluster in clusters:
            if cluster['cascade_risk'] in ['CRITICAL', 'HIGH']:
                distance_from_price = abs(cluster['price_level'] - btc_price) / btc_price * 100
                if distance_from_price < 5:  # Within 5% of current price
                    if cluster['side'] == 'LONG':
                        modifiers['confidence_boost'] -= 10  # Bearish catalyst nearby
                    else:
                        modifiers['confidence_boost'] += 10  # Bullish catalyst nearby
        
        # 4. Positions at immediate risk (< 3% from liq)
        immediate_risk = [p for p in at_risk if p.distance_to_liq_pct < 3]
        if len(immediate_risk) >= 3:
            modifiers['confidence_boost'] += 5  # High volatility expected
        
        return modifiers


# Standalone test
if __name__ == "__main__":
    intel = HyperliquidIntelligence()
    report = intel.get_full_intelligence_report("BTC")
    print(json.dumps(report, indent=2, default=str))
