"""
Hyperliquid Advanced Analyzer
Récupère des données avancées depuis Hyperliquid (DEX avec gros comptes)

Données récupérées:
- Open Interest BTC
- Funding rate et historique
- Positions des gros comptes (whale tracking)
- Liquidations récentes
"""
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import json


class HyperliquidAnalyzer:
    """
    Analyseur avancé pour Hyperliquid
    
    Features:
    - Open Interest en temps réel
    - Funding rate et comparaison avec CEXs
    - Tracking des gros comptes (top traders)
    - Analyse des liquidations
    """
    
    BASE_URL = "https://api.hyperliquid.xyz/info"
    
    # Adresses connues de gros comptes Hyperliquid avec bon winrate
    # Source: données utilisateur vérifiées
    WHALE_ADDRESSES = [
        # Tier 1 - Top Whales
        "0xb83de012dba672c76a7dbbbf3e459cb59d7d6e36",
        "0xb317d2bc2d3d2df5fa441b5bae0ab9d8b07283ae",
        "0xc2a30212a8ddac9e123944d6e29faddce994e5f2",
        "0x4f9a37bc2a4a2861682c0e9be1f9417df03cc27c",
        "0x2ea18c23f72a4b6172c55b411823cdc5335923f4",
        "0x3e10864b0efa14994c350ed247c815966a8fd962",
        "0x3e10d813db3374877e2074cb8126066ca7dad962",
        "0x952044eb3c860b00778ea414670a6c8c22f84ac6",
        "0x99E1E710fAf2EA090E5cFA5A600c1478031640be",
        "0xEe7aE85f2Fe2239E27D9c1E23fFFe168D63b4055",
        # Tier 2 - High Winrate Traders
        "0xa5b0edf6b55128e0ddae8e51ac538c3188401d41",
        "0x9c89f595f5515609ad61f6fda94beff85ae6600e",
        "0x3c363e96d22c056d748f199fb728fc80d70e461a",
        "0xc26cbb6483229e0d0f9a1cab675271eda535b8f4",
        "0x175e7023e8dc93d0c044852685ac33e856b577b4",
        "0x0e41eb80e9a39ae7b887a94f6a88f6c791e26359",
        "0x0ddf9bae2af4b874b96d287a5ad42eb47138a902",
        "0xf97ad6704baec104d00b88e0c157e2b7b3a1ddd1",
        "0xefffa330cbae8d916ad1d33f0eeb16cfa711fa91",
        "0x6f1d35664eab0efa5a796091c28d14f1472d3162",
        "0xe0665019fcf5ffa4a5aea5cf65f84016fcbe78da",
        "0x4b66f4048a0a90fd5ff44abbe5d68332656b78b8",
        "0xa4add8273d7f47318675bdfbcce3e9648cdb4509",
        "0xfd97600ac44b3c4e20ac1a5f23e3b18d10fa5912",
        "0x7583b5364597736a6a72c4ba61ede0a4a37ef4de",
        "0x06cecfbac34101ae41c88ebc2450f8602b3d164b",
        "0x41206f8efb51e5039e5b46e04e7866a4849f72d2",
        "0x7fdafde5cfb5465924316eced2d3715494c517d1",
        # Tier 3 - Additional High Performers
        "0x418aa6bf98a2b2bc93779f810330d88cde488888",
        "0x7ca165f354e3260e2f8d5a7508cc9dd2fa009235",
        "0x2c76be702ee99922754a6df7158091a5e33f7662",
        "0x3e5dacb70247b57aca1d62b927c398ff05b7e570",
        "0x0284bbd3646b59740a167ef78a306028343f3806",
        "0x92b585bdf2d67c0fe321108b863ca4617dd39fe9",
        "0xad572a7894c7b0ba4db67c2a7602dd3376d4f094",
        "0x11eee2e0a613af4f636e23ff295e2dac6a191d1d",
    ]
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def _post(self, data: Dict) -> Any:
        """Execute POST request to Hyperliquid API"""
        try:
            response = self.session.post(self.BASE_URL, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"   ⚠️ Hyperliquid API error: {e}")
            return None
    
    def get_market_data(self) -> Dict[str, Any]:
        """
        Récupère les données de marché BTC
        - Prix, OI, Funding, etc.
        """
        result = {
            'exchange': 'hyperliquid',
            'coin': 'BTC',
            'success': True
        }
        
        # 1. Meta + Asset Contexts (OI, funding, etc.)
        data = self._post({'type': 'metaAndAssetCtxs'})
        if data and len(data) > 1:
            meta = data[0]  # Universe meta
            contexts = data[1]  # Asset contexts
            
            # BTC is usually first, but let's find it
            btc_index = None
            for i, coin in enumerate(meta.get('universe', [])):
                if coin.get('name') == 'BTC':
                    btc_index = i
                    break
            
            if btc_index is not None and btc_index < len(contexts):
                ctx = contexts[btc_index]
                result['open_interest_btc'] = float(ctx.get('openInterest', 0))
                result['funding_rate'] = float(ctx.get('funding', 0))
                result['funding_rate_pct'] = float(ctx.get('funding', 0)) * 100
                result['mark_price'] = float(ctx.get('markPx', 0))
                result['oracle_price'] = float(ctx.get('oraclePx', 0))
                result['premium_pct'] = float(ctx.get('premium', 0)) * 100
                result['prev_day_volume'] = float(ctx.get('dayNtlVlm', 0))
                result['max_leverage'] = meta['universe'][btc_index].get('maxLeverage', 50)
        
        # 2. All mids pour le prix actuel
        mids = self._post({'type': 'allMids'})
        if mids and 'BTC' in mids:
            result['mid_price'] = float(mids['BTC'])
        
        return result
    
    def get_whale_positions(self) -> Dict[str, Any]:
        """
        Récupère les positions des gros comptes
        Analyse le sentiment global des whales
        """
        positions = []
        total_long = 0.0
        total_short = 0.0
        whale_count = 0
        
        for address in self.WHALE_ADDRESSES:
            try:
                data = self._post({
                    'type': 'clearinghouseState',
                    'user': address
                })
                
                if not data or 'assetPositions' not in data:
                    continue
                
                account_value = float(data.get('marginSummary', {}).get('accountValue', 0))
                
                # Chercher la position BTC
                for pos in data['assetPositions']:
                    if pos.get('position', {}).get('coin') == 'BTC':
                        position = pos['position']
                        size = float(position.get('szi', 0))
                        entry_price = float(position.get('entryPx', 0))
                        unrealized_pnl = float(position.get('unrealizedPnl', 0))
                        leverage = float(position.get('leverage', {}).get('value', 1))
                        
                        if size != 0:
                            whale_count += 1
                            positions.append({
                                'address': address[:10] + '...',
                                'size_btc': size,
                                'direction': 'LONG' if size > 0 else 'SHORT',
                                'entry_price': entry_price,
                                'unrealized_pnl': unrealized_pnl,
                                'leverage': leverage,
                                'account_value': account_value
                            })
                            
                            if size > 0:
                                total_long += abs(size)
                            else:
                                total_short += abs(size)
                        break
                        
            except Exception as e:
                continue
        
        # Calculer le ratio et le sentiment
        total_size = total_long + total_short
        long_ratio = (total_long / total_size * 100) if total_size > 0 else 50
        
        if long_ratio > 65:
            sentiment = "STRONG_LONG"
            emoji = "🐋📈"
        elif long_ratio > 55:
            sentiment = "SLIGHTLY_LONG"
            emoji = "🟢"
        elif long_ratio < 35:
            sentiment = "STRONG_SHORT"
            emoji = "🐋📉"
        elif long_ratio < 45:
            sentiment = "SLIGHTLY_SHORT"
            emoji = "🔴"
        else:
            sentiment = "NEUTRAL"
            emoji = "⚪"
        
        return {
            'whale_count': whale_count,
            'positions': positions,
            'total_long_btc': round(total_long, 4),
            'total_short_btc': round(total_short, 4),
            'long_ratio_pct': round(long_ratio, 1),
            'sentiment': sentiment,
            'emoji': emoji,
            'signal_modifier': self._calculate_signal_modifier(long_ratio)
        }
    
    def _calculate_signal_modifier(self, long_ratio: float) -> int:
        """
        Calcule le modificateur de signal basé sur le sentiment whale
        Range: -5 à +5
        """
        if long_ratio > 70:
            return 5
        elif long_ratio > 60:
            return 3
        elif long_ratio > 55:
            return 1
        elif long_ratio < 30:
            return -5
        elif long_ratio < 40:
            return -3
        elif long_ratio < 45:
            return -1
        return 0
    
    def get_funding_comparison(self, cex_funding: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Compare le funding Hyperliquid avec les CEX
        Un écart important peut indiquer une opportunité
        """
        market = self.get_market_data()
        hl_funding = market.get('funding_rate', 0)
        
        result = {
            'hyperliquid_funding': round(hl_funding * 100, 4),
            'hyperliquid_premium': market.get('premium_pct', 0),
        }
        
        if cex_funding:
            # Calculer l'écart moyen
            cex_avg = sum(cex_funding.values()) / len(cex_funding) if cex_funding else 0
            result['cex_avg_funding'] = round(cex_avg * 100, 4)
            result['funding_divergence'] = round((hl_funding - cex_avg) * 100, 4)
            
            # Signal si divergence importante
            if abs(result['funding_divergence']) > 0.01:  # >0.01% divergence
                result['arbitrage_opportunity'] = True
                if result['funding_divergence'] > 0:
                    result['strategy'] = "Short HL, Long CEX"
                else:
                    result['strategy'] = "Long HL, Short CEX"
            else:
                result['arbitrage_opportunity'] = False
        
        return result
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse complète Hyperliquid
        """
        print("   🔷 Hyperliquid: ", end='')
        
        market = self.get_market_data()
        whale = self.get_whale_positions()
        
        # OI en valeur USD
        oi_btc = market.get('open_interest_btc', 0)
        mid_price = market.get('mid_price', 0)
        oi_usd = oi_btc * mid_price if mid_price else 0
        
        print(f"OI {oi_btc:.0f} BTC | Whales {whale['sentiment']} ({whale['whale_count']})")
        
        return {
            'success': market.get('success', False),
            'market': {
                'mid_price': market.get('mid_price'),
                'oracle_price': market.get('oracle_price'),
                'mark_price': market.get('mark_price'),
                'open_interest_btc': oi_btc,
                'open_interest_usd': round(oi_usd, 0),
                'funding_rate_pct': market.get('funding_rate_pct', 0),
                'premium_pct': market.get('premium_pct', 0),
                'volume_24h': market.get('prev_day_volume', 0),
                'max_leverage': market.get('max_leverage', 50)
            },
            'whale_analysis': whale,
            'signal_modifier': whale.get('signal_modifier', 0),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def test_hyperliquid():
    """Test de l'analyseur"""
    analyzer = HyperliquidAnalyzer()
    result = analyzer.analyze()
    
    print("\n=== Résultat ===")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    test_hyperliquid()
