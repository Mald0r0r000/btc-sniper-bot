"""
Whale Tracker - Hyperliquid Position Monitoring & Winrate Analysis
Surveille les positions des whales et envoie des notifications Telegram

Features:
- Calcul du winrate BTC par wallet
- Détection de nouvelles positions
- Notifications Telegram lors d'ouvertures
"""
import requests
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass


@dataclass
class WhalePosition:
    """Position d'une whale"""
    address: str
    size_btc: float
    direction: str
    entry_price: float
    leverage: float
    account_value: float
    unrealized_pnl: float


class WhaleTracker:
    """
    Tracker des whales Hyperliquid avec calcul de winrate et notifications
    """
    
    BASE_URL = "https://api.hyperliquid.xyz/info"
    
    # Adresses des whales (importées de hyperliquid_analyzer)
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
    
    # Cache des positions précédentes pour détecter les changements
    _previous_positions: Dict[str, Optional[WhalePosition]] = {}
    
    def __init__(self, telegram_token: str = None, telegram_chat_id: str = None):
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Telegram config
        self.telegram_token = telegram_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.getenv('TELEGRAM_CHAT_ID')
    
    def _post(self, data: Dict) -> Any:
        """Execute POST request"""
        try:
            response = self.session.post(self.BASE_URL, json=data, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return None
    
    def get_whale_winrate(self, address: str, days: int = 30) -> Dict[str, Any]:
        """
        Calcule le winrate BTC d'une whale sur les N derniers jours
        """
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        fills = self._post({
            'type': 'userFillsByTime',
            'user': address,
            'startTime': start_time,
            'aggregateByTime': False
        })
        
        if not fills:
            return {'address': address, 'has_btc_trades': False}
        
        # Filtrer les trades BTC avec closedPnl (positions fermées)
        btc_fills = [f for f in fills if f.get('coin') == 'BTC']
        closed_trades = [f for f in btc_fills if float(f.get('closedPnl', 0)) != 0]
        
        if not closed_trades:
            return {
                'address': address,
                'has_btc_trades': len(btc_fills) > 0,
                'total_btc_fills': len(btc_fills),
                'closed_trades': 0,
                'winrate': None
            }
        
        wins = sum(1 for f in closed_trades if float(f.get('closedPnl', 0)) > 0)
        losses = sum(1 for f in closed_trades if float(f.get('closedPnl', 0)) < 0)
        total_pnl = sum(float(f.get('closedPnl', 0)) for f in closed_trades)
        
        # Calculer profit factor
        gross_profit = sum(float(f.get('closedPnl', 0)) for f in closed_trades if float(f.get('closedPnl', 0)) > 0)
        gross_loss = abs(sum(float(f.get('closedPnl', 0)) for f in closed_trades if float(f.get('closedPnl', 0)) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'address': address,
            'address_short': address[:10] + '...',
            'has_btc_trades': True,
            'total_btc_fills': len(btc_fills),
            'closed_trades': len(closed_trades),
            'wins': wins,
            'losses': losses,
            'winrate': round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
            'total_pnl': round(total_pnl, 2),
            'profit_factor': round(profit_factor, 2),
            'period_days': days
        }
    
    def get_all_winrates(self, days: int = 30, min_trades: int = 10) -> List[Dict]:
        """
        Calcule le winrate de toutes les whales
        Retourne une liste triée par winrate décroissant
        """
        print("   🐋 Calculating whale winrates...")
        
        results = []
        for address in self.WHALE_ADDRESSES:
            wr = self.get_whale_winrate(address, days)
            if wr.get('closed_trades', 0) >= min_trades:
                results.append(wr)
        
        # Trier par winrate décroissant
        results.sort(key=lambda x: x.get('winrate', 0), reverse=True)
        
        return results
    
    def get_current_position(self, address: str) -> Optional[WhalePosition]:
        """Récupère la position BTC actuelle d'une whale"""
        data = self._post({
            'type': 'clearinghouseState',
            'user': address
        })
        
        if not data or 'assetPositions' not in data:
            return None
        
        account_value = float(data.get('marginSummary', {}).get('accountValue', 0))
        
        for pos in data['assetPositions']:
            if pos.get('position', {}).get('coin') == 'BTC':
                position = pos['position']
                size = float(position.get('szi', 0))
                
                if size != 0:
                    return WhalePosition(
                        address=address,
                        size_btc=size,
                        direction='LONG' if size > 0 else 'SHORT',
                        entry_price=float(position.get('entryPx', 0)),
                        leverage=float(position.get('leverage', {}).get('value', 1)),
                        account_value=account_value,
                        unrealized_pnl=float(position.get('unrealizedPnl', 0))
                    )
        
        return None
    
    def check_position_changes(self) -> List[Dict]:
        """
        Vérifie les changements de position depuis la dernière vérification
        Retourne une liste des nouvelles positions ouvertes
        """
        new_positions = []
        
        for address in self.WHALE_ADDRESSES:
            current = self.get_current_position(address)
            previous = self._previous_positions.get(address)
            
            # Détecter une nouvelle position (pas de position avant, position maintenant)
            if current and not previous:
                new_positions.append({
                    'type': 'NEW_POSITION',
                    'address': address,
                    'address_short': address[:10] + '...',
                    'direction': current.direction,
                    'size_btc': abs(current.size_btc),
                    'entry_price': current.entry_price,
                    'leverage': current.leverage,
                    'account_value': current.account_value
                })
            
            # Détecter un changement de direction
            elif current and previous and current.direction != previous.direction:
                new_positions.append({
                    'type': 'DIRECTION_CHANGE',
                    'address': address,
                    'address_short': address[:10] + '...',
                    'old_direction': previous.direction,
                    'new_direction': current.direction,
                    'size_btc': abs(current.size_btc),
                    'entry_price': current.entry_price,
                    'leverage': current.leverage
                })
            
            # Détecter une augmentation significative de taille (>50%)
            elif current and previous:
                size_change = (abs(current.size_btc) - abs(previous.size_btc)) / abs(previous.size_btc) * 100
                if size_change > 50:
                    new_positions.append({
                        'type': 'SIZE_INCREASE',
                        'address': address,
                        'address_short': address[:10] + '...',
                        'direction': current.direction,
                        'old_size': abs(previous.size_btc),
                        'new_size': abs(current.size_btc),
                        'increase_pct': round(size_change, 1),
                        'leverage': current.leverage
                    })
            
            # Mettre à jour le cache
            self._previous_positions[address] = current
        
        return new_positions
    
    def send_telegram_notification(self, message: str) -> bool:
        """Envoie une notification Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            print("   ⚠️ Telegram not configured")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            response = requests.post(url, json={
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"   ⚠️ Telegram error: {e}")
            return False
    
    def format_position_alert(self, change: Dict) -> str:
        """Formate un changement de position pour Telegram"""
        if change['type'] == 'NEW_POSITION':
            emoji = "🟢" if change['direction'] == 'LONG' else "🔴"
            return (
                f"🐋 <b>WHALE ALERT</b> {emoji}\n\n"
                f"Nouvelle position BTC\n"
                f"📍 {change['address_short']}\n"
                f"📊 {change['direction']} {change['size_btc']:.2f} BTC\n"
                f"💰 Entry: ${change['entry_price']:,.0f}\n"
                f"⚡ Leverage: {change['leverage']:.0f}x\n"
                f"💼 Account: ${change['account_value']:,.0f}"
            )
        
        elif change['type'] == 'DIRECTION_CHANGE':
            return (
                f"🔄 <b>WHALE FLIP</b>\n\n"
                f"Changement de direction!\n"
                f"📍 {change['address_short']}\n"
                f"❌ {change['old_direction']} → ✅ {change['new_direction']}\n"
                f"📊 Size: {change['size_btc']:.2f} BTC\n"
                f"⚡ Leverage: {change['leverage']:.0f}x"
            )
        
        elif change['type'] == 'SIZE_INCREASE':
            emoji = "🟢" if change['direction'] == 'LONG' else "🔴"
            return (
                f"📈 <b>WHALE ADDING</b> {emoji}\n\n"
                f"Position augmentée!\n"
                f"📍 {change['address_short']}\n"
                f"📊 {change['old_size']:.2f} → {change['new_size']:.2f} BTC\n"
                f"💹 +{change['increase_pct']:.0f}%\n"
                f"⚡ Leverage: {change['leverage']:.0f}x"
            )
        
        return str(change)
    
    def monitor_and_alert(self) -> Dict:
        """
        Monitore les whales et envoie des alertes si nécessaire
        """
        changes = self.check_position_changes()
        alerts_sent = 0
        
        for change in changes:
            message = self.format_position_alert(change)
            if self.send_telegram_notification(message):
                alerts_sent += 1
                print(f"   📱 Alert sent: {change['type']} - {change['address_short']}")
        
        return {
            'changes_detected': len(changes),
            'alerts_sent': alerts_sent,
            'changes': changes
        }


def test_whale_tracker():
    """Test du tracker"""
    tracker = WhaleTracker()
    
    # Test winrates
    print("=== Calcul des winrates ===")
    winrates = tracker.get_all_winrates(days=30, min_trades=5)
    
    print(f"\nTop 10 whales par winrate (min 5 trades):")
    for i, wr in enumerate(winrates[:10], 1):
        print(f"{i}. {wr['address_short']}: {wr['winrate']:.1f}% WR | {wr['closed_trades']} trades | PnL ${wr['total_pnl']:,.0f}")
    
    # Test position changes
    print("\n=== Vérification des positions ===")
    changes = tracker.check_position_changes()
    print(f"Changements détectés: {len(changes)}")
    
    return winrates


if __name__ == "__main__":
    test_whale_tracker()
