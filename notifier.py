"""
Telegram Notifier
Envoie des alertes sur Telegram quand un signal fort est détecté
"""
import requests
import os
from typing import Dict, Any, Optional
from datetime import datetime


class TelegramNotifier:
    """Envoie des notifications Telegram pour les signaux de trading"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Args:
            bot_token: Token du bot Telegram (depuis @BotFather)
            chat_id: ID du chat où envoyer les messages
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def is_configured(self) -> bool:
        """Vérifie si le notifier est configuré"""
        return bool(self.bot_token and self.chat_id)
    
    def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Envoie un message texte"""
        if not self.is_configured():
            print("⚠️ Telegram non configuré")
            print(f"   Token: {'Oui' if self.bot_token else 'Non'}")
            print(f"   Chat ID: {'Oui' if self.chat_id else 'Non'}")
            return False
        
        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': parse_mode,
                    'disable_web_page_preview': True
                },
                timeout=10
            )
            if not response.ok:
                print(f"❌ Telegram API Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
            return response.ok
        except Exception as e:
            print(f"❌ Erreur Telegram: {e}")
            return False
    
    def send_signal_alert(self, report: Dict[str, Any], signal_history: list = None, winrate_stats: Dict = None) -> bool:
        """
        Envoie une alerte enrichie pour un signal de trading
        
        Args:
            report: Le rapport d'analyse complet
            signal_history: Historique des signaux pour compter les consécutifs
            winrate_stats: Statistiques de performance (winrate, wins, losses)
        """
        signal = report.get('signal', {})
        confidence = signal.get('confidence', 0)
        signal_type = signal.get('type', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        
        # Emoji selon direction
        if direction == 'LONG' or direction == 'BULLISH':
            direction_emoji = '🟢📈'
            dir_text = 'LONG'
        elif direction == 'SHORT' or direction == 'BEARISH':
            direction_emoji = '🔴📉'
            dir_text = 'SHORT'
        else:
            direction_emoji = '⚪💤'
            dir_text = 'NEUTRAL'
        
        # Prix et timestamp
        price = report.get('price', 0)
        timestamp = datetime.now().strftime('%H:%M UTC')
        
        # ========== TARGETS & P&L AVEC LEVIER x23 ==========
        targets = signal.get('targets', {})
        leverage = 23
        targets_section = ''
        if targets and price > 0:
            tp1 = targets.get('tp1', 0)
            tp2 = targets.get('tp2', 0)
            sl = targets.get('sl', 0)
            
            # Calcul des % bruts
            if dir_text == 'LONG':
                tp1_pct = ((tp1 - price) / price * 100) if tp1 else 0
                tp2_pct = ((tp2 - price) / price * 100) if tp2 else 0
                sl_pct = ((sl - price) / price * 100) if sl else 0
            else:  # SHORT
                tp1_pct = ((price - tp1) / price * 100) if tp1 else 0
                tp2_pct = ((price - tp2) / price * 100) if tp2 else 0
                sl_pct = ((price - sl) / price * 100) if sl else 0
            
            # P&L avec levier
            tp1_leveraged = tp1_pct * leverage
            tp2_leveraged = tp2_pct * leverage
            sl_leveraged = abs(sl_pct) * leverage
            
            # R:R Ratio
            risk = abs(sl_pct) if sl_pct != 0 else 1
            rr_ratio = abs(tp1_pct) / risk if risk > 0 else 0
            
            targets_section = f"""
<b>🎯 Targets (x{leverage}):</b>
  • TP1: ${tp1:,.0f} → <b>+{tp1_leveraged:.0f}%</b>
  • TP2: ${tp2:,.0f} → <b>+{tp2_leveraged:.0f}%</b>
  • SL: ${sl:,.0f} → <b>-{sl_leveraged:.0f}%</b>
  • R:R = {rr_ratio:.1f}:1"""
        
        # ========== CONTEXTE MARCHÉ ENRICHI ==========
        context = report.get('market_context', {})
        indicators = report.get('indicators', {})
        
        # Hyperliquid Whales - Correction des chemins
        hyperliquid = indicators.get('hyperliquid', {})
        whale_analysis = hyperliquid.get('whale_analysis', {})
        whale_sentiment = whale_analysis.get('sentiment', 'N/A')
        whale_count = whale_analysis.get('whale_count', 0)  # Était 'active_whales'
        
        # OI Delta - Récupérer depuis indicators.open_interest ou key_metrics
        oi_data = indicators.get('open_interest', {})
        # Le delta est souvent dans le rapport principal, pas dans indicators
        oi_delta = report.get('open_interest', {}).get('delta', {}).get('1h', {}).get('delta_oi_pct', 0)
        if oi_delta == 0:
            # Fallback: chercher dans key_metrics si disponible
            oi_delta = report.get('key_metrics', {}).get('oi_delta_1h', 0) or 0
        oi_emoji = '📈' if oi_delta > 0 else '📉' if oi_delta < 0 else '➡️'
        
        # Quantum State
        quantum_state = context.get('quantum_state', 'N/A')
        
        # Venturi - Correction du chemin
        fluid = indicators.get('fluid_dynamics', {})
        venturi = fluid.get('venturi', {})
        venturi_dir = venturi.get('direction', 'N/A')
        venturi_prob = venturi.get('breakout_probability', 0)  # Était 'probability'
        
        # VWAP et Exchanges - Correction des chemins (dans indicators.multi_exchange)
        multi_ex = indicators.get('multi_exchange', {})
        vwap = multi_ex.get('vwap', 0) or report.get('vwap_global', 0) or report.get('price', 0)
        exchanges_connected = multi_ex.get('exchanges_connected', 0)
        
        vwap_vs_price = 'AU-DESSUS' if price > vwap else 'EN-DESSOUS' if price < vwap else '='
        
        context_section = f"""
<b>🌍 Contexte:</b>
  • 🐋 Whales: <b>{whale_sentiment}</b> ({whale_count} traders)
  • {oi_emoji} OI Δ1h: <b>{oi_delta:+.1f}%</b>
  • ⚛️ État: <b>{quantum_state}</b>
  • 🌊 Venturi: → <b>{venturi_dir}</b> ({venturi_prob:.0f}%)
  • 💵 Prix vs VWAP: {vwap_vs_price}"""
        
        # ========== COMPTEUR SIGNAUX CONSÉCUTIFS ==========
        consecutive_section = ''
        if signal_history:
            consecutive_count = self._count_consecutive_signals(signal_history, dir_text)
            if consecutive_count > 1:
                ordinal = self._get_french_ordinal(consecutive_count)
                consecutive_section = f"\n\n🔁 <b>{ordinal} signal {dir_text}</b> depuis retournement"
        
        # ========== RAISONS ==========
        reasons = signal.get('reasons', [])
        reasons_section = ''
        if reasons:
            reasons_text = '\n'.join([f"  {r}" for r in reasons[:4]])
            reasons_section = f'\n<b>📝 Raisons:</b>\n{reasons_text}'
        
        # ========== WARNINGS ==========
        warnings = signal.get('warnings', [])
        warnings_section = ''
        if warnings:
            warnings_text = '\n'.join([f"  ⚠️ {w}" for w in warnings[:2]])
            warnings_section = f'\n{warnings_text}'
        
        # ========== WINRATE STATS ==========
        winrate_section = ''
        if winrate_stats:
            wr = winrate_stats.get('winrate_pct', 0)
            wins = winrate_stats.get('wins', 0)
            losses = winrate_stats.get('losses', 0)
            if wins + losses > 0:
                winrate_section = f" | 📊 WR: {wr:.0f}% ({wins}W/{losses}L)"
        
        # ========== MESSAGE FINAL ==========
        message = f"""{direction_emoji} <b>SIGNAL BTC - {signal_type}</b>

<b>💰 Prix:</b> ${price:,.2f}
<b>📊 Confiance:</b> {confidence:.0f}/100
<b>📈 Direction:</b> {dir_text}
<b>⏰</b> {timestamp}{consecutive_section}
{context_section}
{targets_section}
{reasons_section}{warnings_section}

<code>🏦 {exchanges_connected}/12 | VWAP ${vwap:,.0f}{winrate_section}</code>

<i>#BTC #Signal #{signal_type}</i>"""
        
        return self.send_message(message.strip())
    
    def _count_consecutive_signals(self, history: list, current_direction: str) -> int:
        """Compte les signaux consécutifs dans la même direction"""
        if not history or current_direction == 'NEUTRAL':
            return 1
        
        count = 1
        for sig in reversed(history):
            sig_dir = sig.get('signal', {}).get('direction', 'NEUTRAL')
            # Normaliser les directions
            if sig_dir in ['LONG', 'BULLISH']:
                sig_dir = 'LONG'
            elif sig_dir in ['SHORT', 'BEARISH']:
                sig_dir = 'SHORT'
            else:
                sig_dir = 'NEUTRAL'
            
            if sig_dir == current_direction:
                count += 1
            else:
                break
        return count
    
    def _get_french_ordinal(self, n: int) -> str:
        """Convertit un nombre en ordinal français"""
        if n == 1:
            return "1er"
        elif n == 2:
            return "2ème"
        elif n == 3:
            return "3ème"
        else:
            return f"{n}ème"
    
    def send_daily_summary(self, reports: list) -> bool:
        """Envoie un résumé quotidien"""
        if not reports:
            return False
        
        total_signals = len(reports)
        strong_signals = len([r for r in reports if r.get('signal', {}).get('confidence', 0) >= 60])
        
        # Derniers prix
        prices = [r.get('price', 0) for r in reports if r.get('price')]
        if prices:
            high = max(prices)
            low = min(prices)
            last = prices[-1]
            change_pct = ((last - prices[0]) / prices[0] * 100) if prices[0] else 0
        else:
            high = low = last = change_pct = 0
        
        message = f"""
📊 <b>RÉSUMÉ QUOTIDIEN BTC</b>

<b>💰 Prix:</b>
  • Dernier: ${last:,.2f}
  • High: ${high:,.2f}
  • Low: ${low:,.2f}
  • Variation: {change_pct:+.2f}%

<b>📈 Signaux:</b>
  • Total analyses: {total_signals}
  • Signaux forts: {strong_signals}

<i>#BTC #DailyReport</i>
"""
        return self.send_message(message.strip())
    
    def send_startup_message(self) -> bool:
        """Envoie un message de démarrage"""
        message = """
🚀 <b>BTC Sniper Bot V2 démarré</b>

✅ Multi-Exchange (4)
✅ On-Chain Analytics
✅ Options Deribit
✅ Decision Engine V2

<i>Monitoring actif...</i>
"""
        return self.send_message(message.strip())


def test_telegram():
    """Test de la connexion Telegram"""
    notifier = TelegramNotifier()
    
    if not notifier.is_configured():
        print("❌ Telegram non configuré!")
        print("   Ajoutez TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID dans .env")
        return False
    
    # Test avec un message simple
    success = notifier.send_message("🔧 <b>Test BTC Sniper Bot</b>\n\nConnexion OK!")
    
    if success:
        print("✅ Message Telegram envoyé!")
    else:
        print("❌ Échec de l'envoi Telegram")
    
    return success


if __name__ == "__main__":
    test_telegram()
