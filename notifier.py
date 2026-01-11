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
            return response.ok
        except Exception as e:
            print(f"❌ Erreur Telegram: {e}")
            return False
    
    def send_signal_alert(self, report: Dict[str, Any]) -> bool:
        """
        Envoie une alerte pour un signal de trading
        
        Args:
            report: Le rapport d'analyse complet
        """
        signal = report.get('signal', {})
        confidence = signal.get('confidence', 0)
        signal_type = signal.get('type', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        
        # Ne notifier que les signaux forts (>60 de confiance)
        if confidence < 60:
            return False
        
        # Emoji selon direction
        if direction == 'BULLISH':
            direction_emoji = '🟢📈'
        elif direction == 'BEARISH':
            direction_emoji = '🔴📉'
        else:
            direction_emoji = '⚪'
        
        # Construire le message
        price = report.get('price', 0)
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Scores par dimension
        dim_scores = signal.get('dimension_scores', {})
        scores_text = '\n'.join([
            f"  • {dim}: {score:.0f}/100"
            for dim, score in dim_scores.items()
        ])
        
        # Raisons
        reasons = signal.get('reasons', [])
        reasons_text = '\n'.join([f"  {r}" for r in reasons[:3]])
        
        # Warnings
        warnings = signal.get('warnings', [])
        warnings_text = '\n'.join([f"  ⚠️ {w}" for w in warnings[:2]]) if warnings else ''
        
        # Targets
        targets = signal.get('targets', {})
        targets_text = ''
        if targets:
            targets_text = '\n<b>🎯 Targets:</b>\n' + '\n'.join([
                f"  • {k}: ${v:,.0f}" for k, v in targets.items()
            ])
        
        message = f"""
{direction_emoji} <b>SIGNAL BTC - {signal_type}</b>

<b>💰 Prix:</b> ${price:,.2f}
<b>📊 Confiance:</b> {confidence:.0f}/100
<b>📈 Direction:</b> {direction}
<b>⏰ Heure:</b> {timestamp}

<b>📊 Scores:</b>
{scores_text}

<b>📝 Raisons:</b>
{reasons_text}
{targets_text}
{warnings_text}

<i>#BTC #Signal #{signal_type}</i>
"""
        
        return self.send_message(message.strip())
    
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
