"""
BTC Sniper Bot V2 - Runner 24/7
Script optimisé pour GitHub Actions avec notifications Telegram
"""
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any

# Import du bot principal
from main_v2 import run_analysis_v2
from notifier import TelegramNotifier


def run_scheduled_analysis() -> Dict[str, Any]:
    """
    Exécute l'analyse et notifie si signal fort
    Optimisé pour GitHub Actions (exécution rapide)
    """
    print("=" * 60)
    print(f"🤖 BTC Sniper Bot - Scheduled Run")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    # Initialiser le notifier
    notifier = TelegramNotifier()
    telegram_enabled = notifier.is_configured()
    
    if telegram_enabled:
        print("✅ Telegram configuré")
    else:
        print("⚠️ Telegram non configuré (pas d'alertes)")
    
    try:
        # Exécuter l'analyse (mode full pour max de données)
        report = run_analysis_v2(mode='full')
        
        if not report:
            print("❌ Analyse échouée")
            return None
        
        # Sauvegarder le rapport
        with open('analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        # Vérifier si on doit notifier
        signal = report.get('signal', {})
        confidence = signal.get('confidence', 0)
        direction = signal.get('direction', 'NEUTRAL')
        
        print(f"\n📊 Signal: {signal.get('type', 'UNKNOWN')}")
        print(f"📈 Direction: {direction}")
        print(f"📊 Confiance: {confidence:.0f}/100")
        
        # Notifier si signal fort (confiance >= 60%)
        if telegram_enabled and confidence >= 60:
            print("\n📱 Envoi notification Telegram...")
            if notifier.send_signal_alert(report):
                print("✅ Notification envoyée!")
            else:
                print("❌ Échec notification")
        elif confidence >= 60:
            print("\n⚠️ Signal fort mais Telegram non configuré")
        else:
            print(f"\n💤 Signal faible ({confidence:.0f}/100) - Pas de notification")
        
        return report
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        # Notifier l'erreur sur Telegram
        if telegram_enabled:
            notifier.send_message(f"❌ <b>Erreur Bot</b>\n\n{str(e)[:200]}")
        
        return None


def main():
    """Point d'entrée pour GitHub Actions"""
    # Vérifier si c'est un test
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🔧 Mode test - Vérification Telegram")
        from notifier import test_telegram
        success = test_telegram()
        sys.exit(0 if success else 1)
    
    # Exécuter l'analyse
    report = run_scheduled_analysis()
    
    if report:
        print("\n✅ Analyse terminée avec succès")
        sys.exit(0)
    else:
        print("\n❌ Analyse échouée")
        sys.exit(1)


if __name__ == "__main__":
    main()
