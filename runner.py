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
from data_store import GistDataStore
from signal_validator import SignalValidator
from adaptive_scoring import AdaptiveScoringLayer


def run_scheduled_analysis() -> Dict[str, Any]:
    """
    Exécute l'analyse et notifie si signal fort
    Optimisé pour GitHub Actions (exécution rapide)
    """
    print("=" * 60)
    print(f"🤖 BTC Sniper Bot - Scheduled Run")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    # Initialiser le notifier et le data store
    notifier = TelegramNotifier()
    data_store = GistDataStore()
    telegram_enabled = notifier.is_configured()
    
    if telegram_enabled:
        print("✅ Telegram configuré")
    else:
        print("⚠️ Telegram non configuré (pas d'alertes)")
    
    if data_store.github_token and data_store.gist_id:
        print("✅ Stockage Gist configuré")
    elif data_store.github_token:
        print("📝 Gist sera créé à la première sauvegarde")
    else:
        print("⚠️ Stockage désactivé (GITHUB_TOKEN manquant)")
    
    # Récupérer l'historique pour consistency check
    from consistency_checker import ConsistencyChecker
    
    history = data_store.get_recent_signals(count=5)
    consistency_result = {"score": 0, "status": "NEW"}
    
    if history:
        print(f"📊 {len(history)} signaux historiques chargés")
    
    try:
        # Exécuter l'analyse (mode full pour max de données)
        report = run_analysis_v2(mode='full')
        
        if not report:
            print("❌ Analyse échouée")
            return None
        
        # Calculer le consistency check
        current_signal = {
            "direction": report.get('signal', {}).get('direction', 'NEUTRAL'),
            "confidence": report.get('signal', {}).get('confidence', 50)
        }
        
        if history:
            checker = ConsistencyChecker()
            consistency_result = checker.check_consistency(current_signal, history)
            
            # Afficher le résultat
            print(f"\n🔄 Consistency: {consistency_result['status']}")
            if consistency_result['score'] != 0:
                print(f"   Score ajustement: {consistency_result['score']:+d}")
            for detail in consistency_result.get('details', []):
                print(f"   {detail}")
        
        # Ajouter le consistency au rapport
        report['consistency'] = consistency_result
        
        # Sauvegarder le rapport localement
        with open('analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        # Vérifier si on doit notifier
        signal = report.get('signal', {})
        confidence = signal.get('confidence', 0)
        direction = signal.get('direction', 'NEUTRAL')
        
        print(f"\n📊 Signal: {signal.get('type', 'UNKNOWN')}")
        print(f"📈 Direction: {direction}")
        print(f"📊 Confiance: {confidence:.0f}/100")
        
        # Sauvegarder le signal enrichi dans le Gist
        signal_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": report.get("price", 0),
            "signal": {
                "type": signal.get("type"),
                "direction": direction,
                "confidence": confidence,
                "strength": signal.get("strength"),
                "manipulation_penalty": signal.get("manipulation_penalty", 0)
            },
            "dimension_scores": signal.get("dimension_scores", {}),
            "targets": signal.get("targets", {}),
            # Données enrichies pour analyse
            "market_context": report.get("market_context", {}),
            "consistency": report.get("consistency", {}),
            "fluid_dynamics": {
                "venturi": report.get("indicators", {}).get("fluid_dynamics", {}).get("venturi", {}),
                "self_trading": report.get("indicators", {}).get("fluid_dynamics", {}).get("self_trading", {})
            },
            # Métriques clés extraites
            "key_metrics": {
                "vwap": report.get("vwap_global"),
                "fear_greed": report.get("sentiment", {}).get("fear_greed", {}).get("value"),
                "oi_delta_1h": report.get("open_interest", {}).get("delta", {}).get("1h", {}).get("delta_oi_pct"),
                "exchanges_connected": report.get("exchanges_connected", 0)
            }
        }
        data_store.save_signal(signal_record)
        
        # Notifier si signal fort (confiance >= 60%)
        if telegram_enabled and confidence >= 60:
            print("\n📱 Envoi notification Telegram...")
            # Récupérer l'historique pour le compteur de signaux consécutifs
            signal_history = data_store.read_signals()[-20:]  # 20 derniers signaux
            if notifier.send_signal_alert(report, signal_history=signal_history):
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
    
    # Mode validation uniquement
    if len(sys.argv) > 1 and sys.argv[1] == '--validate':
        print("🔍 Mode validation des signaux")
        data_store = GistDataStore()
        result = run_validation_cycle(data_store)
        print(f"Résultat: {result}")
        sys.exit(0)
    
    # Exécuter l'analyse
    report = run_scheduled_analysis()
    
    # Vérifier si on doit valider (toutes les 6 runs = ~30min)
    run_number = int(os.getenv('GITHUB_RUN_NUMBER', '0'))
    if run_number > 0 and run_number % 6 == 0:
        data_store = GistDataStore()
        run_validation_cycle(data_store)
    
    if report:
        print("\n✅ Analyse terminée avec succès")
        sys.exit(0)
    else:
        print("\n❌ Analyse échouée")
        sys.exit(1)


def run_validation_cycle(data_store: GistDataStore) -> Dict[str, Any]:
    """
    Exécute la validation des signaux passés et recalibre les poids
    """
    print("\n" + "=" * 60)
    print("🔍 VALIDATION & RECALIBRATION")
    print("=" * 60)
    
    # Récupérer les signaux non-validés
    signals = data_store.read_signals()
    pending = [s for s in signals if s.get('validation', {}).get('status') not in ['WIN', 'LOSS', 'EXPIRED']]
    
    print(f"⏳ {len(pending)} signaux à valider")
    
    if not pending:
        print("✅ Aucun signal à valider")
        return {'validated': 0}
    
    # Valider les signaux
    validator = SignalValidator()
    validated_signals, performance = validator.run_validation(pending)
    
    # Recalibrer les poids si on a assez de données
    decided_signals = [s for s in validated_signals if s.get('validation', {}).get('status') in ['WIN', 'LOSS']]
    
    if len(decided_signals) >= 5:
        print("\n🧠 Recalibration des poids...")
        scorer = AdaptiveScoringLayer()
        scorer.analyze_signals(decided_signals)
        new_weights = scorer.calculate_adjusted_weights()
        
        # Préparer les données de poids
        import json
        weights_data = {
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'weights': new_weights,
            'performance': performance
        }
        
        # Sauvegarder localement (pour les tests)
        with open('adaptive_weights.json', 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        # Sauvegarder dans le Gist (pour GitHub Actions)
        data_store.save_adaptive_weights(weights_data)
        
        print(f"   ✅ Poids mis à jour: {new_weights}")
    else:
        print(f"   ⏳ Pas assez de données ({len(decided_signals)}/5 requis)")
    
    return {
        'validated': len(validated_signals),
        'performance': performance
    }


if __name__ == "__main__":
    main()

