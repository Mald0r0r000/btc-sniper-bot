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
        
        # Sauvegarder le signal enrichi dans le Gist (données complètes pour ML)
        indicators = report.get("indicators", {})
        
        # Extraire les données order book
        ob_data = indicators.get("order_book", {})
        
        # Extraire les données CVD
        cvd_data = indicators.get("cvd", {})
        
        # Extraire Volume Profile
        vp_data = indicators.get("volume_profile", {})
        
        # Extraire les funding rates multi-exchange
        deriv_data = indicators.get("derivatives", {})
        
        # Extraire cross-asset data
        cross_asset = indicators.get("cross_asset", {})
        macro_indicators = indicators.get("macro", {})
        # Note: cross_asset_data is in market_context from decision engine
        market_ctx = report.get("market_context", {})
        
        # Extraire les indicateurs techniques
        kdj_data = indicators.get("kdj", {})
        adx_data = indicators.get("adx", {})
        macd_data = indicators.get("macd", {})
        
        # Extraire OHLCV snapshot (5 dernières bougies via candles in decision engine data)
        # Ces données sont dans le report mais pas directement - on utilise les données d'entropy
        entropy_data = indicators.get("entropy", {})
        
        # Construire le signal_record enrichi (clés courtes pour compacité)
        signal_record = {
            "ts": datetime.now(timezone.utc).isoformat(),  # timestamp
            "px": report.get("price", 0),  # price
            "sig": {
                "t": signal.get("type"),  # type
                "d": direction,  # direction
                "c": round(confidence, 1),  # confidence
                "s": signal.get("strength"),  # strength
                "mp": signal.get("manipulation_penalty", 0)  # manipulation_penalty
            },
            "ds": {k: round(v, 1) for k, v in signal.get("dimension_scores", {}).items()},  # dimension_scores
            "tgt": signal.get("targets", {}),  # targets
            "ctx": {  # market_context compact
                "qs": market_ctx.get("quantum_state"),
                "vp": market_ctx.get("vp_shape"),
                "mr": market_ctx.get("manipulation_risk"),
                "fg": market_ctx.get("fear_greed"),
                "tb": market_ctx.get("technical_bias"),
                "sb": market_ctx.get("structure_bias"),
                "db": market_ctx.get("derivatives_bias"),
                "stb": market_ctx.get("sentiment_bias"),
                "mb": market_ctx.get("macro_bias")
            },
            "con": {  # consistency
                "st": consistency_result.get("status"),
                "sc": consistency_result.get("score", 0),
                "lv": consistency_result.get("consistency_level"),
                "fc": consistency_result.get("flips_count", 0),
                "ct": consistency_result.get("confidence_trend")
            },
            "fd": {  # fluid_dynamics
                "v": {  # venturi
                    "cd": indicators.get("fluid_dynamics", {}).get("venturi", {}).get("compression_detected"),
                    "cs": indicators.get("fluid_dynamics", {}).get("venturi", {}).get("compression_score"),
                    "dir": indicators.get("fluid_dynamics", {}).get("venturi", {}).get("direction"),
                    "bp": indicators.get("fluid_dynamics", {}).get("venturi", {}).get("breakout_probability")
                },
                "st": {  # self_trading
                    "det": indicators.get("fluid_dynamics", {}).get("self_trading", {}).get("detected"),
                    "pb": indicators.get("fluid_dynamics", {}).get("self_trading", {}).get("probability"),
                    "tp": indicators.get("fluid_dynamics", {}).get("self_trading", {}).get("type")
                }
            },
            # ===== NOUVELLES DONNÉES ML =====
            "ob": {  # order_book
                "br": round(ob_data.get("bid_ratio_pct", 50) / 100, 3),  # bid_ratio 0-1
                "pr": ob_data.get("pressure"),  # pressure string
                "im": round(ob_data.get("imbalance_ratio", 1), 2),  # imbalance
                "sp": ob_data.get("spread_bps")  # spread in bps
            },
            "cvd": {  # cumulative volume delta
                "st": cvd_data.get("trend") if cvd_data.get("mtf_data") else cvd_data.get("status"),
                "ar": round(cvd_data.get("composite_score", 50) / 50 if cvd_data.get("mtf_data") else cvd_data.get("aggression_ratio", 1), 2),
                "d": cvd_data.get("net_cvd") if cvd_data.get("mtf_data") else cvd_data.get("cvd_sum"),
                # New MTF Data fields
                "mtf": cvd_data.get("mtf_data"),
                "cs": cvd_data.get("composite_score"),
                "cf": cvd_data.get("confluence"),
                "tr": cvd_data.get("trend"),
                "em": cvd_data.get("emoji")
            },
            "vp": {  # volume_profile
                "poc": vp_data.get("poc"),
                "vah": vp_data.get("vah"),
                "val": vp_data.get("val"),
                "sh": vp_data.get("shape"),  # shape
                "sk": vp_data.get("skew"),   # skew (new for D-Shape pressure)
                # Price context for contextual VP analysis
                "pctx": "ABOVE_VAH" if report.get("price", 0) > vp_data.get("vah", 0) and vp_data.get("vah", 0) > 0
                       else "BELOW_VAL" if report.get("price", 0) < vp_data.get("val", float('inf')) and vp_data.get("val", 0) > 0
                       else "ABOVE_POC" if report.get("price", 0) > vp_data.get("poc", 0) and vp_data.get("poc", 0) > 0
                       else "BELOW_POC" if vp_data.get("poc", 0) > 0
                       else None
            },
            "fr": indicators.get("multi_exchange", {}).get("funding_divergence"),  # funding rate divergence
            "oi": {  # open_interest
                "t": indicators.get("open_interest", {}).get("total_oi_btc"),  # total (from analysis result)
                "d1h": indicators.get("open_interest", {}).get("delta", {}).get("1h", {}).get("delta_oi_pct"),
                "d24h": indicators.get("open_interest", {}).get("delta", {}).get("24h", {}).get("delta_oi_pct")
            },
            "macro": {  # cross-asset & macro
                "fg": indicators.get("sentiment", {}).get("fear_greed", {}).get("value"),  # fear_greed index
                "re": indicators.get("macro", {}).get("risk_environment", {}).get("environment"),  # risk env
                "ex": indicators.get("multi_exchange", {}).get("exchanges_connected", 0),
                # New Cross-Asset Data
                "dxy": cross_asset.get("dxy", {}).get("value"),
                "spx": cross_asset.get("spx", {}).get("value"),
                "m2": {
                    "v": cross_asset.get("m2", {}).get("current"),  # Value ($B)
                    "yoy": cross_asset.get("m2", {}).get("yoy_change"),  # Year-over-Year change (%)
                    "off": cross_asset.get("m2", {}).get("offset_90d_trend"),  # 90-day offset trend
                    "imp": cross_asset.get("m2", {}).get("btc_impact")  # Impact
                }
            },
            "tech": {  # technical indicators
                "kj": kdj_data.get("values", {}).get("j"),  # kdj J value
                "ks": kdj_data.get("signal"),  # kdj signal
                "cmp": entropy_data.get("compression", {}).get("current"),  # compression (fixed key name)
                "qs": entropy_data.get("quantum_state"),  # quantum state
                # ADX data (from adx_data)
                "adx": adx_data.get("adx"),  # ADX value
                "reg": adx_data.get("regime"),  # ADX regime (TRENDING/RANGING/TRANSITION)
                "atd": adx_data.get("trend_direction"),  # ADX trend direction (BULLISH/BEARISH/NEUTRAL)
                "dip": adx_data.get("plus_di"),  # DI+ value
                "dim": adx_data.get("minus_di"),  # DI- value
                # MACD 3D data (extracted from mtf_data['3d'])
                "mcd": {
                    "h": macd_data.get("mtf_data", {}).get("3d", {}).get("hist"),  # MACD histogram
                    "s": macd_data.get("mtf_data", {}).get("3d", {}).get("signal"),  # MACD signal line
                    "v": macd_data.get("mtf_data", {}).get("3d", {}).get("macd"),  # MACD value
                    "t": macd_data.get("mtf_data", {}).get("3d", {}).get("trend")  # MACD trend (BULLISH/BEARISH/NEUTRAL)
                } if macd_data.get("available") and macd_data.get("mtf_data", {}).get("3d", {}).get("available") else None
            },
            # Hyperliquid whale data (enhanced with two-tier tracking)
            "hl": {
                "ws": indicators.get("hyperliquid", {}).get("whale_analysis", {}).get("sentiment"),
                "lr": indicators.get("hyperliquid", {}).get("whale_analysis", {}).get("long_ratio_pct"),
                "wc": indicators.get("hyperliquid", {}).get("whale_analysis", {}).get("whale_count"),
                "cc": indicators.get("hyperliquid", {}).get("whale_analysis", {}).get("curated_count"),  # Curated whales
                "lc": indicators.get("hyperliquid", {}).get("whale_analysis", {}).get("leaderboard_count"),  # Leaderboard whales
                "wl": indicators.get("hyperliquid", {}).get("whale_analysis", {}).get("weighted_long"),  # Weighted LONG
                "wsh": indicators.get("hyperliquid", {}).get("whale_analysis", {}).get("weighted_short")  # Weighted SHORT
            },
            # MTF MACD data (Multi-Timeframe MACD with divergence detection)
            "mtf": {
                "av": macd_data.get("available", False),  # available
                "cs": round(macd_data.get("composite_score", 0), 1),  # composite_score
                "tr": macd_data.get("trend", "NEUTRAL"),  # trend
                "cf": macd_data.get("confluence", "MIXED"),  # confluence
                "dv": {  # divergence
                    "t": macd_data.get("divergence", {}).get("type", "UNKNOWN"),  # type
                    "sa": macd_data.get("divergence", {}).get("score_adjustment", 0),  # score_adjustment
                    "ds": macd_data.get("divergence", {}).get("description", "")  # description
                } if macd_data.get("divergence") else None,
                "tf": {  # timeframes
                    "1h": {
                        "t": macd_data.get("mtf_data", {}).get("1h", {}).get("trend"),
                        "h": macd_data.get("mtf_data", {}).get("1h", {}).get("hist"),
                        "s": macd_data.get("mtf_data", {}).get("1h", {}).get("slope")
                    } if macd_data.get("mtf_data", {}).get("1h", {}).get("available") else None,
                    "4h": {
                        "t": macd_data.get("mtf_data", {}).get("4h", {}).get("trend"),
                        "h": macd_data.get("mtf_data", {}).get("4h", {}).get("hist"),
                        "s": macd_data.get("mtf_data", {}).get("4h", {}).get("slope")
                    } if macd_data.get("mtf_data", {}).get("4h", {}).get("available") else None,
                    "1d": {
                        "t": macd_data.get("mtf_data", {}).get("1d", {}).get("trend"),
                        "h": macd_data.get("mtf_data", {}).get("1d", {}).get("hist"),
                        "s": macd_data.get("mtf_data", {}).get("1d", {}).get("slope")
                    } if macd_data.get("mtf_data", {}).get("1d", {}).get("available") else None,
                    "3d": {
                        "t": macd_data.get("mtf_data", {}).get("3d", {}).get("trend"),
                        "h": macd_data.get("mtf_data", {}).get("3d", {}).get("hist"),
                        "s": macd_data.get("mtf_data", {}).get("3d", {}).get("slope")
                    } if macd_data.get("mtf_data", {}).get("3d", {}).get("available") else None
                } if macd_data.get("mtf_data") else None
            } if macd_data.get("available") else None,
            # Smart Entry recommendation (fractal zones + MTF context)
            "se": {
                "st": signal.get("smart_entry", {}).get("strategy"),  # strategy
                "oe": signal.get("smart_entry", {}).get("optimal_entry"),  # optimal_entry
                "cp": signal.get("smart_entry", {}).get("current_price"),  # current_price
                "lz": signal.get("smart_entry", {}).get("liq_zone"),  # liq_zone
                "rr": signal.get("smart_entry", {}).get("rr_improvement"),  # rr_improvement
                "to": signal.get("smart_entry", {}).get("timeout_hours")  # timeout_hours
            } if signal.get("smart_entry") else None
        }
        data_store.save_signal(signal_record)
        
        # ========== FILTRAGE QUALITÉ DES SIGNAUX ==========
        # Basé sur backtest 13-17 Jan 2026:
        # - FADE_LOW: 100% WR → INCLURE
        # - LONG_SNIPER: 66.7% WR → INCLURE
        # - FADE_HIGH: 0% WR → EXCLURE TEMPORAIREMENT
        # - NO_SIGNAL: Toujours exclure
        
        QUALITY_SIGNAL_TYPES = [
            'FADE_HIGH',    # BUG FIXÉ: Direction corrigée, maintenant profitable
            'FADE_LOW', 
            'LONG_SNIPER', 'SHORT_SNIPER',
            'SHORT_BREAKOUT',
            'QUANTUM_BUY', 'QUANTUM_SELL' # Nouveaux signaux forts
        ]
        # LONG_BREAKOUT retiré car 0% WR confirmé
        
        signal_type = signal.get('type', 'NO_SIGNAL')
        is_quality_signal = signal_type in QUALITY_SIGNAL_TYPES
        confidence_threshold = 60  # Ramené à 60 car le filtre Consistency nettoie les mauvais signaux
        

        # Notifier uniquement si signal de qualité ET confiance >= 65%
        if telegram_enabled and confidence >= confidence_threshold and is_quality_signal:
            print(f"\n📱 Envoi notification Telegram ({signal_type}, {confidence:.0f}%)...")
            # Récupérer l'historique pour le compteur de signaux consécutifs
            signal_history = data_store.read_signals()[-20:]  # 20 derniers signaux
            # Récupérer les stats de performance (winrate)
            winrate_stats = data_store.get_performance_stats()
            if notifier.send_signal_alert(report, signal_history=signal_history, winrate_stats=winrate_stats):
                print("✅ Notification envoyée!")
            else:
                print("❌ Échec notification")
        elif telegram_enabled and confidence >= confidence_threshold and not is_quality_signal:
            print(f"\n🔇 Signal {signal_type} exclu (faible winrate historique)")
        elif confidence >= confidence_threshold:
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
    
    # Valider les signaux passés à CHAQUE run (était % 6)
    run_number = int(os.getenv('GITHUB_RUN_NUMBER', '0'))
    data_store = GistDataStore()
    validation_result = run_validation_cycle(data_store)
    
    # Afficher le winrate si disponible
    if validation_result.get('performance'):
        perf = validation_result['performance']
        print(f"\n📊 PERFORMANCE: {perf.get('winrate_pct', 0)}% winrate ({perf.get('wins', 0)}W / {perf.get('losses', 0)}L)")
    
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

