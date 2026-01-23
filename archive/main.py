"""
BTC Sniper Bot - Point d'entrée principal
Orchestre tous les analyseurs et génère un rapport consolidé
"""
import json
from datetime import datetime, timezone
from typing import Dict, Any

import config
from exchange import BitgetConnector
from analyzers import (
    OrderBookAnalyzer,
    CVDAnalyzer,
    VolumeProfileAnalyzer,
    FundingLiquidationAnalyzer,
    FVGAnalyzer,
    EntropyAnalyzer
)
from decision_engine import DecisionEngine


def run_analysis() -> Dict[str, Any]:
    """
    Exécute l'analyse complète du marché
    
    Returns:
        Dict avec tous les résultats d'analyse et signaux
    """
    print("=" * 60)
    print("🎯 BTC SNIPER BOT - Analyse en cours...")
    print("=" * 60)
    
    # 1. Connexion à l'exchange
    print("📡 Connexion à Bitget...")
    connector = BitgetConnector()
    
    # 2. Récupération des données
    print("📊 Récupération des données...")
    
    # Prix actuel
    current_price = connector.get_current_price()
    print(f"   💰 Prix actuel: ${current_price:,.2f}")
    
    # OHLCV Multi-Timeframe
    df_micro = connector.fetch_ohlcv(config.TIMEFRAME_MICRO, limit=1000)
    df_meso = connector.fetch_ohlcv(config.TIMEFRAME_MESO, limit=500)
    df_macro = connector.fetch_ohlcv(config.TIMEFRAME_MACRO, limit=30)
    
    print(f"   📈 Données {config.TIMEFRAME_MICRO}: {len(df_micro)} bougies")
    print(f"   📈 Données {config.TIMEFRAME_MESO}: {len(df_meso)} bougies")
    print(f"   📈 Données {config.TIMEFRAME_MACRO}: {len(df_macro)} bougies")
    
    # Order Book
    order_book = connector.fetch_order_book(limit=config.ORDER_BOOK_LIMIT)
    
    # Trades pour CVD
    trades = connector.fetch_trades(limit=config.CVD_TRADES_LIMIT)
    print(f"   🔄 Trades récents: {len(trades)}")
    
    # Open Interest
    oi_data = connector.fetch_open_interest()
    
    # Funding Rate
    funding_data = connector.fetch_funding_rate()
    
    # 3. Analyses
    print("\n🔬 Analyse des indicateurs...")
    
    # Order Book Analysis
    ob_analyzer = OrderBookAnalyzer(order_book, current_price)
    ob_result = ob_analyzer.analyze()
    print(f"   📒 Order Book: {ob_result['pressure']} ({ob_result['bid_ratio_pct']}% vs {ob_result['ask_ratio_pct']}%)")
    
    # CVD Analysis
    cvd_analyzer = CVDAnalyzer(trades)
    cvd_result = cvd_analyzer.analyze()
    print(f"   📊 CVD: {cvd_result['emoji']} {cvd_result['status']} (Net: {cvd_result['net_cvd']:+.2f} BTC)")
    
    # Volume Profile Analysis
    vp_analyzer = VolumeProfileAnalyzer(df_micro)
    vp_result = vp_analyzer.analyze()
    print(f"   📊 Volume Profile: {vp_result['shape']}")
    print(f"      POC: ${vp_result['poc']:,.2f} | VAH: ${vp_result['vah']:,.2f} | VAL: ${vp_result['val']:,.2f}")
    
    # Funding & Liquidation Analysis
    fl_analyzer = FundingLiquidationAnalyzer(funding_data, current_price)
    fl_result = fl_analyzer.analyze()
    print(f"   💸 Funding: {fl_result['funding']['emoji']} {fl_result['funding']['current_pct']:.4f}%")
    print(f"   🧲 {fl_result['magnet']['description']}")
    
    # FVG Analysis (MTF)
    fvg_analyzer = FVGAnalyzer({
        config.TIMEFRAME_MICRO: df_micro,
        config.TIMEFRAME_MESO: df_meso,
        config.TIMEFRAME_MACRO: df_macro
    })
    fvg_result = fvg_analyzer.analyze(current_price)
    print(f"   📦 FVG MTF: {fvg_result['total_active']} gaps actifs")
    
    # Entropy Analysis
    entropy_analyzer = EntropyAnalyzer(df_micro)
    entropy_result = entropy_analyzer.analyze()
    print(f"   ⚛️ Quantum State: {entropy_result['quantum_state']} (Compression: {entropy_result['compression']['current']:.3f})")
    
    # 4. Décision Engine
    print("\n🧠 Génération des signaux...")
    engine = DecisionEngine(
        current_price=current_price,
        order_book_data=ob_result,
        cvd_data=cvd_result,
        volume_profile_data=vp_result,
        funding_liq_data=fl_result,
        fvg_data=fvg_result,
        entropy_data=entropy_result,
        open_interest=oi_data
    )
    decision_result = engine.generate_signals()
    
    # 5. Affichage du signal principal
    primary = decision_result['primary_signal']
    
    print("\n" + "=" * 60)
    print("📢 SIGNAL PRINCIPAL:")
    print("=" * 60)
    print(f"   {primary['emoji']} {primary['description']}")
    print(f"   Direction: {primary['direction']} | Confiance: {primary['confidence']}/10")
    
    if primary['reasons']:
        print("\n   📝 Raisons:")
        for reason in primary['reasons']:
            print(f"      • {reason}")
    
    if primary['targets']:
        print("\n   🎯 Targets:")
        for key, val in primary['targets'].items():
            print(f"      • {key}: ${val:,.2f}")
    
    if primary['warnings']:
        print("\n   ⚠️ Avertissements:")
        for warning in primary['warnings']:
            print(f"      {warning}")
    
    # 6. Warnings généraux
    if decision_result['warnings']:
        print("\n⚠️ NOTES:")
        for warning in decision_result['warnings']:
            print(f"   {warning}")
    
    print("\n" + "=" * 60)
    
    # 7. Construire le rapport final
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': config.SYMBOL,
        'price': current_price,
        'signal': decision_result['primary_signal'],
        'all_signals': decision_result['all_signals'],
        'market_context': decision_result['market_context'],
        'warnings': decision_result['warnings'],
        'indicators': {
            'order_book': ob_result,
            'cvd': cvd_result,
            'volume_profile': vp_result,
            'funding_liquidation': fl_result,
            'fvg': {
                'total_active': fvg_result['total_active'],
                'nearest_bull': fvg_result['nearest_bull'],
                'nearest_bear': fvg_result['nearest_bear']
            },
            'entropy': entropy_result,
            'open_interest': oi_data
        }
    }
    
    return report


def main():
    """Point d'entrée principal"""
    try:
        report = run_analysis()
        
        # Sauvegarder le rapport JSON
        output_file = 'analysis_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n✅ Rapport sauvegardé: {output_file}")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
