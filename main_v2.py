
"""
BTC Sniper Bot V2 - Institutional Grade
Point d'entrée principal avec tous les analyseurs avancés
"""
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any

import config
from exchange import BitgetConnector
from exchange_aggregator import MultiExchangeAggregator
from analyzers import (
    OrderBookAnalyzer,
    CVDAnalyzer,
    SqueezeAnalyzer,
    VolumeProfileAnalyzer,
    FundingLiquidationAnalyzer,
    FVGAnalyzer,
    EntropyAnalyzer,
    SpoofingDetector,
    DerivativesAnalyzer,
    OnChainAnalyzer,
    SentimentAnalyzer,
    MacroAnalyzer,
    OptionsAnalyzer,
    OpenInterestAnalyzer,
    SelfTradingDetector,
    VenturiAnalyzer,
    HyperliquidAnalyzer,
    MACDAnalyzer,
    SpotPerpDivergenceAnalyzer # Added SpotPerpDivergenceAnalyzer
)
from decision_engine_v2 import DecisionEngineV2
from consistency_checker import ConsistencyChecker


def run_analysis_v2(mode: str = 'full') -> Dict[str, Any]:
    """
    Exécute l'analyse complète v2 du marché
    
    Args:
        mode: 'full' (tous les analyseurs) ou 'fast' (core uniquement)
        
    Returns:
        Dict avec tous les résultats d'analyse et signaux
    """
    print("=" * 70)
    print("🏦 BTC SNIPER BOT V2 - INSTITUTIONAL GRADE")
    print("=" * 70)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Mode: {mode.upper()}")
    print("=" * 70)
    
    # ==========================================
    # 1. CONNEXIONS
    # ==========================================
    print("\n📡 Connexion aux exchanges...")
    
    # Bitget (exchange principal)
    connector = BitgetConnector()
    
    # Multi-exchange aggregator
    multi_agg = None # Initialize multi_agg
    if mode == 'full':
        try:
            multi_agg = MultiExchangeAggregator()  # Uses default 9 exchanges with fallback
            multi_exchange_data = multi_agg.get_aggregated_data()
            print(f"   ✅ {multi_exchange_data.get('exchanges_connected', 0)} exchanges connectés")
        except Exception as e:
            print(f"   ⚠️ Multi-exchange partiel: {e}")
            multi_exchange_data = {}
    else:
        multi_exchange_data = {}
    
    # ==========================================
    # 2. DONNÉES DE BASE
    # ==========================================
    print("\n📊 Récupération des données...")
    
    # Prix actuel
    if multi_exchange_data.get('price_analysis'):
        current_price = multi_exchange_data['price_analysis'].get('vwap', 0)
        print(f"   💰 VWAP Global: ${current_price:,.2f}")
    else:
        current_price = connector.get_current_price()
        print(f"   💰 Prix Bitget: ${current_price:,.2f}")
    
    # OHLCV Multi-Timeframe
    df_micro = connector.fetch_ohlcv(config.TIMEFRAME_MICRO, limit=1000)
    df_meso = connector.fetch_ohlcv(config.TIMEFRAME_MESO, limit=500)
    df_macro = connector.fetch_ohlcv(config.TIMEFRAME_MACRO, limit=30)
    
    print(f"   📈 Bougies: {len(df_micro)} (5m) | {len(df_meso)} (1h) | {len(df_macro)} (1d)")
    
    # Convertir en liste de dicts pour les Liquidation Zones et Smart Entry
    candles_5m = df_micro.to_dict('records') if df_micro is not None and len(df_micro) > 0 else []
    candles_1h = df_meso.to_dict('records') if df_meso is not None and len(df_meso) > 0 else []
    
    # Order Book
    order_book = connector.fetch_order_book(limit=config.ORDER_BOOK_LIMIT)
    
    # Trades pour CVD
    trades = connector.fetch_trades(limit=config.CVD_TRADES_LIMIT)
    print(f"   🔄 Trades récents: {len(trades)}")
    
    # Open Interest
    oi_data = connector.fetch_open_interest()
    
    # Funding Rate
    funding_data = connector.fetch_funding_rate()
    
    # ==========================================
    # 3. ANALYSES CORE
    # ==========================================
    print("\n🔬 Analyse des indicateurs core...")
    
    # Init storage
    analysis_results = {
        'indicators': {},
        'signals': {},
        'metadata': {}
    }
    
    # NEW: Liquidation Analysis (1H Swings)
    from analyzers.liquidation import LiquidationAnalyzer
    liq_analyzer = LiquidationAnalyzer(df_meso)
    liq_result = liq_analyzer.analyze(current_price)
    # analysis_results['indicators']['liquidation'] = liq_result # This line was in the instruction but analysis_results is not defined here. Assuming it's a placeholder or should be removed.
    print(f"   💀 Liquidation Heatmap: Longs @ {liq_result['nearest_long_liq']['price']} | Shorts @ {liq_result['nearest_short_liq']['price']}")

    # NEW: Premium Analysis (Spot vs Perp Gap)
    from analyzers.premium import PremiumAnalyzer
    prem_analyzer = PremiumAnalyzer()
    
    # Ensure raw_data exists (fallback to empty dict if multi_exchange_data is empty)
    raw_tickers = multi_exchange_data.get('raw_data', {}).get('tickers', {})
    
    prem_result = prem_analyzer.analyze(raw_tickers)
    analysis_results['indicators']['premium'] = prem_result
    print(f"   🇺🇸 Coinbase Premium: ${prem_result['gap_usd']} ({prem_result['signal']})")
    
    # Order Book Analysis
    ob_analyzer = OrderBookAnalyzer(order_book, current_price)
    ob_result = ob_analyzer.analyze()
    print(f"   📒 Order Book: {ob_result['pressure']} ({ob_result['bid_ratio_pct']}% bids)")
    
    # CVD Analysis (MTF)
    cvd_analyzer = CVDAnalyzer(trades)
    # Pass Hybrid Data: candles_1h (Meso) and candles_1d (Macro)
    cvd_result = cvd_analyzer.analyze_mtf({
        '1h': candles_1h,
        '1d': df_macro.to_dict('records') if df_macro is not None else []
    }) 
    print(f"   📊 CVD (MTF): {cvd_result['emoji']} {cvd_result['trend']} (Score: {cvd_result['composite_score']}) | Confluence: {cvd_result['confluence']}")
    if cvd_result.get('absorption_risk'):
        print("      ⚠️ ABSORPTION DETECTED: Passive liquidity is blocking aggression.")
        
    agg_status = cvd_result.get('aggression_status', 'BALANCED')
    if agg_status != 'BALANCED':
        agg_emoji = "🔥" if agg_status == 'BULLISH_AGGRESSION' else "🩸"
        print(f"      {agg_emoji} {agg_status.replace('_', ' ')}: Active market orders are pushing.")

    if cvd_result.get('mtf_data'):
        print("      Timeframe | Net CVD | Vol Buy | Vol Sell | Ratio | Trend")
        print("      " + "-" * 60)
        for tf in ['5m', '1h', '4h', '1d']:
            data = cvd_result['mtf_data'].get(tf)
            if data:
                print(f"      {tf:<9} | {data['net_cvd']:>7.2f} | {data['buy_volume']:>7.1f} | {data['sell_volume']:>8.1f} | {data['aggression_ratio']:>5.2f} | {data['trend']}")
    
    # Volume Profile Analysis
    vp_analyzer = VolumeProfileAnalyzer(df_micro)
    vp_result = vp_analyzer.analyze(current_price)
    regime_emoji = "⚖️" if vp_result['regime'] == 'BALANCE' else "🚀"
    print(f"   📊 Volume Profile: {regime_emoji} {vp_result['regime']} | Context: {vp_result['context']}")
    if vp_result.get('target_price'):
        print(f"      🎯 Structural Target: ${vp_result['target_price']:,.2f}")
    print(f"      POC: ${vp_result['poc']:,.2f} | VAH: ${vp_result['vah']:,.2f} | VAL: ${vp_result['val']:,.2f}")
    
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
    compression = entropy_result.get('compression', {}).get('current', 1.0)
    print(f"   ⚛️ Quantum State: {entropy_result.get('quantum_state', 'UNKNOWN')} ({compression:.3f})")
    
    # NEW: Oscillator Analysis (KDJ) - 1H/Meso timeframe (User request: reduce noise)
    from analyzers.oscillators import OscillatorAnalyzer
    osc_analyzer = OscillatorAnalyzer(
        high=df_meso['high'],
        low=df_meso['low'],
        close=df_meso['close']
    )
    osc_result = osc_analyzer.analyze()
    kdj = osc_result.get('values', {})
    print(f"   🌊 KDJ (1H): {osc_result['signal']} (J={kdj.get('j', 0):.1f} | Dev={kdj.get('deviation', 0):.1f}) | Score: {osc_result['score']}")
    
    # NEW: ADX Market Regime Analysis (1H)
    from analyzers.adx_analyzer import ADXAnalyzer
    adx_analyzer = ADXAnalyzer(
        high=df_meso['high'],
        low=df_meso['low'],
        close=df_meso['close']
    )
    adx_result = adx_analyzer.analyze()
    regime_emoji = "🔴" if adx_result['regime'] == 'RANGING' else "🟡" if adx_result['regime'] == 'TRANSITION' else "🟢"
    print(f"   📊 ADX (1H): {regime_emoji} {adx_result['regime']} (ADX={adx_result['adx']:.1f}) | Trend: {adx_result['trend_direction']}")
    
    # NEW: HTF Bias Analysis (4H approximated via longer lookback on 1H)
    from analyzers.htf_bias import HTFBiasAnalyzer
    htf_analyzer = HTFBiasAnalyzer(
        high=df_meso['high'],
        low=df_meso['low'],
        close=df_meso['close']
    )
    htf_result = htf_analyzer.analyze()
    bias_emoji = "🟢" if htf_result['bias'] == 'BULLISH' else "🔴" if htf_result['bias'] == 'BEARISH' else "⚪"
    print(f"   📈 HTF Bias: {bias_emoji} {htf_result['bias']} (Structure: {htf_result['structure']} | Momentum: {htf_result['momentum']})")
    
    # NEW: Cross-Asset Intelligence (DXY, SPX, M2)
    from analyzers.cross_asset import CrossAssetAnalyzer
    cross_asset_analyzer = CrossAssetAnalyzer()
    cross_asset_result = cross_asset_analyzer.analyze()
    
    # Display macro data
    regime_emoji = "🟢" if cross_asset_result['overall_regime'] == 'RISK_ON' else "🔴" if cross_asset_result['overall_regime'] == 'RISK_OFF' else "⚪"
    print(f"   🌍 Macro Regime: {regime_emoji} {cross_asset_result['overall_regime']} ({cross_asset_result['bullish_indicators']} Bull / {cross_asset_result['bearish_indicators']} Bear)")
    
    dxy = cross_asset_result.get('dxy', {})
    if dxy.get('available'):
        dxy_emoji = "📈" if dxy['trend'] == 'BULLISH' else "📉" if dxy['trend'] == 'BEARISH' else "➡️"
        print(f"      💵 DXY: {dxy_emoji} {dxy['value']} ({dxy['change_24h']:+.2f}%) → BTC {dxy['btc_impact']}")
    
    spx = cross_asset_result.get('spx', {})
    if spx.get('available'):
        spx_emoji = "📈" if spx['trend'] == 'BULLISH' else "📉" if spx['trend'] == 'BEARISH' else "➡️"
        market_status = "🔴 Closed" if not spx['market_open'] else "🟢 Open"
        print(f"      📊 S&P500: {spx_emoji} {spx['value']:,.0f} ({spx['change_24h']:+.2f}%) | {market_status}")
    
    m2 = cross_asset_result.get('m2', {})
    if m2.get('available'):
        m2_emoji = "💧" if m2['offset_90d_trend'] == 'EXPANDING' else "🏜️" if m2['offset_90d_trend'] == 'CONTRACTING' else "➡️"
        print(f"      💰 M2 Supply: {m2_emoji} ${m2['current']:,.0f}B (YoY: {m2['yoy_change']:+.2f}%) | Offset Trend: {m2['offset_90d_trend']}")
    
    # NEW: Event Calendar Analysis (FOMC, CPI, NFP)
    from analyzers.event_calendar import EventCalendarAnalyzer
    event_analyzer = EventCalendarAnalyzer()
    event_result = event_analyzer.analyze()
    
    if event_result.get('event_active'):
        print(f"   ⚠️ EVENT ACTIF: {event_result['current_event']} ({event_result['time_to_event']}) | Impact: {event_result['impact_level']} | -${event_result['confidence_penalty']} pts")
    else:
        next_event = event_result.get('next_event')
        if next_event:
            print(f"   📅 Prochain Event: {next_event} dans {event_result['next_event_in']} ({event_result['next_event_time_paris']} Paris)")
    
    # ==========================================
    # 4. ANALYSES AVANCÉES (mode full)
    # ==========================================
    spoofing_result = {}
    derivatives_result = {}
    onchain_result = {}
    sentiment_result = {}
    macro_result = {}
    options_result = {}
    venturi_result = {}
    self_trading_result = {}
    hyperliquid_result = {}
    macd_result = {}  # MACD 3D (initialized before mode check)
    squeeze_result = {} # Initialize for DecisionEngine
    spot_perp_results = {} # Initialize for DecisionEngine

    # MACD 3D Analysis (always run - needed for scoring)
    try:
        macd_analyzer = MACDAnalyzer()
        macd_result = macd_analyzer.analyze()
        if macd_result.get('available'):
            macd_emoji = "🟢" if macd_result['trend'] == 'BULLISH' else "🔴" if macd_result['trend'] == 'BEARISH' else "⚪"
            # Access 3D hist from MTF data
            hist_3d = macd_result.get('mtf_data', {}).get('3d', {}).get('hist', 0)
            print(f"   📈 MACD (3D): {macd_emoji} {macd_result['trend']} (Hist: {hist_3d:.2f})")
    except Exception as e:
        print(f"   ⚠️ MACD analysis failed: {e}")
    
    if mode == 'full':
        print("\n🔬 Analyse des indicateurs avancés...")
        
        # Spoofing Detection
        try:
            spoof_detector = SpoofingDetector()
            spoof_detector.add_orderbook_snapshot(
                order_book.get('bids', []),
                order_book.get('asks', [])
            )
            spoof_detector.add_trades_snapshot(trades)
            spoofing_result = spoof_detector.analyze()
            print(f"   🔍 Manipulation: {spoofing_result.get('risk_emoji', '⚪')} {spoofing_result.get('risk_level', 'UNKNOWN')}")
        except Exception as e:
            print(f"   ⚠️ Spoofing analysis failed: {e}")
        
        
        # Hyperliquid Advanced Analysis (R&D)
        hyperliquid_result = {}
        try:
            hl_analyzer = HyperliquidAnalyzer()
            hyperliquid_result = hl_analyzer.analyze()
            market = hyperliquid_result.get('market', {})
            whale = hyperliquid_result.get('whale_analysis', {})
            print(f"   🔷 Hyperliquid: OI {market.get('open_interest_btc', 0):,.0f} BTC | Whales {whale.get('sentiment', 'NEUTRAL')}")
        except Exception as e:
            print(f"   ⚠️ Hyperliquid analysis failed: {e}")

        # Open Interest Analysis (avancée avec delta)
        oi_analysis_result = {}
        target_open_interests = {}
        try:
            oi_analyzer = OpenInterestAnalyzer()
            
            # Récupérer l'OI de tous les exchanges
            if multi_exchange_data.get('open_interest', {}).get('by_exchange'):
                target_open_interests = multi_exchange_data['open_interest']['by_exchange'].copy()
            else:
                target_open_interests = {'bitget': oi_data.get('amount', 0)}
            
            # Ajouter l'OI Hyperliquid s'il est disponible
            if hyperliquid_result and hyperliquid_result.get('success'):
                hl_oi = hyperliquid_result.get('market', {}).get('open_interest_btc', 0)
                target_open_interests['hyperliquid'] = hl_oi
            
            oi_analysis_result = oi_analyzer.analyze(current_price, target_open_interests)
            delta = oi_analysis_result.get('delta', {})
            sig = oi_analysis_result.get('signal', {})
            
            if delta.get('available'):
                delta_1h = delta.get('1h', {}).get('delta_oi_pct', 0)
                print(f"   📊 Open Interest: {sig.get('emoji', '⚪')} {oi_analysis_result.get('total_oi_btc', 0):,.0f} BTC | Δ1h: {delta_1h:+.2f}%")
                
                # Quantum Squeeze Analysis (R&D Point 2) - Moved here to have delta_1h
                squeeze_analyzer = SqueezeAnalyzer(candles_1h, delta_1h)
                squeeze_result = squeeze_analyzer.analyze()
                print(f"   🍋 Squeeze: {squeeze_result['emoji']} {squeeze_result['status']} (Intensity: {squeeze_result['intensity']}) | Score: {squeeze_result['squeeze_score']}")
            else:
                print(f"   📊 Open Interest: {oi_analysis_result.get('total_oi_btc', 0):,.0f} BTC (tracking)")
        except Exception as e:
            print(f"   ⚠️ OI Analysis failed: {e}")
            if not target_open_interests:
                 target_open_interests = {'bitget': oi_data.get('amount', 0)}

        # Derivatives Analysis
        try:
            funding_rates = {}
            open_interests = target_open_interests
            
            if multi_exchange_data.get('funding_analysis'):
                for ex, rate in multi_exchange_data['funding_analysis'].get('by_exchange', {}).items():
                    funding_rates[ex] = rate / 100  # Reconvertir en décimal
            else:
                funding_rates = {'bitget': funding_data.get('current', 0)}
            
            # Injecter Funding Rate Hyperliquid (Normalisé à 8h pour comparaison)
            if hyperliquid_result and hyperliquid_result.get('success'):
                # Hyperliquid donne un taux horaire. CEXs donnent 8h.
                hl_funding_1h = hyperliquid_result.get('market', {}).get('funding_rate', 0)
                funding_rates['hyperliquid'] = hl_funding_1h * 8
            
            deriv_analyzer = DerivativesAnalyzer()
            
            # 2. Analyze Derivatives (using OI analysis result)
            derivatives_result = deriv_analyzer.analyze(current_price, funding_rates, open_interests, oi_analysis_result)
            
            deriv_sentiment = derivatives_result.get('sentiment', {})
            print(f"   📈 Derivatives: {deriv_sentiment.get('emoji', '⚪')} {deriv_sentiment.get('sentiment', 'NEUTRAL')}")
        except Exception as e:
            print(f"   ⚠️ Derivatives analysis failed: {e}")
        
        # On-Chain Analysis
        try:
            onchain_analyzer = OnChainAnalyzer()
            onchain_result = onchain_analyzer.analyze(current_price)
            onchain_score = onchain_result.get('score', {})
            print(f"   ⛓️ On-Chain: {onchain_score.get('emoji', '⚪')} {onchain_score.get('sentiment', 'NEUTRAL')}")
        except Exception as e:
            print(f"   ⚠️ On-chain analysis failed: {e}")
        
        # Sentiment Analysis
        try:
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_result = sentiment_analyzer.analyze()
            fg = sentiment_result.get('fear_greed', {})
            print(f"   😱 Sentiment: {fg.get('emoji', '😐')} Fear & Greed: {fg.get('value', 50)} ({fg.get('classification', 'Neutral')})")
        except Exception as e:
            print(f"   ⚠️ Sentiment analysis failed: {e}")
        
        # Macro Analysis
        try:
            macro_analyzer = MacroAnalyzer()
            macro_result = macro_analyzer.analyze()
            risk_env = macro_result.get('risk_environment', {})
            print(f"   🌍 Macro: {risk_env.get('emoji', '⚪')} {risk_env.get('environment', 'NEUTRAL')}")
        except Exception as e:
            print(f"   ⚠️ Macro analysis failed: {e}")
        
        # Options Deribit Analysis
        try:
            options_analyzer = OptionsAnalyzer()
            options_result = options_analyzer.analyze(current_price)
            mp = options_result.get('max_pain', {})
            pcr = options_result.get('put_call_ratio', {})
            print(f"   🎰 Options: Max Pain ${mp.get('max_pain_price', 0):,.0f} | PCR {pcr.get('pcr_oi', 0):.2f} {pcr.get('emoji', '')}")
        except Exception as e:
            print(f"   ⚠️ Options analysis failed: {e}")
        

        # Open Interest Analysis (Moved up before Derivatives Analysis)

        # Block moved

        # 5. Multi-Exchange CVD (Spot vs Perp)
        # ---------------------------------------------
        try:
            # Fetch Global CVD Data (1h, 4h)
            print("   🌍 Fetching Spot vs Perp data...")
            if multi_agg: # Ensure multi_agg is initialized
                global_cvd_data = multi_agg.fetch_global_cvd_candles(timeframes=['1h', '4h'])
                
                spot_perp_analyzer = SpotPerpDivergenceAnalyzer()
                spot_perp_results = spot_perp_analyzer.analyze(global_cvd_data)
                
                print(f"   📊 Spot/Perp Divergence (1h): {spot_perp_results.get('1h', {}).get('regime', 'N/A')}")
            else:
                print("   ⚠️ Multi-exchange aggregator not available for Spot/Perp analysis.")
        except Exception as e:
            print(f"   ❌ Spot/Perp Analysis Failed: {e}")
            spot_perp_results = {}

        # 6. Décision Finale
        # ---------------------------------------------
        
        # Self-Trading Detection (Fluid Dynamics R&D)
        self_trading_result = {}
        try:
            detector = SelfTradingDetector()
            self_trading_result = detector.analyze(trades, current_price, cvd_result)
            if self_trading_result.get('detected'):
                print(f"   🔍 Self-Trading: ⚠️ {self_trading_result['type']} détecté ({self_trading_result['probability']:.0f}%)")
            else:
                print(f"   🔍 Self-Trading: ✅ Marché sain")
        except Exception as e:
            print(f"   ⚠️ Self-Trading detection failed: {e}")
        
        # Venturi Analysis (Fluid Dynamics R&D)
        venturi_result = {}
        try:
            venturi = VenturiAnalyzer()
            venturi_result = venturi.analyze(order_book)
            if venturi_result.get('compression_detected'):
                print(f"   🌊 Venturi: ⚡ Compression → {venturi_result['direction']} ({venturi_result['breakout_probability']:.0f}%)")
            else:
                print(f"   🌊 Venturi: Marché fluide")
        except Exception as e:
            print(f"   ⚠️ Venturi analysis failed: {e}")
        

    
    # ==========================================
    # 5. DECISION ENGINE V2
    # ==========================================
    print("\n🧠 Génération du signal composite...")
    
    # Consistency data will be empty for now - the notifier handles full consistency
    # The quality filters in DecisionEngineV2 will use defaults when empty
    consistency_data = {}
    
    engine = DecisionEngineV2(
        current_price=current_price,
        order_book_data=ob_result,
        cvd_data=cvd_result,
        volume_profile_data=vp_result,
        fvg_data=fvg_result,
        entropy_data=entropy_result,
        kdj_data=osc_result,  # KDJ Oscillator
        adx_data=adx_result,  # ADX Market Regime
        htf_data=htf_result,  # HTF Bias
        cross_asset_data=cross_asset_result,  # DXY, SPX, M2
        event_data=event_result,  # FOMC, CPI, NFP events
        multi_exchange_data=multi_exchange_data,
        spoofing_data=spoofing_result,
        derivatives_data=derivatives_result,
        onchain_data=onchain_result,
        sentiment_data=sentiment_result,
        macro_data=macro_result,
        open_interest=oi_analysis_result if (mode == 'full' and oi_analysis_result) else oi_data,
        options_data=options_result,
        trading_style='intraday_1h_2d',  # Poids optimisés pour signaux 1H-2D
        consistency_data=consistency_data,  # For quality filters
        candles_5m=candles_5m,  # Pour les zones de liquidation
        candles_1h=candles_1h,  # Pour Smart Entry (Robustesse)
        venturi_data=venturi_result,  # Fluid dynamics - Venturi
        self_trading_data=self_trading_result,  # Fluid dynamics - Self-Trading
        hyperliquid_data=hyperliquid_result,  # Whale tracking sentiment
        macd_data=macd_result,  # MACD 3D for HTF trend confirmation
        squeeze_data=squeeze_result,  # NEW R&D Squeeze metric
        spot_perp_data=spot_perp_results # NEW Spot/Perp Divergence
    )
    
    decision_result = engine.generate_composite_signal()
    
    # ==========================================
    # 6. AFFICHAGE DU SIGNAL
    # ==========================================
    primary = decision_result['primary_signal']
    
    print("\n" + "=" * 70)
    print("📢 SIGNAL COMPOSITE:")
    print("=" * 70)
    print(f"   {primary['emoji']} {primary['description']}")
    print(f"   Direction: {primary['direction']} | Confiance: {primary['confidence']:.0f}/100")
    print(f"   Strength: {decision_result['signal_strength']}")
    
    if primary.get('manipulation_penalty', 0) > 0:
        print(f"   ⚠️ Pénalité manipulation: -{primary['manipulation_penalty']:.0f}")
    
    print("\n📊 Scores par dimension:")
    for dim, score in decision_result['dimension_scores'].items():
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        bias = "🟢" if score > 55 else "🔴" if score < 45 else "⚪"
        print(f"   {dim:15s}: {bar} {score:5.1f} {bias}")
    
    if primary['reasons']:
        print("\n📝 Raisons:")
        for reason in primary['reasons']:
            print(f"   • {reason}")
    
    if primary['targets']:
        print("\n🎯 Targets:")
        for key, val in primary['targets'].items():
            # Skip metadata keys (prefixed with _) and non-numeric values
            if key.startswith('_'):
                continue
            if isinstance(val, (int, float)):
                print(f"   • {key}: ${val:,.2f}")
    
    if primary['warnings']:
        print("\n⚠️ Avertissements:")
        for warning in primary['warnings']:
            print(f"   {warning}")
    
    print("\n" + "=" * 70)
    print(f"💡 Recommandation: {'TRADEABLE' if decision_result['tradeable'] else 'ATTENDRE'}")
    print("=" * 70)
    
    # ==========================================
    # 7. CONSTRUCTION DU RAPPORT
    # ==========================================
    report = {
        'version': '2.0',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'mode': mode,
        'symbol': config.SYMBOL,
        'price': current_price,
        
        # Signal principal
        'signal': decision_result['primary_signal'],
        'dimension_scores': decision_result['dimension_scores'],
        'composite_score': decision_result['composite_score'],
        'signal_strength': decision_result['signal_strength'],
        'tradeable': decision_result['tradeable'],
        'market_context': decision_result['market_context'],
        
        # Détails des indicateurs
        'indicators': {
            'order_book': ob_result,
            'cvd': cvd_result,
            'volume_profile': vp_result,
            'fvg': {
                'total_active': fvg_result.get('total_active', 0),
                'nearest_bull': fvg_result.get('nearest_bull'),
                'nearest_bear': fvg_result.get('nearest_bear')
            },
            'entropy': entropy_result,
            'kdj': osc_result,  # New KDJ result
            'multi_exchange': {
                'exchanges_connected': multi_exchange_data.get('exchanges_connected', 0),
                'vwap': multi_exchange_data.get('price_analysis', {}).get('vwap'),
                'spread_bps': multi_exchange_data.get('price_analysis', {}).get('spread_bps'),
                'funding_divergence': multi_exchange_data.get('funding_analysis', {}).get('divergence_pct'),
                'arbitrage': multi_exchange_data.get('arbitrage')
            } if multi_exchange_data else None,
            'spoofing': spoofing_result if spoofing_result else None,
            'derivatives': {
                'sentiment': derivatives_result.get('sentiment'),
                'liquidations': derivatives_result.get('liquidations', {}).get('magnet')
            } if derivatives_result else None,
            'onchain': onchain_result.get('score') if onchain_result else None,
            'sentiment': {
                'fear_greed': sentiment_result.get('fear_greed'),
                'overall': sentiment_result.get('overall')
            } if sentiment_result else None,
            'macro': {
                'risk_environment': macro_result.get('risk_environment'),
                'btc_impact': macro_result.get('btc_impact')
            } if macro_result else None,
            'options': {
                'max_pain': options_result.get('max_pain'),
                'put_call_ratio': options_result.get('put_call_ratio'),
                'iv_analysis': options_result.get('iv_analysis'),
                'score': options_result.get('score')
            } if options_result else None,
            'squeeze': squeeze_result,
            'open_interest': oi_analysis_result if mode == 'full' else oi_data, # Use full analysis result in full mode
            'adx': adx_result,  # ADX market regime data
            'macd': macd_result,  # MACD MTF data
            'mtf_macd': decision_result['primary_signal'].get('mtf_macd'),  # MTF MACD from signal
            'smart_entry': decision_result['primary_signal'].get('smart_entry'),  # Smart Entry from signal
            'fluid_dynamics': {
                'venturi': venturi_result if venturi_result else None,
                'self_trading': self_trading_result if self_trading_result else None
            },
            'hyperliquid': hyperliquid_result if hyperliquid_result else None,
            'cross_asset': cross_asset_result # Add cross_asset result explicitly
        }
    }
    
    return report


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='BTC Sniper Bot V2 - Institutional Grade')
    parser.add_argument('--mode', choices=['full', 'fast'], default='full',
                       help='Mode d\'analyse: full (tous les indicateurs) ou fast (core uniquement)')
    parser.add_argument('--output', default='analysis_report.json',
                       help='Fichier de sortie JSON')
    
    args = parser.parse_args()
    
    try:
        report = run_analysis_v2(mode=args.mode)
        
        # Sauvegarder le rapport
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n✅ Rapport sauvegardé: {args.output}")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
