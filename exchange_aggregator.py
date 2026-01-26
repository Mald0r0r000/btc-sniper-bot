"""
Multi-Exchange Aggregator
Agrège les données de plusieurs exchanges pour une vision globale du marché
Exchanges supportés: Binance, OKX, Bybit, Bitget
"""
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import time

import config


class ExchangeConnection:
    """Connexion à un exchange individuel"""
    
    def __init__(self, exchange_id: str, api_key: str = '', secret: str = '', password: str = '', market_type: str = 'swap'):
        self.exchange_id = exchange_id
        self.market_type = market_type
        
        exchange_config = {
            'enableRateLimit': True,
            'options': {'defaultType': market_type},
            'timeout': 15000  # 15s timeout
        }

        if api_key:
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = secret
            if password:
                exchange_config['password'] = password
        
        # Initialize exchange
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Exchange {exchange_id} not supported by ccxt")
            
        self.exchange = getattr(ccxt, exchange_id)(exchange_config)
        
        # Standardize symbol based on market type
        # Most exchanges: Spot = BTC/USDT, Perp = BTC/USDT:USDT (or similar)
        # We handle specific quirks here
        if exchange_id == 'kraken' and market_type == 'spot':
            self.symbol = 'BTC/USD'
        elif exchange_id == 'coinbase' and market_type == 'spot':
            self.symbol = 'BTC/USD'
        elif exchange_id == 'hyperliquid':
            self.symbol = 'BTC/USDC:USDC'
        else:
            # Default for major exchanges (Binance, Bybit, OKX, etc.)
            self.symbol = 'BTC/USDT' if market_type == 'spot' else 'BTC/USDT:USDT'
    
    def fetch_ohlcv(self, timeframe: str = '1h', limit: int = 100) -> Dict[str, Any]:
        """
        Fetches OHLCV candles
        Returns standardized data with volume in BTC
        """
        try:
            # Determine limit based on timeframe to optimize data fetch
            candles = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            
            # Simple conversion: timestamp, open, high, low, close, volume
            data = []
            for candle in candles:
                ts = candle[0]
                o, h, l, c, v = map(float, candle[1:6])
                
                # Normalization: ensure volume is in BTC
                # Heuristic: If volume is suspiciously large (> 100,000), it's likely in Quote Currency (USD/USDT) or Contracts
                # because 100k BTC per hour on a single exchange is extremely rare.
                if v > 100000: 
                     # Assume it's USD/Contract volume, convert to BTC
                     if c > 0:
                        v = v / c
                
                data.append({
                    'timestamp': ts,
                    'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
                    'datetime': datetime.fromtimestamp(ts/1000, timezone.utc).isoformat()
                })
                
            return {
                'exchange': self.exchange_id,
                'market_type': self.market_type,
                'timeframe': timeframe,
                'data': data,
                'success': True
            }
        except Exception as e:
            return {
                'exchange': self.exchange_id, 
                'market_type': self.market_type,
                'error': str(e), 
                'success': False
            }

    def fetch_order_book(self, limit: int = 50) -> Dict[str, Any]:
        """Récupère l'order book"""
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=limit)
            return {
                'exchange': self.exchange_id,
                'bids': ob['bids'],
                'asks': ob['asks'],
                'timestamp': ob.get('timestamp', int(time.time() * 1000)),
                'success': True
            }
        except Exception as e:
            return {'exchange': self.exchange_id, 'error': str(e), 'success': False}
    
    def fetch_ticker(self, retries: int = 3) -> Dict[str, Any]:
        """Récupère le ticker avec retry"""
        import time
        
        last_error = None
        for attempt in range(retries):
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                
                # Safe float conversion (Binance renvoie parfois None)
                def safe_float(value, default=0.0):
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                return {
                    'exchange': self.exchange_id,
                    'last': safe_float(ticker.get('last')),
                    'bid': safe_float(ticker.get('bid')),
                    'ask': safe_float(ticker.get('ask')),
                    'volume_24h': safe_float(ticker.get('quoteVolume')),
                    'change_24h': safe_float(ticker.get('percentage')),
                    'success': True
                }
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Délai progressif
        
        return {'exchange': self.exchange_id, 'error': str(last_error), 'success': False}
    
    def fetch_funding_rate(self) -> Dict[str, Any]:
        """Récupère le funding rate"""
        try:
            funding = self.exchange.fetch_funding_rate(self.symbol)
            return {
                'exchange': self.exchange_id,
                'funding_rate': float(funding.get('fundingRate', 0) or 0),
                'next_funding_time': funding.get('fundingDatetime'),
                'success': True
            }
        except Exception as e:
            return {'exchange': self.exchange_id, 'error': str(e), 'success': False}
    
    def fetch_open_interest(self) -> Dict[str, Any]:
        """Récupère l'open interest"""
        try:
            oi = self.exchange.fetch_open_interest(self.symbol)
            return {
                'exchange': self.exchange_id,
                'open_interest': float(oi.get('openInterestAmount', 0) or 0),
                'success': True
            }
        except Exception as e:
            return {'exchange': self.exchange_id, 'error': str(e), 'success': False}


class MultiExchangeAggregator:
    """Agrégateur multi-exchange pour vision globale du marché
    
    Features:
    - Fallback automatique si exchange indisponible (ex: Binance géobloqué)
    - Recalcul dynamique des poids basé sur exchanges disponibles
    - Logging détaillé des connexions
    """
    
    # Poids de volume approximatifs par exchange (basé sur volume réel)
    # Tier 1: Gros volumes (Binance, OKX, Bybit)
    # Tier 2: Volumes moyens (Bitget, MEXC, Gate.io)
    # Tier 3: Volumes plus faibles (Phemex, BitMEX)
    BASE_WEIGHTS = {
        # Tier 1 - Majeurs (géobloqués sur GitHub Actions)
        'binance': 0.30,
        'okx': 0.15,
        'bybit': 0.12,
        # Tier 2 - Secondaires
        'bitget': 0.12,
        'mexc': 0.10,
        'gateio': 0.08,
        # Tier 3 - Fallback US-friendly
        'phemex': 0.05,
        'bitmex': 0.04,
        # US Friendly Replacements (compensent la perte de Binance/Bybit)
        'kraken': 0.10,
        'coinbase': 0.08,
        'binanceus': 0.02,
        # DEX - Pas de géoblocage, gros volumes
        'hyperliquid': 0.08
    }
    
    # Erreurs indiquant un géoblocage
    GEOBLOCK_ERRORS = [
        'Service unavailable',
        'IP has been restricted',
        'forbidden',
        '403',
        'not available in your region',
        'restricted location'
    ]
    
    def __init__(self, exchanges: List[str] = None):
        """
        Args:
            exchanges: Liste des exchanges à utiliser. 
                      Par défaut: 8 exchanges (majeurs + fallback US-friendly)
        """
        if exchanges is None:
            # Full list based on user request (excluding blocked ones)
            # Spot Exchanges
            self.spot_list = [
                'bybit', 'bitget', 'gateio', 'mexc', 
                'kraken', 'coinbase', 'kucoin', 'huobi'
            ]
            # Perp Exchanges
            self.perp_list = [
                'bybit', 'bitget', 'gateio', 'mexc', 'kraken',
                'hyperliquid', 'huobi'
            ]
            # (Note: Binance/OKX excluded to avoid geoblocking issues)
            
            exchanges = list(set(self.spot_list + self.perp_list))
        
        self.requested_exchanges = exchanges
        self.exchanges = {} # Stores connection objects: "exchange_id:market_type" -> Connection
        self.failed_exchanges = {}
        self.dynamic_weights = {}

        # Initialize Connections
        # KEY format: "exchange_id:market_type" (e.g. "bybit:spot", "bybit:swap")
        
        # 1. Initialize Spot Connections
        for ex_id in self.spot_list:
            key = f"{ex_id}:spot"
            try:
                self.exchanges[key] = ExchangeConnection(ex_id, market_type='spot')
            except Exception as e:
                self._handle_conn_error(key, e)

        # 2. Initialize Perp Connections
        for ex_id in self.perp_list:
            key = f"{ex_id}:swap"
            try:
                # Use credentials for Bitget if available
                if ex_id == 'bitget' and config.API_KEY:
                     self.exchanges[key] = ExchangeConnection(
                        ex_id, config.API_KEY, config.API_SECRET, config.API_PASSWORD, market_type='swap'
                    )
                else:
                    self.exchanges[key] = ExchangeConnection(ex_id, market_type='swap')
            except Exception as e:
                 self._handle_conn_error(key, e)

        # Calculer les poids dynamiques
        self._recalculate_weights()
    
    def _handle_conn_error(self, key, e):
        """Log connection errors"""
        error_msg = str(e).lower()
        is_geoblock = any(geo in error_msg for geo in self.GEOBLOCK_ERRORS)
        self.failed_exchanges[key] = {
            'error': str(e),
            'is_geoblock': is_geoblock
        }
        if is_geoblock:
            print(f"   🌍 {key.upper()} géobloqué (IP restriction)")
        else:
            print(f"   ⚠️ Impossible de connecter {key}: {e}")

    def _recalculate_weights(self, available_keys: List[str] = None) -> Dict[str, float]:
        """
        Recalculates weights. For now, we just give equal weight 
        to all available exchanges in a category.
        """
        if available_keys is None:
            available_keys = list(self.exchanges.keys())
            
        if not available_keys:
            return {}
            
        # Simplified Equal Weighting
        count = len(available_keys)
        self.dynamic_weights = {k: 1.0/count for k in available_keys}
        
        return self.dynamic_weights
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Retourne le statut de connexion aux exchanges"""
        return {
            'requested': self.requested_exchanges,
            'connected': list(self.exchanges.keys()),
            'failed': self.failed_exchanges,
            'weights': self.dynamic_weights,
            'data_quality': len(self.exchanges) / len(self.requested_exchanges) * 100
        }
    
    def _parallel_fetch(self, method_name: str, **kwargs) -> Dict[str, Dict]:
        """Exécute une méthode en parallèle sur tous les exchanges"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for ex_id, connection in self.exchanges.items():
                method = getattr(connection, method_name)
                futures[executor.submit(method, **kwargs)] = ex_id
            
            for future in as_completed(futures, timeout=20):
                ex_id = futures[future]
                try:
                    results[ex_id] = future.result()
                except Exception as e:
                    results[ex_id] = {'exchange': ex_id, 'error': str(e), 'success': False}
        
        return results
    
    def fetch_global_cvd_candles(self, timeframes: List[str] = ['1h']) -> Dict[str, Any]:
        """
        Fetches OHLCV data for all Spot and Perp exchanges for CVD analysis.
        """
        print(f"   📡 Fetching Global CVD Data ({len(self.exchanges)} sources)...")
        results = {}
        
        with ThreadPoolExecutor(max_workers=20) as executor: # Higher workers for IO bound
            futures = []
            for key, connection in self.exchanges.items():
                for tf in timeframes:
                    futures.append(executor.submit(connection.fetch_ohlcv, timeframe=tf, limit=100))
            
            for future in as_completed(futures, timeout=30):
                try:
                    res = future.result()
                    if res['success']:
                        ex_id = res['exchange']
                        m_type = res['market_type']
                        tf = res['timeframe']
                        
                        if tf not in results: results[tf] = {}
                        if m_type not in results[tf]: results[tf][m_type] = {}
                        
                        results[tf][m_type][ex_id] = res['data']
                except Exception as e:
                    pass
                    
        return results
    
    def get_aggregated_data(self) -> Dict[str, Any]:
        """
        Récupère toutes les données agrégées
        
        Returns:
            Dict complet avec order books, tickers, funding, OI agrégés
        """
        print("   📡 Agrégation multi-exchange...")
        
        # Fetch en parallèle
        order_books = self._parallel_fetch('fetch_order_book', limit=50)
        tickers = self._parallel_fetch('fetch_ticker')
        funding_rates = self._parallel_fetch('fetch_funding_rate')
        open_interests = self._parallel_fetch('fetch_open_interest')
        
        # Identifier les exchanges avec données valides
        available_exchanges = [
            ex_id for ex_id, ticker in tickers.items() 
            if ticker.get('success')
        ]
        
        # Mettre à jour les poids dynamiques
        self._recalculate_weights(available_exchanges)
        
        # Log status
        connected_list = available_exchanges
        failed_list = [ex for ex in self.requested_exchanges if ex not in available_exchanges]
        
        print(f"   ✅ {len(available_exchanges)} exchanges connectés: {', '.join(connected_list)}")
        if failed_list:
            print(f"   ❌ {len(failed_list)} échecs: {', '.join(failed_list)}")
            
        failed_now = [ex for ex in self.exchanges if ex not in available_exchanges]
        if failed_now:
            for ex in failed_now:
                error = tickers.get(ex, {}).get('error', 'Unknown')
                is_geoblock = any(geo in str(error).lower() for geo in self.GEOBLOCK_ERRORS)
                if is_geoblock:
                    print(f"   🌍 {ex.upper()} géobloqué - poids redistribués")
                else:
                    print(f"   ⚠️ {ex.upper()} indisponible: {error[:50]}")
        
        if len(available_exchanges) < len(self.exchanges):
            print(f"   📊 Poids recalculés: {', '.join(f'{k}:{v:.0%}' for k,v in self.dynamic_weights.items())}")
        
        # Agrégation
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'exchanges_connected': len(available_exchanges),
            'exchanges_requested': len(self.requested_exchanges),
            'exchanges_available': available_exchanges,
            'exchanges_failed': failed_now,
            'dynamic_weights': self.dynamic_weights,
            'data_quality_pct': round(len(available_exchanges) / len(self.requested_exchanges) * 100, 1),
            'global_orderbook': self._aggregate_orderbooks(order_books),
            'price_analysis': self._analyze_prices(tickers),
            'funding_analysis': self._analyze_funding(funding_rates),
            'open_interest': self._aggregate_open_interest(open_interests),
            'arbitrage': self._detect_arbitrage(tickers),
            'raw_data': {
                'order_books': order_books,
                'tickers': tickers,
                'funding_rates': funding_rates,
                'open_interests': open_interests
            }
        }
        
        return result
    
    def _aggregate_orderbooks(self, order_books: Dict[str, Dict]) -> Dict[str, Any]:
        """Agrège les order books de tous les exchanges"""
        all_bids = []
        all_asks = []
        total_bid_volume = 0
        total_ask_volume = 0
        
        for ex_id, ob in order_books.items():
            if not ob.get('success'):
                continue
            
            weight = self.dynamic_weights.get(ex_id, 0.1)
            
            for bid in ob.get('bids', [])[:30]:
                all_bids.append({
                    'price': bid[0],
                    'volume': bid[1],
                    'exchange': ex_id,
                    'weighted_volume': bid[1] * weight
                })
                total_bid_volume += bid[1]
            
            for ask in ob.get('asks', [])[:30]:
                all_asks.append({
                    'price': ask[0],
                    'volume': ask[1],
                    'exchange': ex_id,
                    'weighted_volume': ask[1] * weight
                })
                total_ask_volume += ask[1]
        
        # Trier et consolider
        all_bids.sort(key=lambda x: x['price'], reverse=True)
        all_asks.sort(key=lambda x: x['price'])
        
        # Trouver les plus gros murs
        biggest_bid = max(all_bids, key=lambda x: x['volume']) if all_bids else None
        biggest_ask = max(all_asks, key=lambda x: x['volume']) if all_asks else None
        
        # Calculer la profondeur en USD
        bid_depth_usd = sum(b['price'] * b['volume'] for b in all_bids) / 1_000_000
        ask_depth_usd = sum(a['price'] * a['volume'] for a in all_asks) / 1_000_000
        
        return {
            'total_bid_volume_btc': round(total_bid_volume, 2),
            'total_ask_volume_btc': round(total_ask_volume, 2),
            'bid_depth_m_usd': round(bid_depth_usd, 2),
            'ask_depth_m_usd': round(ask_depth_usd, 2),
            'imbalance_pct': round((total_bid_volume / (total_bid_volume + total_ask_volume)) * 100, 1) if total_bid_volume + total_ask_volume > 0 else 50,
            'biggest_bid_wall': {
                'price': biggest_bid['price'] if biggest_bid else 0,
                'volume': biggest_bid['volume'] if biggest_bid else 0,
                'exchange': biggest_bid['exchange'] if biggest_bid else '',
                'value_m': round(biggest_bid['price'] * biggest_bid['volume'] / 1_000_000, 2) if biggest_bid else 0
            },
            'biggest_ask_wall': {
                'price': biggest_ask['price'] if biggest_ask else 0,
                'volume': biggest_ask['volume'] if biggest_ask else 0,
                'exchange': biggest_ask['exchange'] if biggest_ask else '',
                'value_m': round(biggest_ask['price'] * biggest_ask['volume'] / 1_000_000, 2) if biggest_ask else 0
            },
            'consolidated_bids': all_bids[:20],
            'consolidated_asks': all_asks[:20]
        }
    
    def _analyze_prices(self, tickers: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyse les prix sur tous les exchanges"""
        prices = []
        volumes = []
        valid_tickers = []
        
        for ex_id, ticker in tickers.items():
            if ticker.get('success') and ticker.get('last', 0) > 0:
                prices.append(ticker['last'])
                volumes.append(ticker.get('volume_24h', 0))
                valid_tickers.append(ticker)
        
        if not prices:
            return {'error': 'No valid price data'}
        
        # VWAP (Volume Weighted Average Price)
        total_volume = sum(volumes)
        if total_volume > 0:
            vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        else:
            vwap = np.mean(prices)
        
        # Spread max entre exchanges
        max_price = max(prices)
        min_price = min(prices)
        spread_usd = max_price - min_price
        spread_bps = (spread_usd / min_price) * 10000  # Basis points
        
        # Exchange avec meilleur prix
        best_bid_exchange = max(valid_tickers, key=lambda x: x.get('bid', 0))
        best_ask_exchange = min(valid_tickers, key=lambda x: x.get('ask', float('inf')))
        
        return {
            'vwap': round(vwap, 2),
            'mean_price': round(np.mean(prices), 2),
            'max_price': round(max_price, 2),
            'min_price': round(min_price, 2),
            'spread_usd': round(spread_usd, 2),
            'spread_bps': round(spread_bps, 2),
            'total_volume_24h_usd': round(total_volume, 0),
            'best_bid': {
                'exchange': best_bid_exchange['exchange'],
                'price': best_bid_exchange.get('bid', 0)
            },
            'best_ask': {
                'exchange': best_ask_exchange['exchange'],
                'price': best_ask_exchange.get('ask', 0)
            },
            'by_exchange': {t['exchange']: {'last': t['last'], 'volume_24h': t.get('volume_24h', 0)} 
                          for t in valid_tickers}
        }
    
    def _analyze_funding(self, funding_rates: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyse les funding rates sur tous les exchanges"""
        rates = {}
        valid_rates = []
        
        for ex_id, fr in funding_rates.items():
            if fr.get('success'):
                rate = fr.get('funding_rate', 0)
                rates[ex_id] = round(rate * 100, 4)  # En pourcentage
                valid_rates.append(rate)
        
        if not valid_rates:
            return {'error': 'No valid funding data'}
        
        avg_rate = np.mean(valid_rates)
        max_rate = max(valid_rates)
        min_rate = min(valid_rates)
        
        # Divergence analysis
        divergence = max_rate - min_rate
        expensive_exchange = max(rates.items(), key=lambda x: x[1])[0] if rates else None
        cheap_exchange = min(rates.items(), key=lambda x: x[1])[0] if rates else None
        
        # Signal
        if avg_rate > 0.0005:
            signal = "LONGS_EXPENSIVE"
            signal_emoji = "⚠️"
        elif avg_rate < -0.0001:
            signal = "SHORTS_PAYING"
            signal_emoji = "🚀"
        else:
            signal = "NEUTRAL"
            signal_emoji = "⚪"
        
        return {
            'average_rate_pct': round(avg_rate * 100, 4),
            'max_rate_pct': round(max_rate * 100, 4),
            'min_rate_pct': round(min_rate * 100, 4),
            'divergence_pct': round(divergence * 100, 4),
            'by_exchange': rates,
            'expensive_exchange': expensive_exchange,
            'cheap_exchange': cheap_exchange,
            'signal': signal,
            'signal_emoji': signal_emoji,
            'annualized_avg_pct': round(avg_rate * 3 * 365 * 100, 2)
        }
    
    def _aggregate_open_interest(self, open_interests: Dict[str, Dict]) -> Dict[str, Any]:
        """Agrège l'open interest de tous les exchanges"""
        by_exchange = {}
        total_oi = 0
        
        for ex_id, oi in open_interests.items():
            if oi.get('success'):
                oi_amount = oi.get('open_interest', 0)
                by_exchange[ex_id] = round(oi_amount, 2)
                total_oi += oi_amount
        
        return {
            'total_oi_btc': round(total_oi, 2),
            'total_oi_usd_b': round(total_oi * 90000 / 1_000_000_000, 2),  # Approximatif
            'by_exchange': by_exchange,
            'leader': max(by_exchange.items(), key=lambda x: x[1])[0] if by_exchange else None
        }
    
    def _detect_arbitrage(self, tickers: Dict[str, Dict]) -> Dict[str, Any]:
        """Détecte les opportunités d'arbitrage"""
        valid = [(ex, t) for ex, t in tickers.items() 
                 if t.get('success') and t.get('bid', 0) > 0 and t.get('ask', 0) > 0]
        
        if len(valid) < 2:
            return {'opportunity': False}
        
        # Meilleur bid (où vendre)
        best_bid = max(valid, key=lambda x: x[1].get('bid', 0))
        # Meilleur ask (où acheter)
        best_ask = min(valid, key=lambda x: x[1].get('ask', float('inf')))
        
        spread = best_bid[1]['bid'] - best_ask[1]['ask']
        spread_pct = (spread / best_ask[1]['ask']) * 100 if best_ask[1]['ask'] > 0 else 0
        
        has_opportunity = spread > 0
        
        return {
            'opportunity': has_opportunity,
            'buy_exchange': best_ask[0],
            'buy_price': round(best_ask[1]['ask'], 2),
            'sell_exchange': best_bid[0],
            'sell_price': round(best_bid[1]['bid'], 2),
            'spread_usd': round(spread, 2),
            'spread_pct': round(spread_pct, 4),
            'description': f"Buy {best_ask[0].upper()} @ ${best_ask[1]['ask']:.2f}, Sell {best_bid[0].upper()} @ ${best_bid[1]['bid']:.2f}" if has_opportunity else "No arbitrage opportunity"
        }
    
    def get_global_price(self) -> float:
        """Retourne le VWAP global"""
        tickers = self._parallel_fetch('fetch_ticker')
        analysis = self._analyze_prices(tickers)
        return analysis.get('vwap', 0)
