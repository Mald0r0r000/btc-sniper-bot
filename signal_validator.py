"""
Signal Validator
Valide les signaux passés en vérifiant si les targets ont été atteintes
Permet de calculer le winrate réel du bot
"""
import json
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os


class SignalValidator:
    """
    Valide les signaux passés en vérifiant si TP/SL atteints
    
    Processus:
    1. Récupérer les signaux non-validés du Gist
    2. Pour chaque signal avec TP/SL:
       - Récupérer les prix historiques depuis le signal
       - Vérifier si TP1/TP2/SL atteint en premier
    3. Calculer les métriques de performance
    """
    
    # Durée max pour valider un signal (48h par défaut)
    MAX_VALIDATION_HOURS = 48
    
    def __init__(self, gist_id: str = None, github_token: str = None):
        self.gist_id = gist_id or os.getenv('GIST_ID')
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.api_base = "https://api.github.com"
        
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def fetch_historical_prices(self, start_time: datetime, 
                                 end_time: datetime = None,
                                 exchange: str = 'bitget') -> List[Dict]:
        """
        Récupère les prix historiques depuis un exchange
        Utilise l'API publique CCXT
        """
        try:
            import ccxt
            
            ex = getattr(ccxt, exchange)({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            
            since = int(start_time.timestamp() * 1000)
            
            # Récupérer les bougies 1m pour précision
            ohlcv = ex.fetch_ohlcv(
                'BTC/USDT:USDT',
                timeframe='1m',
                since=since,
                limit=1000  # ~16h de données
            )
            
            return [{
                'timestamp': datetime.fromtimestamp(candle[0]/1000, tz=timezone.utc),
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            } for candle in ohlcv]
            
        except Exception as e:
            print(f"❌ Erreur fetch prix historiques: {e}")
            return []
    
    def validate_signal(self, signal: Dict) -> Dict[str, Any]:
        """
        Valide un signal individuel
        
        Returns:
            Dict avec status (WIN/LOSS/EXPIRED/PENDING), target atteint, temps
        """
        # Extraire les infos du signal
        stored_at = signal.get('stored_at') or signal.get('timestamp')
        if not stored_at:
            return {'status': 'INVALID', 'reason': 'No timestamp'}
        
        signal_data = signal.get('signal', {})
        direction = signal_data.get('direction', 'NEUTRAL')
        targets = signal_data.get('targets', {})
        price_at_signal = signal.get('price', 0)
        
        if direction == 'NEUTRAL' or not targets:
            return {'status': 'SKIP', 'reason': 'No direction or targets'}
        
        tp1 = targets.get('tp1', 0)
        tp2 = targets.get('tp2', 0)
        sl = targets.get('sl', 0)
        
        if not tp1 or not sl:
            return {'status': 'SKIP', 'reason': 'Missing TP1 or SL'}
        
        # Parser le timestamp
        try:
            if isinstance(stored_at, str):
                signal_time = datetime.fromisoformat(stored_at.replace('Z', '+00:00'))
            else:
                signal_time = stored_at
        except Exception as e:
            return {'status': 'INVALID', 'reason': f'Invalid timestamp: {e}'}
        
        # Vérifier si le signal est trop récent
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - signal_time).total_seconds() / 3600
        
        if hours_elapsed < 1:
            return {'status': 'PENDING', 'reason': 'Signal too recent (<1h)'}
        
        # Récupérer les prix historiques
        candles = self.fetch_historical_prices(signal_time)
        
        if not candles:
            return {'status': 'ERROR', 'reason': 'Could not fetch historical prices'}
        
        # Analyser les prix pour trouver si TP/SL atteint
        result = self._check_targets(
            candles=candles,
            direction=direction,
            entry=price_at_signal,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            max_hours=self.MAX_VALIDATION_HOURS
        )
        
        return result
    
    def _check_targets(self, candles: List[Dict], direction: str,
                       entry: float, tp1: float, tp2: float, sl: float,
                       max_hours: float) -> Dict[str, Any]:
        """
        Vérifie si les targets ont été atteintes
        
        Returns:
            Dict avec status, target atteint, temps pour atteindre
        """
        is_long = direction in ['LONG', 'BULLISH']
        
        for i, candle in enumerate(candles):
            high = candle['high']
            low = candle['low']
            elapsed_hours = i / 60  # 1 candle = 1 minute
            
            if elapsed_hours > max_hours:
                break
            
            if is_long:
                # LONG: TP atteint si high >= TP, SL atteint si low <= SL
                if low <= sl:
                    return {
                        'status': 'LOSS',
                        'hit_target': 'SL',
                        'hit_price': sl,
                        'time_to_hit_hours': round(elapsed_hours, 2),
                        'candle_index': i
                    }
                if high >= tp1:
                    # TP1 atteint, vérifier TP2
                    tp_result = {
                        'status': 'WIN',
                        'hit_target': 'TP1',
                        'hit_price': tp1,
                        'time_to_hit_hours': round(elapsed_hours, 2),
                        'candle_index': i
                    }
                    # Continuer pour voir si TP2 atteint
                    for j in range(i, len(candles)):
                        if candles[j]['high'] >= tp2:
                            tp_result['hit_target'] = 'TP2'
                            tp_result['hit_price'] = tp2
                            break
                        if candles[j]['low'] <= sl:
                            break  # SL après TP1
                    return tp_result
            else:
                # SHORT: TP atteint si low <= TP, SL atteint si high >= SL
                if high >= sl:
                    return {
                        'status': 'LOSS',
                        'hit_target': 'SL',
                        'hit_price': sl,
                        'time_to_hit_hours': round(elapsed_hours, 2),
                        'candle_index': i
                    }
                if low <= tp1:
                    tp_result = {
                        'status': 'WIN',
                        'hit_target': 'TP1',
                        'hit_price': tp1,
                        'time_to_hit_hours': round(elapsed_hours, 2),
                        'candle_index': i
                    }
                    for j in range(i, len(candles)):
                        if candles[j]['low'] <= tp2:
                            tp_result['hit_target'] = 'TP2'
                            tp_result['hit_price'] = tp2
                            break
                        if candles[j]['high'] >= sl:
                            break
                    return tp_result
        
        # Aucun target atteint dans la période
        return {
            'status': 'EXPIRED',
            'hit_target': None,
            'reason': f'No target hit within {max_hours}h'
        }
    
    def calculate_performance(self, validated_signals: List[Dict]) -> Dict[str, Any]:
        """
        Calcule les métriques de performance globales
        """
        wins = [s for s in validated_signals if s.get('validation', {}).get('status') == 'WIN']
        losses = [s for s in validated_signals if s.get('validation', {}).get('status') == 'LOSS']
        expired = [s for s in validated_signals if s.get('validation', {}).get('status') == 'EXPIRED']
        
        total_decided = len(wins) + len(losses)
        winrate = (len(wins) / total_decided * 100) if total_decided > 0 else 0
        
        # Par type de signal
        by_type = {}
        for s in validated_signals:
            sig_type = s.get('signal', {}).get('type', 'UNKNOWN')
            if sig_type not in by_type:
                by_type[sig_type] = {'wins': 0, 'losses': 0, 'expired': 0}
            
            status = s.get('validation', {}).get('status')
            if status == 'WIN':
                by_type[sig_type]['wins'] += 1
            elif status == 'LOSS':
                by_type[sig_type]['losses'] += 1
            elif status == 'EXPIRED':
                by_type[sig_type]['expired'] += 1
        
        # Calculer winrate par type
        for sig_type, stats in by_type.items():
            decided = stats['wins'] + stats['losses']
            stats['winrate'] = round(stats['wins'] / decided * 100, 1) if decided > 0 else 0
            stats['total'] = decided + stats['expired']
        
        # Temps moyen pour atteindre target
        win_times = [w.get('validation', {}).get('time_to_hit_hours', 0) for w in wins]
        loss_times = [l.get('validation', {}).get('time_to_hit_hours', 0) for l in losses]
        
        return {
            'total_validated': len(validated_signals),
            'total_decided': total_decided,
            'wins': len(wins),
            'losses': len(losses),
            'expired': len(expired),
            'winrate_pct': round(winrate, 1),
            'by_signal_type': by_type,
            'avg_win_time_hours': round(sum(win_times) / len(win_times), 2) if win_times else 0,
            'avg_loss_time_hours': round(sum(loss_times) / len(loss_times), 2) if loss_times else 0,
            'best_performing_type': max(by_type.items(), key=lambda x: x[1].get('winrate', 0))[0] if by_type else None,
            'worst_performing_type': min(by_type.items(), key=lambda x: x[1].get('winrate', 100))[0] if by_type else None
        }
    
    def run_validation(self, signals: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Valide une liste de signaux et retourne les résultats
        
        Returns:
            (signals_with_validation, performance_metrics)
        """
        print("🔍 Validation des signaux...")
        
        validated = []
        for i, signal in enumerate(signals):
            # Skip si déjà validé
            if signal.get('validation', {}).get('status') in ['WIN', 'LOSS', 'EXPIRED']:
                validated.append(signal)
                continue
            
            result = self.validate_signal(signal)
            signal['validation'] = result
            validated.append(signal)
            
            status_emoji = {
                'WIN': '✅',
                'LOSS': '❌',
                'EXPIRED': '⏰',
                'PENDING': '⏳',
                'SKIP': '⏭️',
                'ERROR': '⚠️'
            }.get(result.get('status'), '❓')
            
            print(f"   {status_emoji} Signal {i+1}: {result.get('status')}", end='')
            if result.get('hit_target'):
                print(f" ({result['hit_target']} @ {result.get('time_to_hit_hours', 0):.1f}h)")
            else:
                print()
        
        # Calculer performance
        performance = self.calculate_performance(validated)
        
        print(f"\n📊 Performance: {performance['winrate_pct']}% winrate ({performance['wins']}W / {performance['losses']}L)")
        
        return validated, performance


def test_validator():
    """Test du validateur avec des signaux simulés"""
    validator = SignalValidator()
    
    # Signal test (simulé)
    test_signal = {
        'timestamp': (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
        'price': 94000,
        'signal': {
            'type': 'LONG_SNIPER',
            'direction': 'LONG',
            'confidence': 70,
            'targets': {
                'tp1': 95000,
                'tp2': 96000,
                'sl': 93000
            }
        }
    }
    
    result = validator.validate_signal(test_signal)
    print(f"Résultat validation: {result}")


if __name__ == "__main__":
    test_validator()
