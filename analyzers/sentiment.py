"""
Sentiment Analysis
Analyse du sentiment de marché via Fear & Greed Index et sources sociales
"""
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import os


class SentimentAnalyzer:
    """
    Analyse multi-source du sentiment de marché
    
    Sources:
    1. Fear & Greed Index (gratuit)
    2. Social volume/sentiment (estimé)
    3. Funding rate sentiment (déjà intégré ailleurs, référencé ici)
    """
    
    FEAR_GREED_API = "https://api.alternative.me/fng/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse complète du sentiment
        
        Returns:
            Dict avec Fear & Greed, sentiment social estimé, et score global
        """
        fear_greed = self._get_fear_greed_index()
        social_estimate = self._estimate_social_sentiment()
        
        # Score global
        overall = self._calculate_overall_sentiment(fear_greed, social_estimate)
        
        return {
            'fear_greed': fear_greed,
            'social': social_estimate,
            'overall': overall,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _get_fear_greed_index(self) -> Dict[str, Any]:
        """Récupère le Fear & Greed Index"""
        try:
            response = self.session.get(f"{self.FEAR_GREED_API}?limit=7")
            
            if not response.ok:
                return {'error': 'Failed to fetch Fear & Greed'}
            
            data = response.json()
            
            if data.get('data'):
                entries = data['data']
                current = entries[0]
                
                value = int(current.get('value', 50))
                classification = current.get('value_classification', 'Neutral')
                
                # Historique pour tendance
                history = []
                for entry in entries:
                    history.append({
                        'value': int(entry.get('value', 50)),
                        'classification': entry.get('value_classification', ''),
                        'date': entry.get('timestamp')
                    })
                
                # Calculer la tendance
                if len(history) >= 2:
                    today = history[0]['value']
                    yesterday = history[1]['value']
                    week_ago = history[-1]['value'] if len(history) >= 7 else yesterday
                    
                    trend_1d = today - yesterday
                    trend_7d = today - week_ago
                else:
                    trend_1d = 0
                    trend_7d = 0
                
                # Interprétation
                if value >= 80:
                    signal = "EXTREME_GREED"
                    contrarian = "BEARISH"
                    interpretation = "Euphorie extrême - Prudence recommandée"
                elif value >= 60:
                    signal = "GREED"
                    contrarian = "SLIGHTLY_BEARISH"
                    interpretation = "Optimisme élevé - Surveiller les excès"
                elif value >= 40:
                    signal = "NEUTRAL"
                    contrarian = "NEUTRAL"
                    interpretation = "Sentiment équilibré"
                elif value >= 20:
                    signal = "FEAR"
                    contrarian = "SLIGHTLY_BULLISH"
                    interpretation = "Peur modérée - Opportunités potentielles"
                else:
                    signal = "EXTREME_FEAR"
                    contrarian = "BULLISH"
                    interpretation = "Panique - Historiquement bon point d'entrée"
                
                return {
                    'value': value,
                    'classification': classification,
                    'signal': signal,
                    'contrarian_signal': contrarian,
                    'interpretation': interpretation,
                    'trend_1d': trend_1d,
                    'trend_7d': trend_7d,
                    'trend_direction': 'UP' if trend_1d > 0 else 'DOWN' if trend_1d < 0 else 'FLAT',
                    'history': history[:7],
                    'emoji': self._get_fear_greed_emoji(value)
                }
            
            return {'error': 'No Fear & Greed data'}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_fear_greed_emoji(self, value: int) -> str:
        """Retourne un emoji basé sur le Fear & Greed"""
        if value >= 80:
            return "🤑"  # Extreme greed
        elif value >= 60:
            return "😀"  # Greed
        elif value >= 40:
            return "😐"  # Neutral
        elif value >= 20:
            return "😰"  # Fear
        else:
            return "😱"  # Extreme fear
    
    def _estimate_social_sentiment(self) -> Dict[str, Any]:
        """
        Estime le sentiment social
        
        Note: Pour des données réelles, intégrer:
        - Twitter/X API (payant)
        - LunarCrush API
        - Santiment API
        """
        # Sans API sociale, on retourne une estimation basée sur F&G
        return {
            'source': 'estimation',
            'twitter_sentiment': 0,  # -1 à +1
            'reddit_sentiment': 0,
            'social_volume_change': 0,  # % vs moyenne
            'influencer_sentiment': 'UNKNOWN',
            'note': 'Données estimées - Intégrer Twitter/LunarCrush pour données réelles',
            'confidence': 'LOW'
        }
    
    def _calculate_overall_sentiment(self, fear_greed: Dict, social: Dict) -> Dict[str, Any]:
        """Calcule le sentiment global"""
        # Principalement basé sur Fear & Greed pour l'instant
        fg_value = fear_greed.get('value', 50)
        
        # Normaliser sur 0-100
        score = fg_value
        
        # Le sentiment contrarian est souvent plus utile
        contrarian_score = 100 - fg_value
        
        # Facteurs
        factors = []
        
        if fg_value >= 75:
            factors.append("⚠️ Euphorie excessive - Contrarian bearish")
        elif fg_value >= 60:
            factors.append("📈 Optimisme - Tendance positive")
        elif fg_value <= 25:
            factors.append("💎 Peur excessive - Contrarian bullish")
        elif fg_value <= 40:
            factors.append("📉 Pessimisme - Prudence")
        
        trend = fear_greed.get('trend_1d', 0)
        if trend > 5:
            factors.append(f"+{trend} points vs hier - Amélioration")
        elif trend < -5:
            factors.append(f"{trend} points vs hier - Détérioration")
        
        return {
            'score': score,
            'contrarian_score': contrarian_score,
            'primary_signal': fear_greed.get('signal', 'NEUTRAL'),
            'contrarian_signal': fear_greed.get('contrarian_signal', 'NEUTRAL'),
            'factors': factors,
            'recommendation': self._get_sentiment_recommendation(fg_value)
        }
    
    def _get_sentiment_recommendation(self, fg_value: int) -> str:
        """Génère une recommandation basée sur le sentiment"""
        if fg_value >= 80:
            return "Réduire l'exposition - Prendre des profits partiels"
        elif fg_value >= 65:
            return "Positions normales - Surveiller les signes de retournement"
        elif fg_value >= 35:
            return "Conditions neutres - Suivre les signaux techniques"
        elif fg_value >= 20:
            return "Considérer accumulation progressive"
        else:
            return "Zone d'accumulation historique - DCA recommandé"


class NewsAnalyzer:
    """
    Analyse des news (optionnel, nécessite API)
    
    Sources potentielles:
    - CryptoCompare News API
    - NewsAPI
    - Custom RSS feeds
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.session = requests.Session()
    
    def get_latest_news_sentiment(self) -> Dict[str, Any]:
        """Récupère et analyse les news récentes"""
        # Placeholder - à implémenter avec API
        return {
            'source': 'placeholder',
            'news_count_24h': 0,
            'sentiment': 'NEUTRAL',
            'major_events': [],
            'note': 'Implémenter avec CryptoCompare ou NewsAPI'
        }
