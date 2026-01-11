"""
Analyseur Funding Rate & Liquidation Clusters
- Funding Rate actuel et prédiction
- Calcul des niveaux de liquidation par levier
- Détection de l'aimant (magnet) le plus proche
"""
from typing import Dict, List, Any

import config


class FundingLiquidationAnalyzer:
    """Analyse le funding rate et calcule les clusters de liquidation"""
    
    def __init__(self, funding_data: Dict[str, float], current_price: float):
        """
        Args:
            funding_data: Dict avec 'current' et 'predicted' funding rates
            current_price: Prix actuel
        """
        self.funding_rate = funding_data.get('current', 0)
        self.predicted_funding = funding_data.get('predicted', 0)
        self.current_price = current_price
        self.leverage_levels = config.LEVERAGE_LEVELS
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse complète funding + liquidations
        
        Returns:
            Dict avec funding_analysis et liquidation_clusters
        """
        funding_analysis = self._analyze_funding()
        liquidation_clusters = self._calculate_liquidation_levels()
        magnet = self._find_closest_magnet(liquidation_clusters)
        
        return {
            'funding': funding_analysis,
            'liquidation_clusters': liquidation_clusters,
            'magnet': magnet
        }
    
    def _analyze_funding(self) -> Dict[str, Any]:
        """Analyse le funding rate"""
        # Annualisé approximatif (3 périodes/jour * 365 jours)
        annualized_pct = self.funding_rate * 3 * 365 * 100
        
        # Interprétation
        if self.funding_rate > 0.0005:
            signal = "TROP_DE_LONGS"
            emoji = "⚠️"
            description = "Funding élevé - Trop de longs (cher pour les longs)"
        elif self.funding_rate < 0:
            signal = "SHORT_SQUEEZE_POSSIBLE"
            emoji = "🚀"
            description = "Funding négatif - Short squeeze possible"
        else:
            signal = "NEUTRE"
            emoji = "⚪"
            description = "Funding normal"
        
        return {
            'current': round(self.funding_rate, 6),
            'current_pct': round(self.funding_rate * 100, 4),
            'predicted': round(self.predicted_funding, 6),
            'annualized_pct': round(annualized_pct, 2),
            'signal': signal,
            'emoji': emoji,
            'description': description,
            'is_expensive_for_longs': self.funding_rate > 0.0005,
            'is_negative': self.funding_rate < 0
        }
    
    def _calculate_liquidation_levels(self) -> List[Dict[str, Any]]:
        """
        Calcule les niveaux de liquidation estimés par levier
        
        Logique: Si quelqu'un Long x100 à prix P, son prix de liq est ~P * (1 - 1/100)
        """
        clusters = []
        
        for leverage in self.leverage_levels:
            # Pourcentage de mouvement pour liquidation
            move_pct = 1 / leverage
            
            # Longs liquidés en dessous (prix - move%)
            long_liq_price = self.current_price * (1 - move_pct)
            
            # Shorts liquidés au dessus (prix + move%)
            short_liq_price = self.current_price * (1 + move_pct)
            
            clusters.append({
                'leverage': leverage,
                'long_liquidation': round(long_liq_price, 2),
                'short_liquidation': round(short_liq_price, 2),
                'move_pct': round(move_pct * 100, 2),
                'distance_to_long_liq': round(abs(self.current_price - long_liq_price), 2),
                'distance_to_short_liq': round(abs(self.current_price - short_liq_price), 2)
            })
        
        return clusters
    
    def _find_closest_magnet(self, clusters: List[Dict]) -> Dict[str, Any]:
        """
        Trouve le cluster de liquidation le plus proche (l'aimant du marché)
        
        Le marché tend à aller chercher les liquidations les plus proches
        """
        if not clusters:
            return {'direction': 'UNKNOWN', 'price': 0, 'leverage': 0, 'type': 'none'}
        
        # Cluster x100 (plus vulnérable)
        closest = clusters[0]
        
        dist_to_short_squeeze = closest['distance_to_short_liq']
        dist_to_long_flush = closest['distance_to_long_liq']
        
        if dist_to_short_squeeze < dist_to_long_flush:
            return {
                'direction': 'HAUSSIER',
                'price': closest['short_liquidation'],
                'leverage': closest['leverage'],
                'type': 'SHORT_SQUEEZE',
                'distance': round(dist_to_short_squeeze, 2),
                'distance_pct': round(dist_to_short_squeeze / self.current_price * 100, 3),
                'emoji': '🧲📈',
                'description': f"Aimant haussier: ${closest['short_liquidation']:.0f} (Liq x{closest['leverage']} Shorts)"
            }
        else:
            return {
                'direction': 'BAISSIER',
                'price': closest['long_liquidation'],
                'leverage': closest['leverage'],
                'type': 'LONG_FLUSH',
                'distance': round(dist_to_long_flush, 2),
                'distance_pct': round(dist_to_long_flush / self.current_price * 100, 3),
                'emoji': '🧲📉',
                'description': f"Aimant baissier: ${closest['long_liquidation']:.0f} (Liq x{closest['leverage']} Longs)"
            }
    
    def is_squeeze_imminent(self, threshold_pct: float = 0.2) -> Dict[str, Any]:
        """
        Vérifie si un squeeze est imminent (aimant très proche)
        
        Args:
            threshold_pct: Seuil de distance en % (défaut 0.2%)
            
        Returns:
            Dict avec is_imminent et détails
        """
        result = self.analyze()
        magnet = result['magnet']
        
        is_imminent = magnet['distance_pct'] < threshold_pct
        
        return {
            'is_imminent': is_imminent,
            'direction': magnet['direction'],
            'target_price': magnet['price'],
            'distance_pct': magnet['distance_pct'],
            'type': magnet['type']
        }
