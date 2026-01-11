"""
Macro Correlation Analysis
Analyse des corrélations avec les marchés traditionnels
"""
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import os


class MacroAnalyzer:
    """
    Analyse des corrélations macro-économiques
    
    Assets surveillés:
    - DXY (Dollar Index) - Corrélation négative avec BTC
    - S&P 500 - Corrélation positive (risk-on/risk-off)
    - Gold - Corrélation partielle (store of value)
    - US 10Y Yield - Impact sur les actifs risqués
    """
    
    # APIs gratuites pour données macro
    YAHOO_FINANCE_QUOTE = "https://query1.finance.yahoo.com/v8/finance/chart"
    TRADING_ECONOMICS_API = "https://api.tradingeconomics.com"
    
    # Tickers
    TICKERS = {
        'dxy': 'DX-Y.NYB',
        'spx': '^GSPC',
        'gold': 'GC=F',
        'us10y': '^TNX',
        'vix': '^VIX'
    }
    
    # Corrélations historiques moyennes avec BTC
    HISTORICAL_CORRELATIONS = {
        'dxy': -0.6,    # Dollar fort = BTC faible
        'spx': 0.5,     # Risk-on corrélé
        'gold': 0.3,    # Store of value partiel
        'us10y': -0.4,  # Taux hauts = pression sur crypto
        'vix': -0.3     # VIX haut = Risk-off
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 10
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyse macro complète
        
        Returns:
            Dict avec données macro et impact estimé sur BTC
        """
        assets = {}
        
        for asset_name, ticker in self.TICKERS.items():
            data = self._fetch_asset_data(ticker)
            if data:
                assets[asset_name] = data
        
        # Calculer l'impact global sur BTC
        btc_impact = self._calculate_btc_impact(assets)
        
        # Risk environment
        risk_env = self._assess_risk_environment(assets)
        
        return {
            'assets': assets,
            'btc_impact': btc_impact,
            'risk_environment': risk_env,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _fetch_asset_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Récupère les données d'un asset via Yahoo Finance"""
        try:
            url = f"{self.YAHOO_FINANCE_QUOTE}/{ticker}"
            params = {
                'range': '5d',
                'interval': '1d'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(url, params=params, headers=headers)
            
            if not response.ok:
                return None
            
            data = response.json()
            
            chart = data.get('chart', {}).get('result', [{}])[0]
            meta = chart.get('meta', {})
            
            current_price = meta.get('regularMarketPrice', 0)
            previous_close = meta.get('previousClose', current_price)
            
            if current_price == 0:
                return None
            
            change = current_price - previous_close
            change_pct = (change / previous_close * 100) if previous_close > 0 else 0
            
            return {
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2),
                'previous_close': round(previous_close, 2)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_btc_impact(self, assets: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calcule l'impact estimé des mouvements macro sur BTC
        
        Utilise les corrélations historiques pour estimer l'impact
        """
        total_impact = 0
        factors = []
        
        for asset_name, data in assets.items():
            if 'error' in data or data is None:
                continue
            
            change_pct = data.get('change_pct', 0)
            correlation = self.HISTORICAL_CORRELATIONS.get(asset_name, 0)
            
            # Impact = mouvement * corrélation
            impact = change_pct * correlation
            total_impact += impact
            
            # Générer les facteurs significatifs
            if abs(change_pct) > 0.5:
                direction = "↑" if change_pct > 0 else "↓"
                btc_effect = "bullish" if impact > 0 else "bearish"
                
                asset_display = {
                    'dxy': 'Dollar (DXY)',
                    'spx': 'S&P 500',
                    'gold': 'Or',
                    'us10y': 'US 10Y Yield',
                    'vix': 'VIX'
                }.get(asset_name, asset_name.upper())
                
                factors.append({
                    'asset': asset_display,
                    'change_pct': change_pct,
                    'direction': direction,
                    'btc_effect': btc_effect,
                    'impact': round(impact, 3)
                })
        
        # Normaliser et interpréter
        if total_impact > 1:
            signal = "STRONGLY_BULLISH"
            interpretation = "Environnement macro très favorable à BTC"
        elif total_impact > 0.3:
            signal = "BULLISH"
            interpretation = "Environnement macro favorable"
        elif total_impact > -0.3:
            signal = "NEUTRAL"
            interpretation = "Pas d'impact macro significatif"
        elif total_impact > -1:
            signal = "BEARISH"
            interpretation = "Pression macro négative"
        else:
            signal = "STRONGLY_BEARISH"
            interpretation = "Environnement macro très défavorable"
        
        return {
            'total_impact': round(total_impact, 3),
            'signal': signal,
            'interpretation': interpretation,
            'significant_factors': sorted(factors, key=lambda x: abs(x['impact']), reverse=True)
        }
    
    def _assess_risk_environment(self, assets: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Évalue l'environnement de risque global
        
        Risk-On: DXY bas, SPX haut, VIX bas
        Risk-Off: DXY haut, SPX bas, VIX haut
        """
        # VIX est le principal indicateur
        vix_data = assets.get('vix', {})
        spx_data = assets.get('spx', {})
        dxy_data = assets.get('dxy', {})
        
        vix_level = vix_data.get('price', 20) if vix_data and 'error' not in vix_data else 20
        spx_change = spx_data.get('change_pct', 0) if spx_data and 'error' not in spx_data else 0
        dxy_change = dxy_data.get('change_pct', 0) if dxy_data and 'error' not in dxy_data else 0
        
        # Score de risque (0 = risk-off extrême, 100 = risk-on extrême)
        risk_score = 50
        
        # VIX impact (-30 à +30 points)
        if vix_level < 15:
            risk_score += 25  # Très calme = risk-on
        elif vix_level < 20:
            risk_score += 10
        elif vix_level > 30:
            risk_score -= 25  # Panique = risk-off
        elif vix_level > 25:
            risk_score -= 15
        
        # SPX momentum (-15 à +15 points)
        if spx_change > 1:
            risk_score += 15
        elif spx_change > 0.5:
            risk_score += 8
        elif spx_change < -1:
            risk_score -= 15
        elif spx_change < -0.5:
            risk_score -= 8
        
        # DXY inverse (-10 à +10 points)
        if dxy_change < -0.3:
            risk_score += 10  # Dollar faible = risk-on
        elif dxy_change > 0.3:
            risk_score -= 10  # Dollar fort = risk-off
        
        # Normaliser
        risk_score = max(0, min(100, risk_score))
        
        if risk_score >= 70:
            environment = "STRONG_RISK_ON"
            emoji = "🟢"
            btc_bias = "BULLISH"
        elif risk_score >= 55:
            environment = "RISK_ON"
            emoji = "🟢"
            btc_bias = "SLIGHTLY_BULLISH"
        elif risk_score >= 45:
            environment = "NEUTRAL"
            emoji = "⚪"
            btc_bias = "NEUTRAL"
        elif risk_score >= 30:
            environment = "RISK_OFF"
            emoji = "🔴"
            btc_bias = "SLIGHTLY_BEARISH"
        else:
            environment = "STRONG_RISK_OFF"
            emoji = "🔴"
            btc_bias = "BEARISH"
        
        return {
            'risk_score': risk_score,
            'environment': environment,
            'emoji': emoji,
            'btc_bias': btc_bias,
            'vix_level': vix_level,
            'vix_assessment': 'CALM' if vix_level < 18 else 'ELEVATED' if vix_level < 25 else 'HIGH'
        }
