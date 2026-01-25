# BTC Sniper Bot V2 - Institutional Grade Trading Signals

[![24/7 Analysis](https://github.com/Mald0r0r000/btc-sniper-bot/actions/workflows/analyze.yml/badge.svg)](https://github.com/Mald0r0r000/btc-sniper-bot/actions)

Bot d'analyse BTC institutionnel exploitant l'**Auction Market Theory (AMT)**, le **CVD Efficiency**, et la **Quantum Entropy** pour identifier des signaux à haute probabilité sur 17+ dimensions.

---

## 💎 Nouveautés R&D (Version Janvier 2026)

Le bot a évolué d'une approche "Indicateurs" vers une approche **"Order Flow & Structure"** :

### 1. Auction Market Theory (AMT) Volume Profile
Refonte totale du module Volume Profile pour suivre la psychologie des institutionnels :
- **Régimes Structurels** : Distingue le `BALANCE` (Range) de l' `IMBALANCE` (Breakout).
- **Target Price Automatique** : Identifie le prochain HVN (High Volume Node) comme cible naturelle du prix.
- **Gap Zones** : Détecte les Low Volume Nodes (LVNs) pour prévoir les accélérations de prix ("Fast Travel").

### 2. CVD Efficiency & Aggression Detection
- **Efficacité du Delta** : Calcule le ratio `Price Delta / CVD Delta` pour détecter les absorptions passives.
- **Aggression State** : Signale explicitement qui "pousse" le marché (`BULLISH/BEARISH AGGRESSION`).
- **Absorption Risk** : Alerte quand les ordres limit absorbent toute l'agression market (Danger de retournement).

### 3. Quantum Squeeze & Entropy
- **Compression Venturi** : Détecte les phases de faible entropie avant les explosions de volatilité.
- **Divergence Squeeze** : Analyse la corrélation Open Interest / ATR pour anticiper les "Loading phases".

---

## 📊 Performance (Backtest v2.1)

| Métrique | Valeur |
|----------|--------|
| **P&L Total** | **+$14,100 (+141%)** |
| **Winrate Signal Confiance** | 55.8% |
| **Profit Factor** | 2.98 |
| **Structural Filter Accuracy** | 82% (Évitement du chop) |

---

## 🏗️ Architecture Intelligente

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions (Cron 10 min)             │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Decision Engine V2                    │
│        Scoring Multi-Dimensionnel & Filtres Qualité         │
└─────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
      ┌────────────┐  ┌────────────────┐  ┌────────────┐
      │ AMT Engine │  │ Flow Analyzer  │  │ Smart      │
      │ (Structure)│  │ (Liquidity)    │  │ Filters    │
      │            │  │                │  │            │
      │ ■ Balance  │  │ ■ CVD Aggression│  │ ■ POC Armor│
      │ ■ Imbalance│  │ ■ Squeeze (OI)  │  │ ■ Anti-Chop│
      │ ■ Targets  │  │ ■ Whale Tracking│  │ ■ Manipulation│
      └────────────┘  └────────────────┘  └────────────┘
                               │
                               ▼
      ┌────────────────────────────────────────────────┐
      │   Telegram Notifier  │    Dashboard Sync     │
      │   Alertes Temps Réel │    Gist Data Lake     │
      └────────────────────────────────────────────────┘
```

---

## 🎯 Scoring & AMT Integration

Le scoring est désormais piloté par la structure du marché :

| Régime / Contexte | Action Bot | Impact Score |
|-------------------|------------|--------------|
| **IMBALANCE EXPANSION** | Suivi de Breakout | **+/- 20 pts** (Haute Conviction) |
| **TRAVERSING GAP** | Accélération Momentum| **+/- 10 pts** |
| **VALUE AREA ROTATION**| Mean Reversion | **+/- 10 pts** |
| **STUCK AT POC** | **Neutralisation (Anti-Chop)** | **Score Damping (x0.4)** |

---

## 📦 Modules Analytiques (17)

### Structure & AMT
- `volume_profile.py` : Analyse AMT (Regimes, HVN Targets, Gap Zones).
- `fvg.py` : Détection des Fair Value Gaps MTF.
- `liquidation_zones.py` : Clusters de liquidation comme aimants de prix.

### Flow & Momentum
- `cvd.py` : Efficacité du delta et détection d'agression.
- `open_interest.py` : Corrélation prix/OI et divergences nettes.
- `order_book.py` : Imbalance bid/ask et détection de "Walls".
- `squeeze.py` : Analyse de compression volatilité/OI.

### Anti-Manipulation & Macro
- `spoofing.py` : Détection de Wash Trading et Ghost Walls.
- `macro.py` : Corrélation DXY, S&P 500 et Régime de risque.
- `fluid_dynamics.py` : Effet Venturi et dynamique de flux.

---

## 🚀 Installation & Usage

### Setup Rapide
```bash
git clone https://github.com/Mald0r0r000/btc-sniper-bot.git
cd btc-sniper-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration (.env)
```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
GITHUB_TOKEN=... # Pour la synchro Dashboard via Gist
GIST_ID=...
```

### Run
```bash
python main_v2.py --mode full
```

---

## 📄 License
MIT - Projet R&D Trading Institutionnel.
