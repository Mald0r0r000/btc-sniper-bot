# BTC Sniper Bot V2 - Institutional Grade

[![24/7 Analysis](https://github.com/Mald0r0r000/btc-sniper-bot/actions/workflows/analyze.yml/badge.svg)](https://github.com/Mald0r0r000/btc-sniper-bot/actions)

Bot d'analyse BTC institutionnel avec **16 modules d'analyse**, scoring multi-dimensionnel et alertes Telegram.

## 🚀 Fonctionnalités

### Multi-Exchange (4)
- **Binance, OKX, Bybit, Bitget** - Agrégation VWAP, détection d'arbitrage

### 16 Analyseurs

| Catégorie | Modules |
|-----------|---------|
| **Core** | Order Book, CVD, Volume Profile, FVG MTF, Entropy, Funding |
| **Advanced** | Spoofing, Derivatives, On-Chain, Sentiment, Macro, Options |
| **R&D** | Fluid Dynamics, Liquidation Zones, Open Interest |

### Decision Engine V2
Scoring pondéré sur **8 dimensions** (0-100):

| Dimension | Poids | Sources |
|-----------|-------|---------|
| Technical | 25% | Order Book, CVD, Volume Profile |
| Structure | 15% | FVG, Entropy, Pivots |
| Multi-Exchange | 10% | VWAP, Arbitrage, Spread |
| Derivatives | 15% | Futures, Options, OI |
| On-Chain | 15% | Whale, Flows |
| Sentiment | 10% | Fear & Greed |
| Macro | 10% | DXY, VIX, S&P 500 |

### Types de Signaux
```
QUANTUM_BUY/SELL    - Breakout après compression
LONG/SHORT_SNIPER   - Confluence forte multi-dimensionnelle
DIAMOND_SETUP       - Setup institutionnel
FADE_HIGH/LOW       - Haut/Bas de range
CONTRARIAN_BUY/SELL - Contre-tendance sur extrêmes
MACRO_ALIGNED       - Alignement macro favorable
```

---

## ⚡ Quick Start

```bash
# Clone
git clone <your-repo>
cd btc-sniper-bot

# Setup
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Éditer .env avec vos clés API

# Exécuter
python main_v2.py --mode full
```

---

## 🔔 Configuration

### Telegram
1. Créer un bot via [@BotFather](https://t.me/BotFather)
2. Obtenir `chat_id` via [@userinfobot](https://t.me/userinfobot)
3. Dans `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=123456789
   ```

### GitHub Actions (24/7)
1. Fork ce repo
2. Settings → Secrets → Actions
3. Ajouter:
   - `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
   - `GIST_TOKEN` / `GIST_ID` (optionnel, historique)
   - `BITGET_API_KEY/SECRET/PASSWORD` (optionnel)

Le bot s'exécute automatiquement **toutes les 5 minutes**.

---

## 📊 Modules Détaillés

### Core Analyzers
| Module | Description |
|--------|-------------|
| **Order Book** | Imbalance bid/ask, murs, pressure |
| **CVD** | Volume Delta Cumulatif, agression ratio |
| **Volume Profile** | POC, VAH, VAL, shape (D/P/b) |
| **FVG MTF** | Fair Value Gaps (5m/1h/1d) |
| **Entropy** | Quantum State, compression, barriers |
| **Funding** | Funding rates, liquidation levels |

### Advanced Analyzers
| Module | Description |
|--------|-------------|
| **Spoofing** | Ghost Walls, Layering, Wash Trading |
| **Derivatives** | Basis, contango/backwardation, liquidations |
| **On-Chain** | Whale tracking, Exchange flows, Network health |
| **Sentiment** | Fear & Greed Index, trend 7j |
| **Macro** | DXY, S&P 500, VIX correlation |
| **Options** | Deribit Max Pain, IV, Put/Call Ratio |

### R&D Analyzers
| Module | Description |
|--------|-------------|
| **Fluid Dynamics** | VenturiAnalyzer (compression → breakout), SelfTradingDetector (wash trading) |
| **Liquidation Zones** | TP/SL dynamiques basés sur zones de liquidation des pivots |
| **Open Interest** | Evolution OI multi-timeframe |

---

## 📁 Structure

```
btc-sniper-bot/
├── main_v2.py              # Point d'entrée principal
├── decision_engine_v2.py   # Scoring multi-dimensionnel
├── exchange_aggregator.py  # Multi-exchange VWAP
├── notifier.py             # Telegram alerts
├── data_store.py           # GitHub Gist persistence
├── runner.py               # Runner GitHub Actions
├── config.py               # Configuration centralisée
├── analyzers/
│   ├── order_book.py       # Core
│   ├── cvd.py
│   ├── volume_profile.py
│   ├── fvg.py
│   ├── entropy.py
│   ├── funding_liquidation.py
│   ├── spoofing.py         # Advanced
│   ├── derivatives.py
│   ├── onchain.py
│   ├── sentiment.py
│   ├── macro.py
│   ├── deribit_options.py
│   ├── fluid_dynamics.py   # R&D
│   ├── liquidation_zones.py
│   └── open_interest.py
└── .github/workflows/
    └── analyze.yml         # Cron 5min
```

---

## 📈 Output Example

```json
{
  "signal": {
    "type": "LONG_SNIPER",
    "direction": "LONG",
    "confidence": 72.5,
    "targets": {
      "tp1": 95000,
      "tp2": 97500,
      "sl": 93500
    }
  },
  "dimension_scores": {
    "technical": 75,
    "structure": 68,
    "derivatives": 80,
    "onchain": 65,
    "sentiment": 70,
    "macro": 60
  }
}
```

---

## 🧪 R&D: Fluid Dynamics

### VenturiAnalyzer
Applique l'effet Venturi au trading:
- Order book fin → compression de liquidité
- Détecte les pre-breakout patterns
- Génère `signal_modifier` (-10 à +10)

### SelfTradingDetector
Détecte le wash trading:
- Volume élevé sans impact prix → suspect
- CVD divergence → accumulation cachée
- Symétrie buy/sell parfaite → manipulation

---

## 📄 License

MIT
