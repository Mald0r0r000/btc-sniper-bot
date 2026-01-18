# BTC Sniper Bot V2 - Institutional Grade Trading Signals

[![24/7 Analysis](https://github.com/Mald0r0r000/btc-sniper-bot/actions/workflows/analyze.yml/badge.svg)](https://github.com/Mald0r0r000/btc-sniper-bot/actions)

Bot d'analyse BTC institutionnel avec **17 modules d'analyse**, scoring multi-dimensionnel, alertes Telegram, et stratégie hybride validée.

---

## 📊 Performance (Backtest Jan 13-17, 2026)

| Métrique | Valeur |
|----------|--------|
| **P&L Total** | **+$14,100 (+141%)** |
| **Winrate** | 55.8% (43W / 34L) |
| **Profit Factor** | 2.98 |
| **Sharpe Ratio** | 8.04 |
| **Sortino Ratio** | 24.45 |
| **Max Drawdown** | 17.2% |
| **Expectancy** | +$183/trade |

### Performance par Signal Type

| Signal | Winrate | P&L | Recommandation |
|--------|---------|-----|----------------|
| 🟢 SHORT_BREAKOUT | 100% (4/4) | +$3,547 | ✅ Production |
| 🟢 SHORT_SNIPER | 100% (2/2) | +$679 | ✅ Production |
| 🟢 FADE_HIGH_SCALP | 83% (5/6) | +$1,339 | ✅ Production |
| 🟢 FADE_LOW | 62% (5/8) | +$4,501 | ✅ Production |
| 🟡 FADE_HIGH | 56% (9/16) | +$1,274 | ⚠️ Monitor |
| 🔴 NO_SIGNAL | 40% (12/30) | +$3,147 | ❌ Filter |
| 🔴 LONG_SNIPER | 33% (1/3) | +$316 | ❌ Filter |
| 🔴 LONG_BREAKOUT | 0% (0/1) | -$270 | ❌ Filter |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions (10 min)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       main_v2.py                            │
│         Orchestre l'analyse complète (Data + Engine)        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌────────────────┐  ┌────────────┐
     │ Exchanges  │  │ Decision       │  │ Analyzers  │
     │ Aggregator │  │ Engine V2      │  │ (17 mods)  │
     │            │  │                │  │            │
     │ Binance    │  │ ■ 8 Dimensions │  │ Technical  │
     │ Bybit      │  │ ■ Pondération  │  │ Structure  │
     │ OKX        │  │ ■ Anti-Manip   │  │ Sentiment  │
     │ Bitget     │  │                │  │ OnChain    │
     └────────────┘  └────────────────┘  └────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌────────────────┐  ┌────────────┐
     │ Smart      │  │ Momentum       │  │ Adaptive   │
     │ Entry      │  │ Analyzer       │  │ Leverage   │
     │            │  │                │  │            │
     │ Wait/Limit │  │ WEAK→Scalp     │  │ 5x-50x     │
     │ Immediate  │  │ STRONG→Swing   │  │ Risk Mgmt  │
     └────────────┘  └────────────────┘  └────────────┘
                              │
                              ▼
     ┌────────────────────────────────────────────────┐
     │   Telegram Notifier  │   GistDataStore        │
     │   Alertes formatées  │   Historique (1000 max)│
     └────────────────────────────────────────────────┘
```

---

## 🧠 Stratégie Hybride (Déployée)

### 1. Smart Entry (Zone Liquidation)
- **WAIT_FOR_DIP** : Attend une correction vers une zone de support
- **LIMIT_ORDER** : Place un ordre limit sur le dip identifié
- **IMMEDIATE** : Entre immédiatement si pas de meilleure opportunité

### 2. Momentum-Based Targets
- **Momentum WEAK + FADE** → Targets Scalp (5m fractals, ~0.5%)
- **Momentum STRONG** → Targets Swing (1h fractals, ~2-3%)

### 3. Adaptive Leverage
- Calcule le levier optimal (5x-50x) basé sur :
  - Distance TP/SL
  - Volatilité actuelle
  - Score Momentum
  - Risk Management (2% max loss par trade)

---

## 📦 Modules (17)

### Core Analyzers
| Module | Description |
|--------|-------------|
| `order_book.py` | Imbalance bid/ask, murs, pressure |
| `cvd.py` | Volume Delta Cumulatif, divergences |
| `volume_profile.py` | POC, VAH, VAL, shape (D/P/b) |
| `fvg.py` | Fair Value Gaps (5m/1h/1d) |
| `entropy.py` | Quantum State, compression, barriers |
| `funding_liquidation.py` | Funding rates, liquidation levels |

### Advanced Analyzers
| Module | Description |
|--------|-------------|
| `spoofing.py` | Ghost Walls, Layering, Wash Trading |
| `derivatives.py` | Basis, contango/backwardation |
| `onchain.py` | Whale tracking, Exchange flows |
| `sentiment.py` | Fear & Greed Index, trend 7j |
| `macro.py` | DXY, S&P 500, VIX correlation |
| `deribit_options.py` | Max Pain, IV, Put/Call Ratio |

### R&D / Enhancement Modules
| Module | Description |
|--------|-------------|
| `fluid_dynamics.py` | Venturi effect, Self-Trading detection |
| `liquidation_zones.py` | TP/SL dynamiques, liq clusters |
| `smart_entry.py` | Wait for dip, limit orders |
| `momentum_analyzer.py` | CVD+OI+Volume score, scalp logic |
| `adaptive_leverage.py` | Dynamic leverage (5x-50x) |

---

## 🏃 Quick Start

```bash
# Clone
git clone https://github.com/Mald0r0r000/btc-sniper-bot.git
cd btc-sniper-bot

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Éditer .env avec vos clés API

# Exécuter localement
python main_v2.py --mode full

# Backtest
python backtest/historical_backtest.py --confidence 0
```

---

## 🔧 Configuration

### Variables d'environnement (.env)
```
# Telegram (Requis)
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789

# GitHub Gist (Optionnel - Historique)
GITHUB_TOKEN=ghp_...
GIST_ID=abc123...

# Macro Data (Requis pour M2)
FRED_API_KEY=...

# Exchanges (Optionnel)
BITGET_API_KEY=...
BITGET_SECRET=...
BITGET_PASSWORD=...
```

### GitHub Actions (24/7)
1. Fork ce repo
2. Settings → Secrets → Actions
3. Ajouter les secrets listés ci-dessus

Le bot s'exécute automatiquement **toutes les 10 minutes**.

---

## 📁 Structure du Projet

```
btc-sniper-bot/
├── main_v2.py              # Point d'entrée principal
├── decision_engine_v2.py   # Scoring multi-dimensionnel (Black Box Recorder)
├── exchange_aggregator.py  # Multi-exchange VWAP
├── notifier.py             # Telegram alerts
├── data_store.py           # GitHub Gist persistence (1000 signals max)
├── runner.py               # Runner GitHub Actions
├── config.py               # Configuration centralisée
├── smart_entry.py          # Smart Entry Analyzer
├── momentum_analyzer.py    # Momentum + Scalp Logic
├── adaptive_leverage.py    # Dynamic Leverage Calculator
├── signal_validator.py     # TP/SL Validation
├── analyzers/              # 17 analysis modules
│   ├── order_book.py
│   ├── cvd.py
│   ├── volume_profile.py
│   ├── fvg.py
│   ├── entropy.py
│   ├── funding_liquidation.py
│   ├── spoofing.py
│   ├── derivatives.py
│   ├── onchain.py
│   ├── sentiment.py
│   ├── macro.py
│   ├── deribit_options.py
│   ├── fluid_dynamics.py
│   ├── liquidation_zones.py
│   └── open_interest.py
├── backtest/              # Backtesting suite
│   ├── historical_backtest.py
│   ├── data_provider.py
│   ├── trade_simulator.py
│   ├── metrics.py
│   └── results/
└── .github/workflows/
    └── analyze.yml         # Cron 10min
```

---

## 🧪 Backtesting

```bash
# Backtest avec tous les signaux
python backtest/historical_backtest.py --confidence 0

# Backtest production (confidence >= 65%)
python backtest/historical_backtest.py --confidence 65
```

### Output inclut:
- P&L Total et par signal type
- Winrate, Sharpe, Sortino, Max Drawdown
- Pattern Discovery Report (corrélation modules/résultats)
- Historique des trades avec Entry/Exit times

---

## 📊 Data Logging (Black Box Recorder)

Chaque signal enregistre automatiquement dans le Gist :
- **Scores détaillés** : Technical, Structure, Sentiment, etc.
- **Momentum state** : Score, Strength, Direction
- **Smart Entry decision** : Strategy, Optimal Price

Ceci permet une analyse rétrospective pour découvrir des patterns invisibles.

---

## 📄 License

MIT
