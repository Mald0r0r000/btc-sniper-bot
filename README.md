# BTC Sniper Bot V2 - Institutional Grade

Bot d'analyse BTC de niveau institutionnel avec 15 modules d'analyse et alertes Telegram.

## 🚀 Fonctionnalités

- **4 Exchanges** - Binance, OKX, Bybit, Bitget
- **On-Chain** - Whale tracking, Exchange flows
- **Options Deribit** - Max Pain, IV, Put/Call Ratio
- **Manipulation Detection** - Spoofing, Ghost Walls
- **Macro** - DXY, S&P 500, VIX correlation
- **Alertes Telegram** - Notifications automatiques

## ⚡ Quick Start

```bash
# Clone
git clone <your-repo>
cd btc-sniper-bot

# Setup
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt

# Configuration (optionnel)
cp .env.example .env
# Éditer .env avec vos clés API

# Exécuter
python main_v2.py --mode full
```

## 🔔 Configuration Telegram

1. Créer un bot via [@BotFather](https://t.me/BotFather)
2. Obtenir votre `chat_id` via [@userinfobot](https://t.me/userinfobot)
3. Ajouter dans `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=123456789
   ```

## ☁️ Déploiement 24/7 (GitHub Actions)

1. Fork ce repo
2. Aller dans Settings → Secrets → Actions
3. Ajouter les secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `BITGET_API_KEY` (optionnel)
   - `BITGET_API_SECRET` (optionnel)
   - `BITGET_API_PASSWORD` (optionnel)
4. Le bot s'exécutera automatiquement toutes les 15 minutes

## 📊 Modules d'Analyse

| Module | Description |
|--------|-------------|
| Order Book | Imbalance, Murs, Pressure |
| CVD | Volume Delta Cumulatif |
| Volume Profile | POC, VAH, VAL, Shape |
| FVG MTF | Fair Value Gaps (5m/1h/1d) |
| Entropy | Quantum State, Compression |
| Multi-Exchange | VWAP global, Arbitrage |
| Spoofing | Ghost Walls, Layering |
| On-Chain | Whale Tracking, Flows |
| Options | Max Pain, IV, PCR |
| Sentiment | Fear & Greed Index |
| Macro | DXY, S&P 500, VIX |

## 📁 Structure

```
btc-sniper-bot/
├── main_v2.py           # Point d'entrée principal
├── runner.py            # Runner pour GitHub Actions
├── notifier.py          # Notifications Telegram
├── decision_engine_v2.py
├── exchange_aggregator.py
├── analyzers/
│   ├── order_book.py, cvd.py, volume_profile.py
│   ├── fvg.py, entropy.py, funding_liquidation.py
│   ├── spoofing.py, derivatives.py, onchain.py
│   ├── deribit_options.py, sentiment.py, macro.py
└── .github/workflows/analyze.yml
```

## 📄 License

MIT
