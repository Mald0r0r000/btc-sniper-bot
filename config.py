"""
Configuration centralisée pour le Bot Sniper BTCUSDT.P
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# EXCHANGE CONFIGURATION
# ==========================================
API_KEY = os.getenv('BITGET_API_KEY', '')
API_SECRET = os.getenv('BITGET_API_SECRET', '')
API_PASSWORD = os.getenv('BITGET_API_PASSWORD', '')

SYMBOL = 'BTC/USDT:USDT'

# ==========================================
# TIMEFRAMES
# ==========================================
TIMEFRAME_MICRO = '5m'    # Micro structure
TIMEFRAME_MESO = '1h'     # Meso structure  
TIMEFRAME_MACRO = '1d'    # Macro structure

# Liste des timeframes pour FVG MTF
MTF_TIMEFRAMES = ['5m', '1h', '1d']

# ==========================================
# ORDER BOOK PARAMETERS
# ==========================================
ORDER_BOOK_LIMIT = 50  # Nombre de niveaux à récupérer

# ==========================================
# CVD PARAMETERS
# ==========================================
CVD_TRADES_LIMIT = 1000  # Derniers trades à analyser

# ==========================================
# VOLUME PROFILE PARAMETERS
# ==========================================
BIN_SIZE = 10  # Taille des bins en USD
VALUE_AREA_PCT = 0.70  # 70% pour Value Area

# ==========================================
# PIVOT / STRUCTURE PARAMETERS
# ==========================================
PIVOT_LEFT_BARS = 10
PIVOT_RIGHT_BARS = 3

# ==========================================
# FVG PARAMETERS
# ==========================================
FVG_WIDTH_BARS = 6
FVG_MIN_SIZE_PCT = 0.1  # Taille minimum en %
VOL_AVG_LENGTH = 20
VOL_MULTIPLIER = 1.5  # High volume threshold

# ==========================================
# ENTROPY / QUANTUM STATE PARAMETERS
# ==========================================
BB_LENGTH = 20
COMPRESSION_LOOKBACK = 100
COMPRESSION_THRESHOLD = 0.8  # < 1.0 = marché comprimé

# ==========================================
# Z-SCORE / ELASTICITY PARAMETERS
# ==========================================
Z_SCORE_LENGTH = 82
Z_SCORE_THRESHOLD = 1.3

# ==========================================
# LIQUIDATION LEVELS
# ==========================================
LEVERAGE_LEVELS = [100, 50, 25]  # Niveaux de levier à surveiller

# ==========================================
# DECISION ENGINE THRESHOLDS
# ==========================================
NEAR_LEVEL_PCT = 0.003  # 0.3% pour être "proche" d'un niveau
WALL_PROXIMITY_PCT = 0.0015  # 0.15% pour un mur dangereux
MIN_WALL_VALUE_M = 1.0  # Minimum 1M$ pour un mur significatif
CVD_BREAKOUT_THRESHOLD = 50  # CVD net minimum pour breakout
AGGRESSION_BULLISH = 1.5  # Ratio agression pour breakout
AGGRESSION_BEARISH = 0.8  # Ratio agression bearish
