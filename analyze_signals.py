import json
import pandas as pd
import numpy as np
from datetime import datetime

import sys

print("Script starting...", flush=True)

# Load the data
file_path = '/Users/antoinebedos/Downloads/b9289f1d093ddfdf94ed0b75eb8e0111-8f90300c1ed8eaedaf9dbb51d450f2a7ec82b5c2/btc_signals_history.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

signals = data.get('signals', [])
if not signals:
    print("No signals found.")
    sys.exit(0)

import requests
import time

def fetch_bitget_candles(start_ts_ms, end_ts_ms):
    url = "https://api.bitget.com/api/v2/mix/market/history-candles"
    symbol = "BTCUSDT"
    product_type = "USDT-FUTURES"
    granularity = "5m"
    limit = "200"
    
    all_candles = []
    current_end = end_ts_ms
    
    print(f"Fetching candles from {pd.to_datetime(start_ts_ms, unit='ms')} to {pd.to_datetime(end_ts_ms, unit='ms')}...")
    
    while current_end > start_ts_ms:
        params = {
            "symbol": symbol,
            "productType": product_type,
            "granularity": granularity,
            "endTime": str(int(current_end)),
            "limit": limit
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('code') != '00000':
                print(f"Error fetching data: {data}")
                break
                
            candles = data.get('data', [])
            if not candles:
                break
                
            # Bitget returns [ts, open, high, low, close, ...]
            # Sort is usually descending by time, but let's check
            # We want to prepend since we go backwards from endTime
            
            # Filter valid candles within our range
            batch_df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'quote_vol'])
            batch_df['ts'] = batch_df['ts'].astype(int)
            
            # Stop if the newest candle in this batch is older than our target end (shouldn't happen with endTime param)
            
            all_candles.extend(candles)
            
            # Bitget returns candles in ASCENDING order (Oldest -> Newest) usually, but let's be robust.
            # We want to find the MINIMUM timestamp in this batch to continue backwards.
            
            timestamps = [int(x[0]) for x in candles]
            min_ts = min(timestamps)
            max_ts_batch = max(timestamps)
            
            print(f"Fetched batch covering {pd.to_datetime(min_ts, unit='ms')} to {pd.to_datetime(max_ts_batch, unit='ms')}")
            
            # If we aren't making progress (min_ts is close to current_end), break to avoid infinite loop
            if min_ts >= current_end:
                 print("Warning: Pagination stuck. Stopping.")
                 break
                 
            current_end = min_ts - 1
            
            # Rate limit
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Exception during fetch: {e}")
            break
            
    print(f"Fetched {len(all_candles)} candles.")
    
    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'quote_vol'])
    df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms').dt.tz_localize('UTC')
    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = df[col].astype(float)
        
    df['price'] = df['close'] # Keep price for compatibility
    df = df.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    return df #[['ts', 'price']] # Return full DF now

# 1. Construct Signal DataFrame
records = []
ts_list = []

for s in signals:
    ts_str = s.get('ts')
    try:
        ts = pd.to_datetime(ts_str)
        ts_list.append(ts)
    except:
        continue

if ts_list:
    min_ts = min(ts_list)
    max_ts = max(ts_list)
    
    # Add buffer: 24h before and 48h after for lookahead, BUT clamp to now
    now_ts = pd.Timestamp.utcnow()
    
    start_fetch = int((min_ts - pd.Timedelta(hours=24)).timestamp() * 1000)
    calculated_end = int((max_ts + pd.Timedelta(hours=48)).timestamp() * 1000)
    end_fetch = min(calculated_end, int(now_ts.timestamp() * 1000))
    
    # Use fetched data as price_df
    price_df = fetch_bitget_candles(start_fetch, end_fetch)
    if price_df.empty:
        print("No market data fetched. Exiting.")
        sys.exit(0)
        
    price_df = price_df.set_index('ts')
else:
    print("No valid timestamps found.")
    sys.exit(0)

# Re-iterate to build records (we don't need price_history array anymore)
for s in signals:
    ts_str = s.get('ts')
    try:
        ts = pd.to_datetime(ts_str)
    except:
        continue
        
    px = s.get('px') # We can still use the signal price as 'entry', but lookups will use fetched data

    
    # Flatten relevant features
    record = {
        'ts': ts,
        'price': px,
        'direction': s.get('sig', {}).get('d'),
        'confidence': s.get('sig', {}).get('c'),
        'market_pressure': s.get('sig', {}).get('mp'),
        
        # Decision Scores
        'score_technical': s.get('ds', {}).get('technical'),
        'score_structure': s.get('ds', {}).get('structure'),
        'score_onchain': s.get('ds', {}).get('onchain'),
        'score_sentiment': s.get('ds', {}).get('sentiment'),
        'score_macro': s.get('ds', {}).get('macro'),
        'score_derivatives': s.get('ds', {}).get('derivatives'),
        
        # Technicals
        'adx': s.get('tech', {}).get('adx'),
        'rsi_kj': s.get('tech', {}).get('kj'), # Keeping it generic if it's KDJ or similar
        'macd_hist': s.get('tech', {}).get('mcd', {}).get('h'),
        
        # Context
        'fear_greed': s.get('ctx', {}).get('fg'),
        'vol_profile': s.get('ctx', {}).get('vp'), # Categorical
        
        # OI
        'oi_d1h': s.get('oi', {}).get('d1h'),
        
        # CVD
        'cvd_ar': s.get('cvd', {}).get('ar'),

        # Venturi / Fluid Dynamics ('fd')
        'venturi_score': s.get('fd', {}).get('v', {}).get('cs'),
        'venturi_pressure': s.get('fd', {}).get('v', {}).get('bp'),
        
        # --- CONFIRMED AVAILABLE DIMENSIONS ---
        
        # 1. Hyperliquid ('hl') - Previously missed
        'hl_long_ratio': s.get('hl', {}).get('lr'),
        'hl_whale_long_vol': s.get('hl', {}).get('wl'),
        'hl_whale_short_vol': s.get('hl', {}).get('wsh'),
        # 'hl_whale_sentiment': s.get('hl', {}).get('ws'), # Categorical: STRONG_LONG etc.
        
        # 2. Context ('ctx')
        'ctx_fear_greed': s.get('ctx', {}).get('fg'),
        # 'ctx_quantum_state': s.get('ctx', {}).get('qs'), # Categorical
        
        # 3. Macro ('macro')
        'score_macro_dxy': s.get('macro', {}).get('dxy'),
        
        # 4. Order Book ('ob')
        'ob_bid_ratio': s.get('ob', {}).get('br'),
        
        # --- MISSING IN LOGS (Confirmed by inspection) ---
        'score_multi_exchange': s.get('ds', {}).get('multi_exchange'), # Score exists, raw data missing
        # 'options_pcr': NaN
        # 'entropy_compression': NaN 
    }
    
    # Calculate derived Hyperliquid Net Whale Vol
    if record['hl_whale_long_vol'] is not None and record['hl_whale_short_vol'] is not None:
        record['hl_net_whale_vol'] = record['hl_whale_long_vol'] - record['hl_whale_short_vol']
    else:
        record['hl_net_whale_vol'] = None

    records.append(record)

df = pd.DataFrame(records)

# Define Optimized Scoring Function (Data-Driven Weights)
def calculate_optimized_score(row):
    # 1. Hyperliquid Long Ratio (Correlation: +0.22)
    # Strongest Positive Predictor.
    hl_lr_score = 0
    if row['hl_long_ratio'] is not None:
        hl_lr_score = (row['hl_long_ratio'] - 50) * 0.5 * 2.5 
        
    # 2. Open Interest (Correlation: +0.20)
    oi_score = 0
    if row['oi_d1h'] is not None:
        oi_score = row['oi_d1h'] * 2.0 
        
    # 3. Hyperliquid Whale Vol (Correlation: +0.19)
    hl_vol_score = 0
    if row['hl_net_whale_vol'] is not None:
        hl_vol_score = (row['hl_net_whale_vol'] / 1000.0) * 2.0
        
    # 4. Multi-Exchange Score (Correlation: +0.17)
    mex_score = 0
    if row['score_multi_exchange'] is not None:
        mex_score = (row['score_multi_exchange'] - 50) / 10.0 * 1.5
        
    # 7. Order Book Bid Ratio (Correlation: -0.16)
    # Negative correlation -> Negative Weight
    ob_score = 0
    if row['ob_bid_ratio'] is not None:
        ob_score = (row['ob_bid_ratio'] - 0.5) * -1.5

    # 5. RSI/KDJ (Correlation: -0.37)
    # Very Strong Negative Correlation. High J -> Bearish.
    kdj_score = 0
    if row['rsi_kj'] is not None:
        kdj_val = (row['rsi_kj'] - 50) / 50
        kdj_score = kdj_val * -3.5 # Massive negative weight
        
    # 6. CVD (Correlation: -0.08)
    # Negative correlation observed -> Invert logic.
    cvd_score = 0
    if row['cvd_ar'] is not None:
        cvd_score = row['cvd_ar'] * -0.5

    final_score = hl_lr_score + oi_score + hl_vol_score + mex_score + ob_score + kdj_score + cvd_score
    return final_score

df['new_composite'] = df.apply(calculate_optimized_score, axis=1)
df = df.sort_values('ts').reset_index(drop=True)

# 2. Calculate Forward Returns
# We'll look at 1-step, 2-step, and roughly 4h/24h forward returns if meaningful
# Since steps are irregular, we'll try to find the closest price point in the future.

# Create a full price series for lookup
# price_df is already created from fetch_bitget_candles

def get_future_change(current_ts, hours_ahead):
    target_time = current_ts + pd.Timedelta(hours=hours_ahead)
    # Find nearest index
    idx = price_df.index.get_indexer([target_time], method='nearest')[0]
    future_price = price_df.iloc[idx]['price']
    future_ts = price_df.index[idx]
    
    # Check if the "nearest" is actually reasonably close (e.g. within 2x hours_ahead radius)
    # Otherwise we might be picking the last data point for a time 3 days later
    time_diff = abs((future_ts - target_time).total_seconds()) / 3600
    if time_diff > max(2, hours_ahead):
        return np.nan
        
    return future_price

df['next_price'] = df['price'].shift(-1)
df['ret_next_signal'] = (df['next_price'] - df['price']) / df['price']

# Calculate fixed horizon returns
for h in [4, 12, 24]:
    df[f'ret_{h}h'] = df.apply(lambda row: (get_future_change(row['ts'], h) - row['price']) / row['price'], axis=1)

# 3. Adjust Return for Signal Direction (Long = +1, Short = -1, Neutral = 0)
def get_dir_mult(d):
    if d == 'LONG': return 1
    if d == 'SHORT': return -1
    return 0 # Neutral signals don't participate

df['dir_mult'] = df['direction'].apply(get_dir_mult)
df['trade_outcome_next'] = df['ret_next_signal'] * df['dir_mult']
df['trade_outcome_4h'] = df[f'ret_4h'] * df['dir_mult']
df['trade_outcome_24h'] = df[f'ret_24h'] * df['dir_mult']

# 4. Correlation Analysis
# Filter for only ACTIVE signals (Long/Short) for "Signal Effectiveness"
active_signals = df[df['dir_mult'] != 0].copy()

# Features to check
numerical_features = [
    'confidence', 'market_pressure', 
    'score_technical', 'score_structure', 'score_onchain', 'score_sentiment', 'score_macro', 'score_derivatives', 'score_multi_exchange',
    'adx', 'rsi_kj', 'macd_hist', 'fear_greed', 'oi_d1h', 'cvd_ar',
    'venturi_score', 'venturi_pressure',
    # Newly Confirmed Valid Features
    'hl_long_ratio', 'hl_net_whale_vol', 'ob_bid_ratio', 'score_macro_dxy'
]


output_file = '/Users/antoinebedos/.gemini/antigravity/scratch/btc-sniper-bot/analysis_results.txt'
with open(output_file, 'w') as f:
    f.write("=== Correlation with Next Signal Outcome (Active Signals) ===\n")
    valid_next = active_signals.dropna(subset=['trade_outcome_next'])
    if not valid_next.empty:
        correlations = valid_next[numerical_features + ['trade_outcome_next']].corr()['trade_outcome_next'].sort_values(ascending=False)
        f.write(correlations.to_string() + "\n")
    else:
        f.write("Not enough data for Next Signal correlation.\n")

    f.write("\n=== Correlation with 4h Outcome (Active Signals) ===\n")
    valid_4h = active_signals.dropna(subset=['trade_outcome_4h'])
    if not valid_4h.empty:
        correlations_4h = valid_4h[numerical_features + ['trade_outcome_4h']].corr()['trade_outcome_4h'].sort_values(ascending=False)
        f.write(correlations_4h.to_string() + "\n")
    else:
        f.write("Not enough data for 4h correlation.\n")

    # 5. Best performing Categorical Features?
    f.write("\n=== Direction Performance ===\n")
    if not active_signals.empty:
        f.write(active_signals.groupby('direction')['trade_outcome_next'].mean().to_string() + "\n")

    f.write("\n=== Volume Profile Performance ===\n")
    if not df.empty:
        f.write(df.groupby('vol_profile')['trade_outcome_next'].mean().to_string() + "\n")

# 6. SIMULATE NEW SCORING LOGIC
print("\n=== Simulating New Scoring Logic ===")

new_kdj_scores = []
new_composite_scores = []

# Weights for Optimized Composite (Based on Findings)
W_OI = 0.4
W_CVD = 0.3
W_KDJ = 0.3 # New KDJ Score
W_TECH_OLD = 0.0 # Removing old tech score impact

for index, row in df.iterrows():
    # 1. New KDJ Score Simulation
    j_val = row.get('rsi_kj', 50)
    score_kdj = 50
    if j_val > 80:
        score_kdj -= (j_val - 80) * 1.5
    elif j_val < 20:
        score_kdj += (20 - j_val) * 1.5
    new_kdj_scores.append(score_kdj)
    
df['new_kdj_score'] = new_kdj_scores

# Normalize for Composite
def safe_normalize(series):
    if series.std() == 0: return series * 0
    return (series - series.mean()) / series.std()

df['norm_oi'] = safe_normalize(df['oi_d1h'].fillna(0))
df['norm_cvd'] = safe_normalize(df['cvd_ar'].fillna(1))
df['norm_kdj'] = safe_normalize(df['new_kdj_score'])

df['new_composite'] = (W_OI * df['norm_oi']) + (W_CVD * df['norm_cvd']) + (W_KDJ * df['norm_kdj'])

with open(output_file, 'a') as f:
    f.write("\n=== Backtest: New Logic Validation ===\n")
    
    # Old vs New KDJ
    corr_old_kj = df['rsi_kj'].corr(df['trade_outcome_next'])
    corr_new_kj = df['new_kdj_score'].corr(df['trade_outcome_next'])
    
    f.write(f"Old Raw J (rsi_kj) Correlation: {corr_old_kj:.4f}\n")
    f.write(f"New KDJ Score Transform Correlation: {corr_new_kj:.4f} (Goal: Positive)\n")
    
    # Old Tech vs New Composite
    corr_old_tech = df['score_technical'].corr(df['trade_outcome_next'])
    corr_new_comp = df['new_composite'].corr(df['trade_outcome_next'])
    
    f.write(f"Old Technical Score Correlation: {corr_old_tech:.4f}\n")
    f.write(f"New Optimized Composite Correlation: {corr_new_comp:.4f} (Goal: > Old)\n")


# 7. DETAILED BACKTEST (Winrate/Sharpe)
print("\n=== Detailed Backtest Simulation ===")

# Simulation Params
ENTRY_THRESHOLD = 0.5  # Z-Score > 0.5 to enter
EXIT_HOURS = 4         # Hold for 4 hours
FEES_BPS = 0.05        # 5 bps taker fee per leg (0.1% round trip)

trades = []
pnl_curve = [100.0] # Start with $100 base

for index, row in df.iterrows():
    # Signal Generation
    signal = 0
    score = row.get('new_composite', 0)
    
    if score > ENTRY_THRESHOLD:
        signal = 1 # LONG
    elif score < -ENTRY_THRESHOLD:
        signal = -1 # SHORT
        
    if signal == 0:
        continue
        
    # Execution
    entry_price = row['price']
    outcome_pct = row.get(f'ret_{EXIT_HOURS}h')
    
    if pd.isna(outcome_pct):
        continue
        
    # PnL Calculation
    raw_pnl = outcome_pct * signal
    net_pnl = raw_pnl - (FEES_BPS * 2 / 100) # Round trip fees
    
    trades.append(net_pnl)
    pnl_curve.append(pnl_curve[-1] * (1 + net_pnl))

if not trades:
    print("No trades generated with current threshold.")
else:
    trades_series = pd.Series(trades)
    
    # Metrics
    total_trades = len(trades)
    wins = trades_series[trades_series > 0]
    losses = trades_series[trades_series <= 0]
    
    winrate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0
    
    # Sharpe (Simplistic: Annualized using trade frequency)
    # Assuming avg 4h hold, but trades might physically overlap in simulation (simple vector backtest)
    avg_return = trades_series.mean()
    std_return = trades_series.std()
    
    sharpe = 0
    if std_return > 0:
        sharpe = (avg_return / std_return) * np.sqrt(365 * 6) # approx 6 trades a day? or just use N trades
        # Or standard Sharpe: PnL Mean / PnL Std 
        sharpe = avg_return / std_return
        
    print(f"Total Trades: {total_trades}")
    print(f"Winrate: {winrate*100:.1f}%")
    print(f"Avg Win: {avg_win*100:.2f}% | Avg Loss: {avg_loss*100:.2f}%")
    print(f"Profit Factor: {abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 'Inf':.2f}")
    print(f"Sharpe Ratio (per trade): {sharpe:.2f}")
    print(f"Final Equity (Base 100): {pnl_curve[-1]:.2f}")

    with open(output_file, 'a') as f:
        f.write("\n=== Detailed Backtest Results ===\n")
        f.write(f"Strategy: Optimized Composite > {ENTRY_THRESHOLD} Z-Score | Hold {EXIT_HOURS}h\n")
        f.write(f"Winrate: {winrate*100:.1f}%\n")
        f.write(f"Profit Factor: {abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 'Inf':.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
        f.write(f"Net PnL: {(pnl_curve[-1]-100):.2f}%\n")


# 8. ADVANCED HYPOTHESIS TESTING (ATR & Filters)
print("\n=== Advanced Hypothesis Testing ===")

# 8.1 Calculate ATR on Price DF
# TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))
price_df['prev_close'] = price_df['close'].shift(1)
price_df['tr1'] = price_df['high'] - price_df['low']
price_df['tr2'] = (price_df['high'] - price_df['prev_close']).abs()
price_df['tr3'] = (price_df['low'] - price_df['prev_close']).abs()
price_df['tr'] = price_df[['tr1', 'tr2', 'tr3']].max(axis=1)
price_df['atr'] = price_df['tr'].rolling(window=14).mean()

# Simulation Engine
def simulate_trade(row, use_atr_exit=False, tp_mult=2.0, sl_mult=1.0, fixed_hours=4):
    """Simulates a trade outcome with either fixed time or ATR targets"""
    
    entry_ts = row['ts']
    entry_price = row['price']
    
    # Signal Direction based on new composite
    score = row.get('new_composite', 0)
    if score > 0.5: direction = 1
    elif score < -0.5: direction = -1
    else: return None
    
    # 1. Look up ATR at entry time
    try:
        idx = price_df.index.get_indexer([entry_ts], method='nearest')[0]
        # Check time diff again to be safe
        if abs((price_df.index[idx] - entry_ts).total_seconds()) > 3600:
            return None # No matching data
            
        atr_val = price_df.iloc[idx]['atr']
        if pd.isna(atr_val) or atr_val == 0: return None
        
    except:
        return None
        
    # 2. Define Targets
    if use_atr_exit:
        if direction == 1: # LONG
            tp_price = entry_price + (atr_val * tp_mult)
            sl_price = entry_price - (atr_val * sl_mult)
        else: # SHORT
            tp_price = entry_price - (atr_val * tp_mult)
            sl_price = entry_price + (atr_val * sl_mult)
    
    # 3. Walk Forward (Path Dependence)
    # Get slice of price_df from entry to max hold time (e.g. 24h to avoid infinite loops)
    # We use 24h max for ATR trades too, to simulate "end of day" exit
    max_time = entry_ts + pd.Timedelta(hours=24) 
    
    future_candles = price_df[entry_ts:max_time]
    
    if future_candles.empty:
        return None
        
    for _, candle in future_candles.iterrows():
        # Check SL first (conservative)
        if direction == 1:
            if candle['low'] <= sl_price:
                return -sl_mult * atr_val / entry_price # Loss % (approx)
            if candle['high'] >= tp_price:
                return tp_mult * atr_val / entry_price # Win %
        else:
            if candle['high'] >= sl_price:
                return -sl_mult * atr_val / entry_price
            if candle['low'] <= tp_price:
                return tp_mult * atr_val / entry_price
                
    # If time runs out: precise exit at last close
    exit_price = future_candles.iloc[-1]['close']
    return (exit_price - entry_price) / entry_price * direction


# Run Scenarios
scenarios = [
    {'name': 'Base (Fixed 4h)', 'atr': False, 'filters': []},
    {'name': 'ATR 2R (TP=2*ATR, SL=1*ATR)', 'atr': True, 'tp': 2.0, 'sl': 1.0, 'filters': []},
    {'name': 'ATR + Filters (No RSI>80)', 'atr': True, 'tp': 2.0, 'sl': 1.0, 'filters': ['rsi_check']},
    {'name': 'ATR + VP Filter (No D-Shape)', 'atr': True, 'tp': 2.0, 'sl': 1.0, 'filters': ['vp_check']},
    {'name': 'All Filters Combined', 'atr': True, 'tp': 2.0, 'sl': 1.0, 'filters': ['rsi_check', 'vp_check']}
]

results = []

for sc in scenarios:
    pnl_log = []
    
    for _, row in df.iterrows():
        # Apply Filters Pre-Trade
        filters = sc.get('filters', [])
        
        # PSI/KDJ Check: Block LONG if RSI > 80 (using our KDJ J proxy)
        if 'rsi_check' in filters:
            score = row.get('new_composite', 0)
            j_val = row.get('rsi_kj', 50)
            if score > 0.5 and j_val > 80: # Trying to LONG the top
                continue
                
        if 'vp_check' in filters:
            if row.get('vol_profile') == 'D-Shape':
                continue
                
        if sc['atr']:
            res = simulate_trade(row, use_atr_exit=True, tp_mult=sc['tp'], sl_mult=sc['sl'])
        else:
             res = row.get(f'ret_4h', 0)
             # Adjust direction
             score = row.get('new_composite', 0)
             if score > 0.5: d = 1
             elif score < -0.5: d = -1
             else: res = None
             
             if res is not None: res *= d
            
        if res is not None and not pd.isna(res):
            pnl_log.append(res - 0.001) # Fees
            
    # Calc Metrics
    if pnl_log:
        ser = pd.Series(pnl_log)
        winrate = (ser > 0).mean()
        pf = abs(ser[ser>0].sum() / ser[ser<0].sum()) if (ser[ser<0].sum()) != 0 else 0
        total_pnl = ser.sum()
        
        results.append({
            'Strategy': sc['name'],
            'Trades': len(ser),
            'Winrate': f"{winrate*100:.1f}%",
            'PF': f"{pf:.2f}",
            'Total PnL': f"{total_pnl*100:.1f}%"
        })

print("\n=== Scenario Comparison ===")
res_df = pd.DataFrame(results)
print(res_df.to_string())

with open(output_file, 'a') as f:
    f.write("\n=== Advanced Hypothesis Testing ===\n")
    f.write(res_df.to_string() + "\n")


