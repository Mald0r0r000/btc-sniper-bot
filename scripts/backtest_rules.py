import pandas as pd
import numpy as np

DATASET_FILE = "dataset.csv"

def backtest_rules():
    print(f"Loading {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)
    
    # Filter for completed trades only
    df = df[df['outcome_result'].isin(['WIN', 'LOSS'])].copy()
    
    if len(df) == 0:
        print("No completed trades.")
        return

    # Base Stats
    total = len(df)
    wins = len(df[df['outcome_result'] == 'WIN'])
    base_winrate = (wins / total) * 100
    
    print(f"\n--- BASELINE ---")
    print(f"Trades: {total}")
    print(f"Winrate: {base_winrate:.2f}%")
    
    # --- RULE 1: POSITIVE GEX (Stability) ---
    # Column: indicators_derivatives_gex_profile_net_gex_usd_m
    gex_col = 'indicators_derivatives_gex_profile_net_gex_usd_m'
    
    # Filter: GEX > 0
    df_gex = df[df[gex_col] > 0]
    wins_gex = len(df_gex[df_gex['outcome_result'] == 'WIN'])
    wr_gex = (wins_gex / len(df_gex)) * 100 if len(df_gex) > 0 else 0
    
    print(f"\n--- RULE 1: Positive GEX (Market Stability) ---")
    print(f"Trades kept: {len(df_gex)} ({len(df_gex)/total*100:.1f}%)")
    print(f"Winrate: {wr_gex:.2f}% (Delta: {wr_gex - base_winrate:+.2f}%)")
    
    # --- RULE 2: LOW VOLATILITY (Low ATR) ---
    # Column: indicators_adx_atr
    atr_col = 'indicators_adx_atr'
    median_atr = df[atr_col].median()
    
    # Filter: ATR < Median
    df_atr = df[df[atr_col] < median_atr]
    wins_atr = len(df_atr[df_atr['outcome_result'] == 'WIN'])
    wr_atr = (wins_atr / len(df_atr)) * 100 if len(df_atr) > 0 else 0
    
    print(f"\n--- RULE 2: Low ATR (Avoid Volatility) (< {median_atr:.2f}) ---")
    print(f"Trades kept: {len(df_atr)} ({len(df_atr)/total*100:.1f}%)")
    print(f"Winrate: {wr_atr:.2f}% (Delta: {wr_atr - base_winrate:+.2f}%)")
    
    # --- RULE 3: TARGETING SHORTS (Momentum) ---
    # Logic: Only take LONGs (since short liqs are up)
    # df_longs = df[df['signal_direction'] == 'LONG']
    # But wait, correlation report said "liquidation_nearest_short_liq_price" correlates with WIN.
    # This implies that when a Short Liq exists/is targeted, we win.
    # Let's test: Filter for trades where signal direction matches the liquidation objective?
    # Actually, simpler: Test the "Holy Trinity" Combination (GEX > 0 AND ATR < Median)
    
    df_combo = df[
        (df[gex_col] > 0) & 
        (df[atr_col] < median_atr)
    ]
    wins_combo = len(df_combo[df_combo['outcome_result'] == 'WIN'])
    wr_combo = (wins_combo / len(df_combo)) * 100 if len(df_combo) > 0 else 0
    
    print(f"\n--- COMBO: Positive GEX + Low ATR ---")
    print(f"Trades kept: {len(df_combo)} ({len(df_combo)/total*100:.1f}%)")
    print(f"Winrate: {wr_combo:.2f}% (Delta: {wr_combo - base_winrate:+.2f}%)")
    
    # --- OPTIMAL QUANTILE SEARCH ---
    # Let's find the best ATR threshold
    print(f"\n--- OPTIMIZATION SEARCH ---")
    best_wr = 0
    best_thresh = 0
    for q in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        thresh = df[atr_col].quantile(q)
        subset = df[df[atr_col] < thresh]
        if len(subset) < 10: continue
        wins_s = len(subset[subset['outcome_result'] == 'WIN'])
        wr_s = (wins_s / len(subset)) * 100
        print(f"ATR < {q*100}th pct ({thresh:.1f}): Winrate {wr_s:.2f}% ({len(subset)} trades)")


if __name__ == "__main__":
    backtest_rules()
