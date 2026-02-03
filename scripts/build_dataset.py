import os
import json
import pandas as pd
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configuration
ARTIFACTS_DIR = "artifacts_archive"
OUTPUT_FILE = "dataset.csv"

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # For lists, we might want to store count or basic stats, 
            # or just convert to string to avoid explosion
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def load_and_flatten_report(file_path: str) -> Dict[str, Any]:
    """Loads a single analysis_report.json and flattens it."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Add source file info/run_id prediction
        # Path is typically artifacts_archive/run_12345/analysis_report.json
        parts = file_path.split(os.sep)
        if len(parts) >= 2 and parts[-2].startswith("run_"):
            run_id = parts[-2].replace("run_", "")
            data["meta_run_id"] = run_id
            
        return flatten_dict(data)
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

def build_dataset():
    print(f"🔍 Scanning {ARTIFACTS_DIR} for analysis_report.json...")
    
    # Find all json files
    pattern = os.path.join(ARTIFACTS_DIR, "**", "analysis_report.json")
    files = glob.glob(pattern, recursive=True)
    
    print(f"   Found {len(files)} reports.")
    
    records = []
    for f in files:
        record = load_and_flatten_report(f)
        if record:
            records.append(record)
            
    if not records:
        print("⚠️ No valid records found.")
        return
        
    print(f"📊 Compiling DataFrame with {len(records)} records...")
    df = pd.DataFrame(records)
    
    # Post-processing: timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        print(f"   ✅ Sorted by timestamp. Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
    print("🔮 Calculating forward outcomes (Signals) AND Future Market Moves (Precursors)...")
    
    # Add columns for Precursor Analysis (The "Truth" of the market)
    df['future_4h_return_pct'] = 0.0
    df['future_4h_volatility_pct'] = 0.0
    
    df['outcome_result'] = 'UNKNOWN'
    df['outcome_pnl'] = 0.0
    
    # Convert to list of dicts for faster iteration than iterrows
    df_records = df.to_dict('records')
    total_records = len(df_records)
    
    # Approx 4 hours converted to 15-min intervals = 16 periods
    LOOKAHEAD_PERIODS = 16 
    
    valid_precursors = 0

    for i, row in enumerate(df_records):
        current_price = row.get('price', 0)
        if current_price == 0: continue
        
        # --- 1. MARKET PRECURSORS (Calculate Truth for every row) ---
        # Look ahead for Return & Volatility
        end_idx = min(i + LOOKAHEAD_PERIODS, total_records)
        # Check timestamps to ensure we are looking into the future, not just next record in list
        # (Though if sorted, i+1 is future)
        
        future_window = [r.get('price', 0) for r in df_records[i+1:end_idx] if r.get('price', 0) > 0]
        
        if future_window and len(future_window) >= 4: # Require at least 4 periods (1h) to be valid
            valid_precursors += 1
            # Return: Price change after 4h
            future_price = future_window[-1]
            val = (future_price - current_price) / current_price * 100
            df_records[i]['future_4h_return_pct'] = val
            df_records[i]['future_4h_volatility_pct'] = (max(future_window) - min(future_window)) / current_price * 100
            
            if i == 0:
                print(f"DEBUG i=0: Current: {current_price}, Future: {future_price}, Val: {val}")
                print(f"DEBUG i=0 Record: {df_records[i].get('future_4h_return_pct')}")
        else:
            df_records[i]['future_4h_return_pct'] = float('nan')
            df_records[i]['future_4h_volatility_pct'] = float('nan')

        # --- 2. SIGNAL OUTCOME (Only for rows with signals) ---
        # Only analyze if there was a signal
        if row.get('signal_type') == 'NO_SIGNAL' or pd.isna(row.get('signal_targets_tp1')):
            continue
            
        entry_price = row.get('price', 0)
        tp1 = row.get('signal_targets_tp1', 0)
        sl = row.get('signal_targets_sl', 0)
        direction = row.get('signal_direction', 'NEUTRAL')
        timestamp = row.get('timestamp')
        
        if direction == 'NEUTRAL' or entry_price == 0:
            continue
            
        # Look ahead
        success = False
        failure = False
        
        # Check future candles (next 48h windows usually)
        for j in range(i + 1, min(i + 96, total_records)): # Look up to 96 reports ahead (~1 day if 15m)
            future_row = df_records[j]
            future_price = future_row.get('price', 0)
            
            if future_price == 0: continue
            
            if direction == 'LONG':
                if future_price >= tp1:
                    success = True
                    break
                if future_price <= sl:
                    failure = True
                    break
            elif direction == 'SHORT':
                if future_price <= tp1:
                    success = True
                    break
                if future_price >= sl:
                    failure = True
                    break
        
        if success:
            df_records[i]['outcome_result'] = 'WIN'
            df_records[i]['outcome_pnl'] = abs(tp1 - entry_price) / entry_price * 100
        elif failure:
            df_records[i]['outcome_result'] = 'LOSS'
            df_records[i]['outcome_pnl'] = -abs(sl - entry_price) / entry_price * 100
        else:
            df_records[i]['outcome_result'] = 'TIMEOUT'
    
    # Recreate DF
    final_df = pd.DataFrame(df_records)
    
    print(f"📊 Valid Precursor Rows (with future data): {valid_precursors}/{total_records}")
    print(f"DEBUG final_df head future_return: {final_df['future_4h_return_pct'].head(1).tolist()}")
    
    # Save
    print(f"💾 Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    build_dataset()
