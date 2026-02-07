import csv
import json
import os
import time
from gemini_client import GeminiClient

def load_csv_data(filepath):
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def run_ai_analysis():
    # Load Blackbox Data (Full Context)
    data = load_csv_data('analysis/blackbox_data.csv')
    if not data:
        return

    # Initialize Client
    api_key = os.getenv('GEMINI_API_KEY')
    client = GeminiClient(api_key=api_key)
    
    print(f"Loaded {len(data)} signals. Selecting test cases...")
    
    # Select specific interesting cases
    # Case 1: Early Short (Winner)
    case_win = data[10] if len(data) > 10 else data[0]
    
    # Case 2: Broadening/Ranging (Potential Zombie?)
    # We look for a LONG that might be a zombie
    case_zombie = next((r for r in data if r['Direction'] == 'LONG'), data[0])
    
    test_cases = [
        {"name": "Winning Short Strategy", "data": case_win},
        {"name": "Potential Zombie Long", "data": case_zombie}
    ]
    
    results = []
    
    print("-" * 50)
    for case in test_cases:
        row = case['data']
        name = case['name']
        
        print(f"Analyzing: {name} ({row['Timestamp']})")
        
        # Construct Context Dict
        context = {
            "timestamp": row['Timestamp'],
            "price": row['Price'],
            "signal": {
                "type": row['Signal_Type'],
                "direction": row['Direction'],
                "confidence": row['Confidence']
            },
            "technical": {
                "score": row['Score_Tech'],
                "structure_score": row.get('Score_Struct', 'N/A')
            },
            "sentiment": {
                "score": row['Score_Sent'],
                "fear_greed": row.get('Fear_Greed', 50)
            },
            "market_profile": {
                "vp_context": row.get('VP_Context', 'N/A'),
                "quantum_state": row.get('Quantum_State', 'N/A'),
                "risk_env": row.get('Risk_Env', 'N/A')
            },
            "kalman": {
                "price": row.get('Kalman_Price', 0),
                "velocity": row.get('Kalman_Velocity', 0)
            }
        }
        
        # Call AI
        start_time = time.time()
        ai_response = client.analyze_market_context(context)
        duration = time.time() - start_time
        
        print(f"AI Response ({duration:.2f}s):")
        print(json.dumps(ai_response, indent=2))
        print("-" * 50)
        
        results.append({
            "case": name,
            "context": context,
            "ai_analysis": ai_response
        })
        
        # Respect Rate Limits if not free tier (though flash is fast)
        time.sleep(1)

if __name__ == "__main__":
    run_ai_analysis()
