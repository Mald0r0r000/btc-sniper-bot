import urllib.request
import urllib.error
import json
import os
import time
import re

class GeminiClient:
    """
    Client for Google Gemini API (REST) - Zero Dependencies (urllib)
    Used for Semantic Analysis of trading signals.
    """
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash-lite"):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            print("⚠️ GeminiClient initialized without API Key.")
            
    def list_models(self):
        """Debug method to see available models"""
        url = f"{self.base_url}/models?key={self.api_key}"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
                return [m['name'].replace('models/', '') for m in data.get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
        except Exception as e:
            return [f"Error listing models: {e}"]

    def generate_content(self, prompt: str) -> dict:
        """
        Sends a prompt to Gemini and returns the response text.
        """
        if not self.api_key:
            return {"error": "No API Key provided"}
            
        # Ensure model has 'models/' prefix only if not provided?
        # The API expects .../models/MODEL_NAME:generateContent
        # If self.model is "gemini-1.5-flash", URL should be .../models/gemini-1.5-flash:generateContent
        
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 8192,
            }
        }
        
        max_retries = 3
        current_retry = 0
        
        # VERBOSE LOGGING
        print(f"🤖 [GEMINI] Calling API...")
        print(f"   Model: {self.model}")
        print(f"   URL: {self.base_url}/models/{self.model}:generateContent")
        print(f"   API Key: {'✅ Present' if self.api_key else '❌ MISSING'}")
        
        start_time = time.time()
        
        while current_retry <= max_retries:
            try:
                req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
                with urllib.request.urlopen(req, timeout=30) as response:
                    elapsed = time.time() - start_time
                    print(f"   ✅ Response: {response.status} in {elapsed:.2f}s")
                    
                    if response.status == 200:
                        raw_data = response.read()
                        print(f"   📦 Data: {len(raw_data)} bytes received")
                        data = json.loads(raw_data.decode('utf-8'))
                        try:
                            text = data['candidates'][0]['content']['parts'][0]['text']
                            print(f"   📝 Text: {len(text)} chars extracted")
                            return {"text": text}
                        except (KeyError, IndexError):
                            print(f"   ❌ Parse Error: Unexpected structure")
                            return {"error": "Unexpected response structure", "raw": data}
                    else:
                        print(f"   ❌ API Error: {response.status}")
                        return {"error": f"API Error {response.status}", "raw": response.read().decode('utf-8')}
                    
            except urllib.error.HTTPError as e:
                elapsed = time.time() - start_time
                error_body = e.read().decode('utf-8')
                print(f"   ❌ HTTP Error {e.code}: {e.reason} (after {elapsed:.2f}s)")
                print(f"   📄 Error Body: {error_body[:200]}...")
                
                # Handle Rate Limit (429)
                if e.code == 429:
                    if current_retry < max_retries:
                        delay = (2 ** current_retry) + 1  # 2s, 3s, 5s...
                        print(f"   ⏳ Rate Limit 429. Retrying in {delay}s... (Attempt {current_retry+1}/{max_retries})")
                        time.sleep(delay)
                        current_retry += 1
                        continue
                    else:
                        print(f"   💀 Rate Limit Exceeded after {max_retries} retries. Giving up.")
                        return {"error": f"API Rate Limit Exceeded (429) after {max_retries} retries", "raw": error_body}
                
                return {"error": f"HTTP Error {e.code}: {e.reason}", "raw": error_body, "url": url}
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"   ❌ Exception: {type(e).__name__}: {e} (after {elapsed:.2f}s)")
                return {"error": f"Request Failed: {e}"}

    def analyze_market_context(self, context: dict) -> dict:
        """
        Analyzes a trading signal context.
        Expects a JSON-like dict with Price, Volume, Scores, Indicators.
        Returns: {
            "sentiment_score": int (0-100),
            "risk_assessment": str,
            "veto": bool
        }
        """
        
        # Construct a structured prompt
        prompt = f"""
You are a Senior Crypto Quantitative Analyst.
Analyze the following market context for a Bitcoin trading signal.

CONTEXT:
{json.dumps(context, indent=2)}

TASK:
1. Assess the alignment of technical indicators.
2. Identify any conflicting signals (e.g. Price making lows but Momentum diverging).
3. Provide a risk assessment.
4. Decide if we should VETO (block) this trade due to high risk or ambiguity.

OUTPUT FORMAT (JSON ONLY):
{{
  "sentiment_score": <0-100, where 0 is Bearish, 100 is Bullish>,
  "risk_assessment": "<One sentence summary of risks>",
  "veto": <true/false>,
  "reason": "<Short explanation>"
}}
"""
        response = self.generate_content(prompt)
        
        if "text" in response:
            raw_text = response['text']
            
            # Use regex to find the first JSON object
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    analysis = json.loads(json_str)
                    return analysis
                except json.JSONDecodeError:
                    return {"error": "Failed to parse Extracted JSON", "raw_text": raw_text}
            else:
                 # Fallback cleanup
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                try:
                    return json.loads(clean_text)
                except:
                    return {"error": "No JSON found in response", "raw_text": raw_text}
        else:
            return response

if __name__ == "__main__":
    # Test
    client = GeminiClient(api_key=os.getenv('GEMINI_API_KEY'))
    
    print("Listing Models...")
    models = client.list_models()
    print(f"Available Models: {models}")
    
    if models and not isinstance(models[0], str) or (isinstance(models[0], str) and "Error" not in models[0]):
        # Try to find a flash model
        flash_models = [m for m in models if 'flash' in m.lower()]
        target_model = flash_models[0] if flash_models else models[0]
        print(f"Testing with model: {target_model}")
        
        client.model = target_model
        
        ctx = {
            "price": 67000,
            "direction": "SHORT",
            "score_technical": 25,
            "score_sentiment": 40,
            "volume_profile": "d_shape",
            "rsi": 30
        }
        print("Testing Gemini Analysis...")
        res = client.analyze_market_context(ctx)
        print(json.dumps(res, indent=2))
    else:
        print("Could not list models or no models available.")
