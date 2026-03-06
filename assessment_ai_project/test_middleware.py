
import urllib.request
import json
import time
import sys

def test_api():
    url = "http://localhost:5001/api/v1/predict"
    payload = {
        "country": "IN",
        "crop": "Wheat",
        "partner": "PartnerA",
        "irrigation": True,
        "hired_workers": 5,
        "area": 10,
        "planYear": 2024,
    }
    
    print(f"Sending request to {url}...")
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print("Success!")
                response_body = response.read().decode('utf-8')
                data = json.loads(response_body)
                print("\n--- Summary ---")
                print(json.dumps(data["summary"], indent=2))
                print("\n--- Auto Filled (Green) ---")
                print(json.dumps(data["groups"]["auto_filled"], indent=2))
                print("\n--- Needs Review (Red/Amber) ---")
                print(json.dumps(data["groups"]["needs_review"], indent=2))
            else:
                print(f"Failed: {response.status}")
                print(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
