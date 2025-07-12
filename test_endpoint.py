import requests
import json

# Test the recommendation endpoint
url = "http://localhost:8000/recommend-hairstyles/"
data = {
    "face_shape": "oval",
    "user_profile": {
        "ethnicity": "caucasian",
        "age": 25,
        "gender": "male",
        "hair_texture": "straight"
    }
}

try:
    print("Testing recommendation endpoint...")
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        recommendations = response.json()
        print(f"Success! Got {len(recommendations)} recommendations")
        for rec in recommendations:
            print(f"  - {rec['name']}: {rec['description'][:50]}...")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Connection error: {e}") 