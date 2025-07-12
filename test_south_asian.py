import requests
import json

# Test the South Asian ethnicity option
url = "http://localhost:8000/create-profile/"
data = {
    "ethnicity": "south_asian",
    "age": 28,
    "gender": "male",
    "hair_texture": "wavy"
}

try:
    print("Testing South Asian ethnicity option...")
    print(f"Sending data: {json.dumps(data, indent=2)}")
    
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        profile = response.json()
        print(f"✅ Success! Profile created with ID: {profile['profile_id']}")
        print(f"Ethnicity: {profile['user_profile']['ethnicity']}")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"Connection error: {e}")
    print("Make sure the backend server is running on localhost:8000") 