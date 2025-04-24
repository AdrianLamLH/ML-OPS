import requests

# Test the wine classification API
url = "http://localhost:8000/predict"

# Example wine features
wine_sample = {
    "alcohol": 13.0,
    "malic_acid": 2.34,
    "ash": 2.36,
    "alcalinity_of_ash": 19.5,
    "magnesium": 99.7,
    "total_phenols": 2.29,
    "flavanoids": 2.03,
    "nonflavanoid_phenols": 0.36,
    "proanthocyanins": 1.59,
    "color_intensity": 5.05,
    "hue": 0.96,
    "od280_od315_of_diluted_wines": 2.61,
    "proline": 746.0
}

# Check if API is running
try:
    health = requests.get("http://localhost:8000/health")
    print("API health check:", "OK" if health.status_code == 200 else "Failed")
except:
    print("Error: API not running at http://localhost:8000")
    exit()

# Send prediction request
print("\nSending wine data to API...")
response = requests.post(url, json=wine_sample)

# Print results
if response.status_code == 200:
    result = response.json()
    print("Prediction successful!")
    print("Wine class:", result["prediction_label"])
    print("Class number:", result["prediction"])
else:
    print("Error:", response.status_code)
    print(response.text)