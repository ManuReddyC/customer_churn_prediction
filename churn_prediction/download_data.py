import urllib.request
import os

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
output_path = "Telco-Customer-Churn.csv"

if not os.path.exists(output_path):
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print("Download complete.")
else:
    print("Dataset already exists.")
