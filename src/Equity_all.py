import requests
import pandas as pd
from io import StringIO

# NSE master equity list
url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

# File path you want
OUTPUT_FILE = "..\\data\\nse_only_symbols.txt"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36"
}

# Fetch CSV directly into memory
response = requests.get(url, headers=headers)
data = StringIO(response.text)

# Load with pandas
df = pd.read_csv(data)

# Save only symbols into the TXT file
df["SYMBOL"].to_csv(OUTPUT_FILE, index=False, header=False)

print(f"✅ NSE symbols saved to {OUTPUT_FILE}")
