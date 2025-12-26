import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# --- Configuration ---
# IMPORTANT: Replace with your actual API key
API_KEY = "" 

# Base URL for the 'everything' endpoint
BASE_URL = "https://newsapi.org/v2/everything" 

# Define your search queries based on your plan [cite: 109]
# (e.g., war, conflict, sanctions, terrorism, economic crisis)
QUERY = "(war OR conflict OR sanctions OR terrorism OR 'economic crisis') AND (geopolitical OR economy)"

# --- Date Range ---
# Your plan starts from 2000 [cite: 82]
# Note: Free NewsAPI plans often limit to 30 days back.
# You may need a paid plan or a different source for full historical data.

# For this example, let's just pull the last 30 days
start_date = datetime.now() - timedelta(days=29)
end_date = datetime.now()

all_headlines = []

# Loop through each day in your range
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"Fetching headlines for {date_str}...")

    params = {
        'q': QUERY,
        'from': date_str,
        'to': date_str,
        'sortBy': 'relevancy',
        'apiKey': API_KEY,
        'language': 'en'
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status() # Raise an error for bad responses (4xx or 5xx)

        data = response.json()

        if data['status'] == 'ok':
            for article in data['articles']:
                all_headlines.append({
                    'date': current_date.date(),
                    'headline': article['title']
                })

        # IMPORTANT: Respect API rate limits
        # Free plans are often limited. Pause between requests.
        time.sleep(1) # Pause for 1 second

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {date_str}: {e}")
        if e.response.status_code == 426: # 426 = Upgrade Required (hit 30-day limit)
            print("Hit free plan 30-day historical limit. Stopping.")
            break
        if e.response.status_code == 429: # 429 = Too Many Requests
            print("Rate limit hit. Sleeping for 60 seconds...")
            time.sleep(60)
            continue # Retry this date

    current_date += timedelta(days=1)

# --- Save Data ---
if all_headlines:
    headline_df = pd.DataFrame(all_headlines)
    headline_df.to_csv("data/raw_headlines.csv", index=False, encoding='utf-8')
    print(f"\nSuccessfully downloaded and saved {len(headline_df)} headlines to data/raw_headlines.csv")
else:
    print("No headlines were downloaded.")
