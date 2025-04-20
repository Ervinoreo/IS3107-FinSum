import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from pytz import timezone

API_KEY = "DrZFwagFdP48WQjiJ7hfLqM61SX3ikGZ"
# BASE_DIR = "/Users/ervinyeoh/Desktop/unimods/is3107/project/airflow"

# yueyaoz
BASE_DIR = "/Users/yueyaoz/Downloads/IS3107-FinSum/airflow"

# YF_OUTPUT_DIR = os.path.join(BASE_DIR, "snp500-yf")
YF_OUTPUT_DIR = os.path.join(BASE_DIR, "snp500-yf-test")
YF_API_URL = "https://api.polygon.io/v2/reference/news"

def run_yf_scraper(tickers):
    """Fetch stock news for each ticker from Yahoo Finance API"""
    os.makedirs(YF_OUTPUT_DIR, exist_ok=True)
    now = datetime.utcnow()
    last_24_hours = now - timedelta(hours=24)
    sgt = timezone("Asia/Singapore")
    now = datetime.now(sgt)
    date_str = now.strftime("%Y%m%d")



    for idx, ticker in enumerate(tickers):
        print(f"🔍 Fetching news for {ticker} ({idx+1}/{len(tickers)})...")

        params = {"ticker": ticker, "limit": 10, "apiKey": API_KEY}
        response = requests.get(YF_API_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            news_list = []
            
            for article in data.get("results", []):
                published_time = article.get("published_utc", "N/A")
                
                if published_time != "N/A":
                    published_dt = datetime.strptime(published_time, "%Y-%m-%dT%H:%M:%SZ")
                    if published_dt >= last_24_hours:
                        news_list.append({
                            "Ticker": ticker,
                            "Title": article.get("title", "N/A"),
                            "Summary": article.get("description", "N/A"),
                            "URL": article.get("article_url", "N/A"),
                            "Published": published_time,
                            "Publisher": article.get("publisher", {}).get("name", "N/A"),
                            "Sentiment": article.get("insights", [{}])[0].get("sentiment", "N/A")
                        })

            if news_list:
                df_news = pd.DataFrame(news_list)
                df_news.to_csv(os.path.join(YF_OUTPUT_DIR, f"{ticker}_yf_{date_str}.csv"), index=False)
                print(f"✅ Saved {len(df_news)} articles for {ticker}")

        time.sleep(12)  # Respect API rate limits

