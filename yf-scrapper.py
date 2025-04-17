# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# import time
# import os

# # Define API key and base URL
# API_KEY = "DrZFwagFdP48WQjiJ7hfLqM61SX3ikGZ"
# BASE_URL = "https://api.polygon.io/v2/reference/news"

# # Path to the S&P 500 tickers CSV file
# sp500_csv_path = "./sp500_companies.csv"  # Ensure this path is correct

# # Load the S&P 500 tickers
# try:
#     sp500_df = pd.read_csv(sp500_csv_path)
#     tickers = sp500_df["Symbol"].tolist()  # Extract tickers from 'Symbol' column
#     print(f"✅ Loaded {len(tickers)} S&P 500 tickers.")
# except Exception as e:
#     print(f"❌ Error loading S&P 500 tickers: {e}")
#     exit()

# # Create a directory to store individual CSV files
# output_dir = "./snp500-yf/"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# # Function to fetch news articles for a given stock ticker
# def fetch_stock_news(ticker, limit=10):
#     params = {
#         "ticker": ticker,
#         "limit": limit,
#         "apiKey": API_KEY
#     }
    
#     response = requests.get(BASE_URL, params=params)
    
#     if response.status_code == 200:
#         data = response.json()
#         if "results" in data:
#             articles = data["results"]
            
#             # Extract relevant fields
#             news_list = []
#             now = datetime.utcnow()
#             last_24_hours = now - timedelta(hours=24)

#             for article in articles:
#                 published_time = article.get("published_utc", "N/A")
                
#                 # Convert to datetime object
#                 if published_time != "N/A":
#                     published_dt = datetime.strptime(published_time, "%Y-%m-%dT%H:%M:%SZ")

#                     # Filter articles from the last 24 hours
#                     if published_dt >= last_24_hours:
#                         news_list.append({
#                             "Ticker": ticker,  # Include ticker for tracking
#                             "Title": article.get("title", "N/A"),
#                             "Summary": article.get("description", "N/A"),
#                             "URL": article.get("article_url", "N/A"),
#                             "Published": published_time,
#                             "Publisher": article["publisher"].get("name", "N/A") if "publisher" in article else "N/A",
#                             "Sentiment": article["insights"][0]["sentiment"] if "insights" in article and article["insights"] else "N/A"
#                         })

#             return news_list  # Return list of articles
#         else:
#             print(f"🔸 No news found for {ticker}.")
#             return []
#     else:
#         print(f"❌ Error fetching data for {ticker}: {response.status_code}")
#         return []

# # Loop through each ticker in S&P 500 and fetch news
# for idx, ticker in enumerate(tickers):
#     print(f"🔍 Fetching news for {ticker} ({idx+1}/{len(tickers)})...")

#     news_data = fetch_stock_news(ticker)

#     # If news articles exist, save them in a separate CSV file
#     if news_data:
#         df_news = pd.DataFrame(news_data)
#         filename = os.path.join(output_dir, f"{ticker}_yf_last24hours.csv")
#         df_news.to_csv(filename, index=False)
#         print(f"✅ Saved {len(df_news)} articles to {filename}")

#     # Respect API rate limits (Polygon.io free tier: 5 requests per minute)
#     time.sleep(12)  # Wait 12 seconds to stay within limits

# print("✅ All news scraping completed! Check the ./sp500_news/ folder.")
