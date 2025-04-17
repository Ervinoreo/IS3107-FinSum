import undetected_chromedriver as uc
import time
import pandas as pd
from selenium.webdriver.common.by import By
import os
from datetime import datetime
from pytz import timezone


# Define Base Directory
BASE_DIR = "/Users/ervinyeoh/Desktop/unimods/is3107/project/airflow"
# REDDIT_OUTPUT_DIR = os.path.join(BASE_DIR, "snp500-reddit")
REDDIT_OUTPUT_DIR = os.path.join(BASE_DIR, "snp500-reddit-test")
sgt = timezone("Asia/Singapore")
now = datetime.now(sgt)
date_str = now.strftime("%Y%m%d")

def run_reddit_scraper(tickers):
    """Scrape Reddit for stock discussions"""
    os.makedirs(REDDIT_OUTPUT_DIR, exist_ok=True)
    
    driver = uc.Chrome(version_main=134)
    # driver = uc.Chrome()

    for ticker in tickers:
        print(f"🔍 Searching Reddit for {ticker}...")

        # Construct Reddit search URL
        # search_url = f"https://www.reddit.com/r/stocks/search/?q={ticker}&restrict_sr=1&t=day"
        search_url = f"https://www.reddit.com/r/wallstreetbets/search/?q={ticker}&restrict_sr=1&t=day"
        driver.get(search_url)
        time.sleep(2)

        post_data = []
        post_index = 0  # Index to track posts

        while True:
            try:
                # Get the list of post titles (refreshes every loop)
                post_titles = driver.find_elements(By.CSS_SELECTOR, 'a[data-testid="post-title"]')

                # If no more posts are available, break
                if post_index >= len(post_titles):
                    print(f"✅ All posts for {ticker} have been processed.")
                    break

                # Click on the current post
                post_titles[post_index].click()
                print(f"✅ Clicked on post {post_index + 1} for {ticker} successfully!")

                # Wait for post content to load
                time.sleep(2)

                # Extract post content
                try:
                    post_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'shreddit-post p')
                    post_text = "\n".join([p.text for p in post_paragraphs]) if post_paragraphs else "N/A"
                except Exception as e:
                    print(f"❌ Error extracting post content: {e}")
                    post_text = "N/A"

                time.sleep(1.5)

                # Extract comments (all comments, no limit)
                try:
                    comment_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'shreddit-comment p')
                    comment_text = "\n".join([f"- {p.text}" for p in comment_paragraphs]) if comment_paragraphs else "N/A"
                except Exception as e:
                    print(f"❌ Error extracting comments: {e}")
                    comment_text = "N/A"

                # Store the extracted data
                post_data.append({"post": post_text, "comments": comment_text})

                # Go back to the search results page
                driver.back()
                print("🔙 Navigated back to search results!")

                # Wait before proceeding to the next post
                time.sleep(2)

                # Move to the next post
                post_index += 1

            except Exception as e:
                print(f"❌ Error clicking post {post_index + 1} for {ticker}: {e}")
                break

        # Save scraped data to CSV
        if post_data:
            df = pd.DataFrame(post_data, columns=["post", "comments"])
            file_name = f"{ticker}_reddit_{date_str}.csv"
            file_path = os.path.join(REDDIT_OUTPUT_DIR, file_name)
            df.to_csv(file_path, index=False)
            print(f"📁 Saved: {file_path}")

    driver.quit()
    print("✅ Done scraping Reddit posts for all S&P 500 tickers.")