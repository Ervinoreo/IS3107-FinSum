# import undetected_chromedriver as uc
# import time
# import pandas as pd
# from selenium.webdriver.common.by import By
# from datetime import datetime
# import os

# output_dir = "./snp500-reddit"

# # Load S&P 500 tickers
# sp500_csv_path = "./sp500_companies.csv"  # Ensure this path is correct
# sp500_df = pd.read_csv(sp500_csv_path)
# tickers = sp500_df["Symbol"].tolist()

# # Initialize the driver
# driver = uc.Chrome()

# # Iterate through each ticker symbol
# for ticker in tickers:
#     print(f"🔍 Searching Reddit for {ticker}...")

#     # Construct Reddit search URL for the ticker
#     search_url = f"https://www.reddit.com/r/stocks/search/?q={ticker}&restrict_sr=1&t=day"
#     driver.get(search_url)

#     # Wait for the page to load
#     time.sleep(2)

#     # Initialize a list to store results
#     data = []

#     post_index = 0  # Index to track posts

#     while True:
#         try:
#             # Get the list of post titles again (since DOM refreshes after each click)
#             post_titles = driver.find_elements(By.CSS_SELECTOR, 'a[data-testid="post-title"]')

#             # If the post index is beyond the available posts, exit the loop
#             if post_index >= len(post_titles):
#                 print(f"✅ All posts for {ticker} have been processed.")
#                 break

#             # Click the current post
#             post_titles[post_index].click()
#             print(f"✅ Clicked on post {post_index + 1} for {ticker} successfully!")

#             # Wait for the post page to load
#             time.sleep(2)

#             ### **1️⃣ Extract `<p>` Tags from the Main Post**
#             try:
#                 post_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'shreddit-comment p')
#                 post_text = "\n".join([p.text for p in post_paragraphs])  # Join all paragraphs
#             except Exception as e:
#                 print("❌ Error extracting post content:", e)
#                 post_text = "N/A"

#             time.sleep(1.5)

#             ### **2️⃣ Extract `<p>` Tags from Comments**
#             try:
#                 comment_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'shreddit-post p')
#                 comment_text = "\n".join([f"- {p.text}" for p in comment_paragraphs[:5]])  # First 5 comments
#             except Exception as e:
#                 print("❌ Error extracting comments:", e)
#                 comment_text = "N/A"

#             # Store the result in the list
#             data.append({"post": post_text, "comments": comment_text})

#             # Go back to search results page
#             driver.back()
#             print("🔙 Navigated back to search results!")

#             # Wait for the page to reload
#             time.sleep(2)

#             # Move to the next post
#             post_index += 1

#         except Exception as e:
#             print(f"❌ Error clicking post {post_index + 1} for {ticker}:", e)
#             break

#     # Save DataFrame to CSV with timestamp
#     if data:
#         df = pd.DataFrame(data, columns=["post", "comments"])
#         file_name = f"{ticker}_reddit_last24hours.csv"
#         file_path = os.path.join(output_dir, file_name)  # Save in snp500 folder
#         df.to_csv(file_path, index=False)
#         print(f"📁 Saved: {file_path}")

# # Close the browser
# driver.quit()

# print("✅ Done scraping Reddit posts for all S&P 500 tickers.")
