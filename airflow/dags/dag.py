
# export AIRFLOW_HOME=$(pwd)/airflow
import pandas as pd
import requests
import sys
import csv
import os
import pendulum
import string
import pandas as pd
import re
import shutil
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from google.cloud import storage
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")  # Check for tokenizers
        except LookupError:
            try:
                nltk.download(resource)
                print(f"✅ Downloaded: {resource}")
            except Exception as e:
                print(f"❌ Failed to download {resource}: {str(e)}")
        try:
            nltk.data.find(f"corpora/{resource}")  # Check for corpora
        except LookupError:
            try:
                nltk.download(resource)
                print(f"✅ Downloaded: {resource}")
            except Exception as e:
                print(f"❌ Failed to download {resource}: {str(e)}")

download_nltk_resources()


# Now Import the Scripts
from load_tickers import load_sp500_tickers
from yf_scraper import run_yf_scraper
from reddit_scraper import run_reddit_scraper
from airflow.exceptions import AirflowSkipException
from airflow.decorators import dag, task
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from datetime import timedelta
from google.cloud import bigquery
from airflow.operators.empty import EmptyOperator


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ervinyeoh/Desktop/unimods/is3107/project/airflow/test-62769-b846daf36f71.json"
# os.environ["GOOGLE_BIGQUERY_CREDENTIALS"] = "/Users/ervinyeoh/Desktop/unimods/is3107/project/airflow/test-62769-dbadfd9eaab9.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ervinyeoh/Desktop/unimods/is3107/project/airflow/test-62769-e009f254cc4b.json"

# ✅ Define Correct Output Directories (Inside airflow/)
# REDDIT_OUTPUT_DIR = os.path.join(AIRFLOW_DIR, "snp500-reddit")
# YF_OUTPUT_DIR = os.path.join(AIRFLOW_DIR, "snp500-yf")

# Get the absolute path of the dags/ directory
DAGS_DIR = os.path.dirname(os.path.abspath(__file__))
AIRFLOW_DIR = os.path.dirname(DAGS_DIR)  # Move up one level to airflow/

REDDIT_OUTPUT_DIR = os.path.join(AIRFLOW_DIR, "snp500-reddit-test")
YF_OUTPUT_DIR = os.path.join(AIRFLOW_DIR, "snp500-yf-test")

DOWNLOAD_DIR = os.path.join(AIRFLOW_DIR, "downloads")
PROCESSED_DIR = os.path.join(AIRFLOW_DIR, "processed")

# Define GCS Bucket
GCS_BUCKET_NAME = "my-data-is3107"
DATASET_ID = "test-62769.finance_project"
TABLE_ID = "preprocessed"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"
PICKLE_PATH = os.path.join(AIRFLOW_DIR, "ollama-dir", "yesterday_data.pkl")
SUMMARY_OUTPUT_PATH = os.path.join(AIRFLOW_DIR, "ollama-dir", "data_summaries.csv")

@dag(
    schedule="@daily",
    start_date=pendulum.today("Asia/Singapore").subtract(days=1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    description="Run Reddit and YF scrapers, then upload results to GCS",
)

def reddit_yf_scraper():

    @task()
    def get_tickers():
        return load_sp500_tickers()

    @task()
    def run_yf(tickers):
        run_yf_scraper(tickers)

    @task()
    def run_reddit(tickers):
        run_reddit_scraper(tickers)
        
    @task()
    def list_gcs_folders(bucket_name: str):
        """List all folders (prefixes) in the specified GCS bucket."""
        hook = GCSHook(gcp_conn_id="google_cloud_default")
        blobs = hook.list(bucket_name=bucket_name)
        folders = set()

        for blob in blobs:
            folder = blob.split('/')[0]  # Extract folder prefix
            folders.add(folder)

        folder_list = list(folders)
        print(f"Found folders: {folder_list}")
        return folder_list

    @task()
    def download_files_from_gcs(bucket_name: str, prefix: str, download_dir: str):
        """Download files from a specific folder in GCS to the local directory."""
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        os.makedirs(download_dir, exist_ok=True)

        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if not blob.name.endswith('/'):  # Ignore folders
                file_name = blob.name.split('/')[-1]  # Get the file name
                local_path = os.path.join(download_dir, file_name)
                print(f"📥 Downloading: {blob.name} to {local_path}")
                blob.download_to_filename(local_path)
                print(f"✅ Downloaded to: {local_path}")

    # ✅ Upload Reddit CSVs to GCS (Uses airflow/snp500-reddit)
    upload_reddit_to_gcs = LocalFilesystemToGCSOperator(
        task_id="upload_reddit_to_gcs",
        src=os.path.join(REDDIT_OUTPUT_DIR, "*.csv"),  # ✅ Updated Path
        dst="reddit-test/",  # Destination folder in GCS
        bucket=GCS_BUCKET_NAME,
        mime_type="text/csv",
        gcp_conn_id="google_cloud_default",  # Uses the configured GCS connection
    )
    
    # ✅ Upload YF CSVs to GCS (Uses airflow/snp500-yf)
    upload_yf_to_gcs = LocalFilesystemToGCSOperator(
        task_id="upload_yf_to_gcs",
        src=os.path.join(YF_OUTPUT_DIR, "*.csv"),  # ✅ Updated Path
        dst="yf-test/",
        bucket=GCS_BUCKET_NAME,
        mime_type="text/csv",
        gcp_conn_id="google_cloud_default",
    )
    
    
    @task()
    def check_files_exist(directory: str):
        """Check if any CSV files exist in the directory, otherwise skip."""
        if not any(fname.endswith(".csv") for fname in os.listdir(directory)):
            raise AirflowSkipException(f"No CSV files found in {directory}, skipping upload.")
        return True
    
    list_gcs_folders_task = list_gcs_folders(bucket_name=GCS_BUCKET_NAME)

    download_reddit_files = download_files_from_gcs(
        bucket_name=GCS_BUCKET_NAME,
        prefix="reddit-test/",
        download_dir=os.path.join(DOWNLOAD_DIR, "reddit"),
    )

    download_yf_files = download_files_from_gcs(
        bucket_name=GCS_BUCKET_NAME,
        prefix="yf-test/",
        download_dir=os.path.join(DOWNLOAD_DIR, "yf"),
    )
    
    
    # ✅ Check for files before uploading
    check_reddit_files = check_files_exist(REDDIT_OUTPUT_DIR)
    check_yf_files = check_files_exist(YF_OUTPUT_DIR)
    
    @task()
    def get_unique_tickers(download_dir: str):
        """Extract unique ticker names from the downloaded files."""
        ticker_set = set()
        date = None

        # Iterate through both Reddit and YF data directories
        data_dirs = [os.path.join(download_dir, "reddit"), os.path.join(download_dir, "yf")]

        for data_dir in data_dirs:
            for file_name in os.listdir(data_dir):
                if file_name.endswith(".csv"):
                    # Extract ticker and date from file name (e.g., AAPL_reddit_20250328.csv)
                    match = re.match(r"^([A-Z]+)_(yf|reddit)_(\d{8})\.csv$", file_name)
                    if match:
                        ticker_set.add(match.group(1))  # Add ticker name to the set
                        date = match.group(3)  # Extract date from file name

        ticker_list = sorted(ticker_set)
        print(f"✅ Unique tickers: {ticker_list}")
        print(f"✅ Date: {date}")
        return ticker_list, date

    
    def preprocess_text(text):
        # Initialize stopwords, stemmer, and lemmatizer
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove digits and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        # Apply lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back into a string
        preprocessed_text = ' '.join(tokens)
        # Remove extra whitespace
        preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text).strip()
        return preprocessed_text

   
    def analyze_sentiment(text):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)['compound'] # Range from -1 to 1

    
    def clean_post(text):
        text = text.strip()  # Remove leading/trailing spaces
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs (http, https)
        text = re.sub(r'\n+', ' ', text) # Remove line breaks
        return text
    
  
    def clean_comment(text):
        if pd.isna(text):  # Handle NaN values
            return ""
        text = text.strip()  # Remove leading/trailing spaces
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs (http, https)
        text = re.sub(r'^\s*-\s*', '', text)  # Remove "-" at the start of the comment
        text = re.sub(r'\n-\s*', '\n', text.strip())  # Remove '-' at the start of new lines
        text = re.sub(r'\n+', ' ', text) # Remove line breaks
        return text
    
    @task()
    def preprocess_and_aggregate(ticker_data):
        ticker_list, date = ticker_data
        client = bigquery.Client()
        date_obj = datetime.strptime(date, "%Y%m%d")  # Parse the date
        date_formatted = date_obj.strftime("%Y-%m-%d")  # Format the date for BigQuery
        
        for ticker in ticker_list:
        # Prepare Reddit and YF file paths
            reddit_file = os.path.join(DOWNLOAD_DIR, "reddit", f"{ticker}_reddit_{date}.csv")
            yf_file = os.path.join(DOWNLOAD_DIR, "yf", f"{ticker}_yf_{date}.csv")

            # Initialize dataframes
            df_reddit, df_yf = pd.DataFrame(), pd.DataFrame()

            # Process Reddit data if the file exists
            if os.path.exists(reddit_file):
                df_reddit = pd.read_csv(reddit_file)
                df_reddit["post"] = df_reddit["post"].astype(str).apply(clean_post)
                df_reddit["comments"] = df_reddit["comments"].astype(str).apply(clean_comment)
                df_reddit["Text"] = df_reddit["post"] + " " + df_reddit["comments"]
                df_reddit["Mentions"] = df_reddit["comments"].apply(lambda x: x.lower().count(ticker.lower()))
                df_reddit.drop(columns=["post", "comments"], inplace=True)
                df_reddit["Sentiment"] = df_reddit["Text"].apply(preprocess_text).apply(analyze_sentiment)
                df_reddit["Date"] = date_formatted
                df_reddit["Ticker"] = ticker
            else:
                df_reddit = pd.DataFrame(columns=["Text", "Mentions", "Date", "Ticker", "Sentiment"])
                
            if os.path.exists(yf_file):
                df_yf = pd.read_csv(yf_file)
                df_yf = df_yf.rename(columns={'Summary': 'Text'})
                df_yf["Mentions"] = 1
                df_yf["Date"] = date_formatted
                df_yf["Ticker"] = ticker
                mapping = {"negative": -0.2, "neutral": 0, "positive": 0.2}
                df_yf["Sentiment"] = df_yf["Sentiment"].replace(mapping)
                df_yf.drop(columns=["Title", "URL", "Published", "Publisher"], inplace=True)
            else:
                df_yf = pd.DataFrame(columns=["Text", "Mentions", "Date", "Ticker", "Sentiment"])
                
            previous_day = date_obj - timedelta(days=1)
            previous_day_str = previous_day.strftime("%Y-%m-%d")
            
            query = f"""
            SELECT Mentions
            FROM `{DATASET_ID}.{TABLE_ID}`
            WHERE Date = DATE '{previous_day_str}' AND Ticker = "{ticker}"
            """
            query_job = client.query(query)
            results = query_job.result()
            
            ystd_mentions = df_reddit["Mentions"].sum() + df_yf["Mentions"].sum()

            for row in results:
                ystd_mentions = row.Mentions  # If a result exists, assign it to ystd_mentions
                break  # Only need the first row
            
            combined_df = pd.concat([df_reddit, df_yf], ignore_index=True)
            aggregated_df = combined_df.groupby(['Date', 'Ticker'], as_index=False).agg(
                # Concatenate all Text values
                Text=('Text', ' '.join),
                # Sum all Mentions
                Mentions=('Mentions', 'sum'),
                # Calculate the average Sentiment
                Sentiment=('Sentiment', 'mean'),
                # Calculate the standard deviation of Sentiment for Sentiment_Volatility
                Sentiment_Volatility=('Sentiment', 'std')
            )
            
            # If there is only one value, let Sentiment_Volatility be 0 instead of NaN
            aggregated_df['Sentiment_Volatility'] = aggregated_df['Sentiment_Volatility'].fillna(0)

            # Round the mean and standard deviation to 3 decimal places
            aggregated_df['Sentiment'] = aggregated_df['Sentiment'].round(3)
            aggregated_df['Sentiment_Volatility'] = aggregated_df['Sentiment_Volatility'].round(3)

            # Calculating "Mentions_Change"
            aggregated_df["Mentions_Change"] = aggregated_df["Mentions"] - ystd_mentions

            # Reordering columns to match table schema
            new_order = ["Ticker", "Sentiment", "Sentiment_Volatility", "Mentions", "Mentions_Change", "Date", "Text"]

            # Reorder the DataFrame columns
            aggregated_df = aggregated_df[new_order]
            
            processed_path = os.path.join(PROCESSED_DIR, f"{ticker}_processed_{date}.csv")
            aggregated_df.to_csv(processed_path, index=False)
            print(f"✅ Processed and saved: {processed_path}")
            
    @task()
    def load_to_bigquery(ticker_data):
        ticker_list, date = ticker_data 
        """Load aggregated CSVs into BigQuery and delete local files."""
        client = bigquery.Client()

        # BigQuery configuration
        dataset_id = "test-62769.finance_project"
        table_id = "preprocessed"
        
        schema = [
            bigquery.SchemaField("Ticker", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("Sentiment", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("Sentiment_Volatility", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("Mentions", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("Mentions_Change", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("Date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("Text", "STRING", mode="REQUIRED"),
        ]

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=False,
            schema=schema
        )

        for ticker in ticker_list:
            processed_file = os.path.join(PROCESSED_DIR, f"{ticker}_processed_{date}.csv")
            
            # Check if the processed file exists before loading
            if os.path.exists(processed_file):
                try:
                    print(f"📤 Uploading {processed_file} to BigQuery...")
                    with open(processed_file, "rb") as source_file:
                        load_job = client.load_table_from_file(
                            source_file, 
                            f"{dataset_id}.{table_id}", 
                            job_config=job_config
                        )
                    load_job.result()  # Wait for the job to complete
                    print(f"✅ Successfully loaded {ticker} data into BigQuery.")
                    
                    # Remove the file after successful upload
                    # os.remove(processed_file)
                    # print(f"🗑️ Deleted processed file: {processed_file}")
                except Exception as e:
                    print(f"❌ Failed to upload {ticker} data: {str(e)}")
            else:
                print(f"⚠️ File not found: {processed_file}")

    @task
    def fetch_yesterday_data():
        # extract all of yesterday data from the aggregated table (cheng yee part)
        yesterday = (datetime.utcnow().date())

        client = bigquery.Client()

        query = f"""
            SELECT * FROM `{DATASET_ID}.{TABLE_ID}`
            WHERE Date = DATE('{yesterday}')
        """

        df = client.query(query).to_dataframe()
        df.to_pickle(PICKLE_PATH)
        
    @task()
    def generate_summaries():
        # read yesterday data
        df = pd.read_pickle(PICKLE_PATH)
        updated_rows = []

        # iterate and read yesterday records for each stock 
        for _, row in df.iterrows():
            ticker = row['Ticker']
            sentiment = row['Sentiment']
            sentiment_volatility = row['Sentiment_Volatility']
            mentions = row['Mentions']
            mention_change = row['Mentions_Change']
            date = row['Date']
            text = row['Text']



            prompt = f"""
            You are a financial sentiment analyst. Based on the following information, write a clear, investor-friendly paragraph summarizing public market sentiment around the stock {ticker}.

            Data:
            - Ticker: {ticker}
            - Sentiment Score: {sentiment}
            - Sentiment Volatility: {sentiment_volatility}
            - Mentions Today:/ {mentions}
            - Mention Change: {mention_change}
            - Date: {date}
            - Combined Reddit + Yahoo Finance Text: {text}

            Your response must be a single paragraph covering:
            - The overall sentiment (positive/neutral/negative) and why.
            - Key opportunities or risks based on the data or text.
            - Close with one sentence starting with “Investor note:” and your investment view (Buy/Hold/Sell).

            Strictly return only the paragraph, without markdown or any additional explanation.
            """
        
            # make request to ollama to summarise
            try:
                response = requests.post(OLLAMA_API_URL, json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                })
                response.raise_for_status()
                raw_response = response.json().get('response', 'No summary returned.')

                # extract the summary without think <think></think> tags
                summary = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

            except Exception as e:
                summary = f"Could not generate summary for {ticker}: {str(e)}"

            # append the summary column to the original data input
            updated_row = row.to_dict()
            updated_row['Summary'] = summary
            updated_rows.append(updated_row)

        pd.DataFrame(updated_rows).to_csv(SUMMARY_OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL,
    doublequote=True,
    escapechar="\\",)

    @task()
    def store_into_summary_table():
        client = bigquery.Client()
        table = "summary"

        # aggregated table + summary
        schema = [
            bigquery.SchemaField("Ticker", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("Sentiment", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("Sentiment_Volatility", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("Mentions", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("Mentions_Change", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("Date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("Text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("Summary", "STRING", mode="REQUIRED"),  # add summary field
        ]

        job_config = bigquery.LoadJobConfig(
            schema=schema,
            skip_leading_rows=1,
            source_format=bigquery.SourceFormat.CSV,
            autodetect=False
        )

        with open(SUMMARY_OUTPUT_PATH, "rb") as source_file:
            load_job = client.load_table_from_file(
                source_file,
                f"{DATASET_ID}.{table}",
                job_config=job_config
            )

        load_job.result()  
        print("✅ CSV successfully uploaded to BigQuery")
    
    @task()
    def cleaning():
        """
        Clear all files inside specified directories.
        """
        directories_to_clean = [
            os.path.join(DOWNLOAD_DIR, "reddit"),
            os.path.join(DOWNLOAD_DIR, "yf"),
            REDDIT_OUTPUT_DIR,
            YF_OUTPUT_DIR,
            PROCESSED_DIR,
            os.path.join(AIRFLOW_DIR, "ollama-dir")
        ]

        for dir_path in directories_to_clean:
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)  # Remove file or symlink
                            print(f"🧹 Deleted file: {file_path}")
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)  # Remove folder and its contents
                            print(f"🧹 Deleted directory: {file_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete {file_path}. Reason: {e}")
            else:
                print(f"⚠️ Directory does not exist: {dir_path}")
        
        # 🧹 Step 2: Clean GCS Buckets
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        prefixes_to_clean = ["reddit-test/", "yf-test/"]

        for prefix in prefixes_to_clean:
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                if not blob.name.endswith('/'):  # Skip folder placeholders
                    print(f"🧹 Deleting GCS file: {blob.name}")
                    blob.delete()

        print("✅ Cleaning of local files and GCS folders completed.")

            
    merge_downloads = EmptyOperator(task_id="merge_downloads")
    
    # Task Dependencies
    tickers = get_tickers()
    yf_task = run_yf(tickers)
    reddit_task = run_reddit(tickers)  
    
    unique_tickers = get_unique_tickers(DOWNLOAD_DIR)
    preprocess_task = preprocess_and_aggregate(unique_tickers)
    load_task = load_to_bigquery(unique_tickers)
    
    summary_read = fetch_yesterday_data()
    summary_gen = generate_summaries()
    summary_store = store_into_summary_table()
    cleaning_task = cleaning()

    
    (yf_task >> check_yf_files >> upload_yf_to_gcs >> list_gcs_folders_task >> download_yf_files >> merge_downloads)
    (reddit_task >> check_reddit_files >> upload_reddit_to_gcs >> list_gcs_folders_task >> download_reddit_files >> merge_downloads)

    merge_downloads >> unique_tickers >> preprocess_task >> load_task >> summary_read >> summary_gen >> summary_store >> cleaning_task
     
reddit_yf_scraper()

