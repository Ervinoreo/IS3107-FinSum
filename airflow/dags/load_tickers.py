import pandas as pd
import os
from google.cloud import bigquery

# BASE_DIR = "/Users/ervinyeoh/Desktop/unimods/is3107/project/airflow"

#yueyaoz
BASE_DIR = "/Users/yueyaoz/Downloads/IS3107-FinSum/airflow"

SP500_CSV_PATH = os.path.join(BASE_DIR, "sp500_companies.csv")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/yueyaoz/Downloads/IS3107-FinSum/airflow/is3107zyy-7bfd94aff019.json"


# def load_sp500_tickers():
#     """Load S&P 500 tickers from CSV"""
#     try:
#         sp500_df = pd.read_csv(SP500_CSV_PATH)
#         tickers = sp500_df["Symbol"].tolist()
#         # return tickers[:1]
#         # return sp500_df["Symbol"].tolist()
#         return tickers[:100]
#     except Exception as e:
#         raise Exception(f"❌ Error loading S&P 500 tickers: {e}")

def load_sp500_tickers():
    """Load S&P 500 tickers from BigQuery table `snp500_companies`"""
    try:
        client = bigquery.Client()
        query = "SELECT Symbol FROM `is3107zyy.finance_project_crypto.snp500-companies`"
        df = client.query(query).to_dataframe()
        tickers = df["Symbol"].dropna().tolist()
        return tickers[:100]
    except Exception as e:
        raise Exception(f"❌ Error loading S&P 500 tickers from BigQuery: {e}")
    
