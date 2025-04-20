import pandas as pd
import os

# BASE_DIR = "/Users/ervinyeoh/Desktop/unimods/is3107/project/airflow"

#yueyaoz
BASE_DIR = "/Users/yueyaoz/Downloads/IS3107-FinSum/airflow"

SP500_CSV_PATH = os.path.join(BASE_DIR, "sp500_companies.csv")

def load_sp500_tickers():
    """Load S&P 500 tickers from CSV"""
    try:
        sp500_df = pd.read_csv(SP500_CSV_PATH)
        tickers = sp500_df["Symbol"].tolist()
        # return tickers[:1]
        # return sp500_df["Symbol"].tolist()
        return tickers[:100]
    except Exception as e:
        raise Exception(f"❌ Error loading S&P 500 tickers: {e}")
    
