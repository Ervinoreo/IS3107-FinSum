import streamlit as st
import pandas as pd
from streamlit.components.v1 import html
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime
import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/tianyi/Documents/y3s2/IS3107/is3107zyy-7bfd94aff019.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/yueyaoz/Downloads/IS3107-FinSum/airflow/is3107zyy-7bfd94aff019.json"
client = bigquery.Client(project="is3107zyy")

# Set page config
st.set_page_config(
    page_title="Main Page",
    page_icon="🏠",
    layout="wide"
)

# Post request time interval: run this query once every 24 hours (86400 seconds)
@st.cache_data(ttl=86400)
def load_data():
    query1 = """
        SELECT *
        FROM `is3107zyy.finance_project_crypto.Company`
    """
    query2 = """
        SELECT * FROM `is3107zyy.finance_project_crypto.discussion`
        WHERE Date IS NOT NULL
    """
    df1 = client.query(query1).to_dataframe()
    df2 = client.query(query2).to_dataframe()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df1["last_fetched"] = now
    df2["last_fetched"] = now
    return df1, df2

df1, df2 = load_data()

# Convert Date column to datetime format
df2["Date"] = pd.to_datetime(df2["Date"])
industry_map = df1[["Ticker", "Industry"]].drop_duplicates()
df2 = df2.merge(industry_map, left_on="Ticker", right_on="Ticker", how="left")

# Load custom CSS from styles.css
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Finance Stock Sentiment Analysis App")
st.write("Predicting stock prices using news and posts sentiment information.")

# Get unique industries from data
all_industries = sorted(df1["Industry"].unique())
default_industries = []

# Initialize session state for industries to display
if 'selected_industries' not in st.session_state:
    st.session_state.selected_industries = default_industries

# Industry filter in sidebar
st.header("Industry Filter")
selected_industries = st.multiselect(
    "Select industries to display:",
    options=all_industries,
    default=st.session_state.selected_industries,
    key="industry_filter"
)
    
# Update session state when filter changes
if st.button("Apply Filter"):
    st.session_state.selected_industries = selected_industries if selected_industries else default_industries
    st.rerun()

# Get the most recent date
most_recent_date = df2["Date"].max()
recent_day_data = df2[df2["Date"] == most_recent_date]

# Create columns for selected industries
num_columns = len(st.session_state.selected_industries)
columns = st.columns(num_columns if num_columns > 0 else 1)  # Ensure at least 1 column

def create_company_block(ticker, sentiment, score, mentions, summary):
    # Create a container for the clickable block
    block_container = st.container(border=True)
    
    with block_container:
        # Show the basic info
        st.markdown(f"""
            <div class="company-block {sentiment}">
                {ticker} - {sentiment.capitalize()}<br>
                Sentiment Score: {score}<br>
                Mentions: {mentions}
            </div>
        """, unsafe_allow_html=True)
        
        # Create the popover
        with st.popover("View details"):
            st.subheader(ticker)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", sentiment.capitalize())
            with col2:
                st.metric("Score", score)
            
            st.write("**Summary:**")
            st.write(summary)
            st.write(f"**Mentions:** {mentions}")

def display_searched_company(ticker, sentiment, score, mentions, summary):
    with st.container(border=True):
        st.subheader(ticker)
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment", sentiment.capitalize())
        with col2:
            st.metric("Score", f"{score:.2f}" if isinstance(score, float) else score)
        with col3:
            st.metric("Mentions", mentions)
        
        # Summary section
        st.divider()
        st.write("**Summary:**")
        st.write(summary)

# Display data for each selected industry
for i, industry in enumerate(st.session_state.selected_industries):
    with columns[i]:
        st.subheader(industry)
        industry_df = recent_day_data[recent_day_data["Industry"] == industry].sort_values("Date", ascending=False)
        
        # Get unique companies
        unique_companies = industry_df.groupby("Ticker").first().reset_index()
        sample_size = min(3, len(unique_companies))
        
        if sample_size > 0:
            # Store sampled companies in session state to maintain consistency
            if f"{industry}_sample" not in st.session_state:
                st.session_state[f"{industry}_sample"] = unique_companies.sample(n=sample_size)
            
            for _, row in st.session_state[f"{industry}_sample"].iterrows():
                sentiment_value = row["Sentiment"]
                if isinstance(sentiment_value, float):
                    sentiment_label = "Positive" if sentiment_value > 0.33 else ("Negative" if sentiment_value < -0.33 else "Neutral")
                    sentiment = sentiment_label.lower()
                else:
                    sentiment = str(sentiment_value).lower()
                
                create_company_block(
                ticker=row["Ticker"],
                sentiment=sentiment,
                score=f"{row['Sentiment']:.2f}" if isinstance(row['Sentiment'], float) else row['Sentiment'],
                mentions=row["Mentions"],
                summary=row["Summary"]
            )
        else:
            st.write("No companies available")


# Add a search function with type-ahead using only selectbox
st.write("---")
st.subheader("Search for a Company")

# Get a list of all unique companies
all_companies = df1["Ticker"].unique()

# Allow the user to type and filter companies directly in the selectbox
selected_company = st.selectbox(
    "Start typing to search for a company:", 
    all_companies, 
    index=None,  # No default selection
    placeholder="Type to search..."
)

# Display the selected company's information
if selected_company:
    company_data = recent_day_data[recent_day_data["Ticker"] == selected_company].groupby("Ticker").first().reset_index()
    if not company_data.empty:
        row = company_data.iloc[0]
        sentiment_value = row["Sentiment"]
        if isinstance(sentiment_value, float):
            sentiment_label = "Positive" if sentiment_value > 0.33 else ("Negative" if sentiment_value < -0.33 else "Neutral")
            sentiment = sentiment_label.lower()
        else:
            sentiment = str(sentiment_value).lower()
        
        display_searched_company(
            ticker=row["Ticker"],
            sentiment=sentiment,
            score=row["Sentiment"],
            mentions=row["Mentions"],
            summary=row["Summary"]
        )
    else:
        st.warning(f"No data found for the company '{selected_company}' on the most recent day.")

# Display the top 10 tickers with the highest counts for the most recent day
st.write("---")
st.subheader("Daily Top 10 Tickers by Count")

# Get the most recent date
most_recent_date = df2["Date"].max()
# Filter the data for the most recent date
recent_day_data = df2[df2["Date"] == most_recent_date]
# Group by Company and sum the Ticker Count for the most recent day
recent_day_ticker_counts = recent_day_data.groupby("Ticker")["Mentions"].sum().reset_index()
# Sort by Ticker Count (descending) and get the top 10
top_10_tickers = recent_day_ticker_counts.sort_values(by="Mentions", ascending=False).head(10)

if not top_10_tickers.empty:
    st.caption(f"Date: {most_recent_date.strftime('%Y-%m-%d')}")
    
    # Row 1: Positions 1-5
    cols = st.columns(5)
    for idx, row in enumerate(top_10_tickers.iloc[:5].itertuples()):
        with cols[idx]:
            st.metric(
                label=row.Ticker,
                value=row.Mentions,
                help=f"Rank {idx+1} | {most_recent_date.strftime('%Y-%m-%d')}"
            )
    
    # Row 2: Positions 6-10 
    cols = st.columns(5)
    for idx, row in enumerate(top_10_tickers.iloc[5:10].itertuples()):
        with cols[idx]:
            st.metric(
                label=row.Ticker,
                value=row.Mentions,
                help=f"Rank {idx+6} | {most_recent_date.strftime('%Y-%m-%d')}"
            )
else:
    st.warning("No data found.")

# Display the top 10 tickers with the highest sentiment scores for the most recent day
st.write("---")
st.subheader("Daily Top 10 Tickers by Sentiment Score")

# Filter the data for the most recent date
recent_day_data = df2[df2["Date"] == most_recent_date]

# Group by Company and average the Sentiment Score for the most recent day
recent_day_ticker_ss = recent_day_data.groupby("Ticker")["Sentiment"].mean().reset_index()

# Sort by Sentiment Score (descending) and get the top 10
top_10_tickers = recent_day_ticker_ss.sort_values(by="Sentiment", ascending=False).head(10)

# Added columns layout for better space utilization
if not top_10_tickers.empty:
    st.caption(f"Date: {most_recent_date.strftime('%Y-%m-%d')}")
    
    # Row 1: Positions 1-5
    cols = st.columns(5)
    for idx, row in enumerate(top_10_tickers.iloc[:5].itertuples()):
        with cols[idx]:
            st.metric(
                label=row.Ticker,
                value=f"{row.Sentiment:.2f}",
                delta="Positive" if row.Sentiment > 0.33 else 
                     ("Negative" if row.Sentiment < -0.33 else "Neutral"),
                help=f"Rank {idx+1} | {most_recent_date.strftime('%Y-%m-%d')}"
            )
    
    # Row 2: Positions 6-10
    cols = st.columns(5)
    for idx, row in enumerate(top_10_tickers.iloc[5:10].itertuples()):
        with cols[idx]:
            st.metric(
                label=row.Ticker,
                value=f"{row.Sentiment:.2f}",
                delta="Positive" if row.Sentiment > 0.33 else 
                     ("Negative" if row.Sentiment < -0.33 else "Neutral"),
                help=f"Rank {idx+6} | {most_recent_date.strftime('%Y-%m-%d')}"
            )
else:
    st.warning("No data found.")