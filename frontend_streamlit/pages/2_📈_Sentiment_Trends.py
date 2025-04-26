import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime
import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/tianyi/Documents/y3s2/IS3107/is3107zyy-7bfd94aff019.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/yueyaoz/Downloads/IS3107-FinSum/airflow/is3107zyy-7bfd94aff019.json"
client = bigquery.Client(project="is3107zyy")

# Set page config
st.set_page_config(
    page_title="Sentiment Trends",
    page_icon="📈",
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


st.title("Sentiment Trends Over Time")

# Main page filters - in a container at the top
with st.container():
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Industry filter
        selected_industry = st.selectbox(
            "Filter by Industry",
            ["All"] + list(df1["Industry"].unique())
        )
    
    with col2:
        # Date range selector
        min_date = df2["Date"].min()
        max_date = df2["Date"].max()
        selected_dates = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    with col3:
        # Just for spacing
        st.write("")  # Empty space for alignment

# Get companies based on industry filter
if selected_industry == "All":
    available_companies = df1["Ticker"].unique()
else:
    available_companies = df1[df1["Industry"] == selected_industry]["Ticker"].unique()

# Company selection below the filters
selected_companies = st.multiselect(
    "Select Companies to Compare", 
    available_companies,
    default=[available_companies[0]] if len(available_companies) > 0 else []
)

if not selected_companies:
    st.warning("Please select at least one company.")
else:
    # Filter data based on selections
    if len(selected_dates) == 2:
        trend_data = df2[
            (df2["Ticker"].isin(selected_companies)) & 
            (df2["Date"] >= pd.to_datetime(selected_dates[0])) &
            (df2["Date"] <= pd.to_datetime(selected_dates[1]))
        ]
    else:
        trend_data = df1[df1["Ticker"].isin(selected_companies)]
    
    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Sentiment Scores Over Time", "Sentiment Volatility Over Time"),
        vertical_spacing=0.15
    )

    # Plot sentiment scores and volatility for each selected company
    for company in selected_companies:
        company_data = trend_data[trend_data["Ticker"] == company].sort_values("Date")
        
        # Sentiment Score trace (top chart)
        fig.add_trace(
            go.Scatter(
                x=company_data["Date"],
                y=company_data["Sentiment"],
                mode="lines+markers",
                name=f"{company} - Sentiment",
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Score:</b> %{y:.2f}",
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # Sentiment Volatility trace (bottom chart)
        fig.add_trace(
            go.Scatter(
                x=company_data["Date"],
                y=company_data["Sentiment_Volatility"],
                mode="lines+markers",
                name=f"{company} - Volatility",
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Volatility:</b> %{y:.4f}",
                line=dict(width=2, dash='dot')
            ),
            row=2, col=1
        )

    # Add sentiment threshold lines
    fig.add_hline(
        y=0.33,
        line_dash="dot",
        line_color="green",
        annotation_text="Positive Threshold",
        annotation_position="bottom right",
        row=1, col=1
    )
    fig.add_hline(
        y=-0.33,
        line_dash="dot",
        line_color="red",
        annotation_text="Negative Threshold",
        annotation_position="bottom right",
        row=1, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment volatility (Std Dev)", row=2, col=1)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics section
    st.subheader("Summary Statistics for Selected Period")
    
    # Calculate metrics
    stats_data = trend_data.groupby("Ticker").agg({
        "Sentiment": ["mean", "max", "min", "std"],
        "Mentions": "sum",
        "Sentiment_Volatility": "mean"
    }).reset_index()
    
    # Flatten multi-index columns
    stats_data.columns = [
        "Ticker", "Avg Sentiment", "Max Sentiment", 
        "Min Sentiment", "Sentiment Std Dev", 
        "Total Mentions", "Avg Volatility"
    ]
    
    # Format numbers
    stats_data["Avg Sentiment"] = stats_data["Avg Sentiment"].round(2)
    stats_data["Sentiment Std Dev"] = stats_data["Sentiment Std Dev"].round(3)
    stats_data["Avg Volatility"] = stats_data["Avg Volatility"].round(4)
    
    # Display stats in columns
    cols = st.columns(len(selected_companies))
    for idx, company in enumerate(selected_companies):
        #company_stats = stats_data[stats_data["Ticker"] == company].iloc[0]
        filtered_stats = stats_data[stats_data["Ticker"] == company]
        if not filtered_stats.empty:
            company_stats = filtered_stats.iloc[0]
            with cols[idx]:
                st.metric(label=f"**{company}**", 
                        value=f"Avg: {company_stats['Avg Sentiment']:.2f}",
                        delta=f"Range: {company_stats['Min Sentiment']:.2f} to {company_stats['Max Sentiment']:.2f}")
                st.write(f"📊 Mentions: {company_stats['Total Mentions']}")
                st.write(f"📈 Volatility: {company_stats['Avg Volatility']:.4f}")
                st.write(f"σ: {company_stats['Sentiment Std Dev']:.3f}")
        else:
            with cols[idx]:
                st.warning(f"No data for {company} in selected range")