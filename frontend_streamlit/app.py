import streamlit as st

st.set_page_config(
    page_title="Finance Stock Sentiment Analysis",
    page_icon="📊",
)

st.write("# Welcome to the Finance Stock Sentiment Analysis App! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This app allows you to analyse stock sentiment and ticker counts based on news and posts.
    **👈 Select a page from the sidebar** to get started!
    ### Pages:
    - **Main Page**: View company info and search for specific companies.
    - **Sentiment Trends**: Visualise sentiment trends over time.
    - **Mention Counts**: Visualise mention counts over time.
"""
)