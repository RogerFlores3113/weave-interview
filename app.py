import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Engineering Impact Dashboard", layout="wide")
st.title("Engineering Impact Dashboard")

# ---------------------------------------------------------------------------
# Load data from pickle files
# ---------------------------------------------------------------------------

DATA_DIR = "data"


@st.cache_data
def load_data():
    df_prs = pd.read_pickle(f"{DATA_DIR}/prs.pkl")
    df_files = pd.read_pickle(f"{DATA_DIR}/pr_files.pkl")
    df_reviews = pd.read_pickle(f"{DATA_DIR}/reviews.pkl")
    return df_prs, df_files, df_reviews


df_prs, df_files, df_reviews = load_data()

st.caption(f"Loaded {len(df_prs)} merged PRs · {len(df_files)} file records · {len(df_reviews)} reviews")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

if not df_prs.empty:
    df_prs["lines_changed"] = df_prs["additions"] + df_prs["deletions"]

    author_stats = (
        df_prs.groupby("author")
        .agg(
            num_prs=("pr_number", "count"),
            total_lines=("lines_changed", "sum"),
        )
        .reset_index()
    )

    fig = px.scatter(
        author_stats,
        x="total_lines",
        y="num_prs",
        color="author",
        hover_name="author",
        log_x=True,
        labels={
            "total_lines": "Lines Changed (log scale)",
            "num_prs": "Merged PRs",
        },
        title="PRs vs Lines Changed by Engineer",
    )
    fig.update_layout(height=500, showlegend=False)
    fig.update_traces(marker=dict(size=10))

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No PR data found. Run fetch_data.py first.")