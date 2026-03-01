import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Engineering Impact Dashboard", layout="wide")
st.title("🔧 PostHog Engineering Impact Dashboard")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_DIR = "data"


@st.cache_data
def load_data():
    df_prs = pd.read_pickle(f"{DATA_DIR}/prs.pkl")
    df_files = pd.read_pickle(f"{DATA_DIR}/pr_files.pkl")
    df_reviews = pd.read_pickle(f"{DATA_DIR}/reviews.pkl")
    return df_prs, df_files, df_reviews


df_prs, df_files, df_reviews = load_data()

if df_prs.empty:
    st.warning("No PR data found. Run fetch_data.py first.")
    st.stop()

# ---------------------------------------------------------------------------
# Date filter
# ---------------------------------------------------------------------------

df_prs["merged_at"] = pd.to_datetime(df_prs["merged_at"])
df_reviews["submitted_at"] = pd.to_datetime(df_reviews["submitted_at"])

min_date = df_prs["merged_at"].min().date()
max_date = df_prs["merged_at"].max().date()

date_range = st.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="MMM DD, YYYY",
)

# Filter all dataframes to selected range
mask_prs = (df_prs["merged_at"].dt.date >= date_range[0]) & (df_prs["merged_at"].dt.date <= date_range[1])
df_prs = df_prs[mask_prs]
pr_nums_in_range = set(df_prs["pr_number"])
df_files = df_files[df_files["pr_number"].isin(pr_nums_in_range)]
df_reviews = df_reviews[df_reviews["pr_number"].isin(pr_nums_in_range)]

if df_prs.empty:
    st.warning("No PRs in selected date range.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

# -- PR-level features --
DELETION_WEIGHT = 1.5  # deletions valued higher — informed removal is high-signal
df_prs["lines_changed"] = df_prs["additions"] + df_prs["deletions"]
df_prs["weighted_lines"] = df_prs["additions"] + (df_prs["deletions"] * DELETION_WEIGHT)
df_prs["log_lines"] = np.log1p(df_prs["weighted_lines"])
df_prs["merged_week"] = df_prs["merged_at"].dt.isocalendar().week.astype(int)


# -- 1. Weighted PR volume: count of PRs, each scaled by log(lines) --
pr_volume = (
    df_prs.groupby("author")
    .agg(
        num_prs=("pr_number", "count"),
        total_lines=("lines_changed", "sum"),
        total_additions=("additions", "sum"),
        total_deletions=("deletions", "sum"),
        weighted_volume=("log_lines", "sum"),
    )
    .reset_index()
)

# -- 2. Review activity --
# Weights: CHANGES_REQUESTED=3, COMMENTED=2, APPROVED=1, other=0.5
REVIEW_WEIGHTS = {"CHANGES_REQUESTED": 3, "COMMENTED": 2, "APPROVED": 1}

if not df_reviews.empty:
    df_reviews["review_weight"] = df_reviews["state"].map(REVIEW_WEIGHTS).fillna(0.5)
    # Bonus for substantive reviews (body > 50 chars)
    df_reviews["substantive"] = (df_reviews["body_length"] > 50).astype(float)
    df_reviews["weighted_score"] = df_reviews["review_weight"] + df_reviews["substantive"]

    review_stats = (
        df_reviews.groupby("reviewer")
        .agg(
            reviews_given=("pr_number", "count"),
            review_score=("weighted_score", "sum"),
        )
        .reset_index()
        .rename(columns={"reviewer": "author"})
    )
else:
    review_stats = pd.DataFrame(columns=["author", "reviews_given", "review_score"])

# -- 3. Codebase breadth: unique top-level directories touched --
if not df_files.empty:
    df_files["top_dir"] = df_files["filename"].str.split("/").str[0]
    breadth = (
        df_files.groupby("author")["top_dir"]
        .nunique()
        .reset_index()
        .rename(columns={"top_dir": "dirs_touched"})
    )
else:
    breadth = pd.DataFrame(columns=["author", "dirs_touched"])

# -- 4. Consistency: distinct weeks with at least one merged PR --
consistency = (
    df_prs.groupby("author")["merged_week"]
    .nunique()
    .reset_index()
    .rename(columns={"merged_week": "active_weeks"})
)
total_weeks = df_prs["merged_week"].nunique()

# -- 5. Review-to-author ratio --
# (computed after merge below)

# -- 6. Bus factor: files where only one author made changes --
if not df_files.empty:
    file_authors = df_files.groupby("filename")["author"].nunique()
    sole_owned_files = set(file_authors[file_authors == 1].index)
    bus_factor = (
        df_files[df_files["filename"].isin(sole_owned_files)]
        .groupby("author")["filename"]
        .nunique()
        .reset_index()
        .rename(columns={"filename": "sole_owned_files"})
    )
else:
    bus_factor = pd.DataFrame(columns=["author", "sole_owned_files"])

# ---------------------------------------------------------------------------
# Merge into single author_stats table
# ---------------------------------------------------------------------------

author_stats = pr_volume.copy()
for df_metric in [review_stats, breadth, consistency, bus_factor]:
    author_stats = author_stats.merge(df_metric, on="author", how="left")

author_stats = author_stats.fillna(0)

# Review-to-author ratio
author_stats["review_ratio"] = (
    author_stats["reviews_given"] / author_stats["num_prs"].replace(0, np.nan)
).fillna(0)

# ---------------------------------------------------------------------------
# Normalize & composite score
# ---------------------------------------------------------------------------

SCORE_COLS = {
    "weighted_volume": "Code Output",
    "review_score": "Review Impact",
    "dirs_touched": "Codebase Breadth",
    "active_weeks": "Consistency",
    "review_ratio": "Force Multiplier",
}

for col in SCORE_COLS:
    max_val = author_stats[col].max()
    author_stats[f"{col}_norm"] = author_stats[col] / max_val if max_val > 0 else 0

WEIGHTS = {
    "weighted_volume_norm": 0.30,
    "review_score_norm": 0.25,
    "dirs_touched_norm": 0.20,
    "active_weeks_norm": 0.15,
    "review_ratio_norm": 0.10,
}

author_stats["impact_score"] = sum(
    author_stats[col] * weight for col, weight in WEIGHTS.items()
)

author_stats = author_stats.sort_values("impact_score", ascending=False).reset_index(drop=True)
top5 = author_stats.head(5)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

st.caption(
    f"Analyzing {len(df_prs)} merged PRs · {len(df_files)} file changes · "
    f"{len(df_reviews)} reviews · {author_stats['author'].nunique()} engineers"
)

# -- Top 5 headline --
st.subheader("Top 5 Most Impactful Engineers")

cols = st.columns(5)
for i, (_, row) in enumerate(top5.iterrows()):
    with cols[i]:
        st.metric(
            label=f"#{i+1}",
            value=row["author"],
            delta=f"{row['num_prs']:.0f} PRs · {row['reviews_given']:.0f} reviews",
        )

# -- Radar chart: top 5 --
st.subheader("Impact Breakdown")

norm_cols = [f"{c}_norm" for c in SCORE_COLS]
labels = list(SCORE_COLS.values())

fig_radar = go.Figure()
for _, row in top5.iterrows():
    values = [row[c] for c in norm_cols]
    values.append(values[0])  # close the polygon
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=labels + [labels[0]],
        name=row["author"],
        fill="toself",
        opacity=0.5,
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    height=450,
    margin=dict(t=30, b=30),
)
st.plotly_chart(fig_radar, width='stretch')

# -- Scoring methodology --
with st.expander("📊 How is impact measured?"):
    st.markdown(f"""
**Dimensions** (each normalized 0–1, then weighted):

| Dimension | Weight | What it captures |
|-----------|--------|-----------------|
| Code Output | 30% | Sum of `log(1 + weighted_lines)` per PR, where deletions count 1.5× — informed removal of code is high-signal |
| Review Impact | 25% | Reviews weighted by depth: changes requested (3×), comments (2×), approvals (1×), +1 for substantive body |
| Codebase Breadth | 20% | Unique top-level directories touched — cross-cutting contributions |
| Consistency | 15% | Distinct weeks with ≥1 merged PR (out of {total_weeks}) — reliable delivery |
| Force Multiplier | 10% | Review-to-PR ratio — unblocking others relative to own output |

Bus factor (sole-owned files) is shown but **not** scored it's a risk signal
""")

    st.dataframe(
        author_stats[["author", "num_prs", "total_additions", "total_deletions", "total_lines",
                       "reviews_given", "dirs_touched",
                       "active_weeks", "review_ratio", "sole_owned_files", "impact_score"]]
        .rename(columns={
            "num_prs": "PRs",
            "total_additions": "Additions",
            "total_deletions": "Deletions",
            "total_lines": "Lines Changed",
            "reviews_given": "Reviews Given",
            "dirs_touched": "Dirs Touched",
            "active_weeks": "Active Weeks",
            "review_ratio": "Review:PR Ratio",
            "sole_owned_files": "Sole-Owned Files",
            "impact_score": "Impact Score",
        })
        .style.format({
            "Additions": "{:,.0f}",
            "Deletions": "{:,.0f}",
            "Lines Changed": "{:,.0f}",
            "Reviews Given": "{:.0f}",
            "Sole-Owned Files": "{:.0f}",
            "Review:PR Ratio": "{:.2f}",
            "Impact Score": "{:.3f}",
        }),
        width='stretch',
    )

# -- Scatter: PRs vs log(lines changed) --
col1, col2 = st.columns(2)

with col1:
    st.subheader("PRs vs Code Volume")
    fig_scatter = px.scatter(
        author_stats,
        x="total_lines",
        y="num_prs",
        color="author",
        hover_name="author",
        hover_data={"weighted_volume": ":.1f", "total_lines": ":,"},
        log_x=True,
        labels={
            "total_lines": "Lines Changed (log scale)",
            "num_prs": "Merged PRs",
        },
    )
    fig_scatter.update_layout(height=400, showlegend=False)
    fig_scatter.update_traces(marker=dict(size=8))
    st.plotly_chart(fig_scatter, width='stretch')

# -- Review activity bar chart --
with col2:
    st.subheader("Review Activity (Top 15)")
    top_reviewers = author_stats.nlargest(15, "review_score")
    fig_reviews = px.bar(
        top_reviewers,
        x="review_score",
        y="author",
        orientation="h",
        color="review_ratio",
        color_continuous_scale="Viridis",
        labels={
            "review_score": "Weighted Review Score",
            "author": "",
            "review_ratio": "Review:PR Ratio",
        },
    )
    fig_reviews.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_reviews, width='stretch')

# -- Consistency + breadth --
col3, col4 = st.columns(2)

with col3:
    st.subheader(f"Shipping Consistency ({total_weeks} weeks)")
    top_consistent = author_stats.nlargest(15, "active_weeks")
    fig_consist = px.bar(
        top_consistent,
        x="active_weeks",
        y="author",
        orientation="h",
        labels={"active_weeks": "Weeks with ≥1 Merged PR", "author": ""},
    )
    fig_consist.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_consist, width='stretch')

with col4:
    st.subheader("Codebase Breadth")
    top_breadth = author_stats.nlargest(15, "dirs_touched")
    fig_breadth = px.bar(
        top_breadth,
        x="dirs_touched",
        y="author",
        orientation="h",
        color="sole_owned_files",
        color_continuous_scale="Reds",
        labels={
            "dirs_touched": "Unique Top-Level Dirs Touched",
            "author": "",
            "sole_owned_files": "Sole-Owned Files",
        },
    )
    fig_breadth.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_breadth, width='stretch')

# -- Additions vs Deletions --
st.subheader("Code Additions vs Deletions (Top 15 by volume)")
top_volume = author_stats.nlargest(15, "total_lines")
add_del_df = top_volume[["author", "total_additions", "total_deletions"]].melt(
    id_vars="author", var_name="type", value_name="lines",
)
add_del_df["type"] = add_del_df["type"].map({"total_additions": "Additions", "total_deletions": "Deletions"})
fig_addel = px.bar(
    add_del_df,
    x="lines",
    y="author",
    color="type",
    orientation="h",
    color_discrete_map={"Additions": "#2ecc71", "Deletions": "#e74c3c"},
    labels={"lines": "Lines", "author": "", "type": ""},
)
fig_addel.update_layout(height=400, yaxis=dict(autorange="reversed"), barmode="group")
st.plotly_chart(fig_addel, width='stretch')

