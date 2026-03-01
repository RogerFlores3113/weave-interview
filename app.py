import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Engineering Impact Dashboard", layout="wide")
st.title("Engineering Impact Dashboard")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    TOKEN = st.secrets["GITHUB_TOKEN"]
except:
    TOKEN = os.getenv("GITHUB_TOKEN")

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
REPO = "PostHog/posthog"
BASE_URL = f"https://api.github.com/repos/{REPO}"
BOT_AUTHORS = {"github-actions[bot]", "posthog-bot", "claude[bot]", "dependabot[bot]"}
DAYS = 90


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> requests.Response | None:
    """GET with rate-limit awareness. Returns None on rate limit exhaustion."""
    resp = requests.get(url, headers=HEADERS, params=params)
    
    if resp.status_code != 200:
        st.warning(f"GitHub API returned {resp.status_code}: {resp.text[:200]}")
        return None

    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", 0))
        wait = reset - int(time.time())
        if wait > 60:
            # Don't block the app for minutes — fail soft
            return None
        time.sleep(max(wait, 1))
        resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        return None
    return resp


def _paginate(url: str, params: dict | None = None) -> list[dict]:
    params = dict(params or {})
    params.setdefault("per_page", 100)
    results = []
    page = 1
    while True:
        params["page"] = page
        resp = _get(url, params)
        if resp is None:
            break
        batch = resp.json()
        if not batch:
            break
        results.extend(batch)
        if len(batch) < params["per_page"]:
            break
        page += 1
    return results


# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching PRs from GitHub...")
def load_data(days: int = DAYS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_str = since.strftime("%Y-%m-%d")
    query = f"repo:{REPO} is:pr is:merged merged:>={since_str}"

    # 1. Search for merged PRs
    raw_prs = []
    page = 1
    while True:
        resp = _get(
            "https://api.github.com/search/issues",
            params={"q": query, "per_page": 100, "page": page, "sort": "updated"},
        )
        if resp is None:
            break
        data = resp.json()
        items = data.get("items", [])
        raw_prs.extend(items)
        if len(raw_prs) >= data.get("total_count", 0) or not items:
            break
        page += 1

    # Build lookup, filter bots
    search_lookup = {}
    for item in raw_prs:
        author = item["user"]["login"]
        if author in BOT_AUTHORS:
            continue
        search_lookup[item["number"]] = {
            "title": item["title"],
            "author": author,
            "created_at": item["created_at"],
            "merged_at": item.get("pull_request", {}).get("merged_at"),
            "body": (item.get("body") or "")[:500],
        }

    # 2. Enrich with files + reviews
    pr_rows = []
    file_rows = []
    review_rows = []
    progress = st.progress(0, text="Enriching PRs...")
    total = len(search_lookup)

    for i, (pr_num, meta) in enumerate(search_lookup.items()):
        author = meta["author"]
        progress.progress((i + 1) / total, text=f"Enriching PR #{pr_num} ({i+1}/{total})")

        # Files
        files = _paginate(f"{BASE_URL}/pulls/{pr_num}/files")
        total_add = sum(f["additions"] for f in files)
        total_del = sum(f["deletions"] for f in files)

        pr_rows.append({
            "pr_number": pr_num,
            "title": meta["title"],
            "author": author,
            "created_at": meta["created_at"],
            "merged_at": meta["merged_at"],
            "additions": total_add,
            "deletions": total_del,
            "changed_files": len(files),
            "body": meta["body"],
        })

        for f in files:
            file_rows.append({
                "pr_number": pr_num,
                "author": author,
                "filename": f["filename"],
                "additions": f["additions"],
                "deletions": f["deletions"],
                "changes": f["changes"],
                "status": f["status"],
            })

        # Reviews
        reviews = _paginate(f"{BASE_URL}/pulls/{pr_num}/reviews")
        for r in reviews:
            reviewer = r["user"]["login"] if r["user"] else "unknown"
            if reviewer in BOT_AUTHORS:
                continue
            review_rows.append({
                "pr_number": pr_num,
                "pr_author": author,
                "reviewer": reviewer,
                "state": r["state"],
                "submitted_at": r["submitted_at"],
                "body_length": len(r.get("body") or ""),
            })

        # If we hit rate limits mid-enrichment, stop and use what we have
        rl_resp = requests.get("https://api.github.com/rate_limit", headers=HEADERS)
        remaining = rl_resp.json().get("resources", {}).get("core", {}).get("remaining", 999)
        if remaining < 20:
            st.warning(f"⚠️ Rate limit low ({remaining} remaining). Using {i+1}/{total} PRs.")
            break

    progress.empty()

    df_prs = pd.DataFrame(pr_rows)
    df_files = pd.DataFrame(file_rows)
    df_reviews = pd.DataFrame(review_rows)

    return df_prs, df_files, df_reviews


# ---------------------------------------------------------------------------
# Load & display
# ---------------------------------------------------------------------------

df_prs, df_files, df_reviews = load_data()

st.caption(f"Loaded {len(df_prs)} merged PRs from the last {DAYS} days")

# Compute per-author stats
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
    st.warning("No PR data loaded.")