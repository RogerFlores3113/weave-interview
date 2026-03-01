"""
Fetch PR data from PostHog/posthog GitHub repo.
Pulls merged PRs, their file changes, and reviews.
Saves to pickle files in data/ for local development.

Usage:
    python fetch_data.py              # last 2 days (testing)
    python fetch_data.py --days 90    # last 90 days (production)
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
REPO = "PostHog/posthog"
BASE_URL = f"https://api.github.com/repos/{REPO}"

BOT_AUTHORS = {"github-actions[bot]", "posthog-bot", "claude[bot]", "dependabot[bot]"}

DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET with rate-limit awareness."""
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset - int(time.time()), 5)
        print(f"  ⏳ Rate limited. Sleeping {wait}s ...")
        time.sleep(wait)
        resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp


def _paginate(url: str, params: dict | None = None) -> list[dict]:
    """Fetch all pages from a paginated GitHub endpoint."""
    params = dict(params or {})
    params.setdefault("per_page", 100)
    results = []
    page = 1
    while True:
        params["page"] = page
        resp = _get(url, params)
        batch = resp.json()
        if not batch:
            break
        results.extend(batch)
        # If we got fewer than per_page, we're on the last page
        if len(batch) < params["per_page"]:
            break
        page += 1
    return results


def _check_rate_limit():
    """Print current rate limit status."""
    resp = _get("https://api.github.com/rate_limit")
    core = resp.json()["resources"]["core"]
    print(f"  API calls remaining: {core['remaining']}/{core['limit']}")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_merged_prs(since: datetime) -> list[dict]:
    """Fetch all merged PRs since `since` using the search API."""
    since_str = since.strftime("%Y-%m-%d")
    query = f"repo:{REPO} is:pr is:merged merged:>={since_str}"

    print(f"Searching merged PRs since {since_str} ...")
    results = []
    page = 1
    while True:
        resp = _get(
            "https://api.github.com/search/issues",
            params={"q": query, "per_page": 100, "page": page, "sort": "updated"},
        )
        data = resp.json()
        items = data.get("items", [])
        results.extend(items)
        total = data.get("total_count", 0)
        print(f"  Page {page}: got {len(items)} items ({len(results)}/{total} total)")
        if len(results) >= total or not items:
            break
        page += 1

    return results


def enrich_pr(pr_number: int) -> dict:
    """Fetch full PR details (additions/deletions/merge info)."""
    resp = _get(f"{BASE_URL}/pulls/{pr_number}")
    return resp.json()


def fetch_pr_files(pr_number: int) -> list[dict]:
    """Fetch files changed in a PR."""
    return _paginate(f"{BASE_URL}/pulls/{pr_number}/files")


def fetch_pr_reviews(pr_number: int) -> list[dict]:
    """Fetch reviews on a PR."""
    return _paginate(f"{BASE_URL}/pulls/{pr_number}/reviews")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(days: int = 2):
    os.makedirs(DATA_DIR, exist_ok=True)
    since = datetime.now(timezone.utc) - timedelta(days=days)

    _check_rate_limit()

    # 1. Get merged PRs via search
    raw_prs = fetch_merged_prs(since)
    print(f"\nFound {len(raw_prs)} merged PRs total.")

    # Extract PR numbers and filter bots early
    pr_numbers = []
    for item in raw_prs:
        author = item["user"]["login"]
        if author in BOT_AUTHORS:
            continue
        pr_numbers.append(item["number"])

    print(f"After filtering bots: {len(pr_numbers)} PRs to enrich.\n")

    # 2. Enrich each PR with full details, files, and reviews
    pr_rows = []
    file_rows = []
    review_rows = []

    for i, pr_num in enumerate(pr_numbers):
        print(f"  [{i+1}/{len(pr_numbers)}] Enriching PR #{pr_num} ...", end=" ")

        # Full PR data (additions, deletions, merge date, etc.)
        pr_data = enrich_pr(pr_num)
        author = pr_data["user"]["login"]
        pr_rows.append({
            "pr_number": pr_num,
            "title": pr_data["title"],
            "author": author,
            "created_at": pr_data["created_at"],
            "merged_at": pr_data["merged_at"],
            "additions": pr_data["additions"],
            "deletions": pr_data["deletions"],
            "changed_files": pr_data["changed_files"],
            "body": (pr_data.get("body") or "")[:500],  # truncate
        })

        # Files changed
        files = fetch_pr_files(pr_num)
        for f in files:
            file_rows.append({
                "pr_number": pr_num,
                "author": author,
                "filename": f["filename"],
                "additions": f["additions"],
                "deletions": f["deletions"],
                "changes": f["changes"],
                "status": f["status"],  # added/removed/modified/renamed
            })

        # Reviews
        reviews = fetch_pr_reviews(pr_num)
        for r in reviews:
            reviewer = r["user"]["login"] if r["user"] else "unknown"
            if reviewer in BOT_AUTHORS:
                continue
            review_rows.append({
                "pr_number": pr_num,
                "pr_author": author,
                "reviewer": reviewer,
                "state": r["state"],  # APPROVED, CHANGES_REQUESTED, COMMENTED, etc.
                "submitted_at": r["submitted_at"],
                "body_length": len(r.get("body") or ""),
            })

        print(f"({len(files)} files, {len(reviews)} reviews)")

    # 3. Save to pickle
    df_prs = pd.DataFrame(pr_rows)
    df_files = pd.DataFrame(file_rows)
    df_reviews = pd.DataFrame(review_rows)

    df_prs.to_pickle(f"{DATA_DIR}/prs.pkl")
    df_files.to_pickle(f"{DATA_DIR}/pr_files.pkl")
    df_reviews.to_pickle(f"{DATA_DIR}/reviews.pkl")

    print(f"\n✅ Saved to {DATA_DIR}/:")
    print(f"   prs.pkl       — {len(df_prs)} PRs")
    print(f"   pr_files.pkl  — {len(df_files)} file records")
    print(f"   reviews.pkl   — {len(df_reviews)} review records")

    _check_rate_limit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch GitHub PR data")
    parser.add_argument("--days", type=int, default=2, help="Number of days to look back (default: 2)")
    args = parser.parse_args()
    run(days=args.days)