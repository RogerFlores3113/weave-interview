"""
Fetch PR data from PostHog/posthog GitHub repo.
Pulls merged PRs, their file changes, and reviews.
Upserts into pickle files in data/.

Usage:
    python fetch_data.py                                # last 2 days (testing)
    python fetch_data.py --from-days 90 --to-days 60   # window: 90 to 60 days ago
    python fetch_data.py --from-days 60 --to-days 30   # window: 60 to 30 days ago
    python fetch_data.py --from-days 30 --to-days 0    # window: 30 days ago to now
    python fetch_data.py --from-days 90                 # 90 days ago to now (full)
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Connection-pooled session (thread-safe, reuses TCP connections)
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# Thread-safe rate limit tracker (read from response headers — no extra API calls)
_rate_lock = threading.Lock()
_rate_remaining = 5000


def _update_rate(resp: requests.Response):
    global _rate_remaining
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        with _rate_lock:
            _rate_remaining = int(remaining)


def _get_rate_remaining() -> int:
    with _rate_lock:
        return _rate_remaining


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET with rate-limit awareness."""
    resp = SESSION.get(url, params=params)
    _update_rate(resp)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset - int(time.time()), 5)
        print(f"  ⏳ Rate limited. Sleeping {wait}s ...")
        time.sleep(wait)
        resp = SESSION.get(url, params=params)
        _update_rate(resp)
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
        if len(batch) < params["per_page"]:
            break
        page += 1
    return results


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_merged_prs(since_str: str, until_str: str) -> list[dict]:
    """Fetch all merged PRs in date range using the search API."""
    query = f"repo:{REPO} is:pr is:merged merged:{since_str}..{until_str}"

    print(f"Searching merged PRs from {since_str} to {until_str} ...")
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


def fetch_pr_files(pr_number: int) -> list[dict]:
    return _paginate(f"{BASE_URL}/pulls/{pr_number}/files")


def fetch_pr_reviews(pr_number: int) -> list[dict]:
    return _paginate(f"{BASE_URL}/pulls/{pr_number}/reviews")


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def _load_existing_pkl(filename: str) -> pd.DataFrame:
    """Load an existing pickle or return empty DataFrame."""
    path = f"{DATA_DIR}/{filename}"
    if os.path.exists(path):
        df = pd.read_pickle(path)
        print(f"  Loaded {len(df)} existing rows from {filename}")
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(from_days: int = 2, to_days: int = 0):
    os.makedirs(DATA_DIR, exist_ok=True)

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=from_days)
    until = now - timedelta(days=to_days)
    since_str = since.strftime("%Y-%m-%d")
    until_str = until.strftime("%Y-%m-%d")

    print(f"Rate limit remaining: {_get_rate_remaining()}")

    # Load existing data to skip already-fetched PRs
    existing_prs = _load_existing_pkl("prs.pkl")
    existing_pr_nums = set(existing_prs["pr_number"].tolist()) if not existing_prs.empty else set()

    # 1. Get merged PRs via search
    raw_prs = fetch_merged_prs(since_str, until_str)
    print(f"\nFound {len(raw_prs)} merged PRs in window.")

    # Build lookup, filter bots + already-fetched
    search_lookup = {}
    skipped_bots = 0
    skipped_existing = 0
    for item in raw_prs:
        author = item["user"]["login"]
        if author in BOT_AUTHORS:
            skipped_bots += 1
            continue
        if item["number"] in existing_pr_nums:
            skipped_existing += 1
            continue
        search_lookup[item["number"]] = {
            "title": item["title"],
            "author": author,
            "created_at": item["created_at"],
            "merged_at": item.get("pull_request", {}).get("merged_at"),
            "body": (item.get("body") or "")[:500],
        }

    pr_numbers = list(search_lookup.keys())
    print(f"  Filtered: {skipped_bots} bots, {skipped_existing} already fetched")
    print(f"  New PRs to enrich: {len(pr_numbers)}")
    print(f"  Estimated API calls: ~{len(pr_numbers) * 2}")
    print(f"  Rate limit remaining: {_get_rate_remaining()}\n")

    if not pr_numbers:
        print("Nothing new to fetch!")
        return

    # 2. Enrich concurrently
    pr_rows = []
    file_rows = []
    review_rows = []

    def _enrich_pr(pr_num: int) -> dict:
        meta = search_lookup[pr_num]
        author = meta["author"]
        files = fetch_pr_files(pr_num)
        reviews = fetch_pr_reviews(pr_num)
        return {
            "pr_num": pr_num,
            "meta": meta,
            "author": author,
            "files": files,
            "reviews": reviews,
            "total_additions": sum(f["additions"] for f in files),
            "total_deletions": sum(f["deletions"] for f in files),
        }

    print(f"Enriching {len(pr_numbers)} PRs (50 concurrent)...")
    with ThreadPoolExecutor(max_workers=50) as pool:
        futures = {pool.submit(_enrich_pr, n): n for n in pr_numbers}
        done = 0
        for future in as_completed(futures):
            done += 1
            pr_num = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"  ⚠️ PR #{pr_num} failed: {e}")
                continue

            meta = result["meta"]
            author = result["author"]

            pr_rows.append({
                "pr_number": pr_num,
                "title": meta["title"],
                "author": author,
                "created_at": meta["created_at"],
                "merged_at": meta["merged_at"],
                "additions": result["total_additions"],
                "deletions": result["total_deletions"],
                "changed_files": len(result["files"]),
                "body": meta["body"],
            })

            for f in result["files"]:
                file_rows.append({
                    "pr_number": pr_num,
                    "author": author,
                    "filename": f["filename"],
                    "additions": f["additions"],
                    "deletions": f["deletions"],
                    "changes": f["changes"],
                    "status": f["status"],
                })

            for r in result["reviews"]:
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

            if done % 50 == 0 or done == len(pr_numbers):
                print(f"  [{done}/{len(pr_numbers)}] enriched (rate limit: {_get_rate_remaining()})")

    # 3. Upsert into pickles
    df_new_prs = pd.DataFrame(pr_rows)
    df_new_files = pd.DataFrame(file_rows)
    df_new_reviews = pd.DataFrame(review_rows)

    existing_files = _load_existing_pkl("pr_files.pkl")
    existing_reviews = _load_existing_pkl("reviews.pkl")

    # PRs: dedupe on pr_number
    df_prs = pd.concat([existing_prs, df_new_prs], ignore_index=True)
    if not df_prs.empty:
        df_prs = df_prs.drop_duplicates(subset=["pr_number"], keep="last")

    # Files: dedupe on (pr_number, filename)
    df_files = pd.concat([existing_files, df_new_files], ignore_index=True)
    if not df_files.empty:
        df_files = df_files.drop_duplicates(subset=["pr_number", "filename"], keep="last")

    # Reviews: dedupe on (pr_number, reviewer, submitted_at)
    df_reviews = pd.concat([existing_reviews, df_new_reviews], ignore_index=True)
    if not df_reviews.empty:
        df_reviews = df_reviews.drop_duplicates(subset=["pr_number", "reviewer", "submitted_at"], keep="last")

    df_prs.to_pickle(f"{DATA_DIR}/prs.pkl")
    df_files.to_pickle(f"{DATA_DIR}/pr_files.pkl")
    df_reviews.to_pickle(f"{DATA_DIR}/reviews.pkl")

    print(f"\n✅ Saved to {DATA_DIR}/:")
    print(f"   prs.pkl       — {len(df_prs)} PRs ({len(df_new_prs)} new)")
    print(f"   pr_files.pkl  — {len(df_files)} file records ({len(df_new_files)} new)")
    print(f"   reviews.pkl   — {len(df_reviews)} review records ({len(df_new_reviews)} new)")
    print(f"   Rate limit remaining: {_get_rate_remaining()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch GitHub PR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_data.py                                  # last 2 days (test)
  python fetch_data.py --from-days 90 --to-days 60     # 90-60 days ago
  python fetch_data.py --from-days 60 --to-days 30     # 60-30 days ago
  python fetch_data.py --from-days 30                   # 30 days ago to now
        """,
    )
    parser.add_argument("--from-days", type=int, default=2, help="Start of window (days ago, default: 2)")
    parser.add_argument("--to-days", type=int, default=0, help="End of window (days ago, default: 0 = now)")
    args = parser.parse_args()
    run(from_days=args.from_days, to_days=args.to_days)