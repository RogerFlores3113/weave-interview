import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Test: fetch repo info
resp = requests.get("https://api.github.com/repos/PostHog/posthog", headers=HEADERS)
print(f"Status: {resp.status_code}")
print(f"Rate limit remaining: {resp.headers.get('X-RateLimit-Remaining')}")
print(f"Repo stars: {resp.json().get('stargazers_count')}")