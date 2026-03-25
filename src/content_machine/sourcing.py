"""Stage 1: obtain raw potential content from Reddit and X."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request

from .config import load_json
from .http_utils import request_json_with_retries
from .models import RawPost

logger = logging.getLogger(__name__)


_REDDIT_URL = "https://www.reddit.com/r/{subreddit}/top.json?t=day&limit=50"
_X_SEARCH_URLS = (
    "https://api.x.com/2/tweets/search/recent",
    "https://api.twitter.com/2/tweets/search/recent",
)


def _parse_datetime(value: str) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)

    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")

    return datetime.fromisoformat(value)


def _is_within_lookback(created_at: datetime, lookback_hours: int) -> bool:
    threshold = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)
    return created_at >= threshold


def _request_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(url=url, headers=headers or {})
    return request_json_with_retries(request, operation=f"GET {url}", timeout=20, max_attempts=3)


def fetch_reddit_posts() -> list[RawPost]:
    """Fetch and normalize Reddit posts according to config thresholds."""

    cfg = load_json("config/sources.json").get("reddit", {})
    if not cfg.get("enabled", False):
        return []

    min_score = int(cfg.get("min_score", 0))
    min_comments = int(cfg.get("min_comments", 0))
    lookback_hours = int(cfg.get("lookback_hours", 72))
    subreddits = cfg.get("subreddits", [])

    normalized: list[RawPost] = []

    for subreddit in subreddits:
        try:
            payload = _request_json(
                _REDDIT_URL.format(subreddit=subreddit),
                headers={"User-Agent": "content-machine/0.1"},
            )
        except Exception:
            continue

        children = payload.get("data", {}).get("children", [])
        for item in children:
            data = item.get("data", {})

            score = int(data.get("score", 0))
            comments = int(data.get("num_comments", 0))
            created_utc = float(data.get("created_utc", 0))
            created_at = datetime.fromtimestamp(created_utc, tz=timezone.utc)
            if score < min_score or comments < min_comments:
                continue
            if not _is_within_lookback(created_at, lookback_hours):
                continue

            title = (data.get("title") or "").strip()
            body = (data.get("selftext") or "").strip()
            combined_text = "\n\n".join(part for part in (title, body) if part).strip()
            if not combined_text:
                continue

            has_media = bool(
                data.get("is_video")
                or data.get("is_gallery")
                or data.get("post_hint") in {"image", "hosted:video", "rich:video"}
            )

            normalized.append(
                RawPost(
                    source="reddit",
                    source_id=str(data.get("id", "")),
                    author=data.get("author", ""),
                    text=combined_text,
                    metrics={
                        "score": score,
                        "comments": comments,
                        "upvote_ratio": float(data.get("upvote_ratio", 0.0)),
                        "subreddit": subreddit,
                        "permalink": f"https://reddit.com{data.get('permalink', '')}",
                        "has_media": has_media,
                    },
                    created_at=created_at.isoformat(),
                )
            )

    return normalized


def fetch_x_posts() -> list[RawPost]:
    """Fetch and normalize X posts according to config thresholds."""

    cfg = load_json("config/sources.json").get("x", {})
    if not cfg.get("enabled", False):
        return []

    bearer_token = os.getenv("X_BEARER_TOKEN", "").strip()
    if not bearer_token:
        logger.warning("Skipping X sourcing because X_BEARER_TOKEN is not set.")
        return []

    min_likes = int(cfg.get("min_likes", 0))
    min_replies = int(cfg.get("min_replies", 0))
    lookback_hours = int(cfg.get("lookback_hours", 72))
    accounts = cfg.get("accounts", [])

    normalized: list[RawPost] = []

    for account in accounts:
        handle = account.lstrip("@").strip()
        if not handle:
            continue

        query = quote_plus(f"from:{handle} -is:retweet -is:reply lang:en")
        url_suffix = (
            f"?query={query}&max_results=50"
            "&tweet.fields=created_at,public_metrics,entities,attachments"
            "&expansions=author_id&user.fields=username"
        )

        payload: dict[str, Any] | None = None
        for base_url in _X_SEARCH_URLS:
            try:
                payload = _request_json(
                    f"{base_url}{url_suffix}",
                    headers={"Authorization": f"Bearer {bearer_token}"},
                )
                break
            except Exception:
                payload = None

        if not payload:
            logger.error("Unable to fetch X posts for @%s after trying all configured API endpoints.", handle)
            continue

        users = {
            user.get("id", ""): user.get("username", "")
            for user in payload.get("includes", {}).get("users", [])
        }

        for tweet in payload.get("data", []):
            metrics = tweet.get("public_metrics", {})
            likes = int(metrics.get("like_count", 0))
            replies = int(metrics.get("reply_count", 0))
            if likes < min_likes or replies < min_replies:
                continue

            created_at_text = tweet.get("created_at", "")
            created_at = _parse_datetime(created_at_text)
            if not _is_within_lookback(created_at, lookback_hours):
                continue

            text = (tweet.get("text") or "").strip()
            if not text:
                continue

            entity_urls = tweet.get("entities", {}).get("urls", [])
            has_media = bool(tweet.get("attachments", {}).get("media_keys"))
            has_external_url = bool(entity_urls)

            author_id = tweet.get("author_id", "")
            author = users.get(author_id, handle)

            normalized.append(
                RawPost(
                    source="x",
                    source_id=str(tweet.get("id", "")),
                    author=author,
                    text=text,
                    metrics={
                        "likes": likes,
                        "replies": replies,
                        "reposts": int(metrics.get("retweet_count", 0)),
                        "quotes": int(metrics.get("quote_count", 0)),
                        "bookmarks": int(metrics.get("bookmark_count", 0)),
                        "views": int(metrics.get("impression_count", 0)),
                        "account": handle,
                        "url": f"https://x.com/{author}/status/{tweet.get('id', '')}",
                        "has_media": has_media,
                        "has_external_url": has_external_url,
                    },
                    created_at=created_at.isoformat(),
                )
            )

    return normalized


def collect_raw_posts() -> list[RawPost]:
    """Collect all raw posts from enabled sources."""

    return [*fetch_reddit_posts(), *fetch_x_posts()]
