"""Stage 1: obtain raw potential content from Reddit and X."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request

from .config import load_json
from .http_utils import request_json_with_retries
from .models import RawPost

logger = logging.getLogger(__name__)


_REDDIT_URL = "https://www.reddit.com/r/{subreddit}/top.json?t=day&limit=50"
_TWITTERAPI_IO_URL = "https://api.twitterapi.io/twitter/user/last_tweets"


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
    """Fetch and normalize X posts via twitterapi.io according to config thresholds."""

    cfg = load_json("config/sources.json").get("x", {})
    if not cfg.get("enabled", False):
        return []

    api_key = os.getenv("TWITTERAPI_IO_KEY", "").strip()
    if not api_key:
        logger.warning("Skipping X sourcing because TWITTERAPI_IO_KEY is not set.")
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

        url = f"{_TWITTERAPI_IO_URL}?userName={handle}"
        try:
            payload = _request_json(url, headers={"X-API-Key": api_key})
        except Exception:
            logger.error("Unable to fetch tweets for @%s from twitterapi.io.", handle)
            continue

        for tweet in payload.get("tweets", []):
            # Skip retweets and replies
            text = (tweet.get("text") or "").strip()
            if not text or text.startswith("RT ") or tweet.get("isReply"):
                continue

            likes = int(tweet.get("likeCount", 0) or 0)
            replies = int(tweet.get("replyCount", 0) or 0)
            if likes < min_likes or replies < min_replies:
                continue

            created_at = _parse_datetime(tweet.get("createdAt", ""))
            if not _is_within_lookback(created_at, lookback_hours):
                continue

            tweet_id = str(tweet.get("id", ""))
            author = (tweet.get("author") or {}).get("userName", handle)
            entities = tweet.get("entities") or {}
            entity_urls = entities.get("urls") or []
            has_external_url = bool(entity_urls)
            has_media = any(
                urlparse(u.get("expanded_url") or "").hostname == "pic.twitter.com"
                or urlparse(u.get("display_url") or "").hostname == "pic.twitter.com"
                for u in entity_urls
            )

            normalized.append(
                RawPost(
                    source="x",
                    source_id=tweet_id,
                    author=author,
                    text=text,
                    metrics={
                        "likes": likes,
                        "replies": replies,
                        "reposts": int(tweet.get("retweetCount", 0) or 0),
                        "quotes": int(tweet.get("quoteCount", 0) or 0),
                        "views": int(tweet.get("viewCount", 0) or 0),
                        "account": handle,
                        "url": tweet.get("url", f"https://x.com/{author}/status/{tweet_id}"),
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
