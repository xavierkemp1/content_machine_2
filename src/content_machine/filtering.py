"""Stage 2: filtering and virality ranking."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import logging
import math
import os
from typing import Any
from urllib.request import Request

from .config import load_json
from .http_utils import request_json_with_retries
from .models import RawPost, RankedPost

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _word_count(text: str) -> int:
    return len(text.split())


def _length_bucket(word_count: int, short_max_words: int, medium_max_words: int) -> str:
    if word_count <= short_max_words:
        return "short"
    if word_count <= medium_max_words:
        return "medium"
    return "long"


def _is_media_dependent(post: RawPost) -> bool:
    has_media = bool(post.metrics.get("has_media"))
    has_external_url = bool(post.metrics.get("has_external_url"))
    word_count = _word_count(post.text)
    return (has_media or has_external_url) and word_count < 120


def _dedupe_posts(posts: list[RawPost], similarity_threshold: float) -> list[RawPost]:
    deduped: list[RawPost] = []
    normalized_seen: list[str] = []

    for post in posts:
        normalized = _normalize_text(post.text)
        if not normalized:
            continue

        duplicate = False
        for seen in normalized_seen:
            if normalized == seen:
                duplicate = True
                break
            ratio = SequenceMatcher(None, normalized, seen).ratio()
            if ratio >= similarity_threshold:
                duplicate = True
                break

        if not duplicate:
            deduped.append(post)
            normalized_seen.append(normalized)

    return deduped


def apply_rules(posts: list[RawPost]) -> list[RawPost]:
    """Apply blacklist, length, media-dependent filtering, and dedupe checks."""

    cfg = load_json("config/filtering.json")
    blacklist = [phrase.lower() for phrase in cfg.get("blacklist_phrases", [])]

    length_cfg = cfg.get("length_buckets", {})
    short_max_words = int(length_cfg.get("short_max_words", 80))
    medium_max_words = int(length_cfg.get("medium_max_words", 200))
    max_words = int(length_cfg.get("max_words", 450))
    min_words = int(length_cfg.get("min_words", 10))

    dedupe_cfg = cfg.get("dedupe", {})
    dedupe_enabled = bool(dedupe_cfg.get("enabled", True))
    similarity_threshold = float(dedupe_cfg.get("similarity_threshold", 0.9))

    filtered: list[RawPost] = []

    for post in posts:
        text = post.text.strip()
        text_lower = text.lower()
        if not text:
            continue

        word_count = _word_count(text)
        if word_count < min_words or word_count > max_words:
            continue

        if any(phrase in text_lower for phrase in blacklist):
            continue

        if _is_media_dependent(post):
            continue

        filtered.append(post)

    if dedupe_enabled:
        filtered = _dedupe_posts(filtered, similarity_threshold)

    # Precompute bucket hints so ranking can use consistent values in one place.
    for post in filtered:
        post.metrics["length_bucket"] = _length_bucket(
            _word_count(post.text),
            short_max_words,
            medium_max_words,
        )

    return filtered


def _heuristic_viral_score(post: RawPost) -> tuple[float, dict[str, float]]:
    text = post.text
    words = _word_count(text)

    engagement = 0.0
    if post.source == "reddit":
        engagement += math.log1p(float(post.metrics.get("score", 0))) * 10
        engagement += math.log1p(float(post.metrics.get("comments", 0))) * 8
    if post.source == "x":
        engagement += math.log1p(float(post.metrics.get("likes", 0))) * 9
        engagement += math.log1p(float(post.metrics.get("replies", 0))) * 11
        engagement += math.log1p(float(post.metrics.get("reposts", 0))) * 6

    hook = 8.0 if any(c in text for c in ("?", "!")) else 4.0
    emotion = min(10.0, text.count("!") * 1.5 + text.lower().count("i ") * 0.5)
    clarity = max(0.0, 12.0 - abs(words - 120) / 10.0)
    relatability = 6.0 if any(
        phrase in text.lower()
        for phrase in ("i ", "my ", "we ", "our ", "friend", "family", "relationship")
    ) else 3.0
    comment_bait = 7.0 if "?" in text else 3.5
    short_form = 10.0 if 40 <= words <= 220 else max(1.5, 8.0 - abs(words - 130) / 30)

    score_reasons = {
        "engagement": round(engagement, 3),
        "hook_strength": round(hook, 3),
        "emotional_charge": round(emotion, 3),
        "clarity": round(clarity, 3),
        "relatability": round(relatability, 3),
        "comment_bait": round(comment_bait, 3),
        "short_form_suitability": round(short_form, 3),
    }

    weighted = (
        engagement * 0.35
        + hook * 0.1
        + emotion * 0.1
        + clarity * 0.15
        + relatability * 0.1
        + comment_bait * 0.1
        + short_form * 0.1
    )
    return round(weighted, 3), score_reasons


def _openai_rank(posts: list[RawPost]) -> dict[str, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not posts:
        if posts and not api_key:
            logger.warning("Skipping OpenAI ranking because OPENAI_API_KEY is not set.")
        return {}

    model = os.getenv("OPENAI_RANKING_MODEL", "gpt-4.1-mini")
    prompt_data = [
        {
            "source": post.source,
            "source_id": post.source_id,
            "text": post.text,
            "metrics": post.metrics,
        }
        for post in posts
    ]

    prompt = (
        "Score each post for viral short-form potential. Return only JSON with key 'scores' where "
        "scores is a list of {source_id, score, reasons}. Score must be 0-100. Reasons should include "
        "hook_strength, emotional_charge, clarity, relatability, comment_bait, short_form_suitability."
    )

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(prompt_data)}],
            },
        ],
        "text": {"format": {"type": "json_object"}},
    }

    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    body = request_json_with_retries(
        request,
        operation="OpenAI virality ranking request",
        timeout=45,
        max_attempts=4,
    )
    if not body:
        return {}

    output_text = body.get("output_text", "")
    if not output_text:
        logger.error("OpenAI ranking response did not include output_text; falling back to heuristic scores.")
        return {}

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        logger.error("OpenAI ranking output_text was not valid JSON; falling back to heuristic scores.")
        return {}

    scores = {}
    for row in parsed.get("scores", []):
        source_id = str(row.get("source_id", ""))
        if not source_id:
            continue
        scores[source_id] = {
            "score": float(row.get("score", 0.0)),
            "reasons": row.get("reasons", {}),
        }

    return scores


def rank_for_virality(posts: list[RawPost]) -> list[RankedPost]:
    """Score posts for virality using OpenAI when available, else heuristic fallback."""

    cfg = load_json("config/filtering.json")
    length_cfg = cfg.get("length_buckets", {})
    short_max_words = int(length_cfg.get("short_max_words", 80))
    medium_max_words = int(length_cfg.get("medium_max_words", 200))

    ai_scores = _openai_rank(posts)
    ranked_posts: list[RankedPost] = []

    for post in posts:
        word_count = _word_count(post.text)
        length_bucket = post.metrics.get("length_bucket") or _length_bucket(
            word_count,
            short_max_words,
            medium_max_words,
        )

        source_id = str(post.source_id)
        if source_id in ai_scores:
            ai_row = ai_scores[source_id]
            viral_score = float(ai_row.get("score", 0.0))
            score_reasons = ai_row.get("reasons", {})
        else:
            viral_score, score_reasons = _heuristic_viral_score(post)

        ranked_posts.append(
            RankedPost(
                raw=post,
                length_bucket=str(length_bucket),
                viral_score=viral_score,
                score_reasons=score_reasons,
            )
        )

    ranked_posts.sort(key=lambda item: item.viral_score, reverse=True)
    return ranked_posts
