"""Stage 3: rewrite selected posts for short-form delivery."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib.request import Request

from .http_utils import request_json_with_retries
from .models import EnhancedContent, RankedPost

logger = logging.getLogger(__name__)


def _clean_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or [text.strip()]


def _fallback_enhancement(post: RankedPost) -> EnhancedContent:
    lines = _clean_lines(post.raw.text)
    first = lines[0] if lines else post.raw.text
    hook = first if len(first) <= 130 else f"{first[:127].rstrip()}..."
    narration = " ".join(lines)
    title = hook[:70].strip(".!?") or "Story time"
    caption = f"{title} — would you have handled it the same way?"

    hashtag_tokens = re.findall(r"[a-z0-9]+", narration.lower())
    unique = []
    for token in hashtag_tokens:
        if len(token) < 4 or token in unique:
            continue
        unique.append(token)
        if len(unique) >= 4:
            break
    hashtags = [f"#{token}" for token in (unique or ["storytime", "viral", "shorts"])]

    return EnhancedContent(
        source_post=post,
        title=title,
        hook=hook,
        narration=narration,
        caption=caption,
        rewritten_caption_script=narration,
        rewritten_tts_script=narration,
        hashtags=hashtags,
    )


def _normalize_script_outputs(ai_row: dict[str, Any], fallback: EnhancedContent) -> tuple[str, str]:
    """Return (caption_script, tts_script) with sensible fallback behaviour."""
    caption_script = str(
        ai_row.get("rewritten_caption_script")
        or ai_row.get("caption_script")
        or ai_row.get("caption")
        or ""
    ).strip()
    tts_script = str(
        ai_row.get("rewritten_tts_script")
        or ai_row.get("tts_script")
        or ai_row.get("narration")
        or ""
    ).strip()

    if caption_script and not tts_script:
        tts_script = caption_script
    elif tts_script and not caption_script:
        caption_script = tts_script

    if not caption_script:
        caption_script = fallback.rewritten_caption_script or fallback.narration
    if not tts_script:
        tts_script = fallback.rewritten_tts_script or fallback.narration

    return caption_script, tts_script


def _openai_enhance(posts: list[RankedPost]) -> dict[str, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not posts:
        if posts and not api_key:
            logger.warning("Skipping OpenAI enhancement because OPENAI_API_KEY is not set.")
        return {}

    model = os.getenv("OPENAI_REWRITE_MODEL", "gpt-4.1-mini")
    prompt_data = [
        {
            "source_id": post.raw.source_id,
            "source": post.raw.source,
            "text": post.raw.text,
            "viral_score": post.viral_score,
        }
        for post in posts
    ]
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Rewrite each item for short-form delivery. Return JSON object with key "
                            "'items': [{source_id, title, hook, rewritten_caption_script, "
                            "rewritten_tts_script, narration, caption, hashtags}] only. "
                            "Rules: rewritten_caption_script must be concise and punchy for on-screen "
                            "captions. rewritten_tts_script must be optimized for natural narration "
                            "with spoken-friendly wording, expanded abbreviations when useful, and "
                            "pause-friendly punctuation."
                        ),
                    }
                ],
            },
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
        operation="OpenAI enhancement request",
        timeout=60,
        max_attempts=4,
    )
    if not body:
        return {}
    try:
        parsed = json.loads(body.get("output_text", "{}"))
    except json.JSONDecodeError:
        logger.error("OpenAI enhancement output_text was not valid JSON; using fallback enhancement.")
        return {}

    enhanced: dict[str, dict[str, Any]] = {}
    for item in parsed.get("items", []):
        source_id = str(item.get("source_id", ""))
        if source_id:
            enhanced[source_id] = item
    return enhanced


def enhance_posts(posts: list[RankedPost]) -> list[EnhancedContent]:
    """Enhance ranked posts using OpenAI when available, then fallback rewrite logic."""

    ai_items = _openai_enhance(posts)
    output: list[EnhancedContent] = []
    for post in posts:
        ai_row = ai_items.get(post.raw.source_id)
        fallback = _fallback_enhancement(post)
        if not ai_row:
            output.append(fallback)
            continue
        caption_script, tts_script = _normalize_script_outputs(ai_row, fallback)

        output.append(
            EnhancedContent(
                source_post=post,
                title=str(ai_row.get("title", "")).strip() or fallback.title,
                hook=str(ai_row.get("hook", "")).strip() or fallback.hook,
                narration=str(ai_row.get("narration", "")).strip() or tts_script or fallback.narration,
                caption=str(ai_row.get("caption", "")).strip() or fallback.caption,
                rewritten_caption_script=caption_script,
                rewritten_tts_script=tts_script,
                hashtags=[
                    str(tag).strip()
                    for tag in ai_row.get("hashtags", [])
                    if str(tag).strip()
                ]
                or fallback.hashtags,
            )
        )
    return output
