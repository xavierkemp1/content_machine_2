"""Stage 3: rewrite selected posts for short-form delivery."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.request import Request, urlopen

from .models import EnhancedContent, RankedPost


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
        hashtags=hashtags,
    )


def _openai_enhance(posts: list[RankedPost]) -> dict[str, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not posts:
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
                        "type": "text",
                        "text": (
                            "Rewrite each item for short-form delivery. Return JSON object with key "
                            "'items': [{source_id, title, hook, narration, caption, hashtags}] only."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(prompt_data)}],
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
    try:
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        parsed = json.loads(body.get("output_text", "{}"))
    except Exception:
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
        if not ai_row:
            output.append(_fallback_enhancement(post))
            continue

        output.append(
            EnhancedContent(
                source_post=post,
                title=str(ai_row.get("title", "")).strip() or _fallback_enhancement(post).title,
                hook=str(ai_row.get("hook", "")).strip() or _fallback_enhancement(post).hook,
                narration=str(ai_row.get("narration", "")).strip() or _fallback_enhancement(post).narration,
                caption=str(ai_row.get("caption", "")).strip() or _fallback_enhancement(post).caption,
                hashtags=[
                    str(tag).strip()
                    for tag in ai_row.get("hashtags", [])
                    if str(tag).strip()
                ]
                or _fallback_enhancement(post).hashtags,
            )
        )
    return output
