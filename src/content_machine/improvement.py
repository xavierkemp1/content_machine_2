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

_ABBREVIATION_MAP = {
    "aita": "am i the asshole",
    "wibta": "would i be wrong",
    "til": "today i learned",
    "tifu": "today i messed up",
    "tl;dr": "too long, didn't read",
    "tldr": "too long, didn't read",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "idk": "i don't know",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "rn": "right now",
}


def _clean_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or [text.strip()]


def _fallback_enhancement(post: RankedPost) -> EnhancedContent:
    lines = _clean_lines(post.raw.text)
    first = lines[0] if lines else post.raw.text
    hook = first if len(first) <= 130 else f"{first[:127].rstrip()}..."
    narration = normalize_tts_text(" ".join(lines))
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
        rewritten_caption_script=" ".join(lines),
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

    return caption_script, normalize_tts_text(tts_script)


def _extract_response_output_text(body: dict[str, Any]) -> str:
    output_text = str(body.get("output_text", "")).strip()
    if output_text:
        return output_text
    output_parts: list[str] = []
    for item in body.get("output", []):
        for part in item.get("content", []):
            if part.get("type") in {"output_text", "text"}:
                text_value = str(part.get("text", "")).strip()
                if text_value:
                    output_parts.append(text_value)
    return "".join(output_parts).strip()


def _normalize_numeric_text(text: str) -> str:
    text = re.sub(
        r"£\s*(\d+)\s*k\b",
        lambda m: f"{int(m.group(1)):,} pounds".replace(",", " thousand "),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d{1,2})\s*(am|pm)\b",
        lambda m: f"{m.group(1)} {'in the morning' if m.group(2).lower() == 'am' else 'in the evening'}",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _expand_abbreviations(text: str) -> str:
    expanded = text
    expanded = re.sub(r"\bPOV\b", "point of view", expanded, flags=re.IGNORECASE)
    for short, long_form in _ABBREVIATION_MAP.items():
        expanded = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(short)}(?![A-Za-z0-9])",
            long_form,
            expanded,
            flags=re.IGNORECASE,
        )
    return expanded


def _repair_punctuation(text: str) -> str:
    normalized = " ".join(text.replace("\n", " ").split())
    if not normalized:
        return ""
    normalized = re.sub(r"\s+([,.;!?])", r"\1", normalized)
    normalized = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()
    if normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def normalize_tts_text(text: str) -> str:
    """Normalize story text for speech with expanded shorthand and cleaner punctuation."""
    return _repair_punctuation(_normalize_numeric_text(_expand_abbreviations(text)))


def _openai_enhance_batch(posts: list[RankedPost], model: str, api_key: str) -> dict[str, dict[str, Any]]:
    if not api_key or not posts:
        return {}
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
                            "with spoken-friendly wording, explicit abbreviation expansion, fixed punctuation, "
                            "clear sentence boundaries, and no run-on sentences. "
                            "TTS script must read naturally out loud."
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
    output_text = _extract_response_output_text(body)
    if not output_text:
        logger.error("OpenAI enhancement response did not include output text.")
        return {}
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        logger.error("OpenAI enhancement output_text was not valid JSON; using fallback enhancement.")
        return {}

    enhanced: dict[str, dict[str, Any]] = {}
    for item in parsed.get("items", []):
        source_id = str(item.get("source_id", ""))
        if source_id:
            enhanced[source_id] = item
    return enhanced


def _openai_enhance(posts: list[RankedPost]) -> dict[str, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not posts:
        if posts and not api_key:
            logger.warning("Skipping OpenAI enhancement because OPENAI_API_KEY is not set.")
        return {}

    model = os.getenv("OPENAI_REWRITE_MODEL", "gpt-4.1-mini")
    try:
        batch_size = max(1, int(os.getenv("OPENAI_REWRITE_BATCH_SIZE", "5")))
    except ValueError:
        batch_size = 5

    enhanced: dict[str, dict[str, Any]] = {}
    for start in range(0, len(posts), batch_size):
        batch = posts[start : start + batch_size]
        batch_result = _openai_enhance_batch(batch, model=model, api_key=api_key)
        if not batch_result:
            logger.warning(
                "OpenAI enhancement batch failed for items %d-%d; using local fallback for that batch.",
                start,
                start + len(batch) - 1,
            )
            continue
        enhanced.update(batch_result)
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
                rewritten_tts_script=normalize_tts_text(tts_script),
                hashtags=[
                    str(tag).strip()
                    for tag in ai_row.get("hashtags", [])
                    if str(tag).strip()
                ]
                or fallback.hashtags,
            )
        )
    return output
