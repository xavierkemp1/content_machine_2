"""Stage 3: lightly copy-edit selected posts for short-form delivery."""

from __future__ import annotations

import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any
from urllib.request import Request

from .http_utils import request_json_with_retries
from .models import EnhancedContent, RankedPost

logger = logging.getLogger(__name__)

_ABBREVIATION_MAP = {
    "aita": "am I the asshole",
    "aitah": "am I the asshole",
    "iata": "I am the asshole",
    "wibta": "would I be wrong",
    "til": "today I learned",
    "tifu": "today I messed up",
    "tl;dr": "too long, didn't read",
    "tldr": "too long, didn't read",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "idk": "I don't know",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "rn": "right now",
    "bc": "because",
    "tbh": "to be honest",
    "ngl": "not going to lie",
    "smh": "shaking my head",
    "atm": "at the moment",
    "omg": "oh my God",
    "omfg": "oh my God",
    "irl": "in real life",
    "fml": "my life is ruined",
    "bff": "best friend",
    "dw": "do not worry",
    "nvm": "never mind",
    "hmu": "hit me up",
    "dm": "direct message",
}

_FILLER_TOKENS_TO_REMOVE = {
    "lol",
    "lmao",
    "rofl",
}

_SMALL_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}


def _clean_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or [text.strip()]


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def _small_number_to_words(n: int) -> str:
    if n in _SMALL_NUMBER_WORDS:
        return _SMALL_NUMBER_WORDS[n]
    if 20 <= n < 100:
        tens_words = {
            20: "twenty",
            30: "thirty",
            40: "forty",
            50: "fifty",
            60: "sixty",
            70: "seventy",
            80: "eighty",
            90: "ninety",
        }
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return tens_words[tens]
        return f"{tens_words[tens]}-{_SMALL_NUMBER_WORDS[ones]}"
    return str(n)


def _remove_filler_tokens(text: str) -> str:
    cleaned = text
    for token in _FILLER_TOKENS_TO_REMOVE:
        cleaned = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _normalize_numeric_text(text: str) -> str:
    def pounds_k(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return f"{_small_number_to_words(value)} thousand pounds"

    def dollars_k(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return f"{_small_number_to_words(value)} thousand dollars"

    def euros_k(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return f"{_small_number_to_words(value)} thousand euros"

    def am_pm(match: re.Match[str]) -> str:
        value = int(match.group(1))
        suffix = match.group(2).lower()
        return f"{_small_number_to_words(value)} {'a.m.' if suffix == 'am' else 'p.m.'}"

    normalized = text
    normalized = re.sub(r"£\s*(\d+)\s*k\b", pounds_k, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\$\s*(\d+)\s*k\b", dollars_k, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"€\s*(\d+)\s*k\b", euros_k, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(\d{1,2})\s*(am|pm)\b", am_pm, normalized, flags=re.IGNORECASE)
    return normalized


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


def _speech_friendly_capitalization(text: str) -> str:
    if not text:
        return ""

    parts = re.split(r"([.!?]\s+)", text)
    rebuilt: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            stripped = part.lstrip()
            if not stripped:
                rebuilt.append(part)
                continue
            leading_spaces = part[: len(part) - len(stripped)]
            rebuilt.append(leading_spaces + stripped[:1].upper() + stripped[1:])
        else:
            rebuilt.append(part)
    return "".join(rebuilt).strip()


def _repair_punctuation(text: str) -> str:
    normalized = _collapse_whitespace(text)
    if not normalized:
        return ""

    normalized = re.sub(r"\s+([,.;!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;!?])([A-Za-z])", r"\1 \2", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()

    # Very light run-on support for common patterns.
    normalized = re.sub(r"\bbut then\b", ". But then", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\band then\b", ". And then", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bso then\b", ". So then", normalized, flags=re.IGNORECASE)

    normalized = _speech_friendly_capitalization(normalized)

    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def preclean_source_text(text: str) -> str:
    """Light cleanup before sending text to the model."""
    cleaned = _collapse_whitespace(text)
    cleaned = _remove_filler_tokens(cleaned)
    return cleaned


def postclean_ai_script(text: str) -> str:
    """Minimal cleanup for accepted AI output without flattening its sentence flow."""
    cleaned = _collapse_whitespace(text)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;!?])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = _speech_friendly_capitalization(cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def fallback_tts_text(text: str) -> str:
    """Stronger deterministic fallback for narration-safe text."""
    normalized = preclean_source_text(text)
    normalized = _expand_abbreviations(normalized)
    normalized = _normalize_numeric_text(normalized)
    normalized = _repair_punctuation(normalized)
    return normalized


def normalize_tts_text(text: str) -> str:
    """Backward-compatible alias for fallback narration normalization."""
    return fallback_tts_text(text)


def _fallback_enhancement(post: RankedPost) -> EnhancedContent:
    lines = _clean_lines(post.raw.text)
    joined = " ".join(lines)
    final_script = fallback_tts_text(joined)

    first = lines[0] if lines else post.raw.text
    first = _collapse_whitespace(first)
    hook = first if len(first) <= 130 else f"{first[:127].rstrip()}..."
    title = hook[:70].strip(".!?") or "Story time"
    caption = f"{title} — would you have handled it the same way?"

    hashtag_tokens = re.findall(r"[a-z0-9]+", final_script.lower())
    unique: list[str] = []
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
        narration=final_script,
        caption=caption,
        final_script=final_script,
        rewritten_caption_script=final_script,
        rewritten_tts_script=final_script,
        hashtags=hashtags,
    )


def _extract_candidate_final_script(ai_row: dict[str, Any]) -> str:
    return str(
        ai_row.get("final_script")
        or ai_row.get("rewritten_tts_script")
        or ai_row.get("rewritten_caption_script")
        or ai_row.get("tts_script")
        or ai_row.get("caption_script")
        or ai_row.get("narration")
        or ""
    ).strip()


def _is_too_far_from_source(source_text: str, edited_text: str) -> bool:
    if not edited_text.strip():
        return True

    source_words = source_text.split()
    edited_words = edited_text.split()
    if not source_words:
        return False

    edited_ratio = len(edited_words) / max(1, len(source_words))
    min_ratio = float(os.getenv("OPENAI_EDIT_MIN_WORD_RATIO", "0.75"))
    max_ratio = float(os.getenv("OPENAI_EDIT_MAX_WORD_RATIO", "1.35"))
    if edited_ratio < min_ratio or edited_ratio > max_ratio:
        return True

    min_char_ratio = float(os.getenv("OPENAI_EDIT_MIN_CHAR_RATIO", "0.70"))
    edited_char_ratio = len(edited_text) / max(1, len(source_text))
    if edited_char_ratio < min_char_ratio:
        return True

    similarity = SequenceMatcher(None, source_text.lower(), edited_text.lower()).ratio()
    min_similarity = float(os.getenv("OPENAI_EDIT_MIN_SIMILARITY", "0.60"))
    return similarity < min_similarity


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


def _openai_enhance_batch(posts: list[RankedPost], model: str, api_key: str) -> dict[str, dict[str, Any]]:
    if not api_key or not posts:
        return {}

    prompt_data = [
        {
            "source_id": post.raw.source_id,
            "source": post.raw.source,
            "text": preclean_source_text(post.raw.text),
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
                            "You are a careful copy editor for narrated short-form videos.\n\n"
                            "For each item, return ONLY a lightly edited final_script.\n\n"
                            "Your job is to make the source text sound natural when spoken aloud while preserving it closely.\n\n"
                            "Rules:\n"
                            "1. Preserve the original wording, order, meaning, and structure as closely as possible.\n"
                            "2. Make only minimal edits:\n"
                            "   - fix grammar\n"
                            "   - add missing punctuation\n"
                            "   - fix capitalization\n"
                            "   - split obvious run-on sentences\n"
                            "   - expand abbreviations that sound bad in speech\n"
                            "   - lightly normalize time, number, and currency phrasing into natural spoken English\n"
                            "3. Do NOT heavily paraphrase.\n"
                            "4. Do NOT aggressively shorten.\n"
                            "5. Do NOT add dramatic flair.\n"
                            "6. Do NOT make it more viral.\n"
                            "7. Keep the output close in length to the source.\n"
                            "8. Return natural written English that also sounds good in TTS.\n\n"
                            "Helpful speech expansions:\n"
                            "- AITA -> \"Am I the asshole\"\n"
                            "- WIBTA -> \"Would I be wrong\"\n"
                            "- idk -> \"I don't know\"\n"
                            "- bc -> \"because\"\n"
                            "- bf / gf -> \"boyfriend\" / \"girlfriend\"\n"
                            "- tbh -> \"to be honest\"\n"
                            "- 3am -> \"three a.m.\" or another natural spoken equivalent\n"
                            "- £20k -> \"twenty thousand pounds\"\n\n"
                            "Return JSON only in this exact format:\n"
                            "{\"items\": [{\"source_id\": \"...\", \"final_script\": \"...\"}]}"
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(prompt_data, ensure_ascii=False),
                    }
                ],
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
        source_id = str(item.get("source_id", "")).strip()
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
    """Enhance ranked posts using light AI copy-editing with strict drift guardrails and fallback."""

    ai_items = _openai_enhance(posts)
    output: list[EnhancedContent] = []

    for post in posts:
        fallback = _fallback_enhancement(post)
        ai_row = ai_items.get(post.raw.source_id)

        if not ai_row:
            output.append(fallback)
            continue

        source_text = preclean_source_text(" ".join(_clean_lines(post.raw.text)))
        raw_candidate_script = _extract_candidate_final_script(ai_row)
        candidate_script = postclean_ai_script(raw_candidate_script)
        accepted_ai_edit = not _is_too_far_from_source(source_text, candidate_script)

        final_script = candidate_script if accepted_ai_edit else fallback.final_script
        if not accepted_ai_edit:
            logger.info(
                "Rejected aggressive AI edit for %s; using deterministic normalization fallback.",
                post.raw.source_id,
            )

        output.append(
            EnhancedContent(
                source_post=post,
                title=fallback.title,
                hook=fallback.hook,
                narration=final_script,
                caption=fallback.caption,
                final_script=final_script,
                rewritten_caption_script=final_script,
                rewritten_tts_script=final_script,
                hashtags=fallback.hashtags,
            )
        )

        logger.info(
            "Enhancement stats for %s: source_len=%d final_len=%d ai_edit_accepted=%s source_preview=%r",
            post.raw.source_id,
            len(source_text),
            len(final_script),
            accepted_ai_edit,
            source_text[:80],
        )
        logger.info(
            "final_script preview for %s: %r",
            post.raw.source_id,
            final_script[:120],
        )

    return output
