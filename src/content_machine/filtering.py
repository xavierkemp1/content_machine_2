"""Stage 2: filtering and virality ranking skeleton."""

from .models import RawPost, RankedPost


def apply_rules(posts: list[RawPost]) -> list[RawPost]:
    """TODO: blacklist, dedupe, length, and context checks."""

    return posts


def rank_for_virality(posts: list[RawPost]) -> list[RankedPost]:
    """TODO: score with OpenAI using hook/emotion/clarity/etc criteria."""

    return []
