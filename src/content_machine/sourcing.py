"""Stage 1: obtain raw potential content from Reddit and X."""

from .models import RawPost


def fetch_reddit_posts() -> list[RawPost]:
    """TODO: fetch and normalize Reddit posts according to config thresholds."""

    return []


def fetch_x_posts() -> list[RawPost]:
    """TODO: fetch and normalize X posts according to config thresholds."""

    return []


def collect_raw_posts() -> list[RawPost]:
    """Collect all raw posts from enabled sources."""

    return [*fetch_reddit_posts(), *fetch_x_posts()]
