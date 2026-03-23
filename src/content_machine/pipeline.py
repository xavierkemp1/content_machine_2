"""Orchestrates the full short-form content pipeline skeleton."""

from .sourcing import collect_raw_posts
from .filtering import apply_rules, rank_for_virality
from .improvement import enhance_posts


def run_pipeline() -> None:
    """Run the end-to-end pipeline (skeleton)."""

    raw_posts = collect_raw_posts()
    filtered = apply_rules(raw_posts)
    ranked = rank_for_virality(filtered)
    _enhanced = enhance_posts(ranked)
    # Production/export are intentionally left uninvoked in this skeleton.
