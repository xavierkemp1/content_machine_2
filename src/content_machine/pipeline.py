"""Orchestrates the full short-form content pipeline skeleton."""

from .exporting import export_outputs
from .sourcing import collect_raw_posts
from .filtering import apply_rules, rank_for_virality
from .improvement import enhance_posts
from .production import produce_all


def run_pipeline() -> None:
    """Run the end-to-end pipeline."""

    raw_posts = collect_raw_posts()
    filtered = apply_rules(raw_posts)
    ranked = rank_for_virality(filtered)
    enhanced = enhance_posts(ranked)
    artifacts = produce_all(enhanced)
    export_outputs(list(zip(enhanced, artifacts, strict=False)))
