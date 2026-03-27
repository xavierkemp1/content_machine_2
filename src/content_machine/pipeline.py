"""Orchestrates the full short-form content pipeline skeleton."""

import logging

from .exporting import export_outputs
from .sourcing import collect_raw_posts
from .filtering import apply_rules, rank_for_virality
from .improvement import enhance_posts
from .production import produce_all

logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Run the end-to-end pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== Content Machine pipeline starting ===")

    try:
        raw_posts = collect_raw_posts()
        logger.info("Stage 1 — sourcing: collected %d raw post(s).", len(raw_posts))

        filtered = apply_rules(raw_posts)
        logger.info("Stage 2 — filtering: %d post(s) passed rules.", len(filtered))

        ranked = rank_for_virality(filtered)
        logger.info("Stage 2 — ranking: %d post(s) scored.", len(ranked))

        enhanced = enhance_posts(ranked)
        logger.info("Stage 3 — improvement: %d post(s) enhanced.", len(enhanced))

        artifacts = produce_all(enhanced)
        logger.info(
            "Stage 4 — production: %d video(s) rendered to output/_build/videos/.",
            len(artifacts),
        )

        exported = export_outputs(list(zip(enhanced, artifacts, strict=False)))
        logger.info(
            "Stage 5 — export: %d item(s) written to output/ (organised by theme).",
            len(exported),
        )

        logger.info("=== Pipeline finished successfully. ===")

    except RuntimeError as exc:
        logger.error(
            "Pipeline stopped with an error:\n\n  %s\n\n"
            "Fix the issue above and re-run.  "
            "Tip: set FFMPEG_VERBOSE=1 in your .env for full ffmpeg output.",
            exc,
        )
        raise
