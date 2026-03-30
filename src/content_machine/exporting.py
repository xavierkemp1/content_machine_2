"""Stage 5: export video files and metadata."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from .models import EnhancedContent, ProductionArtifact


def _slugify(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in clean.split("-") if part)[:64] or "item"


def _determine_theme(content: EnhancedContent) -> str:
    raw = content.source_post.raw
    return str(raw.metrics.get("subreddit") or raw.metrics.get("account") or raw.source)


def export_outputs(
    items: list[tuple[EnhancedContent, ProductionArtifact]],
    base_dir: str = "output",
) -> list[ProductionArtifact]:
    """Save videos and metadata grouped by source theme, returning updated artifacts."""

    output_root = Path(base_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    exported: list[ProductionArtifact] = []
    for content, artifact in items:
        theme = _slugify(_determine_theme(content))
        destination = output_root / theme
        destination.mkdir(parents=True, exist_ok=True)

        stem = f"{content.source_post.raw.source}-{_slugify(content.source_post.raw.source_id)}"
        video_dest = destination / f"{stem}.mp4"
        subtitles_dest = destination / f"{stem}.srt"
        metadata_dest = destination / f"{stem}.json"

        source_video = Path(artifact.video_path)
        if not source_video.exists():
            raise RuntimeError(
                f"Expected video artifact does not exist — cannot export.\n"
                f"  missing file: {source_video}\n"
                f"  This usually means ffmpeg failed silently during the production stage.\n"
                f"  Re-run with FFMPEG_VERBOSE=1 in your .env to see full ffmpeg output, "
                f"or check FFMPEG_TIMEOUT (default 300 s) if rendering stalled."
            )
        if source_video.stat().st_size == 0:
            raise RuntimeError(
                f"Video artifact exists but is 0 bytes — cannot export.\n"
                f"  empty file: {source_video}\n"
                f"  ffmpeg may have crashed or been interrupted before finishing. "
                f"Re-run with FFMPEG_VERBOSE=1 to see full ffmpeg output."
            )
        shutil.copy2(source_video, video_dest)

        source_subs = Path(artifact.subtitles_path)
        if source_subs.exists():
            shutil.copy2(source_subs, subtitles_dest)

        metadata = {
            "title": content.title,
            "hook": content.hook,
            "narration": content.narration,
            "caption": content.caption,
            "final_script": content.final_script or content.narration,
            "rewritten_caption_script": content.final_script or content.rewritten_caption_script,
            "rewritten_tts_script": content.final_script or content.rewritten_tts_script,
            "hashtags": content.hashtags,
            "source": {
                "platform": content.source_post.raw.source,
                "source_id": content.source_post.raw.source_id,
                "author": content.source_post.raw.author,
                "created_at": content.source_post.raw.created_at,
                "metrics": content.source_post.raw.metrics,
            },
            "viral_score": content.source_post.viral_score,
            "score_reasons": content.source_post.score_reasons,
            "assets": {
                "video_path": str(video_dest),
                "subtitles_path": str(subtitles_dest),
                "audio_path": artifact.audio_path,
                "background_path": artifact.background_path,
            },
        }
        metadata_dest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        exported.append(
            ProductionArtifact(
                video_path=str(video_dest),
                subtitles_path=str(subtitles_dest),
                metadata_path=str(metadata_dest),
                audio_path=artifact.audio_path,
                background_path=artifact.background_path,
            )
        )

    return exported
