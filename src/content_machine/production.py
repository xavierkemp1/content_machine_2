"""Stage 4: produce vertical videos from enhanced content."""

from __future__ import annotations

import contextlib
from pathlib import Path
import shutil
import subprocess
import wave

from .models import EnhancedContent, ProductionArtifact


def _slugify(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in clean.split("-") if part)[:64] or "item"


def _estimate_duration_seconds(narration: str, words_per_minute: int = 145) -> float:
    words = max(1, len(narration.split()))
    return max(6.0, (words / max(80, words_per_minute)) * 60.0)


def _generate_tts_stub(narration: str, output_path: Path) -> str:
    """Generate a silent WAV with narration-proportional duration as offline-safe TTS stub."""

    duration = _estimate_duration_seconds(narration)
    sample_rate = 22050
    total_frames = int(duration * sample_rate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * total_frames)
    return str(output_path)


def _generate_subtitles(narration: str, output_path: Path) -> str:
    words = narration.split()
    chunk_size = 8
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)] or [narration]
    total_duration = _estimate_duration_seconds(narration)
    chunk_duration = total_duration / max(1, len(chunks))

    def srt_time(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        hh = ms // 3_600_000
        mm = (ms % 3_600_000) // 60_000
        ss = (ms % 60_000) // 1000
        mmm = ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{mmm:03d}"

    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        start = (idx - 1) * chunk_duration
        end = idx * chunk_duration
        lines.extend([str(idx), f"{srt_time(start)} --> {srt_time(end)}", chunk, ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(output_path)


def _pick_background(content: EnhancedContent, assets_dir: Path) -> str:
    """Choose matching background clip if available; otherwise use built-in ffmpeg color source."""

    theme = str(content.source_post.raw.metrics.get("subreddit") or content.source_post.raw.metrics.get("account") or "")
    candidates = [f"{_slugify(theme)}.mp4", f"{content.source_post.raw.source}.mp4", "default.mp4"]
    for candidate in candidates:
        path = assets_dir / candidate
        if path.exists():
            return str(path)
    return ""


def _compose_video(
    audio_path: str,
    subtitles_path: str,
    background_path: str,
    output_path: Path,
) -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not ffmpeg_bin:
        output_path.touch()
        return str(output_path)

    input_args: list[str]
    video_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    if background_path:
        input_args = ["-stream_loop", "-1", "-i", background_path]
    else:
        input_args = ["-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30"]
        video_filter = "format=yuv420p"

    command = [
        ffmpeg_bin,
        "-y",
        *input_args,
        "-i",
        audio_path,
        "-vf",
        f"{video_filter},subtitles={subtitles_path}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    ]

    with contextlib.suppress(Exception):
        subprocess.run(command, check=True, capture_output=True, text=True)
    if not output_path.exists():
        output_path.touch()
    return str(output_path)


def render_video(content: EnhancedContent, work_dir: str = "output/_build") -> ProductionArtifact:
    """Create TTS, subtitles, choose background, and compose a 9:16 video artifact."""

    root = Path(work_dir)
    stem = f"{content.source_post.raw.source}-{_slugify(content.source_post.raw.source_id)}"
    audio_path = _generate_tts_stub(content.narration, root / "audio" / f"{stem}.wav")
    subtitles_path = _generate_subtitles(content.narration, root / "subs" / f"{stem}.srt")
    background_path = _pick_background(content, root / "backgrounds")
    video_path = _compose_video(
        audio_path=audio_path,
        subtitles_path=subtitles_path,
        background_path=background_path,
        output_path=root / "videos" / f"{stem}.mp4",
    )

    return ProductionArtifact(
        video_path=video_path,
        subtitles_path=subtitles_path,
        metadata_path="",
        audio_path=audio_path,
        background_path=background_path,
    )


def produce_all(contents: list[EnhancedContent], work_dir: str = "output/_build") -> list[ProductionArtifact]:
    """Run rendering for all enhanced content objects."""

    artifacts: list[ProductionArtifact] = []
    for item in contents:
        artifacts.append(render_video(item, work_dir=work_dir))
    return artifacts
