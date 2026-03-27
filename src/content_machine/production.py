"""Stage 4: produce vertical videos from enhanced content."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
import wave

from .models import EnhancedContent, ProductionArtifact

logger = logging.getLogger(__name__)


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


_DEFAULT_PIPER_VOICE = "en_GB-northern_english_male-medium"


def _generate_tts(narration: str, output_path: Path) -> str:
    """Generate TTS audio via Piper when configured, otherwise fall back to the silent stub.

    Environment variables
    ---------------------
    PIPER_EXE        Path to the piper executable.  When unset or the file is
                     absent the silent stub is used instead (offline-safe).
    PIPER_VOICES_DIR Directory that contains the ``.onnx`` voice model files.
    PIPER_VOICE      Voice basename without extension
                     (default: ``en_GB-northern_english_male-medium``).
    """
    piper_exe = os.getenv("PIPER_EXE", "").strip()
    if not piper_exe or not Path(piper_exe).is_file():
        if piper_exe:
            logger.warning(
                "PIPER_EXE is set to %r but the file was not found; "
                "falling back to silent TTS stub.",
                piper_exe,
            )
        return _generate_tts_stub(narration, output_path)

    voices_dir = os.getenv("PIPER_VOICES_DIR", "").strip()
    voice = os.getenv("PIPER_VOICE", _DEFAULT_PIPER_VOICE).strip() or _DEFAULT_PIPER_VOICE
    model_path = Path(voices_dir) / f"{voice}.onnx" if voices_dir else Path(f"{voice}.onnx")

    if not model_path.is_file():
        logger.warning(
            "Piper voice model not found at %r; falling back to silent TTS stub.",
            str(model_path),
        )
        return _generate_tts_stub(narration, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        piper_exe,
        "--model",
        str(model_path),
        "--output_file",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            input=narration,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            logger.warning(
                "Piper exited with code %d: %s; falling back to silent TTS stub.",
                result.returncode,
                result.stderr[-500:],
            )
            return _generate_tts_stub(narration, output_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Piper TTS failed (%s); falling back to silent TTS stub.",
            exc,
        )
        return _generate_tts_stub(narration, output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        logger.warning(
            "Piper produced no output at %r; falling back to silent TTS stub.",
            str(output_path),
        )
        return _generate_tts_stub(narration, output_path)

    logger.info("Piper TTS generated %r using voice %r.", str(output_path), voice)
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


def _escape_subtitles_filter_path(path: str) -> str:
    """Return an ffmpeg-filter-safe subtitle path.

    Resolves to an absolute path, converts backslashes to forward slashes, and
    escapes the drive-letter colon (e.g. ``C:`` → ``C\\:``) so the path is
    valid inside an ffmpeg ``-vf`` filter expression on Windows.
    """
    resolved = str(Path(path).resolve()).replace("\\", "/")
    if len(resolved) >= 2 and resolved[1] == ":":
        resolved = resolved[0] + r"\:" + resolved[2:]
    return resolved


def _compose_video(
    audio_path: str,
    subtitles_path: str,
    output_path: Path,
) -> str:
    """Render a 9:16 black-background video with burned-in subtitles and audio.

    Raises ``RuntimeError`` if ffmpeg is not found, if ffmpeg exits with a
    non-zero return code, or if the output file is missing / empty after the run.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure it is accessible."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    escaped_subs = _escape_subtitles_filter_path(subtitles_path)
    vf = f"format=yuv420p,subtitles='{escaped_subs}'"

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1080x1920:r=30",
        "-i",
        audio_path,
        "-vf",
        vf,
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

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_excerpt = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
        raise RuntimeError(
            f"ffmpeg exited with code {result.returncode}:\n{stderr_excerpt}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg produced no output or an empty file at {output_path}"
        )

    return str(output_path)


def render_video(content: EnhancedContent, work_dir: str = "output/_build") -> ProductionArtifact:
    """Create TTS, subtitles, and compose a 9:16 black-screen video artifact."""

    root = Path(work_dir)
    stem = f"{content.source_post.raw.source}-{_slugify(content.source_post.raw.source_id)}"
    audio_path = _generate_tts(content.narration, root / "audio" / f"{stem}.wav")
    subtitles_path = _generate_subtitles(content.narration, root / "subs" / f"{stem}.srt")
    video_path = _compose_video(
        audio_path=audio_path,
        subtitles_path=subtitles_path,
        output_path=root / "videos" / f"{stem}.mp4",
    )

    return ProductionArtifact(
        video_path=video_path,
        subtitles_path=subtitles_path,
        metadata_path="",
        audio_path=audio_path,
        background_path="",
    )


def produce_all(contents: list[EnhancedContent], work_dir: str = "output/_build") -> list[ProductionArtifact]:
    """Run rendering for all enhanced content objects."""

    artifacts: list[ProductionArtifact] = []
    for item in contents:
        artifacts.append(render_video(item, work_dir=work_dir))
    return artifacts
