"""Stage 4: produce vertical videos from enhanced content."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import random
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


def _get_wav_duration(wav_path: str) -> float:
    """Return the exact duration in seconds of a WAV file using the wave module.

    Returns 0.0 if the file cannot be read (missing, corrupt, or empty).
    """
    try:
        with wave.open(wav_path, "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0 and frames > 0:
                return frames / float(rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read WAV duration from %r: %s", wav_path, exc)
    return 0.0


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
    if not piper_exe:
        logger.info(
            "PIPER_EXE is not set — using silent TTS stub. "
            "To enable real voice synthesis set PIPER_EXE to the full path of "
            "the piper executable in your .env file "
            "(e.g. PIPER_EXE=C:\\Users\\xavie\\OneDrive\\Documents\\piper\\piper.exe)."
        )
        return _generate_tts_stub(narration, output_path)

    if not Path(piper_exe).is_file():
        logger.warning(
            "PIPER_EXE is set to %r but no file was found at that path. "
            "Check that you have pointed to the piper executable itself "
            "(e.g. piper.exe), not just its containing folder. "
            "Falling back to silent TTS stub.",
            piper_exe,
        )
        return _generate_tts_stub(narration, output_path)

    voices_dir = os.getenv("PIPER_VOICES_DIR", "").strip()
    voice = os.getenv("PIPER_VOICE", _DEFAULT_PIPER_VOICE).strip() or _DEFAULT_PIPER_VOICE
    model_path = Path(voices_dir) / f"{voice}.onnx" if voices_dir else Path(f"{voice}.onnx")

    if not model_path.is_file():
        logger.warning(
            "Piper voice model not found at %r. "
            "Ensure PIPER_VOICES_DIR points to the folder containing *.onnx voice files "
            "(e.g. PIPER_VOICES_DIR=C:\\Users\\xavie\\OneDrive\\Documents\\piper\\voices) "
            "and that PIPER_VOICE matches a file in that folder without the .onnx extension "
            "(current voice: %r). Falling back to silent TTS stub.",
            str(model_path),
            voice,
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
            stderr_excerpt = result.stderr[-500:] if result.stderr else "(no stderr)"
            logger.warning(
                "Piper exited with code %d while generating %r.\n"
                "  command:  %s\n"
                "  stderr:   %s\n"
                "Falling back to silent TTS stub.",
                result.returncode,
                str(output_path),
                " ".join(command),
                stderr_excerpt,
            )
            return _generate_tts_stub(narration, output_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Piper TTS failed with exception %r while running %r — "
            "falling back to silent TTS stub.",
            str(exc),
            " ".join(command),
        )
        return _generate_tts_stub(narration, output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        logger.warning(
            "Piper ran successfully (exit 0) but produced no output at %r. "
            "The voice model may be incompatible with the installed Piper version. "
            "Falling back to silent TTS stub.",
            str(output_path),
        )
        return _generate_tts_stub(narration, output_path)

    logger.info("Piper TTS generated %r using voice %r.", str(output_path), voice)
    return str(output_path)


def _generate_subtitles(narration: str, output_path: Path, audio_duration: float | None = None) -> str:
    """Write an SRT subtitle file whose timings fit exactly within *audio_duration*.

    When *audio_duration* is ``None`` the duration is estimated from word count.
    Chunk durations are allocated proportional to character count so that longer
    chunks stay on screen longer; the final subtitle always ends at *audio_duration*.
    """
    words = narration.split()
    chunk_size = 8
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)] or [narration]
    total_duration = audio_duration if (audio_duration and audio_duration > 0) else _estimate_duration_seconds(narration)

    total_chars = max(1, sum(len(c) for c in chunks))

    def srt_time(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        hh = ms // 3_600_000
        mm = (ms % 3_600_000) // 60_000
        ss = (ms % 60_000) // 1000
        mmm = ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{mmm:03d}"

    lines: list[str] = []
    elapsed = 0.0
    for idx, chunk in enumerate(chunks, start=1):
        start = elapsed
        chunk_fraction = len(chunk) / total_chars
        elapsed += total_duration * chunk_fraction
        # Last chunk must end exactly at total_duration for perfect sync
        end = total_duration if idx == len(chunks) else elapsed
        lines.extend([str(idx), f"{srt_time(start)} --> {srt_time(end)}", chunk, ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(output_path)


def _probe_media_duration_seconds(media_path: Path) -> float:
    """Read media duration using ffprobe; return 0.0 if unavailable."""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return 0.0

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return 0.0
        return max(0.0, float((result.stdout or "").strip() or 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def _select_background_clips(background_dir: Path, target_duration: float) -> list[Path]:
    """Select random clips and repeat as needed until *target_duration* is covered."""
    supported_suffixes = {".mp4", ".mov", ".mkv", ".webm"}
    clips = sorted(
        path
        for path in background_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    if not clips:
        raise RuntimeError(
            f"No background clips found in {background_dir}. "
            "Add .mp4 (or .mov/.mkv/.webm) clips to this folder."
        )

    if target_duration <= 0:
        return [random.choice(clips)]

    selected: list[Path] = []
    total = 0.0
    attempts = 0
    max_attempts = max(20, len(clips) * 10)
    while total < target_duration:
        attempts += 1
        if attempts > max_attempts and not selected:
            raise RuntimeError(
                "Could not determine durations for background clips via ffprobe. "
                "Ensure clips are valid media files and ffprobe is installed."
            )
        if attempts > max_attempts and selected:
            logger.warning(
                "Background selection ended early after %d attempts; accumulated %.2fs for target %.2fs.",
                attempts,
                total,
                target_duration,
            )
            break
        clip = random.choice(clips)
        duration = _probe_media_duration_seconds(clip)
        if duration <= 0:
            logger.warning("Could not probe duration for background clip %s; skipping.", clip)
            continue
        selected.append(clip)
        total += duration

    return selected


def _write_concat_manifest(clips: list[Path], manifest_path: Path) -> str:
    """Write ffmpeg concat demuxer file for clip list."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for clip in clips:
        clip_path = str(clip.resolve()).replace("'", "'\\''")
        lines.append(f"file '{clip_path}'")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(manifest_path)


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
    background_manifest_path: str,
    output_path: Path,
    audio_duration: float = 0.0,
    timeout: int = 300,
    verbose: bool = False,
) -> str:
    """Render a 9:16 background video with burned-in subtitles and audio.

    Parameters
    ----------
    audio_path:
        Path to the source WAV audio file.
    subtitles_path:
        Path to the SRT subtitle file.
    output_path:
        Destination MP4 path.
    audio_duration:
        Exact audio duration in seconds.  When > 0 a ``-t`` flag is added so
        ffmpeg cannot run past the end of the audio, preventing indefinite hangs.
    timeout:
        Maximum seconds to wait for ffmpeg before raising ``RuntimeError``.
        Defaults to 300 s (5 min).  Set via ``FFMPEG_TIMEOUT`` env var.
    verbose:
        When ``True`` ffmpeg stdout/stderr are streamed to the console instead
        of being captured.  Set via ``FFMPEG_VERBOSE=1`` env var.

    Raises ``RuntimeError`` if ffmpeg is not found, times out, exits with a
    non-zero return code, or produces a missing/empty output file.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure it is accessible."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    escaped_subs = _escape_subtitles_filter_path(subtitles_path)
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        f"subtitles='{escaped_subs}',"
        "format=yuv420p"
    )

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error" if not verbose else "info",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        background_manifest_path,
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
    ]

    if audio_duration > 0:
        command += ["-t", f"{audio_duration:.3f}"]

    command.append(str(output_path))

    logger.info("ffmpeg command: %s", " ".join(command))
    logger.info(
        "Rendering video → %s  (build dir; final output expected in output/ by theme)",
        output_path,
    )

    try:
        result = subprocess.run(
            command,
            capture_output=not verbose,
            text=not verbose,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg timed out after {timeout}s rendering {output_path}. "
            "Check that ffmpeg is working correctly and that input files are valid.\n"
            f"  audio:     {audio_path}\n"
            f"  subtitles: {subtitles_path}"
        )

    if result.returncode != 0:
        stderr_excerpt = ""
        if result.stderr:
            stderr_excerpt = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
        raise RuntimeError(
            f"ffmpeg exited with code {result.returncode} rendering {output_path}:\n"
            f"{stderr_excerpt}\n"
            f"  audio:     {audio_path}\n"
            f"  subtitles: {subtitles_path}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg produced no output or an empty file at {output_path}\n"
            f"  audio:     {audio_path}\n"
            f"  subtitles: {subtitles_path}"
        )

    return str(output_path)


def render_video(content: EnhancedContent, work_dir: str = "output/_build") -> ProductionArtifact:
    """Create TTS, subtitles, and compose a 9:16 background video artifact.

    Subtitle timings are derived from the *actual* WAV duration so captions
    sync perfectly to the voice-over.  ffmpeg is constrained to that same
    duration via ``-t`` to prevent indefinite hangs.

    Environment variables
    ---------------------
    FFMPEG_TIMEOUT   Seconds before ffmpeg is killed (default: 300).
    FFMPEG_VERBOSE   Set to ``1`` / ``true`` / ``yes`` to stream ffmpeg output
                     to the console instead of capturing it silently.
    """

    root = Path(work_dir)
    stem = f"{content.source_post.raw.source}-{_slugify(content.source_post.raw.source_id)}"

    audio_path = _generate_tts(content.narration, root / "audio" / f"{stem}.wav")

    # Derive exact duration from the generated WAV for accurate subtitle sync
    audio_duration = _get_wav_duration(audio_path)
    if audio_duration <= 0:
        logger.warning(
            "Could not determine audio duration for %r; subtitle timing will use estimate.",
            audio_path,
        )

    subtitles_path = _generate_subtitles(
        content.narration,
        root / "subs" / f"{stem}.srt",
        audio_duration=audio_duration if audio_duration > 0 else None,
    )
    background_dir = Path(os.getenv("BACKGROUND_CLIPS_DIR", "background_clips"))
    if not background_dir.is_absolute():
        background_dir = Path.cwd() / background_dir
    selected_backgrounds = _select_background_clips(
        background_dir=background_dir,
        target_duration=audio_duration if audio_duration > 0 else _estimate_duration_seconds(content.narration),
    )
    manifest_path = _write_concat_manifest(
        selected_backgrounds,
        root / "backgrounds" / f"{stem}.txt",
    )

    verbose = os.getenv("FFMPEG_VERBOSE", "").strip().lower() in ("1", "true", "yes")
    try:
        timeout = int(os.getenv("FFMPEG_TIMEOUT", "300"))
    except ValueError:
        timeout = 300

    video_path = _compose_video(
        audio_path=audio_path,
        subtitles_path=subtitles_path,
        background_manifest_path=manifest_path,
        output_path=root / "videos" / f"{stem}.mp4",
        audio_duration=audio_duration,
        timeout=timeout,
        verbose=verbose,
    )

    logger.info(
        "Production complete for %r. Build artifacts in %s; "
        "final exports expected in output/ organised by theme after export stage.",
        stem,
        root,
    )

    return ProductionArtifact(
        video_path=video_path,
        subtitles_path=subtitles_path,
        metadata_path="",
        audio_path=audio_path,
        background_path="|".join(str(path) for path in selected_backgrounds),
    )


def produce_all(contents: list[EnhancedContent], work_dir: str = "output/_build") -> list[ProductionArtifact]:
    """Run rendering for all enhanced content objects."""

    artifacts: list[ProductionArtifact] = []
    for item in contents:
        artifacts.append(render_video(item, work_dir=work_dir))
    return artifacts
