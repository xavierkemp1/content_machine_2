"""Stage 4: produce vertical videos from enhanced content."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass

from .models import EnhancedContent, ProductionArtifact

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptionRenderConfig:
    style_mode: str = "active_word"
    active_word_highlight_color: str = "&H0038FF&"
    font_size: int = 62
    margin_v: int = 210
    words_per_chunk: int = 7
    min_chunk_seconds: float = 0.9
    hook_scale: float = 1.12


@dataclass(frozen=True)
class ProductionRuntimeConfig:
    background_safety_buffer_seconds: float = 0.75
    background_randomize: bool = True
    allow_immediate_background_repeats: bool = False
    background_rng_seed: int | None = None
    voice_profile: str = "en_GB-northern_english_male-medium"
    caption: CaptionRenderConfig = CaptionRenderConfig()


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


def _load_runtime_config() -> ProductionRuntimeConfig:
    """Load runtime options from environment with safe defaults."""
    style_mode = os.getenv("CAPTION_STYLE_MODE", "active_word").strip().lower() or "active_word"
    if style_mode not in {"active_word", "plain"}:
        style_mode = "active_word"
    try:
        safety_buffer = float(os.getenv("BACKGROUND_SAFETY_BUFFER_SECONDS", "0.75"))
    except ValueError:
        safety_buffer = 0.75
    safety_buffer = max(0.0, min(2.0, safety_buffer))
    randomize = os.getenv("BACKGROUND_RANDOMIZE", "1").strip().lower() in {"1", "true", "yes"}
    allow_repeats = os.getenv("BACKGROUND_ALLOW_IMMEDIATE_REPEATS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    seed_text = os.getenv("BACKGROUND_RANDOM_SEED", "").strip()
    try:
        background_rng_seed = int(seed_text) if seed_text else None
    except ValueError:
        background_rng_seed = None

    voice_profile = os.getenv("VOICE_PROFILE", _DEFAULT_PIPER_VOICE).strip() or _DEFAULT_PIPER_VOICE
    highlight_color = os.getenv("CAPTION_ACTIVE_WORD_COLOR", "&H0038FF&").strip() or "&H0038FF&"

    return ProductionRuntimeConfig(
        background_safety_buffer_seconds=safety_buffer,
        background_randomize=randomize,
        allow_immediate_background_repeats=allow_repeats,
        background_rng_seed=background_rng_seed,
        voice_profile=voice_profile,
        caption=CaptionRenderConfig(
            style_mode=style_mode,
            active_word_highlight_color=highlight_color,
        ),
    )


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
    voice = (
        os.getenv("PIPER_VOICE", "").strip()
        or os.getenv("VOICE_PROFILE", "").strip()
        or _DEFAULT_PIPER_VOICE
    )
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


def _chunk_words_for_captions(words: list[str], words_per_chunk: int) -> list[list[str]]:
    if not words:
        return [[]]
    chunks: list[list[str]] = []
    idx = 0
    while idx < len(words):
        end = min(len(words), idx + words_per_chunk)
        chunks.append(words[idx:end])
        idx = end
    return chunks


def _estimate_word_weights(chunk_words: list[str]) -> list[float]:
    weights = []
    for word in chunk_words:
        clean = re.sub(r"[^a-zA-Z0-9]", "", word)
        weights.append(max(1.0, float(len(clean) or 1)))
    return weights


def _seconds_to_ass_time(seconds: float) -> str:
    total_cs = int(round(max(0.0, seconds) * 100))
    hh = total_cs // 360000
    mm = (total_cs % 360000) // 6000
    ss = (total_cs % 6000) // 100
    cs = total_cs % 100
    return f"{hh}:{mm:02d}:{ss:02d}.{cs:02d}"


def _ass_escape_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _build_ass_dialogue_lines(
    caption_script: str,
    audio_duration: float,
    caption_cfg: CaptionRenderConfig,
) -> list[str]:
    words = caption_script.split()
    chunks = _chunk_words_for_captions(words, words_per_chunk=max(2, caption_cfg.words_per_chunk))
    total_chars = max(1, sum(len(" ".join(chunk)) for chunk in chunks if chunk))
    lines: list[str] = []
    elapsed = 0.0

    for idx, chunk_words in enumerate(chunks):
        chunk_text = " ".join(chunk_words).strip()
        if not chunk_text:
            continue
        base_fraction = len(chunk_text) / total_chars
        chunk_duration = max(caption_cfg.min_chunk_seconds, audio_duration * base_fraction)
        if idx == len(chunks) - 1:
            chunk_end = audio_duration
        else:
            chunk_end = min(audio_duration, elapsed + chunk_duration)
        chunk_start = min(elapsed, audio_duration)
        chunk_span = max(0.05, chunk_end - chunk_start)
        word_weights = _estimate_word_weights(chunk_words)
        total_weight = max(1.0, sum(word_weights))
        word_elapsed = chunk_start
        for word_idx, word in enumerate(chunk_words):
            weight_fraction = word_weights[word_idx] / total_weight
            word_end = chunk_end if word_idx == len(chunk_words) - 1 else min(chunk_end, word_elapsed + chunk_span * weight_fraction)
            rendered_words: list[str] = []
            for highlight_idx, token in enumerate(chunk_words):
                escaped = _ass_escape_text(token)
                if highlight_idx == word_idx and caption_cfg.style_mode == "active_word":
                    rendered_words.append(
                        "{\\c"
                        + caption_cfg.active_word_highlight_color
                        + "}"
                        + escaped
                        + "{\\c&HFFFFFF&}"
                    )
                else:
                    rendered_words.append(escaped)
            rendered_text = " ".join(rendered_words)
            style = "Hook" if idx == 0 else "Default"
            lines.append(
                "Dialogue: 0,"
                f"{_seconds_to_ass_time(word_elapsed)},{_seconds_to_ass_time(word_end)},"
                f"{style},,0,0,0,,{rendered_text}"
            )
            word_elapsed = word_end
        elapsed = chunk_end
    return lines


def _generate_ass_subtitles(
    caption_script: str,
    output_path: Path,
    audio_duration: float,
    caption_cfg: CaptionRenderConfig,
) -> str:
    dialogue_lines = _build_ass_dialogue_lines(caption_script, max(0.01, audio_duration), caption_cfg)
    ass = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
            "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
            "Alignment,MarginL,MarginR,MarginV,Encoding",
            f"Style: Default,Arial,{caption_cfg.font_size},&H00FFFFFF&,&H00FFFFFF&,&H00303030&,&H64000000&,"
            "1,0,0,0,100,100,0,0,1,2.2,1.2,2,80,80,"
            f"{caption_cfg.margin_v},1",
            f"Style: Hook,Arial,{int(caption_cfg.font_size * caption_cfg.hook_scale)},&H00FFFFFF&,&H00FFFFFF&,&H00303030&,&H64000000&,"
            "1,0,0,0,100,100,0,0,1,2.4,1.3,2,80,80,"
            f"{caption_cfg.margin_v + 20},1",
            "",
            "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
            *dialogue_lines,
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ass, encoding="utf-8")
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


def build_background_timeline(
    background_dir: Path,
    target_duration: float,
    safety_buffer_seconds: float = 0.75,
    rng_seed: int | None = None,
    randomize: bool = True,
    allow_immediate_repeats: bool = False,
) -> tuple[list[Path], float]:
    """Select clips until duration covers target+buffer, with reuse when needed."""
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

    valid_clips: list[tuple[Path, float]] = []
    for clip in clips:
        duration = _probe_media_duration_seconds(clip)
        if duration <= 0.15:
            logger.warning("Skipping background clip %s (duration unavailable or too short).", clip)
            continue
        valid_clips.append((clip, duration))
    if not valid_clips:
        raise RuntimeError(
            f"No valid background clips with measurable duration found in {background_dir}. "
            "Ensure clips are readable and ffprobe is available."
        )

    if target_duration <= 0:
        clip, duration = valid_clips[0]
        return [clip], duration

    rng = random.Random(rng_seed) if randomize else None
    desired_total = target_duration + max(0.0, safety_buffer_seconds)
    selected: list[Path] = []
    total_duration = 0.0
    previous: Path | None = None
    deterministic_idx = 0
    attempts = 0
    max_attempts = max(40, len(valid_clips) * 20)

    while total_duration < desired_total and attempts < max_attempts:
        attempts += 1
        candidates = (
            [item for item in valid_clips if allow_immediate_repeats or item[0] != previous]
            or valid_clips
        )
        if randomize:
            clip, duration = rng.choice(candidates)  # type: ignore[union-attr]
        else:
            clip, duration = candidates[deterministic_idx % len(candidates)]
            deterministic_idx += 1
        selected.append(clip)
        total_duration += duration
        previous = clip

    if total_duration < desired_total:
        logger.warning(
            "Background timeline did not reach desired %.2fs after %d attempts (reached %.2fs).",
            desired_total,
            attempts,
            total_duration,
        )
    selected_summary = ", ".join(path.name for path in selected)
    logger.info(
        "Background clip order (%d clip(s), total %.2fs): %s",
        len(selected),
        total_duration,
        selected_summary,
    )
    return selected, total_duration


def _select_background_clips(background_dir: Path, target_duration: float) -> list[Path]:
    """Backward-compatible wrapper around background timeline builder."""
    selected, _ = build_background_timeline(background_dir, target_duration)
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
    background_clips: list[Path],
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
    if not background_clips:
        raise RuntimeError("No background clips were selected for composition.")

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error" if not verbose else "info",
        "-y",
    ]
    for clip in background_clips:
        command.extend(["-i", str(clip)])
    command.extend(["-i", audio_path])

    normalized_labels: list[str] = []
    filter_parts: list[str] = []
    for idx in range(len(background_clips)):
        label = f"v{idx}"
        normalized_labels.append(f"[{label}]")
        filter_parts.append(
            f"[{idx}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,setsar=1[{label}]"
        )
    concat_label = "bg" if len(normalized_labels) > 1 else "v0"
    if len(normalized_labels) > 1:
        filter_parts.append(
            "".join(normalized_labels) + f"concat=n={len(normalized_labels)}:v=1:a=0[{concat_label}]"
        )
    target_duration = max(0.01, audio_duration)
    filter_parts.append(
        f"[{concat_label}]trim=duration={target_duration:.3f},"
        "setpts=PTS-STARTPTS,"
        f"subtitles='{escaped_subs}',"
        "format=yuv420p[vout]"
    )
    filter_complex = ";".join(filter_parts)

    audio_input_idx = len(background_clips)
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            f"{audio_input_idx}:a:0",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ]
    )

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

    runtime_cfg = _load_runtime_config()
    final_script = (content.final_script or content.narration or "").strip()
    if not final_script:
        raise RuntimeError(f"Cannot render {stem}: final_script is empty.")

    audio_path = _generate_tts(final_script, root / "audio" / f"{stem}.wav")

    # Derive exact duration from the generated WAV for accurate subtitle sync
    audio_duration = _get_wav_duration(audio_path)
    if audio_duration <= 0:
        logger.warning(
            "Could not determine audio duration for %r; subtitle timing will use estimate.",
            audio_path,
        )

    if runtime_cfg.caption.style_mode in {"active_word", "plain"}:
        subtitles_path = _generate_ass_subtitles(
            caption_script=final_script,
            output_path=root / "subs" / f"{stem}.ass",
            audio_duration=audio_duration if audio_duration > 0 else _estimate_duration_seconds(final_script),
            caption_cfg=runtime_cfg.caption,
        )
    else:
        subtitles_path = _generate_subtitles(
            final_script,
            root / "subs" / f"{stem}.srt",
            audio_duration=audio_duration if audio_duration > 0 else None,
        )
    background_dir = Path(os.getenv("BACKGROUND_CLIPS_DIR", "background_clips"))
    if not background_dir.is_absolute():
        background_dir = Path.cwd() / background_dir
    selected_backgrounds, assembled_duration = build_background_timeline(
        background_dir=background_dir,
        target_duration=audio_duration if audio_duration > 0 else _estimate_duration_seconds(final_script),
        safety_buffer_seconds=runtime_cfg.background_safety_buffer_seconds,
        rng_seed=runtime_cfg.background_rng_seed,
        randomize=runtime_cfg.background_randomize,
        allow_immediate_repeats=runtime_cfg.allow_immediate_background_repeats,
    )
    logger.info(
        "Background timeline assembled %.2fs for target %.2fs (buffer %.2fs).",
        assembled_duration,
        audio_duration if audio_duration > 0 else _estimate_duration_seconds(final_script),
        runtime_cfg.background_safety_buffer_seconds,
    )
    verbose = os.getenv("FFMPEG_VERBOSE", "").strip().lower() in ("1", "true", "yes")
    try:
        timeout = int(os.getenv("FFMPEG_TIMEOUT", "300"))
    except ValueError:
        timeout = 300

    video_path = _compose_video(
        audio_path=audio_path,
        subtitles_path=subtitles_path,
        background_clips=selected_backgrounds,
        output_path=root / "videos" / f"{stem}.mp4",
        audio_duration=audio_duration,
        timeout=timeout,
        verbose=verbose,
    )
    output_duration = _probe_media_duration_seconds(Path(video_path))
    logger.info(
        "Render stats for %s: final_script_len=%d audio_duration=%.2fs background_duration=%.2fs output_duration=%.2fs",
        stem,
        len(final_script),
        audio_duration,
        assembled_duration,
        output_duration,
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
