"""Stage 4: produce vertical videos from enhanced content."""

from __future__ import annotations

import hashlib
import json
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
    words_per_chunk: int = 4
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

    _default_voice = ""
    voice_profile = os.getenv("VOICE_PROFILE", _default_voice).strip()
    highlight_color = os.getenv("CAPTION_ACTIVE_WORD_COLOR", "&H0038FF&").strip() or "&H0038FF&"

    _jitter_rng = random.Random()  # unseeded = different each run
    font_size_jitter = _jitter_rng.randint(-3, 3)
    margin_v_jitter = _jitter_rng.randint(-8, 8)
    logger.debug("Caption jitter: font_size_jitter=%d margin_v_jitter=%d", font_size_jitter, margin_v_jitter)

    return ProductionRuntimeConfig(
        background_safety_buffer_seconds=safety_buffer,
        background_randomize=randomize,
        allow_immediate_background_repeats=allow_repeats,
        background_rng_seed=background_rng_seed,
        voice_profile=voice_profile,
        caption=CaptionRenderConfig(
            style_mode=style_mode,
            active_word_highlight_color=highlight_color,
            font_size=62 + font_size_jitter,
            margin_v=210 + margin_v_jitter,
        ),
    )


def _generate_tts(narration: str, output_path: Path) -> str:
    """Generate TTS audio via ElevenLabs when configured, otherwise fall back to the silent stub.

    Environment variables
    ---------------------
    ELEVENLABS_API_KEY   ElevenLabs API key.
    ELEVENLABS_VOICE_ID  Voice ID to use for synthesis.
    ELEVENLABS_MODEL_ID  Model to use (default: eleven_multilingual_v2).

    When either key is absent the silent WAV stub is used (offline-safe).
    The output will be MP3 when ElevenLabs is used; WAV for the stub.
    """
    from .elevenlabs_client import ElevenLabsClient  # local import to keep module load light

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

    if not api_key or not voice_id:
        logger.info(
            "ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID is not set — using silent TTS stub. "
            "To enable real voice synthesis set both variables in your .env file."
        )
        stub_path = output_path.with_suffix(".wav")
        return _generate_tts_stub(narration, stub_path)

    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip() or "eleven_multilingual_v2"
    mp3_path = output_path.with_suffix(".mp3")
    client = ElevenLabsClient(api_key=api_key, voice_id=voice_id, model_id=model_id)
    try:
        return client.generate_speech(narration, str(mp3_path))
    except RuntimeError as exc:
        logger.warning(
            "ElevenLabs TTS failed (%s) — falling back to silent TTS stub.",
            exc,
        )
        stub_path = output_path.with_suffix(".wav")
        return _generate_tts_stub(narration, stub_path)


def _trim_audio_silence(input_path: Path, output_path: Path) -> Path:
    """Use ffmpeg silenceremove to strip leading/trailing and inter-sentence silence.

    The *output_path* suffix is replaced with the *input_path* suffix so that
    MP3 → MP3 and WAV → WAV trimming both work correctly.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return input_path
    if os.getenv("TRIM_SILENCE", "1").strip().lower() not in {"1", "true", "yes"}:
        return input_path
    # Preserve the input format (e.g. .mp3 or .wav) in the output path
    trimmed_path = output_path.with_suffix(input_path.suffix)
    try:
        result = subprocess.run(
            [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_path),
                "-af",
                # stop_periods=-1: remove all silence segments (not just leading/trailing)
                # stop_duration=0.3: minimum silence duration (seconds) to remove
                # stop_threshold=-40dB: audio below -40 dB is considered silence
                "silenceremove=stop_periods=-1:stop_duration=0.3:stop_threshold=-40dB",
                str(trimmed_path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and trimmed_path.exists() and trimmed_path.stat().st_size > 0:
            logger.info("Silence trimmed: %s -> %s", input_path.name, trimmed_path.name)
            return trimmed_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Silence trim failed (%s); using original audio.", exc)
    return input_path


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


_NATURAL_BREAK_CONJUNCTIONS = {"and", "but", "so", "or", "because", "that", "when", "if"}


def _smart_chunk_words(words: list[str], target_size: int) -> list[list[str]]:
    """Split words into chunks aligned with natural speech pauses.

    A chunk boundary is inserted when:
    - the current chunk has reached *target_size* words, OR
    - the current chunk has at least 2 words AND the next word is a natural
      break conjunction (``and``, ``but``, ``so``, etc.) that follows a
      comma-terminated word.
    All input words are preserved; none are dropped.
    """
    if not words:
        return []
    chunks: list[list[str]] = []
    current: list[str] = []
    for word in words:
        if current and len(current) >= target_size:
            chunks.append(current)
            current = [word]
            continue
        if (
            len(current) >= 2
            and word.lower().rstrip(",.!?") in _NATURAL_BREAK_CONJUNCTIONS
            and current[-1].endswith(",")
        ):
            chunks.append(current)
            current = [word]
            continue
        current.append(word)
    if current:
        chunks.append(current)
    return chunks


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
    chunks = _smart_chunk_words(words, target_size=max(2, caption_cfg.words_per_chunk))
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
            style = "Hook" if idx <= 1 else "Default"
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
    ass = "\n".join([*_ass_header(caption_cfg), *dialogue_lines, ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ass, encoding="utf-8")
    return str(output_path)


def _ass_header(caption_cfg: CaptionRenderConfig) -> list[str]:
    """Return the ASS [Script Info] + [V4+ Styles] + [Events] header lines."""
    return [
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
        "1,0,0,0,100,100,0,0,1,2.4,1.3,5,80,80,"
        "0,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]


def _generate_ass_subtitles_from_alignment(
    word_timings: list[dict],
    output_path: Path,
    caption_cfg: CaptionRenderConfig,
) -> str:
    """Generate an ASS subtitle file using exact word timestamps from the alignment API.

    Parameters
    ----------
    word_timings:
        List of ``{"word": str, "start": float, "end": float}`` dicts.
    output_path:
        Destination ``.ass`` file path.
    caption_cfg:
        Caption rendering options.

    Returns ``str(output_path)``.
    """
    words = [t["word"] for t in word_timings]
    chunks = _smart_chunk_words(words, target_size=max(2, caption_cfg.words_per_chunk))

    dialogue_lines: list[str] = []
    word_cursor = 0

    for chunk_idx, chunk_words in enumerate(chunks):
        chunk_timings = word_timings[word_cursor : word_cursor + len(chunk_words)]
        word_cursor += len(chunk_words)

        if not chunk_timings:
            continue

        style = "Hook" if chunk_idx <= 1 else "Default"

        for word_idx, (word, timing) in enumerate(zip(chunk_words, chunk_timings)):
            w_start = timing["start"]
            w_end = timing["end"]

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
            dialogue_lines.append(
                "Dialogue: 0,"
                f"{_seconds_to_ass_time(w_start)},{_seconds_to_ass_time(w_end)},"
                f"{style},,0,0,0,,{rendered_text}"
            )

    ass = "\n".join([*_ass_header(caption_cfg), *dialogue_lines, ""])
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


def _short_hash(text: str) -> str:
    """Return a short (12-char) hex SHA-256 of *text* for compact log identifiers."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def render_video(content: EnhancedContent, work_dir: str = "output/_build") -> ProductionArtifact:
    """Create TTS, subtitles, and compose a 9:16 background video artifact.

    Subtitle timings are derived from the *actual* WAV duration so captions
    sync perfectly to the voice-over.  ffmpeg is constrained to that same
    duration via ``-t`` to prevent indefinite hangs.

    Environment variables
    ---------------------
    FFMPEG_TIMEOUT    Seconds before ffmpeg is killed (default: 300).
    FFMPEG_VERBOSE    Set to ``1`` / ``true`` / ``yes`` to stream ffmpeg output
                      to the console instead of capturing it silently.
    STRICT_TEXT_SYNC  Default ``1``.  When enabled, aborts if the exact string
                      passed to TTS differs from the one passed to caption
                      generation.  Set to ``0`` to log the mismatch and continue.
    """

    root = Path(work_dir)
    stem = f"{content.source_post.raw.source}-{_slugify(content.source_post.raw.source_id)}"

    runtime_cfg = _load_runtime_config()
    final_script = (content.final_script or content.narration or "").strip()
    if not final_script:
        raise RuntimeError(f"Cannot render {stem}: final_script is empty.")

    # ── PART 3/4: capture exact runtime TTS input ──────────────────────────
    exact_tts_input: str = final_script
    raw_audio_path = _generate_tts(exact_tts_input, root / "audio" / f"{stem}.wav")
    trimmed_audio_path = _trim_audio_silence(
        Path(raw_audio_path), root / "audio" / f"{stem}.trimmed.wav"
    )
    audio_path = str(trimmed_audio_path)

    # Derive exact duration from the generated audio for accurate subtitle sync
    audio_duration = _probe_media_duration_seconds(Path(audio_path))
    if audio_duration <= 0:
        logger.warning(
            "Could not determine audio duration for %r; subtitle timing will use estimate.",
            audio_path,
        )

    # ── PART 3/4: capture exact runtime caption input ──────────────────────
    exact_caption_input: str = final_script
    if runtime_cfg.caption.style_mode in {"active_word", "plain"}:
        # Attempt ElevenLabs forced alignment for exact word timestamps
        el_api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        el_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
        word_timings: list[dict] = []
        if el_api_key and el_voice_id:
            from .elevenlabs_client import ElevenLabsClient
            el_model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip() or "eleven_multilingual_v2"
            el_client = ElevenLabsClient(api_key=el_api_key, voice_id=el_voice_id, model_id=el_model_id)
            word_timings = el_client.get_alignment(audio_path, exact_caption_input)

        if word_timings:
            subtitles_path = _generate_ass_subtitles_from_alignment(
                word_timings=word_timings,
                output_path=root / "subs" / f"{stem}.ass",
                caption_cfg=runtime_cfg.caption,
            )
        else:
            if el_api_key and el_voice_id:
                logger.info("ElevenLabs alignment unavailable, falling back to estimated caption timing")
            subtitles_path = _generate_ass_subtitles(
                caption_script=exact_caption_input,
                output_path=root / "subs" / f"{stem}.ass",
                audio_duration=audio_duration if audio_duration > 0 else _estimate_duration_seconds(final_script),
                caption_cfg=runtime_cfg.caption,
            )
    else:
        subtitles_path = _generate_subtitles(
            exact_caption_input,
            root / "subs" / f"{stem}.srt",
            audio_duration=audio_duration if audio_duration > 0 else None,
        )

    # ── PART 4: STRICT_TEXT_SYNC enforcement ───────────────────────────────
    strict_sync = os.getenv("STRICT_TEXT_SYNC", "1").strip().lower() not in {"0", "false", "no"}
    inputs_identical = exact_tts_input == exact_caption_input
    if not inputs_identical:
        tts_hash = _short_hash(exact_tts_input)
        caption_hash = _short_hash(exact_caption_input)
        msg = (
            f"TEXT SYNC MISMATCH for {stem}: "
            f"TTS input (hash={tts_hash}) != caption input (hash={caption_hash}). "
            "These strings must be identical for spoken audio and visible captions to match."
        )
        if strict_sync:
            raise RuntimeError(msg)
        logger.error("STRICT_TEXT_SYNC=0 — continuing despite mismatch. %s", msg)

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

    # ── PART 3: write debug JSON artifact ──────────────────────────────────
    subtitle_actual_path = Path(subtitles_path)
    subtitle_actual_format = subtitle_actual_path.suffix.lstrip(".").lower()
    debug_payload: dict = {
        "final_script": final_script,
        "exact_tts_input_text": exact_tts_input,
        "exact_caption_input_text": exact_caption_input,
        "tts_input_hash": _short_hash(exact_tts_input),
        "caption_input_hash": _short_hash(exact_caption_input),
        "inputs_identical": inputs_identical,
        "subtitle_file_actual_path": str(subtitle_actual_path),
        "subtitle_file_actual_format": subtitle_actual_format,
        "output_video_path": video_path,
        "audio_path": audio_path,
        "source_id": content.source_post.raw.source_id,
    }
    debug_dir = root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{stem}.debug.json"
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    logger.info("Debug artifact written to %s (inputs_identical=%s)", debug_path, inputs_identical)

    logger.info(
        "Production complete for %r. Build artifacts in %s; "
        "final exports expected in output/ organised by theme after export stage.",
        stem,
        root,
    )

    return ProductionArtifact(
        video_path=video_path,
        subtitles_path=subtitles_path,
        metadata_path=str(debug_path),
        audio_path=audio_path,
        background_path="|".join(str(path) for path in selected_backgrounds),
    )


def produce_all(contents: list[EnhancedContent], work_dir: str = "output/_build") -> list[ProductionArtifact]:
    """Run rendering for all enhanced content objects."""

    artifacts: list[ProductionArtifact] = []
    for item in contents:
        artifacts.append(render_video(item, work_dir=work_dir))
    return artifacts
