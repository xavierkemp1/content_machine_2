"""ElevenLabs TTS client — speech generation and forced-alignment word timestamps."""

from __future__ import annotations

import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_CHUNK_SIZE = 8192


class ElevenLabsClient:
    """Thin stdlib-only client for ElevenLabs TTS and forced-alignment APIs."""

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_request(self, url: str, body: dict) -> Request:
        payload = json.dumps(body).encode("utf-8")
        req = Request(url, data=payload, method="POST")
        req.add_header("xi-api-key", self._api_key)
        req.add_header("Content-Type", "application/json")
        return req

    def _voice_body(self, text: str) -> dict:
        return {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
            },
        }

    # ------------------------------------------------------------------
    # Speech generation
    # ------------------------------------------------------------------

    def generate_speech(self, text: str, output_path: str) -> str:
        """Generate speech for *text* and stream the MP3 to *output_path*.

        Returns *output_path* on success.  Raises ``RuntimeError`` after
        three failed attempts (retries on transient HTTP errors).
        """
        url = f"{_ELEVENLABS_BASE_URL}/{self._voice_id}"
        req = self._build_request(url, self._voice_body(text))
        logger.info("Generating speech…")

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                with urlopen(req, timeout=120) as response:
                    import pathlib
                    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as fh:
                        while True:
                            chunk = response.read(_CHUNK_SIZE)
                            if not chunk:
                                break
                            fh.write(chunk)
                logger.info("ElevenLabs TTS generated %s", output_path)
                return output_path
            except HTTPError as exc:
                status = int(getattr(exc, "code", 0))
                if status in _RETRYABLE_STATUS_CODES and attempt < 3:
                    delay = 1.0 * (2 ** (attempt - 1))
                    logger.warning(
                        "ElevenLabs TTS transient error (status=%s) on attempt %s/3; retrying in %.2fs.",
                        status,
                        attempt,
                        delay,
                    )
                    time.sleep(delay)
                    last_exc = exc
                    continue
                last_exc = exc
                break
            except (URLError, OSError) as exc:
                if attempt < 3:
                    delay = 1.0 * (2 ** (attempt - 1))
                    logger.warning(
                        "ElevenLabs TTS network error on attempt %s/3: %s. Retrying in %.2fs.",
                        attempt,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    last_exc = exc
                    continue
                last_exc = exc
                break

        raise RuntimeError(
            f"ElevenLabs TTS failed after 3 attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Forced alignment
    # ------------------------------------------------------------------

    def get_alignment(self, audio_path: str, text: str) -> list[dict]:  # noqa: ARG002
        """Fetch word-level timing data from the ElevenLabs alignment endpoint.

        *audio_path* is accepted for API symmetry but the alignment call re-
        synthesises audio server-side; the local file is not uploaded.

        Returns a list of ``{"word": str, "start": float, "end": float}``
        dicts on success, or ``[]`` on failure (caller should fall back).
        """
        url = f"{_ELEVENLABS_BASE_URL}/{self._voice_id}/with-timestamps"
        req = self._build_request(url, self._voice_body(text))
        logger.info("Fetching ElevenLabs alignment…")

        for attempt in range(1, 4):
            try:
                with urlopen(req, timeout=120) as response:
                    body = response.read().decode("utf-8")
                data = json.loads(body)
                word_timings = _parse_alignment(data)
                logger.info("Alignment received: %d words", len(word_timings))
                return word_timings
            except HTTPError as exc:
                status = int(getattr(exc, "code", 0))
                if status in _RETRYABLE_STATUS_CODES and attempt < 3:
                    delay = 1.0 * (2 ** (attempt - 1))
                    logger.warning(
                        "ElevenLabs alignment transient error (status=%s) on attempt %s/3; retrying in %.2fs.",
                        status,
                        attempt,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                logger.warning(
                    "ElevenLabs alignment failed with HTTP %s on attempt %s/3.",
                    status,
                    attempt,
                )
                return []
            except (URLError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                if attempt < 3:
                    delay = 1.0 * (2 ** (attempt - 1))
                    logger.warning(
                        "ElevenLabs alignment error on attempt %s/3: %s. Retrying in %.2fs.",
                        attempt,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                logger.warning("ElevenLabs alignment failed after 3 attempts: %s", exc)
                return []

        return []


# ---------------------------------------------------------------------------
# Character → word timing parser
# ---------------------------------------------------------------------------


def _parse_alignment(data: dict) -> list[dict]:
    """Convert ElevenLabs character-level alignment data into word-level timings.

    The response structure is::

        {
          "alignment": {
            "characters": ["H","e","l","l","o"," ","w","o","r","l","d"],
            "character_start_times_seconds": [0.0, 0.05, ...],
            "character_end_times_seconds":   [0.05, 0.10, ...]
          }
        }

    Characters are grouped into words by splitting on whitespace/punctuation
    boundaries.  Each word's ``start`` is the start of its first character and
    ``end`` is the end of its last character.
    """
    alignment = data.get("alignment", {})
    characters: list[str] = alignment.get("characters", [])
    starts: list[float] = alignment.get("character_start_times_seconds", [])
    ends: list[float] = alignment.get("character_end_times_seconds", [])

    if not characters or len(characters) != len(starts) or len(characters) != len(ends):
        return []

    word_timings: list[dict] = []
    current_chars: list[str] = []
    current_start: float | None = None
    current_end: float = 0.0

    for char, start, end in zip(characters, starts, ends):
        if char.isspace():
            # Flush the current word (if any)
            if current_chars and current_start is not None:
                word = "".join(current_chars)
                word_timings.append({"word": word, "start": current_start, "end": current_end})
            current_chars = []
            current_start = None
        else:
            if current_start is None:
                current_start = start
            current_chars.append(char)
            current_end = end

    # Flush last word
    if current_chars and current_start is not None:
        word_timings.append({"word": "".join(current_chars), "start": current_start, "end": current_end})

    return word_timings
