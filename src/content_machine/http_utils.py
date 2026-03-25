"""HTTP helpers with retries and actionable error logging."""

from __future__ import annotations

from email.utils import parsedate_to_datetime
import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _retry_after_seconds(header_value: str | None) -> float:
    if not header_value:
        return 0.0

    cleaned = header_value.strip()
    if cleaned.isdigit():
        return float(cleaned)

    try:
        retry_at = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, retry_at.timestamp() - time.time())


def request_json_with_retries(
    request: Request,
    *,
    operation: str,
    timeout: int = 20,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    """Execute an HTTP request with retry logic and structured error logs."""

    if max_attempts < 1:
        max_attempts = 1

    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except HTTPError as err:
            status_code = int(getattr(err, "code", 0))
            body = ""
            try:
                body = err.read().decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""

            if status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
                retry_after = _retry_after_seconds(err.headers.get("Retry-After"))
                delay = retry_after if retry_after > 0 else backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s rate-limited/transient error (status=%s) on attempt %s/%s; retrying in %.2fs.",
                    operation,
                    status_code,
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)
                continue

            logger.error(
                "%s failed with HTTP %s on attempt %s/%s. Response body: %s",
                operation,
                status_code,
                attempt,
                max_attempts,
                body or "<empty>",
            )
            return {}
        except (URLError, TimeoutError) as err:
            if attempt < max_attempts:
                delay = backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "%s network error on attempt %s/%s: %s. Retrying in %.2fs.",
                    operation,
                    attempt,
                    max_attempts,
                    err,
                    delay,
                )
                time.sleep(delay)
                continue

            logger.error("%s failed after %s attempts due to network error: %s", operation, max_attempts, err)
            return {}
        except json.JSONDecodeError as err:
            logger.error("%s returned invalid JSON payload: %s", operation, err)
            return {}
        except Exception as err:  # noqa: BLE001
            logger.exception("%s failed due to unexpected error: %s", operation, err)
            return {}

    logger.error("%s failed without a successful response.", operation)
    return {}
