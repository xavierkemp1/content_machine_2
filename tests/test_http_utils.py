import io
import json
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request

from content_machine import http_utils


class TestHttpUtils(unittest.TestCase):
    def test_retries_then_succeeds_on_rate_limit(self):
        calls = {"count": 0}
        slept: list[float] = []

        class FakeResponse:
            def __init__(self, payload: dict):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout=20):
            calls["count"] += 1
            if calls["count"] == 1:
                raise HTTPError(
                    request.full_url,
                    429,
                    "Too Many Requests",
                    {"Retry-After": "0"},
                    io.BytesIO(b'{"error":"rate_limited"}'),
                )
            return FakeResponse({"ok": True})

        original_urlopen = http_utils.urlopen
        original_sleep = http_utils.time.sleep
        try:
            http_utils.urlopen = fake_urlopen
            http_utils.time.sleep = lambda delay: slept.append(delay)
            result = http_utils.request_json_with_retries(
                Request("https://example.com"),
                operation="rate-limit test",
                max_attempts=2,
                backoff_seconds=0.01,
            )
        finally:
            http_utils.urlopen = original_urlopen
            http_utils.time.sleep = original_sleep

        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, calls["count"])
        self.assertEqual([0.01], slept)

    def test_returns_empty_after_network_failures(self):
        calls = {"count": 0}
        slept: list[float] = []

        def fake_urlopen(_request, timeout=20):
            calls["count"] += 1
            raise URLError("dns failure")

        original_urlopen = http_utils.urlopen
        original_sleep = http_utils.time.sleep
        try:
            http_utils.urlopen = fake_urlopen
            http_utils.time.sleep = lambda delay: slept.append(delay)
            result = http_utils.request_json_with_retries(
                Request("https://example.com"),
                operation="network test",
                max_attempts=3,
                backoff_seconds=0.01,
            )
        finally:
            http_utils.urlopen = original_urlopen
            http_utils.time.sleep = original_sleep

        self.assertEqual({}, result)
        self.assertEqual(3, calls["count"])
        self.assertEqual([0.01, 0.02], slept)


if __name__ == "__main__":
    unittest.main()
