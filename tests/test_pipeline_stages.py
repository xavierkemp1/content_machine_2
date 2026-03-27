import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from content_machine import exporting, filtering, improvement, pipeline, production, sourcing
from content_machine.models import EnhancedContent, ProductionArtifact, RawPost, RankedPost


class TestPipelineStages(unittest.TestCase):
    def test_sourcing_collect_combines_sources(self):
        original_reddit = sourcing.fetch_reddit_posts
        original_x = sourcing.fetch_x_posts
        try:
            sourcing.fetch_reddit_posts = lambda: [RawPost(source="reddit", source_id="r1", author="a", text="hello world one two three four five six seven eight nine ten")]
            sourcing.fetch_x_posts = lambda: [RawPost(source="x", source_id="x1", author="b", text="hello world one two three four five six seven eight nine ten")]
            posts = sourcing.collect_raw_posts()
        finally:
            sourcing.fetch_reddit_posts = original_reddit
            sourcing.fetch_x_posts = original_x

        self.assertEqual(2, len(posts))
        self.assertEqual({"reddit", "x"}, {p.source for p in posts})

    def test_filtering_apply_rules_and_rank(self):
        raw_posts = [
            RawPost(source="reddit", source_id="1", author="a", text="This is a long enough story with many words to pass minimum limit and be unique for testing purposes right now", metrics={"score": 100, "comments": 20, "has_media": False}),
            RawPost(source="reddit", source_id="2", author="a", text="This is a long enough story with many words to pass minimum limit and be unique for testing purposes right now", metrics={"score": 95, "comments": 18, "has_media": False}),
        ]
        filtered = filtering.apply_rules(raw_posts)
        self.assertEqual(1, len(filtered), "duplicate posts should be removed")

        ranked = filtering.rank_for_virality(filtered)
        self.assertEqual(1, len(ranked))
        self.assertGreaterEqual(ranked[0].viral_score, 0.0)

    def test_improvement_enhance_posts_fallback(self):
        ranked = RankedPost(
            raw=RawPost(
                source="reddit",
                source_id="abc",
                author="user",
                text="I quit my job and started a tiny business that changed my life in six months.",
                metrics={},
            ),
            length_bucket="short",
            viral_score=52.0,
        )

        enhanced = improvement.enhance_posts([ranked])
        self.assertEqual(1, len(enhanced))
        self.assertTrue(enhanced[0].hook)
        self.assertTrue(enhanced[0].narration)
        self.assertTrue(enhanced[0].hashtags)

    def test_production_render_video_creates_assets(self):
        content = EnhancedContent(
            source_post=RankedPost(
                raw=RawPost(source="x", source_id="42", author="a", text="Story text with enough words for rendering path coverage in tests.", metrics={"account": "business"}),
                length_bucket="short",
                viral_score=70.0,
            ),
            title="Title",
            hook="Hook",
            narration="This is narration text for generating subtitles and tts stub.",
            caption="Caption",
            hashtags=["#test"],
        )
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not available in this environment")
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = production.render_video(content, work_dir=tmpdir)
            self.assertTrue(Path(artifact.audio_path).exists())
            self.assertTrue(Path(artifact.subtitles_path).exists())
            self.assertTrue(Path(artifact.video_path).exists())

    def test_exporting_writes_metadata(self):
        content = EnhancedContent(
            source_post=RankedPost(
                raw=RawPost(source="reddit", source_id="55", author="u", text="some text some text some text some text some text", metrics={"subreddit": "aita"}),
                length_bucket="short",
                viral_score=65.0,
            ),
            title="My title",
            hook="My hook",
            narration="Narration",
            caption="Caption",
            hashtags=["#one"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "v.mp4"
            subs = Path(tmpdir) / "s.srt"
            video.write_bytes(b"fake mp4 content")  # non-empty so export guard passes
            subs.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            artifact = ProductionArtifact(video_path=str(video), subtitles_path=str(subs), metadata_path="", audio_path="a.wav")
            exported = exporting.export_outputs([(content, artifact)], base_dir=tmpdir)

            self.assertEqual(1, len(exported))
            metadata_path = Path(exported[0].metadata_path)
            self.assertTrue(metadata_path.exists())
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual("My title", payload["title"])

    def test_exporting_raises_on_empty_video(self):
        """export_outputs must raise RuntimeError when the source video is 0 bytes."""
        content = EnhancedContent(
            source_post=RankedPost(
                raw=RawPost(source="reddit", source_id="99", author="u", text="text", metrics={"subreddit": "test"}),
                length_bucket="short",
                viral_score=50.0,
            ),
            title="T",
            hook="H",
            narration="N",
            caption="C",
            hashtags=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "empty.mp4"
            subs = Path(tmpdir) / "s.srt"
            video.write_bytes(b"")  # 0-byte file
            subs.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            artifact = ProductionArtifact(video_path=str(video), subtitles_path=str(subs), metadata_path="", audio_path="a.wav")
            with self.assertRaises(RuntimeError):
                exporting.export_outputs([(content, artifact)], base_dir=tmpdir)

    def test_exporting_raises_on_missing_video(self):
        """export_outputs must raise RuntimeError when the source video does not exist."""
        content = EnhancedContent(
            source_post=RankedPost(
                raw=RawPost(source="reddit", source_id="100", author="u", text="text", metrics={"subreddit": "test"}),
                length_bucket="short",
                viral_score=50.0,
            ),
            title="T",
            hook="H",
            narration="N",
            caption="C",
            hashtags=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = ProductionArtifact(video_path=str(Path(tmpdir) / "nonexistent.mp4"), subtitles_path="", metadata_path="", audio_path="")
            with self.assertRaises(RuntimeError):
                exporting.export_outputs([(content, artifact)], base_dir=tmpdir)

    def test_escape_subtitles_filter_path_no_backslashes(self):
        """_escape_subtitles_filter_path returns a path with no unescaped backslashes."""
        from content_machine.production import _escape_subtitles_filter_path
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            tmp_path = f.name
        try:
            result = _escape_subtitles_filter_path(tmp_path)
            # Remove escaped colons before checking for stray backslashes
            cleaned = result.replace(r"\:", "")
            self.assertNotIn("\\", cleaned)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_escape_subtitles_filter_path_drive_letter_colon(self):
        """Drive-letter colon is escaped when the resolved path contains one."""
        from content_machine.production import _escape_subtitles_filter_path
        import os
        if os.name != "nt":
            self.skipTest("Drive-letter colon escaping is only relevant on Windows")
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            tmp_path = f.name
        try:
            result = _escape_subtitles_filter_path(tmp_path)
            # e.g. C\:/Users/... — the drive letter colon must be escaped
            self.assertRegex(result, r"^[A-Za-z]\\:")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pipeline_calls_all_stages(self):
        called = []
        original_collect = pipeline.collect_raw_posts
        original_apply = pipeline.apply_rules
        original_rank = pipeline.rank_for_virality
        original_enhance = pipeline.enhance_posts
        original_produce = pipeline.produce_all
        original_export = pipeline.export_outputs
        try:
            pipeline.collect_raw_posts = lambda: called.append("collect") or [RawPost(source="x", source_id="1", author="a", text="one two three four five six seven eight nine ten eleven", metrics={})]
            pipeline.apply_rules = lambda posts: called.append("filter") or posts
            pipeline.rank_for_virality = lambda posts: called.append("rank") or [RankedPost(raw=posts[0], length_bucket="short", viral_score=10.0)]
            pipeline.enhance_posts = lambda posts: called.append("enhance") or [EnhancedContent(source_post=posts[0], title="t", hook="h", narration="narration words", caption="c", hashtags=["#x"])]
            pipeline.produce_all = lambda contents: called.append("produce") or [ProductionArtifact(video_path="v", subtitles_path="s", metadata_path="")]
            pipeline.export_outputs = lambda items: called.append("export") or []

            pipeline.run_pipeline()
        finally:
            pipeline.collect_raw_posts = original_collect
            pipeline.apply_rules = original_apply
            pipeline.rank_for_virality = original_rank
            pipeline.enhance_posts = original_enhance
            pipeline.produce_all = original_produce
            pipeline.export_outputs = original_export

        self.assertEqual(["collect", "filter", "rank", "enhance", "produce", "export"], called)

    # ------------------------------------------------------------------
    # Piper TTS tests
    # ------------------------------------------------------------------

    def test_generate_tts_no_piper_exe_uses_stub(self):
        """When PIPER_EXE is not set, _generate_tts falls back to the silent stub."""
        from content_machine.production import _generate_tts
        env_backup = os.environ.pop("PIPER_EXE", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir) / "audio" / "test.wav"
                result = _generate_tts("Hello world", out)
                self.assertEqual(result, str(out))
                self.assertTrue(out.exists())
                self.assertGreater(out.stat().st_size, 0)
        finally:
            if env_backup is not None:
                os.environ["PIPER_EXE"] = env_backup

    def test_generate_tts_missing_exe_warns_and_uses_stub(self):
        """When PIPER_EXE points to a non-existent file, log warning and use stub."""
        from content_machine.production import _generate_tts
        import logging
        env_backup = os.environ.get("PIPER_EXE")
        os.environ["PIPER_EXE"] = "/nonexistent/path/to/piper"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir) / "audio" / "test.wav"
                with self.assertLogs("content_machine.production", level=logging.WARNING):
                    result = _generate_tts("Hello world", out)
                self.assertEqual(result, str(out))
                self.assertTrue(out.exists())
        finally:
            if env_backup is None:
                os.environ.pop("PIPER_EXE", None)
            else:
                os.environ["PIPER_EXE"] = env_backup

    # ------------------------------------------------------------------
    # OpenAI ranking response parsing tests
    # ------------------------------------------------------------------

    def test_openai_rank_output_text_field(self):
        """_openai_rank correctly parses the legacy output_text field."""
        from content_machine import filtering as flt
        scores_json = json.dumps({"scores": [{"source_id": "abc", "score": 80.0, "reasons": {}}]})
        fake_body = {"output_text": scores_json}

        original_fn = flt.request_json_with_retries
        flt.request_json_with_retries = lambda *a, **kw: fake_body
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            post = RawPost(source="reddit", source_id="abc", author="u", text="x " * 20, metrics={})
            result = flt._openai_rank([post])
            self.assertIn("abc", result)
            self.assertAlmostEqual(result["abc"]["score"], 80.0)
        finally:
            flt.request_json_with_retries = original_fn
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_openai_rank_output_array_field(self):
        """_openai_rank correctly parses the modern output array format."""
        from content_machine import filtering as flt
        scores_json = json.dumps({"scores": [{"source_id": "xyz", "score": 90.0, "reasons": {}}]})
        fake_body = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": scores_json}
                    ],
                }
            ]
        }

        original_fn = flt.request_json_with_retries
        flt.request_json_with_retries = lambda *a, **kw: fake_body
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            post = RawPost(source="reddit", source_id="xyz", author="u", text="x " * 20, metrics={})
            result = flt._openai_rank([post])
            self.assertIn("xyz", result)
            self.assertAlmostEqual(result["xyz"]["score"], 90.0)
        finally:
            flt.request_json_with_retries = original_fn
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_openai_rank_missing_output_falls_back(self):
        """_openai_rank returns empty dict when no output is present."""
        from content_machine import filtering as flt
        fake_body = {"id": "resp_123", "model": "gpt-4.1-mini"}

        original_fn = flt.request_json_with_retries
        flt.request_json_with_retries = lambda *a, **kw: fake_body
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            post = RawPost(source="reddit", source_id="nope", author="u", text="x " * 20, metrics={})
            import logging
            with self.assertLogs("content_machine.filtering", level=logging.ERROR):
                result = flt._openai_rank([post])
            self.assertEqual(result, {})
        finally:
            flt.request_json_with_retries = original_fn
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_openai_rank_invalid_json_falls_back(self):
        """_openai_rank returns empty dict when output text is not valid JSON."""
        from content_machine import filtering as flt
        fake_body = {"output_text": "not valid json {{{{"}

        original_fn = flt.request_json_with_retries
        flt.request_json_with_retries = lambda *a, **kw: fake_body
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key"
        try:
            post = RawPost(source="reddit", source_id="bad", author="u", text="x " * 20, metrics={})
            import logging
            with self.assertLogs("content_machine.filtering", level=logging.ERROR):
                result = flt._openai_rank([post])
            self.assertEqual(result, {})
        finally:
            flt.request_json_with_retries = original_fn
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key


if __name__ == "__main__":
    unittest.main()
