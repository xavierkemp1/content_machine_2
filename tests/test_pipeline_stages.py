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

    def test_select_background_clips_reaches_target_duration(self):
        from content_machine import production as prod

        with tempfile.TemporaryDirectory() as tmpdir:
            bg_dir = Path(tmpdir) / "background_clips"
            bg_dir.mkdir(parents=True, exist_ok=True)
            for name in ("a.mp4", "b.mp4"):
                (bg_dir / name).write_bytes(b"fake")

            original_probe = prod._probe_media_duration_seconds
            try:
                prod._probe_media_duration_seconds = lambda _path: 1.5
                clips, assembled = prod.build_background_timeline(
                    bg_dir,
                    target_duration=4.0,
                    safety_buffer_seconds=0.5,
                    rng_seed=1,
                )
            finally:
                prod._probe_media_duration_seconds = original_probe

            self.assertGreaterEqual(len(clips), 3)
            self.assertGreaterEqual(assembled, 4.5)
            self.assertTrue(all(path.suffix == ".mp4" for path in clips))

    def test_background_builder_handles_single_clip_reuse(self):
        from content_machine import production as prod

        with tempfile.TemporaryDirectory() as tmpdir:
            bg_dir = Path(tmpdir) / "background_clips"
            bg_dir.mkdir(parents=True, exist_ok=True)
            (bg_dir / "only.mp4").write_bytes(b"fake")
            original_probe = prod._probe_media_duration_seconds
            try:
                prod._probe_media_duration_seconds = lambda _path: 1.0
                clips, assembled = prod.build_background_timeline(
                    bg_dir,
                    target_duration=3.0,
                    safety_buffer_seconds=0.5,
                    rng_seed=1,
                )
            finally:
                prod._probe_media_duration_seconds = original_probe

            self.assertGreaterEqual(len(clips), 3)
            self.assertGreaterEqual(assembled, 3.5)
            self.assertTrue(all(path.name == "only.mp4" for path in clips))

    def test_background_builder_avoids_immediate_duplicates_when_possible(self):
        from content_machine import production as prod

        with tempfile.TemporaryDirectory() as tmpdir:
            bg_dir = Path(tmpdir) / "background_clips"
            bg_dir.mkdir(parents=True, exist_ok=True)
            (bg_dir / "a.mp4").write_bytes(b"fake")
            (bg_dir / "b.mp4").write_bytes(b"fake")
            original_probe = prod._probe_media_duration_seconds
            try:
                prod._probe_media_duration_seconds = lambda _path: 1.0
                clips, _assembled = prod.build_background_timeline(
                    bg_dir,
                    target_duration=4.0,
                    safety_buffer_seconds=0.0,
                    randomize=False,
                    allow_immediate_repeats=False,
                )
            finally:
                prod._probe_media_duration_seconds = original_probe

            for previous, current in zip(clips, clips[1:]):
                self.assertNotEqual(previous, current)

    def test_generate_ass_subtitles_plain_mode_disables_active_word_highlighting(self):
        from content_machine.production import CaptionRenderConfig, _generate_ass_subtitles

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = Path(tmpdir) / "subs.ass"
            _generate_ass_subtitles(
                caption_script="This is a caption script for subtitle timing checks",
                output_path=ass_path,
                audio_duration=6.0,
                caption_cfg=CaptionRenderConfig(style_mode="plain"),
            )
            body = ass_path.read_text(encoding="utf-8")
            self.assertIn("Dialogue:", body)
            self.assertNotIn("{\\c", body)

    def test_improvement_enhance_posts_uses_single_final_script(self):
        ranked = RankedPost(
            raw=RawPost(
                source="reddit",
                source_id="xyz",
                author="u",
                text="AITA for charging my friend £20k after a long dispute?",
                metrics={},
            ),
            length_bucket="short",
            viral_score=70.0,
        )
        original_openai = improvement._openai_enhance
        try:
            improvement._openai_enhance = lambda _posts: {
                "xyz": {
                    "source_id": "xyz",
                    "title": "Title",
                    "hook": "Hook",
                    "final_script": "Am I the asshole for charging my friend twenty thousand pounds after a long dispute?",
                    "hashtags": ["#storytime"],
                }
            }
            enhanced = improvement.enhance_posts([ranked])[0]
        finally:
            improvement._openai_enhance = original_openai

        self.assertEqual(enhanced.final_script, enhanced.rewritten_caption_script)
        self.assertEqual(enhanced.final_script, enhanced.rewritten_tts_script)
        self.assertTrue(enhanced.final_script)

    def test_improvement_rejects_over_short_ai_rewrite(self):
        ranked = RankedPost(
            raw=RawPost(
                source="reddit",
                source_id="shrink",
                author="u",
                text="AITA for leaving at 3am bc idk what else to do after our argument went on for hours?",
                metrics={},
            ),
            length_bucket="short",
            viral_score=70.0,
        )
        original_openai = improvement._openai_enhance
        try:
            improvement._openai_enhance = lambda _posts: {
                "shrink": {
                    "source_id": "shrink",
                    "title": "Title",
                    "hook": "Hook",
                    "final_script": "I left.",
                    "hashtags": ["#storytime"],
                }
            }
            enhanced = improvement.enhance_posts([ranked])[0]
        finally:
            improvement._openai_enhance = original_openai

        self.assertNotEqual(enhanced.final_script, "I left.")
        self.assertIn("am i the asshole", enhanced.final_script.lower())

    def test_normalize_tts_text_expands_common_abbreviations(self):
        text = "AITA for saying idk at 3am after a £20k argument?"
        normalized = improvement.normalize_tts_text(text)
        self.assertIn("am i the asshole", normalized.lower())
        self.assertIn("i don't know", normalized.lower())
        self.assertIn("in the morning", normalized.lower())
        self.assertIn("pounds", normalized.lower())
        self.assertTrue(normalized.endswith(("?", ".", "!")))

    def test_render_video_uses_single_canonical_script(self):
        content = EnhancedContent(
            source_post=RankedPost(
                raw=RawPost(source="x", source_id="canon", author="a", text="Story text", metrics={}),
                length_bucket="short",
                viral_score=70.0,
            ),
            title="Title",
            hook="Hook",
            narration="Narration should not be used.",
            caption="Caption",
            final_script="Canonical script for tts and captions.",
            rewritten_caption_script="Different caption script",
            rewritten_tts_script="Different tts script",
            hashtags=["#test"],
        )

        from content_machine import production as prod

        captured = {}
        original_generate_tts = prod._generate_tts
        original_generate_ass = prod._generate_ass_subtitles
        original_build_bg = prod.build_background_timeline
        original_compose = prod._compose_video
        original_wav_duration = prod._get_wav_duration
        original_probe_duration = prod._probe_media_duration_seconds
        try:
            def _fake_generate_tts(narration, output_path):
                captured["tts_script"] = narration
                return str(output_path)

            def _fake_generate_ass_subtitles(caption_script, output_path, audio_duration, caption_cfg):
                captured["caption_script"] = caption_script
                return str(output_path)

            prod._generate_tts = _fake_generate_tts
            prod._generate_ass_subtitles = _fake_generate_ass_subtitles
            prod.build_background_timeline = lambda **kwargs: ([Path(__file__)], 5.0)
            prod._compose_video = lambda **kwargs: kwargs["audio_path"].replace(".wav", ".mp4")
            prod._get_wav_duration = lambda _path: 5.0
            prod._probe_media_duration_seconds = lambda _path: 5.0
            with tempfile.TemporaryDirectory() as tmpdir:
                Path(tmpdir, "audio").mkdir(parents=True, exist_ok=True)
                Path(tmpdir, "subs").mkdir(parents=True, exist_ok=True)
                prod.render_video(content, work_dir=tmpdir)
        finally:
            prod._generate_tts = original_generate_tts
            prod._generate_ass_subtitles = original_generate_ass
            prod.build_background_timeline = original_build_bg
            prod._compose_video = original_compose
            prod._get_wav_duration = original_wav_duration
            prod._probe_media_duration_seconds = original_probe_duration

        self.assertEqual(content.final_script, captured["tts_script"])
        self.assertEqual(content.final_script, captured["caption_script"])

    def test_write_concat_manifest_contains_clip_paths(self):
        from content_machine.production import _write_concat_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            clip1 = Path(tmpdir) / "clip 1.mp4"
            clip2 = Path(tmpdir) / "clip2.mp4"
            clip1.write_bytes(b"1")
            clip2.write_bytes(b"2")
            manifest = Path(tmpdir) / "manifest.txt"
            manifest_path = _write_concat_manifest([clip1, clip2], manifest)

            body = Path(manifest_path).read_text(encoding="utf-8")
            self.assertIn("file '", body)
            self.assertIn(str(clip1.resolve()), body)
            self.assertIn(str(clip2.resolve()), body)

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


    # ------------------------------------------------------------------
    # _get_wav_duration tests
    # ------------------------------------------------------------------

    def test_get_wav_duration_returns_correct_duration(self):
        """_get_wav_duration reads the exact duration from a WAV file written by the wave module."""
        from content_machine.production import _get_wav_duration
        import wave

        sample_rate = 22050
        duration_s = 3.0
        frames = int(duration_s * sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            with wave.open(tmp_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"\x00\x00" * frames)
            result = _get_wav_duration(tmp_path)
            self.assertAlmostEqual(result, duration_s, places=2)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_get_wav_duration_returns_zero_for_missing_file(self):
        """_get_wav_duration returns 0.0 when the WAV file does not exist."""
        from content_machine.production import _get_wav_duration
        result = _get_wav_duration("/nonexistent/path/audio.wav")
        self.assertEqual(result, 0.0)

    # ------------------------------------------------------------------
    # _generate_subtitles timing tests
    # ------------------------------------------------------------------

    def test_generate_subtitles_final_end_matches_audio_duration(self):
        """The last SRT entry must end exactly at the given audio_duration."""
        from content_machine.production import _generate_subtitles
        narration = "one two three four five six seven eight nine ten eleven twelve"
        audio_duration = 7.5
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "subs.srt"
            _generate_subtitles(narration, out, audio_duration=audio_duration)
            content = out.read_text(encoding="utf-8")
        # Last timestamp line ends with the HH:MM:SS,mmm for audio_duration
        # 7.5 s → 00:00:07,500
        self.assertIn("00:00:07,500", content)

    def test_generate_subtitles_proportional_chunks(self):
        """Longer chunks (more characters) get more on-screen time than shorter ones."""
        from content_machine.production import _generate_subtitles
        # Two clean 8-word groups: first group has very short words, second has very long words.
        # chunk_size=8 means chunking splits exactly at word 9.
        short_words = " ".join(["a"] * 8)           # 8 one-letter words → ~15 chars
        long_words  = " ".join(["superlongword"] * 8)  # 8 long words → ~111 chars
        narration = short_words + " " + long_words
        audio_duration = 10.0
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "subs.srt"
            _generate_subtitles(narration, out, audio_duration=audio_duration)
            lines = out.read_text(encoding="utf-8").splitlines()
        # Extract end times from timestamp lines (contain " --> ")
        end_times: list[float] = []
        for line in lines:
            if " --> " in line:
                end_str = line.split(" --> ")[1].strip()
                h, m, rest = end_str.split(":")
                s, ms = rest.split(",")
                end_times.append(int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000)
        self.assertEqual(len(end_times), 2)
        # First chunk (short words) ends well before midpoint
        self.assertLess(end_times[0], audio_duration / 2)
        # Last chunk ends exactly at audio_duration
        self.assertAlmostEqual(end_times[-1], audio_duration, places=2)

    def test_generate_subtitles_falls_back_to_estimate_when_no_duration(self):
        """When audio_duration is None, subtitles are still written without error."""
        from content_machine.production import _generate_subtitles
        narration = "hello world this is a test of the subtitle generation code"
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "subs.srt"
            result = _generate_subtitles(narration, out, audio_duration=None)
            self.assertTrue(Path(result).exists())
            self.assertGreater(Path(result).stat().st_size, 0)

    # ------------------------------------------------------------------
    # _compose_video timeout / error message tests
    # ------------------------------------------------------------------

    def test_compose_video_timeout_raises_runtime_error(self):
        """A TimeoutExpired from subprocess is converted to RuntimeError with path info."""
        import subprocess
        import unittest.mock as mock
        from content_machine.production import _compose_video

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "a.wav"
            subs = Path(tmpdir) / "s.srt"
            manifest = Path(tmpdir) / "bg.txt"
            audio.write_bytes(b"\x00" * 100)
            subs.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
            manifest.write_text("file '/tmp/fake.mp4'\n", encoding="utf-8")
            out = Path(tmpdir) / "v.mp4"

            with mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
                mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 1)):
                with self.assertRaises(RuntimeError) as ctx:
                    _compose_video(str(audio), str(subs), str(manifest), out, timeout=1)

        msg = str(ctx.exception)
        self.assertIn("timed out", msg)
        self.assertIn(str(audio), msg)
        self.assertIn(str(subs), msg)

    def test_compose_video_nonzero_exit_includes_stderr_and_paths(self):
        """Non-zero ffmpeg exit code produces RuntimeError with stderr excerpt and file paths."""
        import subprocess
        import unittest.mock as mock
        from content_machine.production import _compose_video

        fake_result = mock.MagicMock()
        fake_result.returncode = 1
        fake_result.stderr = "Error opening filters!"

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "a.wav"
            subs = Path(tmpdir) / "s.srt"
            manifest = Path(tmpdir) / "bg.txt"
            audio.write_bytes(b"\x00" * 100)
            subs.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
            manifest.write_text("file '/tmp/fake.mp4'\n", encoding="utf-8")
            out = Path(tmpdir) / "v.mp4"

            with mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
                mock.patch("subprocess.run", return_value=fake_result):
                with self.assertRaises(RuntimeError) as ctx:
                    _compose_video(str(audio), str(subs), str(manifest), out)

        msg = str(ctx.exception)
        self.assertIn("Error opening filters!", msg)
        self.assertIn(str(audio), msg)
        self.assertIn(str(subs), msg)

    # ------------------------------------------------------------------
    # Piper TTS – improved error message tests
    # ------------------------------------------------------------------

    def test_generate_tts_piper_exe_points_to_folder_warns_clearly(self):
        """When PIPER_EXE is a directory (not an exe), the warning says so clearly."""
        from content_machine.production import _generate_tts
        env_backup = os.environ.get("PIPER_EXE")
        # Use a path that exists (the tmpdir itself) but is a directory, not a file
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PIPER_EXE"] = tmpdir  # folder, not file
            out = Path(tmpdir) / "audio" / "out.wav"
            with self.assertLogs("content_machine.production", level="WARNING") as cm:
                result = _generate_tts("hello", out)
            # Should fall back to stub successfully
            self.assertTrue(Path(result).exists())
            # Warning must tell user to point to the exe, not the folder
            joined = "\n".join(cm.output)
            self.assertIn("executable", joined.lower())
        if env_backup is None:
            os.environ.pop("PIPER_EXE", None)
        else:
            os.environ["PIPER_EXE"] = env_backup

    # ------------------------------------------------------------------
    # pipeline.run_pipeline – error propagation test
    # ------------------------------------------------------------------

    def test_pipeline_reraises_runtime_error_from_stage(self):
        """run_pipeline must re-raise RuntimeError so the caller sees it."""
        original_collect = pipeline.collect_raw_posts

        def always_fail():
            raise RuntimeError("sourcing exploded")

        try:
            pipeline.collect_raw_posts = always_fail
            with self.assertRaises(RuntimeError):
                pipeline.run_pipeline()
        finally:
            pipeline.collect_raw_posts = original_collect


if __name__ == "__main__":
    unittest.main()
