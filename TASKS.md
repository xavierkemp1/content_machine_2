# Project Tasks

## Tasks Done

- [x] Create initial project skeleton files for source, filtering/ranking, improvement, production, and export stages.
- [x] Add shared data models for pipeline objects.
- [x] Add starter configuration files (`config/sources.json`, `config/filtering.json`).
- [x] Add `.env.example` for API key/secrets placeholders.
- [x] Add pipeline entrypoint script (`scripts/run_pipeline.py`).

## Tasks To Do

- [ ] Implement Reddit API ingestion and normalization.
- [ ] Implement X API ingestion and normalization.
- [ ] Implement rule-based filtering (blacklist, length, media-dependent filtering).
- [ ] Implement duplicate detection and similarity-based removal.
- [ ] Implement OpenAI ranking for virality scoring.
- [ ] Implement OpenAI rewrite/enhancement for hooks, narration, title, caption, hashtags.
- [ ] Implement TTS generation.
- [ ] Implement subtitle generation.
- [ ] Implement background video selection logic.
- [ ] Implement ffmpeg composition for 9:16 video export.
- [ ] Implement metadata JSON export per video.
- [ ] Add tests for each pipeline stage.
