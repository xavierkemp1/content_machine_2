# Project Tasks

## Tasks Done

- [x] Create initial project skeleton files for source, filtering/ranking, improvement, production, and export stages.
- [x] Add shared data models for pipeline objects.
- [x] Add starter configuration files (`config/sources.json`, `config/filtering.json`).
- [x] Add `.env.example` for API key/secrets placeholders.
- [x] Add pipeline entrypoint script (`scripts/run_pipeline.py`).

## Tasks To Do

- [x] Implement Reddit API ingestion and normalization.
- [x] Implement X API ingestion and normalization.
- [x] Implement rule-based filtering (blacklist, length, media-dependent filtering).
- [x] Implement duplicate detection and similarity-based removal.
- [x] Implement OpenAI ranking for virality scoring.
- [x] Implement OpenAI rewrite/enhancement for hooks, narration, title, caption, hashtags.
- [x] Implement TTS generation.
- [x] Implement subtitle generation.
- [x] Implement background video selection logic.
- [x] Implement ffmpeg composition for 9:16 video export.
- [x] Implement metadata JSON export per video.
- [x] Add tests for each pipeline stage.
