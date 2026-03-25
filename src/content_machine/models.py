"""Shared data models for the short-form content pipeline skeleton."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawPost:
    """Normalized source post object from Reddit/X."""

    source: str
    source_id: str
    author: str
    text: str
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class RankedPost:
    """Filtered and virality-scored post."""

    raw: RawPost
    length_bucket: str
    viral_score: float
    score_reasons: dict[str, float] = field(default_factory=dict)


@dataclass
class EnhancedContent:
    """LLM-enhanced content object for video production."""

    source_post: RankedPost
    title: str
    hook: str
    narration: str
    caption: str
    hashtags: list[str] = field(default_factory=list)


@dataclass
class ProductionArtifact:
    """Rendered output references."""

    video_path: str
    subtitles_path: str
    metadata_path: str
    audio_path: str = ""
    background_path: str = ""
