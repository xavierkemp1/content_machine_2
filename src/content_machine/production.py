"""Stage 4: produce vertical videos from enhanced content."""

from .models import EnhancedContent, ProductionArtifact


def render_video(content: EnhancedContent) -> ProductionArtifact:
    """TODO: create TTS, subtitles, and merge assets into 9:16 video."""

    raise NotImplementedError("Video rendering skeleton only")


def produce_all(contents: list[EnhancedContent]) -> list[ProductionArtifact]:
    """Run rendering for all enhanced content objects."""

    artifacts: list[ProductionArtifact] = []
    for item in contents:
        artifacts.append(render_video(item))
    return artifacts
