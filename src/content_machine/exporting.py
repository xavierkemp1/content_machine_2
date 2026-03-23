"""Stage 5: export video files and metadata."""

from pathlib import Path

from .models import EnhancedContent, ProductionArtifact


def export_outputs(items: list[tuple[EnhancedContent, ProductionArtifact]], base_dir: str = "output") -> None:
    """TODO: save videos and metadata grouped by content type/theme."""

    Path(base_dir).mkdir(parents=True, exist_ok=True)
