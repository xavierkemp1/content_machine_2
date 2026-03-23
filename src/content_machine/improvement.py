"""Stage 3: rewrite selected posts for short-form delivery."""

from .models import EnhancedContent, RankedPost


def enhance_posts(posts: list[RankedPost]) -> list[EnhancedContent]:
    """TODO: use OpenAI to generate hook, narration, title, caption, hashtags."""

    return []
