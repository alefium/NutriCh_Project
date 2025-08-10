import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent storage and retrieval in RAG:
    - Unicode NFKC
    - Standardize quotes/apostrophes and dashes
    - Normalize ellipsis and NBSP
    - Remove zero-width characters
    - Collapse whitespace and trim
    - Lowercase

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if text is None:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace typographic characters with ASCII equivalents
    replacements = {
        # Quotes/apostrophes
        "\u2018": "'", "\u2019": "'", "\u201B": "'", "\u2032": "'",
        "\u201C": '"', "\u201D": '"', "\u2033": '"',
        # Dashes/hyphens/minus
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
        # Ellipsis
        "\u2026": "...",
        # Non-breaking space
        "\u00A0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Remove zero-width characters
    for zw in ["\u200B", "\u200C", "\u200D", "\u200E", "\u200F", "\uFEFF"]:
        text = text.replace(zw, "")

    # Collapse whitespace and trim
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase for consistent embeddings
    return text.lower()


