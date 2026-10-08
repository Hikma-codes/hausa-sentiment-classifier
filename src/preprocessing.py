def clean_text(text: str) -> str:
    """
    Apply basic text normalization before inference.
    """

    if text is None:
        return ""

    text = str(text).strip().lower()

    return text
