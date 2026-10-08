def clean_text(text: str) -> str:
    """
    Apply the same basic text normalization used during
    the Hausa sentiment classification project.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    return text
