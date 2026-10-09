
def clean_text(text):
    """Basic text cleaning for Hausa sentiment prediction."""
    if text is None:
        return ""

    return str(text).strip().lower()
