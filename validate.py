def validate_user_text(text: str, max_len: int = 2000) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty input.")
    if len(text) > max_len:
        raise ValueError(f"Input too long (>{max_len} chars).")
    return text
