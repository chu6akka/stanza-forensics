def normalize_text(text: str, normalize_quotes: bool = True, normalize_dash: bool = True) -> str:
    out = text
    if normalize_quotes:
        out = out.replace('“', '"').replace('”', '"').replace('«', '"').replace('»', '"')
    if normalize_dash:
        out = out.replace('—', '-').replace('–', '-')
    return out
