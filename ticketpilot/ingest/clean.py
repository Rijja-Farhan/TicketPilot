"""Stage 2, clean: normalise unicode and whitespace, strip page numbers and repeated headers and footers."""
import re
import unicodedata

PAGE_NUMBER = re.compile(r"^(?:page\s+\d+(?:\s+of\s+\d+)?|-?\s*\d{1,3}\s*-?)$", re.I)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    pages = [[ln.strip() for ln in page.split("\n")] for page in text.split("\f")]
    pages = [p for p in pages if any(p)]
    if len(pages) >= 2:  # a line on every page is a running header or footer
        common = set.intersection(*(set(filter(None, p)) for p in pages))
        pages = [[ln for ln in p if ln not in common] for p in pages]
    lines = [ln for p in pages for ln in p if not PAGE_NUMBER.match(ln)]
    out = re.sub(r"[ \t]+", " ", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", out).strip()
