import re
import threading
from urllib.parse import urlparse

# ============================================================
# AILAF - SECURITY HELPERS
# ============================================================
# 1. Cleaning of text saved to CSV files (prevents spreadsheet
#    formula injection)
# 2. Student ID and URL validation
# 3. A lock so two sessions never write to a file at the same time
# ============================================================

WRITE_LOCK = threading.Lock()

FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def safe_text(value, max_len=2000):
    # trims text, removes null bytes and neutralizes spreadsheet formulas
    text = str(value if value is not None else "").replace("\x00", "").strip()[:max_len]

    if text.startswith(FORMULA_START):
        text = "'" + text

    return text


def clean_id(value):
    # returns "" for empty, the ID if valid, or None if invalid
    text = str(value if value is not None else "").strip()

    if text == "":
        return ""

    if re.fullmatch(r"[A-Za-z0-9_-]{3,20}", text):
        return text

    return None


def valid_url(value):
    # only normal web links are allowed
    try:
        parsed = urlparse(str(value).strip())
    except ValueError:
        return False

    return parsed.scheme in ("http", "https") and bool(parsed.netloc)
