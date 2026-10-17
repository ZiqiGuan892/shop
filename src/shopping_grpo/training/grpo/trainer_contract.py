"""Reviewed 0.9.0.dev trainer sources, with portable line endings and final LF."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_ORIGINAL_SHA256 = "13eb175028e2af2f5f1893c227d104831874c04ac48ab8b31f3ceeee17282d60"
EXPECTED_TRAINER_SHA256 = "0ad8221a5c89cda79b1332a5a244228cfea781eba97949ffe66a014579b49015"


def normalized_source_bytes(path: Path) -> bytes:
    """Allow CRLF and trailing empty lines without ignoring source whitespace."""
    return path.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n"


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(normalized_source_bytes(path)).hexdigest()
