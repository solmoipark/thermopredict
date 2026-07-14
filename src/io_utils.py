from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]


def read_csv_with_fallback(source) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(_rewind(source), encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise ValueError("CSV file could not be read.")


def _rewind(source):
    if isinstance(source, (str, Path)):
        return source
    if isinstance(source, bytes):
        return BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)
    return source

