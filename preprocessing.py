from typing import List, Dict
import pandas as pd

WANTED_COLS = ["id", "source", "title", "text", "year", "date", "url", "authors", "venue", "extra"]


def build_corpus_dataframe(docs: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(docs)
    # Normalize
    for c in WANTED_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[WANTED_COLS].fillna("")

    # Basic cleaning
    df["title"] = df["title"].astype(str).str.strip()
    df["text"] = df["text"].astype(str)

    # Deduplicate by id or title
    df = df.drop_duplicates(subset=["id"]).drop_duplicates(subset=["title"])

    # Ensure types
    def _to_int_safe(x):
        try:
            return int(x)
        except Exception:
            return None
    df["year"] = df["year"].map(_to_int_safe)

    return df.reset_index(drop=True)
