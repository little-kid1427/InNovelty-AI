import os
import io
import re
import time
import json
import logging
from collections import deque
from typing import List, Tuple
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

_rl_tokens: dict[str, deque] = {}


def init_logging(path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
    )


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def rate_limiter(limit: int, window_seconds: int, key: str) -> bool:
    now = time.time()
    dq = _rl_tokens.setdefault(key, deque())
    while dq and now - dq[0] > window_seconds:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extract_key_phrases(text: str, k: int = 15) -> List[str]:
    # Simple heuristic: split on punctuation, dedupe, drop short tokens
    tokens = re.split(r"[^a-z0-9+]+", text)
    tokens = [t for t in tokens if len(t) > 3 and not t.isdigit()]
    # Keep phrases by also taking 2-grams
    unigrams = tokens
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
    phrases = list(dict.fromkeys(unigrams + bigrams))  # preserve order
    return phrases[:k]


def compute_novelty_and_confidence(hits) -> Tuple[float, float]:
    if not hits:
        return 100.0, 40.0  # empty index -> assume novel but low confidence
    top_sims = [max(0.0, min(1.0, h.score)) for h in hits[:10]]
    max_sim = max(top_sims) if top_sims else 0
    # Novelty inversely proportional to strongest match; penalize if many strong matches
    strong_count = sum(1 for s in top_sims if s > 0.7)
    novelty = max(0.0, (1.0 - max_sim)) * 100.0 - strong_count * 3.0
    novelty = max(0.0, min(100.0, novelty))
    # Confidence grows with #hits and agreement among them
    n = len(hits)
    dispersion = max(0.02, (sum(abs(s - max_sim) for s in top_sims) / max(1, len(top_sims))))
    conf = min(100.0, 50.0 + min(50.0, (n/50.0)*30.0) + (1.0/dispersion)*10.0)
    conf = max(0.0, min(100.0, conf))
    return novelty, conf


def save_search_history_row(data_dir, session_id: str, title: str, domain: str, novelty: float, confidence: float, key_phrases: List[str]):
    path = data_dir / "search_history.csv"
    from datetime import datetime
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session": session_id,
        "title": title,
        "domain": domain,
        "novelty": round(novelty, 1),
        "confidence": round(confidence, 1),
        "key_phrases": ", ".join(key_phrases),
    }
    try:
        if path.exists():
            df = pd.read_csv(path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(path, index=False)
    except Exception as e:
        logging.getLogger("novelty_app").warning("Failed to save history: %s", e)


def export_results_csv(patents_df: pd.DataFrame, papers_df: pd.DataFrame) -> bytes:
    # Combine with an origin column
    patents = patents_df.copy()
    patents["origin"] = "patent"
    papers = papers_df.copy()
    papers["origin"] = "paper"
    combo = pd.concat([patents, papers], ignore_index=True)
    return combo.to_csv(index=False).encode("utf-8")


def export_results_pdf(title: str, inputs: dict, novelty: float, confidence: float, patents: pd.DataFrame, papers: pd.DataFrame, recommendations_md: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    def draw_text(x, y, text, size=10):
        c.setFont("Helvetica", size)
        for line in text.split("\n"):
            c.drawString(x, y, line[:110])
            y -= 12
        return y

    y = height - 20 * mm
    y = draw_text(20 * mm, y, f"Novelty Report — {title}", size=14)
    y = draw_text(20 * mm, y, f"Novelty: {novelty:.0f}%    Confidence: {confidence:.0f}%", size=12)
    y -= 6
    y = draw_text(20 * mm, y, "Inputs:")
    for k, v in inputs.items():
        y = draw_text(25 * mm, y, f"• {k}: {v}")
    y -= 6
    y = draw_text(20 * mm, y, "Top Similar Patents:")
    for _, row in patents.head(10).iterrows():
        y = draw_text(25 * mm, y, f"• {row.get('title')} ({row.get('year')})  sim={row.get('score')}")
        if y < 30 * mm:
            c.showPage(); y = height - 20 * mm
    y -= 6
    y = draw_text(20 * mm, y, "Top Similar Papers:")
    for _, row in papers.head(10).iterrows():
        y = draw_text(25 * mm, y, f"• {row.get('title')} ({row.get('year')})  sim={row.get('score')}")
        if y < 30 * mm:
            c.showPage(); y = height - 20 * mm
    if recommendations_md:
        y -= 6
        y = draw_text(20 * mm, y, "Recommendations:")
        for line in recommendations_md.split("\n"):
            y = draw_text(25 * mm, y, line)
            if y < 30 * mm:
                c.showPage(); y = height - 20 * mm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
