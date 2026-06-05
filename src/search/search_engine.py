from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

@dataclass
class SearchHit:
    idx: int
    score: float
    record: Dict[str, Any]


class VectorSearch:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.index = faiss.read_index(str(self.data_dir / "vector_index.faiss"))
        self.meta = pd.read_pickle(self.data_dir / "metadata.pkl")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def semantic_search(self, text: str, top_k: int = 50, recency_boost: bool = True) -> List[SearchHit]:
        q_emb = self.model.encode([text], normalize_embeddings=True)
        sims, idxs = self.index.search(np.asarray(q_emb, dtype="float32"), top_k)
        hits = []
        for i, idx in enumerate(idxs[0]):
            rec = self.meta.iloc[idx].to_dict()
            score = float(sims[0][i])  # cosine similarity in [-1,1]
            if recency_boost and rec.get("year"):
                # Stronger recency boost: papers from 2020+ get up to +0.05 boost
                # Formula: (year - 2015) * 0.005 gives 2025 -> +0.05, 2020 -> +0.025
                year = int(rec["year"]) if rec.get("year") else 0
                score += (max(0, year - 2015) * 0.005)
            hits.append(SearchHit(idx=idx, score=score, record=rec))
        # sort by boosted score desc, then by year desc
        hits.sort(key=lambda h: (h.score, h.record.get("year") or 0), reverse=True)
        return hits
