from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim


def ensure_index_ready(df: pd.DataFrame, data_dir: Path, force_rebuild: bool = False) -> None:
    idx_path = data_dir / "vector_index.faiss"
    meta_path = data_dir / "metadata.pkl"

    if idx_path.exists() and meta_path.exists() and not force_rebuild:
        return

    # Build embeddings
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    texts = (df["title"].fillna("") + " \n" + df["text"].fillna("")).tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Build FAISS index (cosine similarity via inner product)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    faiss.write_index(index, str(idx_path))
    df.to_pickle(meta_path)
