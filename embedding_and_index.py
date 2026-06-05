import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- Configuration for embeddings and indexing ---
DATA_FILE = 'arxiv_papers.csv'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_FILE = 'papers.faiss'

def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

def create_embeddings(df, model_name):
    """Creates text embeddings for a DataFrame's text column."""
    print("Loading Sentence-Transformer model...")
    model = SentenceTransformer(model_name)
    print("Generating embeddings for documents...")
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    return embeddings, model

def create_faiss_index(embeddings):
    """Creates a FAISS index from the embeddings."""
    print("Creating FAISS index...")
    # The dimension of the embeddings (384 for all-MiniLM-L6-v2)
    embedding_dim = embeddings.shape[1]
    
    # Create an index
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def main():
    """Main function to run the embedding and indexing process."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found. Please run data_collection.py first.")
        return

    # Load data from the CSV file
    df = load_data(DATA_FILE)
    if df is None or df.empty:
        print("DataFrame is empty. Exiting.")
        return

    # Create embeddings and get the model object
    embeddings, model = create_embeddings(df, EMBEDDING_MODEL_NAME)

    # Create the FAISS index
    faiss_index = create_faiss_index(embeddings)

    # Save the index to a file
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    print(f"\nFAISS index created and saved to '{FAISS_INDEX_FILE}'.")
    
    # Save the original DataFrame as well to map IDs back to results later
    df.to_pickle('papers_dataframe.pkl')
    print("Original DataFrame saved as 'papers_dataframe.pkl'.")

if __name__ == '__main__':
    main()