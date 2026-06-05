import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Configuration ---
FAISS_INDEX_FILE = 'papers.faiss'
DATA_PICKLE_FILE = 'papers_dataframe.pkl'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def load_resources(index_file, data_file, model_name):
    """Loads the FAISS index, original DataFrame, and embedding model."""
    print("Loading resources...")
    try:
        index = faiss.read_index(index_file)
        df = pd.read_pickle(data_file)
        model = SentenceTransformer(model_name)
        print("Resources loaded successfully!")
        return index, df, model
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure you've run 'embedding_and_index.py' first.")
        return None, None, None

def search_documents(query, index, df, model, top_k=5):
    """
    Performs a semantic search on the index.
    Args:
        query (str): The user's search query.
        index (faiss.Index): The loaded FAISS index.
        df (pd.DataFrame): The original DataFrame with document metadata.
        model (SentenceTransformer): The embedding model.
        top_k (int): The number of top results to return.
    Returns:
        A list of dictionaries with search results.
    """
    # Embed the user's query
    query_embedding = model.encode([query])

    # Search the FAISS index for the most similar vectors
    distances, indices = index.search(query_embedding, top_k)

    # Get the results from the original DataFrame using the indices
    results = []
    for i, idx in enumerate(indices[0]):
        doc_data = df.iloc[idx].to_dict()
        doc_data['similarity_score'] = 1 - distances[0][i] # Convert L2 distance to similarity score
        results.append(doc_data)
        
    return results

# --- Main script logic to test the search functionality ---
if __name__ == '__main__':
    # Load the necessary components
    index, df, model = load_resources(FAISS_INDEX_FILE, DATA_PICKLE_FILE, EMBEDDING_MODEL_NAME)
    
    # Corrected logic to check if resources are loaded
    if index is None or df is None or model is None:
        exit()

    # --- Test the search tool with a sample query ---
    print("\n--- Testing Search Tool ---")
    
    test_query = "new methods for improving quantum computing efficiency"
    print(f"Searching for: '{test_query}'")
    
    search_results = search_documents(test_query, index, df, model, top_k=3)
    
    print("\n--- Search Results ---")
    for result in search_results:
        print(f"Title: {result['title']}")
        print(f"Source: {result['source']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print("-" * 20)