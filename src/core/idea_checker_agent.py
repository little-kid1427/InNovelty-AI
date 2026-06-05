import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- 1. Load your pre-built resources ---
FAISS_INDEX_FILE = 'data/papers.faiss'
DATA_PICKLE_FILE = 'data/papers_dataframe.pkl'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

try:
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    df_papers = pd.read_pickle(DATA_PICKLE_FILE)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except FileNotFoundError as e:
    print(f"Error: {e}. Please run 'embedding_and_index.py' first.")
    exit()

# --- 2. Define the search tool for the agent ---
@tool
def search_research_papers(query: str) -> str:
    """
    Searches a database of academic research papers for a given query.
    This tool is useful for checking if a new idea is similar to existing research.
    It returns the title, abstract, and similarity score of the top 3 results.
    """
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, 3)

    results = []
    for i, idx in enumerate(indices[0]):
        doc_data = df_papers.iloc[idx].to_dict()
        similarity_score = 1 - distances[0][i]
        
        # Format the result as a string for the LLM to easily read
        result_str = (f"Title: {doc_data['title']}\n"
                      f"Summary: {doc_data['text'][:200]}...\n"
                      f"Similarity Score: {similarity_score:.4f}\n")
        results.append(result_str)
        
    return "\n---\n".join(results)

# --- 3. Set up the LLM and Agent ---
# Load Gemini API key from environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Please set it in your .env file.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=gemini_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical idea and novelty evaluator. Your job is to assess the originality of a user's technical idea by using your tools to search for similar existing work. When a user provides an idea, you must use the 'search_research_papers' tool to find relevant documents. Finally, you will synthesize the results and provide a clear, concise report on the idea's novelty."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [search_research_papers]

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. Run the Agent with a Sample Idea ---
if __name__ == '__main__':
    user_idea = "I have an idea for a new semiconductor design that uses carbon nanotubes to increase data transfer speeds and reduce heat."
    
    print(f"Evaluating the idea: '{user_idea}'\n")
    
    response = agent_executor.invoke({"input": user_idea})
    
    print("\n--- Final Agent Response ---")
    print(response['output'])