import os
from typing import List, Dict, Any
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class LLMResult:
    reasoning_md: str
    recommendations_md: str
    top_patents: List[Dict[str, Any]]
    top_papers: List[Dict[str, Any]]


SYSTEM_PROMPT = (
    "You are an expert technical novelty evaluator. Given a user's idea and a set of similar items, "
    "explain overlap, estimate novelty drivers, and recommend next steps. Respond in concise markdown."
)


def _make_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Use a dummy that will raise only when invoked; app also shows model-agnostic scores
        raise RuntimeError("GEMINI_API_KEY not set. Set it to use AI analysis.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)


def analyze_idea(title: str, domain: str, problem: str, approach: str, description: str, features: List[str], specs: str, key_phrases: List[str], vector_hits: List[dict | Any]) -> Dict[str, Any]:
    # Prepare top lists
    patents = [
        {
            "title": h.record.get("title"),
            "year": h.record.get("year"),
            "score": round(h.score, 4),
            "url": h.record.get("url"),
            "link": h.record.get("url"),  # Add link field for UI compatibility
            "authors": h.record.get("authors", ["N/A"])[0] if h.record.get("authors") else "N/A",
            "abstract": h.record.get("text", "")[:500] if h.record.get("text") else "",
        }
        for h in vector_hits if h.record.get("source") == "patent"
    ]
    papers = [
        {
            "title": h.record.get("title"),
            "year": h.record.get("year"),
            "score": round(h.score, 4),
            "url": h.record.get("url"),
            "link": h.record.get("url"),  # Add link field for UI compatibility
            "authors": ", ".join(h.record.get("authors", ["N/A"])[:3]) if h.record.get("authors") else "N/A",
            "abstract": h.record.get("text", "")[:500] if h.record.get("text") else "",
        }
        for h in vector_hits if h.record.get("source") == "paper"
    ]

    # Sort by score first (relevance), then by year (recency) as tiebreaker
    # This ensures highly relevant papers rank high even if older
    patents_sorted = sorted(patents, key=lambda x: (x["score"], x.get("year") or 0), reverse=True)[:10]
    papers_sorted = sorted(papers, key=lambda x: (x["score"], x.get("year") or 0), reverse=True)[:10]

    # Try LLM reasoning; if key missing, fall back to templated text
    try:
        llm = _make_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", (
                "Idea Title: {title}\nDomain: {domain}\nProblem: {problem}\nApproach: {approach}\n"
                "Description: {description}\nFeatures: {features}\nKey Phrases: {key_phrases}\n\n"
                "Similar Patents (title | year | sim): {patents}\n"
                "Similar Papers (title | year | sim): {papers}\n\n"
                "Tasks: 1) Summarize novelty drivers and overlaps; 2) If novel, draft 3 claim ideas;"
                " if not novel, propose 3 research directions; 3) Provide rationale with references to titles."
            )),
        ])
        resp = llm.invoke(prompt.format_messages(
            title=title,
            domain=domain,
            problem=problem,
            approach=approach,
            description=description,
            features=", ".join(features),
            key_phrases=", ".join(key_phrases[:15]),
            patents=[(p["title"], p.get("year"), p["score"]) for p in patents_sorted],
            papers=[(p["title"], p.get("year"), p["score"]) for p in papers_sorted],
        ))
        reasoning_md = resp.content
        recommendations_md = ""
    except Exception as e:
        reasoning_md = (
            "**LLM analysis unavailable** (missing API key or offline).\n\n"
            "Below are the most similar items; use them to refine your idea or draft claims."
        )
        recommendations_md = ""

    return {
        "reasoning_md": reasoning_md,
        "recommendations_md": recommendations_md,
        "top_patents": patents_sorted,
        "top_papers": papers_sorted,
    }
