import os
import time
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import pandas as pd

# Load environment variables from .env file
load_dotenv()

from src.utils.utils import (
    init_logging,
    require_env,
    rate_limiter,
    clean_text,
    extract_key_phrases,
    compute_novelty_and_confidence,
    save_search_history_row,
    export_results_pdf,
    export_results_csv,
)
from src.data_ingestion.data_collection import collect_all_sources
from src.data_ingestion.preprocessing import build_corpus_dataframe
from src.vector_store.embedding_generator import ensure_index_ready
from src.search.search_engine import VectorSearch
from src.core.ai_analyzer import analyze_idea

# ------------------------------
# App Setup
# ------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = APP_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

init_logging(LOGS_DIR / "app.log")
logger = logging.getLogger("novelty_app")

st.set_page_config(
    page_title="Patent & Research Novelty Checker",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Improve container styling */
    .stContainer {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }

    /* Better link buttons */
    .stLinkButton > a {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 5px 10px;
        text-decoration: none;
        font-size: 0.9em;
    }

    .stLinkButton > a:hover {
        background-color: #0052a3;
    }

    /* Improve expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1em;
    }

    /* Better metric cards */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e0e0e0;
    }

    /* Divider styling */
    hr {
        margin: 20px 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar: Mode toggle & Admin
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select mode", ["User", "Admin Dashboard"], index=0)

# Sidebar: Search History
st.sidebar.divider()
hist_path = DATA_DIR / "search_history.csv"
col_hist, col_clear = st.sidebar.columns([3, 1])
with col_hist:
    st.subheader("📜 Search History")
with col_clear:
    if st.button("🗑️", key="clear_all_history", help="Clear all history"):
        if hist_path.exists():
            hist_path.unlink()
            st.rerun()
if hist_path.exists():
    df_hist = pd.read_csv(hist_path)
    if len(df_hist) > 0:
        # Show latest 5 searches
        for idx, row in df_hist.tail(5).iterrows():
            with st.sidebar.expander(f"🔍 {row['title'][:30]}...", expanded=False):
                st.caption(f"**Domain:** {row.get('domain', 'N/A')}")
                st.caption(f"**Novelty:** {row.get('novelty', 0):.0f}%")
                st.caption(f"**Confidence:** {row.get('confidence', 0):.0f}%")
                st.caption(f"**Date:** {row.get('timestamp', 'N/A')}")
                if st.button("🗑️ Delete", key=f"delete_{idx}", use_container_width=True):
                    # Delete this row from the CSV
                    df_hist = df_hist.drop(idx)
                    df_hist.to_csv(hist_path, index=False)
                    st.rerun()
    else:
        st.sidebar.caption("No history yet")
else:
    st.sidebar.caption("No history yet")

# Sidebar: Current Results
if "analysis_results" in st.session_state:
    st.sidebar.divider()
    st.sidebar.subheader("📊 Current Results")
    results = st.session_state.analysis_results
    st.sidebar.metric("Novelty Score", f"{results['novelty_score']:.0f}%")
    st.sidebar.metric("Confidence", f"{results['confidence']:.0f}%")

    if results['novelty_score'] >= 70:
        st.sidebar.success("✅ Can file patent")
    else:
        st.sidebar.error("❌ Cannot file patent")
        st.sidebar.info("✅ Can write paper")

    # Quick stats
    st.sidebar.caption(f"**Title:** {results['title']}")
    st.sidebar.caption(f"**Domain:** {results['domain']}")

    # Top patents/papers count
    num_patents = len(results['analysis'].get('top_patents', []))
    num_papers = len(results['analysis'].get('top_papers', []))
    st.sidebar.caption(f"**Similar Patents:** {num_patents}")
    st.sidebar.caption(f"**Similar Papers:** {num_papers}")

# ------------------------------
# User Mode UI
# ------------------------------
if mode == "User":
    st.title("🔎 Patent & Research Novelty Checker")
    st.caption("Analyze your idea against existing patents & papers. Get novelty scores, overlaps, and next-step recommendations.")

    # Add helpful info section
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        **This tool helps you:**
        1. 🔍 Search across multiple sources (arXiv, IEEE Xplore, DOAJ, Google Scholar, Patents)
        2. 🤖 AI analyzes novelty using Google Gemini
        3. 📊 Get a novelty score (≥70% = Patent ready, <70% = Research paper)
        4. 📜 View similar patents and research papers with clickable links
        5. 💬 Chat with AI about your results
        6. 📄 Export reports as PDF or CSV
        """)

    with st.form("idea_form", clear_on_submit=False):
        st.subheader("📝 Enter Your Idea Details")
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Project/Idea Title *", max_chars=200, help="Give your idea a clear, descriptive title")
            domain = st.text_input("Target Industry / Domain *", placeholder="e.g., FinTech, MedTech, IoT, AI/ML", help="What field does your idea belong to?")
            problem = st.text_area("Problem Being Solved *", height=90, help="What problem does your idea address?")
            approach = st.text_area("Proposed Solution Approach *", height=120, help="How does your idea solve the problem?")
        with col2:
            description = st.text_area("Detailed Description *", height=180, help="Provide a comprehensive description of your idea")
            features = st.text_area("Key Features & Innovations", placeholder="feature1, feature2, feature3...", height=90, help="Comma-separated list of unique features")
            specs = st.text_area("Technical Specifications", height=90, help="Technical details, technologies used, etc.")

        st.caption("* Required fields")
        submitted = st.form_submit_button("🚀 Run Novelty Analysis", use_container_width=True, type="primary")

    if submitted:
        # Validate
        if not title or not description or not problem or not approach:
            st.error("Please fill in the title, description, problem, and proposed solution approach.")
            st.stop()

        # Debounce / rate limit (5 analyses per 60s per session)
        if not rate_limiter(limit=5, window_seconds=60, key=f"run:{st.session_state.session_id}"):
            st.warning("You're going too fast. Please wait a moment and try again.")
            st.stop()

        # Build a single analysis text block for embeddings
        features_list = [f.strip() for f in (features or "").split(",") if f.strip()]
        user_blob = "\n".join([
            f"Title: {title}",
            f"Domain: {domain}",
            f"Problem: {problem}",
            f"Approach: {approach}",
            f"Description: {description}",
            f"Features: {', '.join(features_list)}",
            f"Specs: {specs}",
        ])
        user_blob_clean = clean_text(user_blob)
        key_phrases = extract_key_phrases(user_blob_clean)

        with st.status("Collecting documents from sources (arXiv, IEEE Xplore, DOAJ, Google Scholar, Patents)…", expanded=False) as s:
            try:
                docs = collect_all_sources(query_terms=key_phrases, max_per_source=200, data_dir=DATA_DIR)
                if len(docs) == 0:
                    st.error("⚠️ No documents found for your query. Try:\n- Using different keywords\n- Broadening your search terms\n- Checking if your idea is very niche")
                    st.stop()
                s.update(label=f"Collected {len(docs)} records (sorted by recency).")
            except Exception as e:
                logger.exception("Data collection failed")
                st.error(f"❌ Data collection failed: {e}\n\nPossible causes:\n- API rate limits reached\n- Network connectivity issues\n- Invalid API keys")
                st.stop()

        with st.status("Preprocessing & building corpus…", expanded=False) as s:
            df = build_corpus_dataframe(docs)
            s.update(label=f"Corpus size: {len(df)}")

        with st.status("Ensuring embeddings & FAISS index…", expanded=False) as s:
            try:
                ensure_index_ready(df=df, data_dir=DATA_DIR)
                s.update(label="Index ready.")
            except Exception as e:
                logger.exception("Index build failed")
                st.error(f"Index build failed: {e}")
                st.stop()

        with st.status("Searching vector index…", expanded=False) as s:
            vs = VectorSearch(data_dir=DATA_DIR)
            results = vs.semantic_search(user_blob_clean, top_k=50, recency_boost=True)
            s.update(label=f"Found {len(results)} similar items.")

        with st.status("Running AI analysis…", expanded=False) as s:
            try:
                analysis = analyze_idea(
                    title=title,
                    domain=domain,
                    problem=problem,
                    approach=approach,
                    description=description,
                    features=features_list,
                    specs=specs,
                    key_phrases=key_phrases,
                    vector_hits=results,
                )
                s.update(label="Analysis complete.")
            except Exception as e:
                logger.exception("AI analysis failed")
                st.error(f"❌ AI analysis failed: {e}\n\nPossible causes:\n- GEMINI_API_KEY not set or invalid\n- API quota exceeded\n- Network connectivity issues\n\nNote: Novelty scores are still calculated and shown below.")
                # Don't stop - novelty scores work without AI
                analysis = {
                    "reasoning_md": "**AI analysis unavailable.** Gemini API error. Please check your API key and quota.",
                    "recommendations_md": "",
                    "top_patents": [],
                    "top_papers": []
                }

        # Compute novelty and confidence from vector hits (model-agnostic, reproducible)
        novelty_score, confidence = compute_novelty_and_confidence(results)

        # Save to session state first
        patents_df = pd.DataFrame(analysis["top_patents"]).head(10)
        papers_df = pd.DataFrame(analysis["top_papers"]).head(10)

        # Save history
        save_search_history_row(
            data_dir=DATA_DIR,
            session_id=st.session_state.session_id,
            title=title,
            domain=domain,
            novelty=novelty_score,
            confidence=confidence,
            key_phrases=key_phrases,
        )

        # Store analysis results in session state for chat and persistent display
        st.session_state.analysis_results = {
            "title": title,
            "domain": domain,
            "problem": problem,
            "approach": approach,
            "description": description,
            "features": features_list,
            "specs": specs,
            "novelty_score": novelty_score,
            "confidence": confidence,
            "analysis": analysis,
            "key_phrases": key_phrases,
            "patents_df": patents_df,
            "papers_df": papers_df,
        }

        # Clear chat history for new analysis (avoid showing old chats)
        if "current_analysis_id" not in st.session_state or st.session_state.current_analysis_id != title:
            st.session_state.chat_messages = []
            st.session_state.current_analysis_id = title

    # Display results section - persistent across chat interactions
    if "analysis_results" in st.session_state:
        results = st.session_state.analysis_results
        novelty_score = results["novelty_score"]
        confidence = results["confidence"]
        analysis = results["analysis"]
        patents_df = results["patents_df"]
        papers_df = results["papers_df"]

        # Results Header
        st.divider()
        st.header("📊 Analysis Results")

        # Quick navigation
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
        with nav_col1:
            st.markdown("**Quick Jump:**")
        with nav_col2:
            st.markdown("[📜 Patents](#similar-patents)")
        with nav_col3:
            st.markdown("[📚 Papers](#similar-research-papers)")
        with nav_col4:
            st.markdown("[💬 Chat](#chat-with-ai)")

        st.divider()

        # Decision logic
        metric_col1, metric_col2, metric_col3 = st.columns([2, 2, 3])
        with metric_col1:
            st.metric("Novelty Score", f"{novelty_score:.0f}%", help="Higher score = More novel idea")
        with metric_col2:
            st.metric("Confidence", f"{confidence:.0f}%", help="Reliability of the novelty score")
        with metric_col3:
            if novelty_score >= 70:
                st.success("✅ YES, you can file/create a patent on your idea.")
            else:
                st.error("❌ NO, you cannot file/create a patent on this idea.")
                st.info("✅ BUT you can create a research paper on this topic.")

        # Reasoning & Recommendations
        with st.expander("🧠 AI Reasoning & Recommendations", expanded=True):
            st.markdown(analysis["reasoning_md"], unsafe_allow_html=True)
            if analysis.get("recommendations_md"):
                st.divider()
                st.markdown(analysis["recommendations_md"], unsafe_allow_html=True)

        # Patents Section
        st.divider()
        patent_col1, patent_col2 = st.columns([3, 1])
        with patent_col1:
            st.subheader("📜 Similar Patents")
            st.caption(f"Found {len(patents_df)} similar patents (sorted by recency and relevance)")
        with patent_col2:
            if len(patents_df) > 0:
                st.caption(f"Showing top {min(10, len(patents_df))} results")

        if len(patents_df) > 0:
            # Display patents with clickable links
            for idx, patent in patents_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        title_text = patent.get('title', 'Untitled')
                        link = patent.get('link', '')
                        if link:
                            st.markdown(f"**{idx + 1}. [{title_text}]({link})**")
                        else:
                            st.markdown(f"**{idx + 1}. {title_text}**")

                        # Display patent details
                        detail_cols = st.columns(4)
                        with detail_cols[0]:
                            st.caption(f"📅 Year: {patent.get('year', 'N/A')}")
                        with detail_cols[1]:
                            st.caption(f"🔍 Similarity: {patent.get('score', 'N/A')}")
                        with detail_cols[2]:
                            st.caption(f"👤 Author: {patent.get('authors', 'N/A')}")
                        with detail_cols[3]:
                            if link:
                                st.link_button("🔗 View Patent", link, use_container_width=True)

                    # Abstract/Summary if available
                    if patent.get('abstract'):
                        with st.expander("📄 Abstract"):
                            st.caption(patent['abstract'][:500] + "..." if len(patent.get('abstract', '')) > 500 else patent.get('abstract', ''))

                    st.divider()
        else:
            st.info("No similar patents found.")

        # Research Papers Section
        paper_col1, paper_col2 = st.columns([3, 1])
        with paper_col1:
            st.subheader("📚 Similar Research Papers")
            st.caption(f"Found {len(papers_df)} similar research papers (sorted by recency and relevance)")
        with paper_col2:
            if len(papers_df) > 0:
                st.caption(f"Showing top {min(10, len(papers_df))} results")

        if len(papers_df) > 0:
            # Display papers with clickable links
            for idx, paper in papers_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        title_text = paper.get('title', 'Untitled')
                        link = paper.get('link', '')
                        if link:
                            st.markdown(f"**{idx + 1}. [{title_text}]({link})**")
                        else:
                            st.markdown(f"**{idx + 1}. {title_text}**")

                        # Display paper details
                        detail_cols = st.columns(4)
                        with detail_cols[0]:
                            st.caption(f"📅 Year: {paper.get('year', 'N/A')}")
                        with detail_cols[1]:
                            st.caption(f"🔍 Similarity: {paper.get('score', 'N/A')}")
                        with detail_cols[2]:
                            st.caption(f"👤 Authors: {paper.get('authors', 'N/A')}")
                        with detail_cols[3]:
                            if link:
                                st.link_button("🔗 View Paper", link, use_container_width=True)

                    # Abstract if available
                    if paper.get('abstract'):
                        with st.expander("📄 Abstract"):
                            st.caption(paper['abstract'][:500] + "..." if len(paper.get('abstract', '')) > 500 else paper.get('abstract', ''))

                    st.divider()
        else:
            st.info("No similar research papers found.")

        # Chat Section - Always visible when results exist
        st.divider()
        st.header("💬 Chat with AI")
        chat_col1, chat_col2 = st.columns([3, 1])
        with chat_col1:
            st.caption("Ask questions about your novelty analysis, patents, or research papers")
        with chat_col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

        # Initialize chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your results..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        from langchain.prompts import ChatPromptTemplate

                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=os.getenv("GEMINI_API_KEY"))

                        # Get analysis results from session state
                        results = st.session_state.analysis_results
                        title = results["title"]
                        domain = results["domain"]
                        problem = results["problem"]
                        approach = results["approach"]
                        novelty_score = results["novelty_score"]
                        confidence = results["confidence"]
                        analysis = results["analysis"]
                        key_phrases = results["key_phrases"]

                        # Build detailed context from analysis results
                        patents_list = "\n".join([f"  {i+1}. {p['title']} ({p.get('year', 'N/A')}) - Similarity: {p['score']}"
                                                   for i, p in enumerate(analysis['top_patents'][:5])])
                        papers_list = "\n".join([f"  {i+1}. {p['title']} ({p.get('year', 'N/A')}) - Similarity: {p['score']}"
                                                  for i, p in enumerate(analysis['top_papers'][:5])])

                        context = f"""
Analysis Results:
- Title: {title}
- Domain: {domain}
- Problem: {problem}
- Approach: {approach}
- Novelty Score: {novelty_score:.1f}%
- Confidence: {confidence:.1f}%
- Key Phrases: {', '.join(key_phrases[:10])}
- Patent Decision: {"✅ Can file patent (novelty ≥ 70%)" if novelty_score >= 70 else "❌ Cannot file patent, but can write research paper"}

Top 5 Similar Patents:
{patents_list if patents_list else "  (None found)"}

Top 5 Similar Papers:
{papers_list if papers_list else "  (None found)"}

AI Reasoning: {analysis['reasoning_md']}
"""

                        # Build conversation history for multi-turn chat
                        messages = [
                            ("system", "You are a helpful assistant analyzing patent and research novelty results. "
                             "Answer questions based on the analysis results provided. Be concise and helpful. "
                             "Reference specific papers/patents by their numbers when relevant."),
                            ("user", f"Here is the context for our conversation:\n{context}")
                        ]

                        # Add previous conversation history
                        for msg in st.session_state.chat_messages[:-1]:  # Exclude the last message (current prompt)
                            if msg["role"] == "user":
                                messages.append(("user", msg["content"]))
                            else:
                                messages.append(("assistant", msg["content"]))

                        # Add current question
                        messages.append(("user", prompt))

                        chat_prompt = ChatPromptTemplate.from_messages(messages)

                        response = llm.invoke(chat_prompt.format_messages())
                        response_text = response.content

                        st.markdown(response_text)

                        # Add assistant response to chat history
                        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

                    except Exception as e:
                        error_msg = f"Chat error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})

# ------------------------------
# Admin Dashboard
# ------------------------------
if mode == "Admin Dashboard":
    st.title("🛠️ Admin Dashboard")
    st.caption("Manage datasets, indexes, and view analytics.")

    admin_ok = False
    pwd = st.text_input("Admin password", type="password")
    if pwd and pwd == os.getenv("NOVELTY_ADMIN_PASSWORD", ""):
        admin_ok = True
    if not admin_ok:
        st.info("Enter admin password to continue. Set NOVELTY_ADMIN_PASSWORD in your environment.")
        st.stop()

    # Analytics
    hist_path = DATA_DIR / "search_history.csv"
    if hist_path.exists():
        df_hist = pd.read_csv(hist_path)
        st.metric("Total analyses", len(df_hist))
        st.dataframe(df_hist.tail(50), use_container_width=True)
    else:
        st.caption("No search history yet.")

    st.divider()
    st.subheader("Rebuild vector index from current cache")
    if st.button("Rebuild index"):
        try:
            # Load any cached corpora
            cached_corpus = DATA_DIR / "corpus_cache.parquet"
            if cached_corpus.exists():
                df_cache = pd.read_parquet(cached_corpus)
                ensure_index_ready(df=df_cache, data_dir=DATA_DIR, force_rebuild=True)
                st.success("Index rebuilt from cache.")
            else:
                st.warning("No cached corpus found. Run a user analysis first to populate cache.")
        except Exception as e:
            st.error(f"Failed to rebuild index: {e}")
