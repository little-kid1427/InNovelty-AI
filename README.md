# 🔎 Patent & Research Novelty Checker

An AI-powered tool to analyze the novelty of your ideas against existing patents and research papers. Get novelty scores, identify overlaps, and receive AI-powered recommendations.

## ✨ Features

- 🔍 **Multi-Source Search**: arXiv, IEEE Xplore, DOAJ, Google Scholar, Google Patents
- 🤖 **AI Analysis**: Google Gemini AI for detailed novelty evaluation
- 📊 **Smart Scoring**: Vector similarity-based novelty calculation
- 💬 **Interactive Chat**: AI-powered Q&A about your results
- 📄 **Export**: Download PDF reports and CSV data
- 📜 **History**: Track and manage previous searches
- 🎯 **Clear Decision**: Patent filing guidance

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
SERPAPI_KEY=your_serpapi_key_here
```

Get your API keys:
- **GEMINI_API_KEY**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **SERPAPI_KEY**: [SerpAPI](https://serpapi.com/) (100 free searches/month)

### 3. Run the App
```bash
streamlit run app.py
```

Open: `http://localhost:8501`

### 4. Test Setup (Optional)
```bash
python test_setup.py
```

## 📖 How to Use

1. **Fill the form** - Enter your idea details
2. **Run analysis** - Click "Run Novelty Analysis" (30-60s wait)
3. **Review results** - Check novelty score, patents, papers
4. **Chat with AI** - Ask questions about findings
5. **Export** - Download PDF/CSV reports

## 📊 Understanding Results

### Novelty Score
- **≥70%**: ✅ Can file a patent
- **<70%**: ❌ Write research paper instead

### Confidence Score
Indicates reliability based on:
- Corpus size
- Result distribution
- Similarity patterns

## 🔑 API Keys

### Required
- **GEMINI_API_KEY** - AI analysis & chat
- **SERPAPI_KEY** - Google Scholar & Patents

### Optional
- **IEEE_API_KEY** - IEEE papers ([Get here](https://developer.ieee.org/))
- **SEARCHAPI_KEY** - Alternative to SerpAPI
- **NOVELTY_ADMIN_PASSWORD** - Admin dashboard

## 📦 Data Sources

| Source | Type | API Key |
|--------|------|---------|
| arXiv | Papers | Free |
| DOAJ | Journals | Free |
| Google Scholar | Papers | SerpAPI |
| Google Patents | Patents | SerpAPI |
| IEEE Xplore | Papers | Optional |

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI**: Google Gemini (gemini-2.5-flash)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Search**: FAISS vector index
- **Data**: Pandas, NumPy
- **Export**: ReportLab (PDF)

## 📁 Project Structure

```
├── app.py                 # Main application
├── data_collection.py     # Multi-source fetching
├── ai_analyzer.py        # Gemini AI integration
├── search_engine.py      # Vector search
├── embedding_generator.py # Embeddings
├── utils.py              # Utilities
├── test_setup.py         # Setup checker
└── requirements.txt      # Dependencies
```

## 🐛 Troubleshooting

**No documents found**
- Use broader keywords
- Try different terms

**AI analysis failed**
- Check GEMINI_API_KEY
- Verify quota
- Novelty scores still work!

**Chat not showing**
- Run analysis first
- Refresh page

**Logs**: Check `logs/app.log`

## ⚡ Rate Limits

- App: 5 analyses/minute
- SerpAPI: 100/month (free)
- Gemini: Per quota

## 🔮 Future Scope

- 🚀 **Smart Innovation Suggestions**: If an idea cannot be patented, the AI will suggest innovations by analyzing the "Future Scope" from reference papers.
- 🎙️ **Voice Integration**: Support for voice messages (Whisper-like flow) for hands-free idea submission.
- 📖 **Literature Review**: Automated literature review generation from research papers.
- ⭐ **Negative Novelty Detector**: System flags "dangerous" or overly generic claim elements to avoid rejection.
    - *Example*: "Using blockchain for security" → ❌ Too generic
    - *Example*: "AI-based optimization" → ⚠️ Over-claimed
    - *Output*: High rejection risk vs Safe innovation zones.

## 📝 License

Educational and research use. Follow API terms of service.

## 🤝 Contributing

Issues and PRs welcome!
