# API Keys Reference

## 🔑 Required vs Optional API Keys

### ✅ **REQUIRED for Core Functionality**

| API Key | Purpose | Get It From | Free Tier |
|---------|---------|-------------|-----------|
| `GEMINI_API_KEY` | AI analysis & chat | [Google AI Studio](https://makersuite.google.com/app/apikey) | ✅ Yes |
| `SERPAPI_KEY` | Google Scholar & Patents | [SerpAPI](https://serpapi.com/) | ✅ 100 searches/month |

**Without these keys:**
- ❌ No AI-powered novelty analysis
- ❌ No Google Scholar papers
- ❌ No Google Patents results
- ❌ No chat functionality

---

### ⚠️ **OPTIONAL (Enhances Results)**

| API Key | Purpose | Get It From | Free Tier |
|---------|---------|-------------|-----------|
| `IEEE_API_KEY` | IEEE Xplore papers | [IEEE Developer](https://developer.ieee.org/) | ✅ 200 requests/day |
| `SEARCHAPI_KEY` | Alternative to SerpAPI | [SearchAPI.io](https://www.searchapi.io/) | ✅ 100 searches/month |

**Without these keys:**
- ⚠️ Still works, but with fewer results
- ⚠️ Missing IEEE research papers (if no IEEE key)

---

### 📊 **What Works Without Any Keys?**

Even with NO API keys configured, you still get:
- ✅ **arXiv** papers (public API)
- ✅ **DOAJ** journal articles (public API)
- ✅ Basic novelty scoring (vector similarity)

---

## 🔧 Current Setup

Check your `.env` file to see which keys are configured:

```bash
cat .env
```

Expected format:
```env
GEMINI_API_KEY=your_key_here
SERPAPI_KEY=your_key_here
IEEE_API_KEY=your_key_here        # Optional
SEARCHAPI_KEY=your_key_here       # Optional
NOVELTY_ADMIN_PASSWORD=admin123   # Optional
```

---

## 🚨 Troubleshooting

### Problem: "No documents found"
**Cause:** No API keys configured or API limits reached
**Solution:**
1. Add `SERPAPI_KEY` to access Google Scholar & Patents
2. Try again with different keywords
3. Check API quota at [SerpAPI Dashboard](https://serpapi.com/dashboard)

### Problem: "AI analysis failed"
**Cause:** Missing or invalid `GEMINI_API_KEY`
**Solution:**
1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `.env` file: `GEMINI_API_KEY=your_key_here`
3. Restart the app

### Problem: "IEEE papers missing"
**Cause:** No `IEEE_API_KEY` configured
**Solution:**
- This is OPTIONAL - app still works fine without it
- To add IEEE papers, get a free key from [IEEE Developer Portal](https://developer.ieee.org/)

---

## 📈 Data Sources Summary

| Source | Type | Requires Key | Fallback |
|--------|------|--------------|----------|
| arXiv | Papers | ❌ No | Always available |
| DOAJ | Papers | ❌ No | Always available |
| IEEE Xplore | Papers | ⚠️ Optional | Skips if no key |
| Google Scholar | Papers | ✅ Yes (SERPAPI) | Returns empty if no key |
| Google Patents | Patents | ✅ Yes (SERPAPI or SEARCHAPI) | Returns empty if no key |

---

## 💡 Best Practice Setup

**Minimal setup (FREE):**
```env
GEMINI_API_KEY=get_from_google_ai_studio
SERPAPI_KEY=get_from_serpapi_com
```

**Full setup (FREE):**
```env
GEMINI_API_KEY=get_from_google_ai_studio
SERPAPI_KEY=get_from_serpapi_com
IEEE_API_KEY=get_from_ieee_developer
NOVELTY_ADMIN_PASSWORD=your_secure_password
```

---

## 🔒 Security

- ✅ Never commit `.env` file to git (already in `.gitignore`)
- ✅ Use `.env.example` as template
- ✅ Keep API keys secret
- ✅ Rotate keys if exposed

---

Last updated: 2025-10-03
