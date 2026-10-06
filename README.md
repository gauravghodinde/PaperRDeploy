# Read · See · Synthesize 🔬

**Multi-LLM Related Work Generation with Graphical Abstract Intelligence**

A Streamlit application that automates academic related work generation through a three-stage pipeline:

| Stage | Name | What it does |
|-------|------|--------------|
| 📖 **1 — Read** | Extract | Upload a research paper PDF → extracts title, abstract, supplementary & complementary evidence |
| 👁️ **2 — See** | Classify | ML classifier (Random Forest) labels each graphical abstract detail as *Supplementary* or *Complementary* |
| ⚡ **3 — Synthesize** | Generate | Sends structured evidence to Gemini, LLaMA-3.3-70B, GPT-OSS-120B, DeepSeek → produces an ensemble related work section |

---

## 📁 Project Structure

```
streamlit_app/
├── app.py                        ← Streamlit entry point
├── requirements.txt
├── .env.example                  ← Copy to .env and fill in API keys
├── .gitignore
│
├── .streamlit/
│   ├── config.toml               ← Dark theme + upload size settings
│   └── secrets.toml              ← API keys (gitignored — never commit!)
│
├── src/
│   ├── pipeline.py               ← Main orchestration pipeline
│   ├── see_stage.py              ← Stage 2: ML classification bridge
│   ├── chains/
│   │   ├── related_work_chain.py ← Multi-LLM synthesis chains
│   │   ├── compressive_chain.py
│   │   └── summary_chain.py
│   ├── clients/
│   │   ├── pdf_extractor.py      ← Stage 1: PDF parsing
│   │   ├── keyword_extractor.py  ← KeyBERT keyword extraction
│   │   └── semantic_scholar.py   ← Related paper retrieval
│   ├── models/
│   │   └── paper.py              ← Data models
│   └── utils/
│       ├── config.py             ← Pydantic settings
│       └── llm_factory.py        ← LLM provider factory
│
└── classification/
    ├── details_classifier.py     ← Supp/Comp ML classifier
    └── Augumented/               ← Training data (JSON)
        ├── Combined.json
        ├── Biomedical.json
        ├── Wireless.json
        └── nlp_output.json
```

---

## ⚡ Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API keys
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Minimum required key (free):
```
GROQ_API_KEY=gsk-...   # Get free at https://console.groq.com/keys
```

### 3. Run
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk-..."
   GOOGLE_API_KEY = "..."
   SEMANTIC_SCHOLAR_API_KEY = "..."
   ```
5. Click **Deploy** (~2–3 min)

---

## 🔑 API Keys

| Key | Required | Where to get |
|-----|----------|--------------|
| `GROQ_API_KEY` | ✅ Min required | [console.groq.com](https://console.groq.com/keys) — **free** |
| `GOOGLE_API_KEY` | Recommended | [aistudio.google.com](https://aistudio.google.com) |
| `SEMANTIC_SCHOLAR_API_KEY` | Recommended | [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) |
| `DEEPSEEK_API_KEY` | Optional | [platform.deepseek.com](https://platform.deepseek.com) |
| `OPENAI_API_KEY` | Optional | [platform.openai.com](https://platform.openai.com) |

---

## 📄 Citation

If you use this tool in your research, please cite the associated paper.
