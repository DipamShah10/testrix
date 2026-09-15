# Testrix — AI-Powered QA Automation System

## Project Overview

Testrix is an AI-powered QA automation platform. It analyzes bug descriptions, generates API test cases and security payloads, and runs visual regression tests comparing Figma designs against live Shopify storefronts. Uses Groq LLaMA 3.3 70B (text) and LLaMA 3.2 Vision (images) with RAG for context-aware analysis.

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Web Framework | FastAPI + Uvicorn |
| LLM (text) | Groq API — `llama-3.3-70b-versatile` |
| LLM (vision) | Groq API — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Vector Store | FAISS + HuggingFace Sentence-Transformers |
| Browser Automation | Playwright (sync, runs in thread pool) |
| Image Processing | Pillow |
| Database | MongoDB (pymongo) |
| Frontend | Vanilla HTML/CSS/JS (single-page app) |

## Project Structure

```
testrix/
├── app.py                          # FastAPI entry point, all API routes
├── main.py                         # CLI pipeline entry point (--requirements, --url)
├── requirements.txt
├── .env                            # API keys (not committed)
├── .env.example                    # Key reference — copy to .env
│
├── agents/
│   ├── agent_manager.py            # Orchestrates bug + test case flow (parallel)
│   ├── bug_agent.py                # Bug analysis → structured JSON
│   ├── visual_qa_agent.py          # Visual QA pipeline orchestrator
│   ├── ai_crawl_agent.py           # AI crawl pipeline orchestrator
│   ├── llm_client.py               # Configurable LLM client (groq/openai/claude)
│   ├── ai_reviewer.py              # GO/NO-GO reviewer for AI crawl
│   └── requirement_analyzer.py     # Requirements file parser for AI crawl
│
├── ai_engine/
│   ├── llm.py                      # Groq AsyncOpenAI wrapper
│   ├── prompts.py                  # Reusable LLM prompt templates
│   └── utils.py                    # JSON extraction helpers
│
├── services/
│   ├── db.py                       # MongoDB: history + visual_qa_jobs + ai_crawl_jobs
│   ├── bug_analysis_service.py     # RAG-enhanced bug analysis
│   ├── test_case_service.py        # RAG-enhanced test case generation
│   ├── test_runner.py              # Playwright API test runner
│   ├── figma_extractor.py          # Figma REST API — cached, throttled, retry
│   ├── shopify_scraper.py          # Playwright sync scraper (SSRF-safe)
│   ├── site_crawler.py             # Sitemap + BFS HTML link crawler
│   ├── visual_comparator.py        # Pillow pixel diff + BFS region detection
│   ├── visual_ai_analyzer.py       # Groq vision — batched region analysis
│   ├── severity_classifier.py      # Rule-based + LLM severity scoring
│   └── bug_report_generator.py     # Page + full report builder
│
├── rag/
│   ├── data_loader.py              # Loads domain knowledge from /data
│   └── vector_store.py             # FAISS vector store, lazy-loaded & cached
│
├── data/
│   ├── bugs.txt                    # Known bugs (RAG training data)
│   └── test_cases.txt              # Example test case templates
│
├── visual/                         # Section-level visual comparison pipeline
│   ├── figma_section_extractor.py  # Crops named sections from Figma frame PNG
│   ├── section_matcher.py          # Greedy Figma↔DOM section pairing (semantic + positional)
│   ├── section_comparator.py       # SSIM + pixel diff per matched section pair
│   └── section_alignment_engine.py # DOM coordinate normalization + region→section lookup
│
├── qa/
│   └── fix_recommendation_engine.py # AI-generated developer-ready fix guides per issue
│
├── tests/                          # pytest suite (104 tests, no external calls)
│   ├── test_utils.py
│   ├── test_severity_classifier.py
│   ├── test_visual_comparator.py
│   ├── test_section_alignment.py
│   ├── test_section_matcher.py
│   ├── test_figma_extractor.py
│   ├── test_shopify_scraper.py
│   └── test_fix_recommendation.py
│
└── ui/
    └── index.html                  # Frontend SPA (Bug QA + Visual QA + AI Crawl tabs)
```

## Running the Project

```bash
pip install -r requirements.txt
playwright install chromium
cp .env.example .env   # then fill in keys

uvicorn app:app --reload
```

Frontend: `http://localhost:8000/ui/index.html`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/qa-ai` | Main — bug analysis + test case generation |
| `POST` | `/qa-ai/stream` | Same, SSE streaming |
| `POST` | `/run-tests` | Execute test cases via Playwright |
| `POST` | `/visual-qa` | Start Visual QA job (202 async, returns job_id) |
| `GET` | `/visual-qa/{job_id}` | Poll Visual QA job status + results |
| `GET` | `/history` | List recent analyses |
| `GET` | `/history/{id}` | Fetch single history record |
| `DELETE` | `/history/{id}` | Delete history record |
| `POST` | `/analyze-bug` | Legacy — agent bug analysis only |
| `POST` | `/test-cases` | Legacy — test case generation only |
| `POST` | `/bug-analysis` | Legacy — basic bug analysis only |

### Main Request (`POST /qa-ai`)

```json
{ "input_text": "Login fails when password contains special characters" }
```

### Visual QA Request (`POST /visual-qa`)

```json
{
  "shopify_url": "https://your-store.myshopify.com",
  "figma_url": "https://www.figma.com/design/FILE_KEY/...",
  "pages": ["home", "product", "collection", "cart"],
  "diff_threshold": 0.05
}
```

Returns `{ "job_id": "..." }`. Poll `GET /visual-qa/{job_id}` until `status` is `complete` or `failed`.

## Data Flow

```
Bug QA:
User Input → Agent Manager → Bug Agent + Test Case Service (parallel)
                           → Groq LLM + FAISS RAG → JSON → MongoDB history

Visual QA:
POST /visual-qa → MongoDB job (pending) → BackgroundTask
    ├── Figma REST API → frames PNG @2x + typography tokens
    ├── Playwright (thread pool) → Shopify screenshots + DOM section extraction
    ├── Section pipeline (preferred):
    │   ├── Crop Figma sections + match to live DOM sections (semantic + positional)
    │   ├── SSIM + pixel diff per section pair
    │   └── Groq vision → issue analysis per significant section pair
    ├── Full-page fallback (when section pipeline has < 2 pairs):
    │   ├── Pillow pixel diff → BFS regions
    │   └── Groq vision → issue analysis per region
    ├── Rule-based + LLM severity classification
    ├── AI fix recommendation per issue (root cause, CSS snippet, effort estimate)
    └── MongoDB job (complete) → UI polls result
```

## Key Architecture Decisions

- **CORS** is open (`allow_origins=["*"]`) — restrict before production.
- **LLM text model**: `llama-3.3-70b-versatile` via Groq's OpenAI-compatible endpoint.
- **LLM vision model**: `meta-llama/llama-4-scout-17b-16e-instruct` — called once per page with all diff regions batched.
- **Playwright**: sync API running in `run_in_threadpool` — avoids Windows `SelectorEventLoop` subprocess error.
- **RAG retrieval**: Top-2 FAISS similarity results from `data/bugs.txt` and `data/test_cases.txt`.
- **Vector store**: Lazy-loaded on first request and cached in memory.
- **Severity classification**: Critical/Low trusted from rules; LLM only called for borderline Medium/High.
- **MongoDB**: base64 images returned in API response but stripped before saving to DB.
- **JSON extraction**: LLM outputs parsed with regex fallback to handle markdown-wrapped JSON.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq LLM service — text + vision |
| `GROQ_MODEL` | No | Defaults to `llama-3.3-70b-versatile` |
| `FIGMA_API_TOKEN` | For Visual QA | Figma Personal Access Token |
| `MONGODB_URI` | Yes | Defaults to `mongodb://localhost:27017` |
| `CORS_ORIGINS` | No | Defaults to `*` — restrict before production |
| `TESTRIX_API_KEY` | No | Enables `X-API-Key` header auth on all endpoints; leave blank for dev mode |

## Running Tests

```bash
pytest           # run all 104 tests
pytest -q        # quiet mode
pytest tests/test_visual_comparator.py  # single file
```

Tests cover pure/deterministic code only. LLM-dependent paths are mocked with `unittest.mock`. No external API calls are made during tests.

## Known Limitations

- CORS allows all origins — restrict `CORS_ORIGINS` before production.
- Authentication is optional (`TESTRIX_API_KEY`) — disabled by default in dev mode.
- Rate limiting via `slowapi`: 10 req/min on `/qa-ai` and `/visual-qa`, 20 req/min on legacy endpoints.
- LLM responses can occasionally fail JSON parsing; regex fallback returns raw text.
- Playwright Visual QA runs sync in a thread pool — concurrent jobs share the default pool.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore