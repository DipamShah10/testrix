import os
import json
import logging
import asyncio

from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

# generate_test_cases pulls in langchain/FAISS (rag.vector_store) transitively,
# which is only needed by the deprecated /test-cases endpoint below — imported
# lazily inside that handler so a broken optional RAG dependency (e.g. a
# blocked native DLL) can't take down the entire app, including Visual QA,
# which never touches this code path.
from services.bug_analysis_service import analyze_bug as basic_bug_analysis
from agents.bug_agent import analyze_bug as agent_analyze_bug
from agents.agent_manager import run_qa_ai
from services.test_runner import run_tests
from agents.visual_qa_agent import run_visual_qa, run_single_section_qa, run_multi_section_qa
from agents.ai_crawl_agent import run_ai_crawl
from services.db import (
    get_history, get_history_item, delete_history_item,
    create_vqa_job, get_vqa_job,
    create_ai_crawl_job, get_ai_crawl_job,
)
from services.site_crawler import discover_site

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Fail fast on missing required env vars — surface the problem at startup, not mid-request
_REQUIRED_ENV = ["GROQ_API_KEY"]
_OPTIONAL_ENV = {"FIGMA_API_TOKEN": "/visual-qa"}
for _var in _REQUIRED_ENV:
    if not os.environ.get(_var):
        raise RuntimeError(f"Missing required environment variable: {_var}. Check your .env file.")

limiter = Limiter(key_func=get_remote_address)

# ── Optional API key auth ─────────────────────────────────────────────────────
# Set TESTRIX_API_KEY in .env to enable. Leave unset to run without auth (dev mode).
_API_KEY_VALUE = os.environ.get("TESTRIX_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str | None = Security(_api_key_header)) -> None:
    if not _API_KEY_VALUE:
        return  # auth disabled — dev mode
    if key != _API_KEY_VALUE:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app = FastAPI(
    title="Testrix",
    version="2.0.0",
    description="AI-powered QA automation",
    dependencies=[Depends(verify_api_key)],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Serve per-job region screenshots.  Create the dir first so StaticFiles
# doesn't throw if no VQA job has run yet.
from pathlib import Path as _Path
_Path("artifacts/screenshots").mkdir(parents=True, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory="artifacts/screenshots"), name="screenshots")

_Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory="artifacts/reports"), name="reports")


# ---------- Request models ----------

class FeatureRequest(BaseModel):
    feature: str = Field(..., min_length=3, max_length=2000)


class BugRequest(BaseModel):
    bug: str = Field(..., min_length=3, max_length=2000)


class QARequest(BaseModel):
    input_text: str = Field(..., min_length=5, max_length=2000)


class RunTestsRequest(BaseModel):
    base_url: str = Field(..., min_length=1, max_length=500)
    test_cases: list[dict] = Field(...)
    timeout_ms: int = Field(default=5000, ge=1000, le=30000)
    auth_token: str | None = None


class VisualQARequest(BaseModel):
    shopify_url: str = Field(..., min_length=8, max_length=500, description="Live Shopify store URL (https only)")
    figma_url: str = Field(..., min_length=8, max_length=500, description="Figma file or frame URL")
    pages: list[str] = Field(
        default=["home", "product", "collection", "cart"],
        description="Pages to test: home, product, collection, cart",
    )
    shopify_password: str | None = Field(default=None, description="Password for password-protected stores")
    diff_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Pixel diff sensitivity (0–1)")
    exclude_sections: list[str] = Field(
        default_factory=list,
        description="Section types to skip entirely, e.g. ['header', 'footer', 'collection']",
    )
    target_section: str | None = Field(
        default=None,
        description=(
            "Free-text section name/heading to scope the run to a single section, "
            "e.g. 'Curated Selection (Featured Pieces)', instead of the whole page. "
            "When set, only the first entry in `pages` is used. Compares typography, "
            "element dimensions, and spacing for that section — full pixel/vision "
            "diff scoped to one section isn't supported yet. Mutually exclusive with "
            "`section_limit`."
        ),
    )
    section_limit: int | None = Field(
        default=None, ge=1, le=20,
        description=(
            "Scope the run to the first N real-content sections of the page, in page "
            "order, after dropping any excluded section types (see exclude_sections) "
            "— e.g. section_limit=3 tests the first 3 sections a visitor would "
            "actually scroll through, skipping the header/footer even if excluded "
            "types would otherwise be counted. Only the first entry in `pages` is "
            "used. Mutually exclusive with `target_section`."
        ),
    )
    include_typography: bool = Field(
        default=True,
        description=(
            "Whether to run typography comparison. Only applies to target_section/"
            "section_limit runs — set to False to check only dimensions/spacing when "
            "typography has already been separately verified."
        ),
    )


class CrawlRequest(BaseModel):
    seed_url: str = Field(..., min_length=8, max_length=500, description="HTTPS entry URL (same host is crawled)")
    max_pages: int = Field(default=50, ge=1, le=200)
    max_depth: int = Field(default=2, ge=0, le=5)
    persist: bool = Field(default=True, description="Write artifacts/crawl/latest.json")


class AICrawlRequest(BaseModel):
    seed_url: str = Field(..., min_length=8, max_length=500, description="HTTPS entry URL to crawl and QA-test")
    max_pages: int = Field(default=20, ge=1, le=100, description="Max pages to discover and inspect")
    max_depth: int = Field(default=2, ge=0, le=4, description="BFS link depth from seed/sitemap")


# ---------- Health ----------

@app.get("/")
async def home():
    return {"message": "Testrix Running", "version": "2.0.0"}


# ---------- Site discovery (sitemap + bounded crawl) ----------

@app.post("/crawl", tags=["Discovery"])
@limiter.limit("5/minute")
async def crawl_site_api(request: Request, body: CrawlRequest):
    """
    Discover URLs on the same host as ``seed_url`` via sitemap.xml (+ index) and HTML link BFS.
    Returns ``routes_for_pipeline`` path strings suitable for ``main.py --pages`` / ``BrowserAgent``.
    """
    from services.shopify_scraper import validate_url

    try:
        validate_url(body.seed_url)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    try:
        result = await run_in_threadpool(
            discover_site,
            body.seed_url,
            body.max_pages,
            body.max_depth,
            body.persist,
        )
        return result
    except Exception as e:
        logger.error(f"/crawl error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------- AI Crawl (full pipeline: discover → inspect → QA → AI review) ----------

@app.post("/ai-crawl", tags=["AI Crawl"], status_code=202)
@limiter.limit("2/minute")
async def ai_crawl_start(request: Request, body: AICrawlRequest, background_tasks: BackgroundTasks):
    """
    Start an AI-powered crawl job. Discovers pages, takes Playwright screenshots,
    runs accessibility checks, QA evaluation, and generates an AI GO/NO-GO report.
    Returns immediately with a job_id. Poll GET /ai-crawl/{job_id} for results.
    """
    from services.shopify_scraper import validate_url

    try:
        validate_url(body.seed_url)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    job_id = await run_in_threadpool(
        create_ai_crawl_job, body.seed_url, body.max_pages, body.max_depth
    )

    background_tasks.add_task(
        run_ai_crawl,
        job_id=job_id,
        seed_url=body.seed_url,
        max_pages=body.max_pages,
        max_depth=body.max_depth,
    )

    logger.info(f"AI crawl job started — id={job_id}, url={body.seed_url}, max_pages={body.max_pages}")
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job started. Poll /ai-crawl/{job_id} for results.",
    }


@app.get("/ai-crawl/{job_id}", tags=["AI Crawl"])
async def ai_crawl_status(job_id: str):
    """Poll AI crawl job status. status: pending | running | complete | failed."""
    job = await run_in_threadpool(get_ai_crawl_job, job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job


# ---------- Legacy endpoints ----------

@app.post("/test-cases", deprecated=True, tags=["Legacy"])
@limiter.limit("20/minute")
async def test_cases(request: Request, body: FeatureRequest):
    try:
        from services.test_case_service import generate_test_cases
        result = await generate_test_cases(body.feature)
        return {"result": result}
    except Exception as e:
        logger.error(f"/test-cases error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/bug-analysis", deprecated=True, tags=["Legacy"])
@limiter.limit("20/minute")
async def bug_analysis(request: Request, body: BugRequest):
    try:
        result = await basic_bug_analysis(body.bug)
        return {"result": result}
    except Exception as e:
        logger.error(f"/bug-analysis error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze-bug", deprecated=True, tags=["Legacy"])
@limiter.limit("20/minute")
async def analyze_bug_api(request: Request, body: BugRequest):
    try:
        result = await agent_analyze_bug(body.bug)
        return {"result": result}
    except Exception as e:
        logger.error(f"/analyze-bug error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------- Main QA endpoint ----------

@app.post("/qa-ai", tags=["QA"])
@limiter.limit("10/minute")
async def qa_ai_api(request: Request, body: QARequest):
    try:
        logger.info(f"QA request: {body.input_text[:100]!r}")
        result = await run_qa_ai(body.input_text)
        return {"result": result}
    except Exception as e:
        logger.error(f"/qa-ai error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/qa-ai/stream", tags=["QA"])
@limiter.limit("10/minute")
async def qa_ai_stream(request: Request, body: QARequest):
    """SSE endpoint — sends a progress event then the full result."""

    async def event_stream():
        try:
            yield f"data: {json.dumps({'status': 'analyzing', 'message': 'Running bug analysis and generating test cases in parallel…'})}\n\n"

            result = await run_qa_ai(body.input_text)

            payload = {"status": "complete", "result": result}
            yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            logger.error(f"/qa-ai/stream error: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- Playwright test runner ----------

@app.post("/run-tests", tags=["QA"])
@limiter.limit("5/minute")
async def run_tests_api(request: Request, body: RunTestsRequest):
    if not body.test_cases:
        return JSONResponse(status_code=422, content={"error": "test_cases must not be empty"})
    try:
        logger.info(f"Running {len(body.test_cases)} tests against {body.base_url}")
        result = await run_tests(
            base_url=body.base_url,
            test_cases=body.test_cases,
            timeout_ms=body.timeout_ms,
            auth_token=body.auth_token,
        )
        return result
    except Exception as e:
        logger.error(f"/run-tests error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------- Visual QA ----------

@app.post("/visual-qa", tags=["Visual QA"], status_code=202)
@limiter.limit("2/minute")
async def visual_qa_start(request: Request, body: VisualQARequest, background_tasks: BackgroundTasks):
    """
    Start a visual QA job. Returns immediately with a job_id.
    Poll GET /visual-qa/{job_id} for status and results.
    """
    # Warn if optional keys are missing — don't block the request
    missing = [v for v in ("FIGMA_API_TOKEN",) if not os.environ.get(v)]
    if missing:
        logger.warning(f"/visual-qa called but missing env vars: {missing}")

    try:
        from services.shopify_scraper import validate_url
        validate_url(body.shopify_url)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    job_id = await run_in_threadpool(
        create_vqa_job, body.shopify_url, body.figma_url, body.pages
    )

    if body.target_section:
        page_name = body.pages[0] if body.pages else "home"
        background_tasks.add_task(
            run_single_section_qa,
            job_id=job_id,
            shopify_url=body.shopify_url,
            figma_url=body.figma_url,
            page_name=page_name,
            target_section=body.target_section,
            shopify_password=body.shopify_password,
            include_typography=body.include_typography,
        )
        logger.info(
            f"Visual QA job started — id={job_id}, page={page_name}, "
            f"target_section={body.target_section!r}, include_typography={body.include_typography}"
        )
    elif body.section_limit:
        page_name = body.pages[0] if body.pages else "home"
        background_tasks.add_task(
            run_multi_section_qa,
            job_id=job_id,
            shopify_url=body.shopify_url,
            figma_url=body.figma_url,
            page_name=page_name,
            section_limit=body.section_limit,
            exclude_sections=body.exclude_sections,
            shopify_password=body.shopify_password,
            include_typography=body.include_typography,
        )
        logger.info(
            f"Visual QA job started — id={job_id}, page={page_name}, "
            f"section_limit={body.section_limit}, exclude={body.exclude_sections}, "
            f"include_typography={body.include_typography}"
        )
    else:
        background_tasks.add_task(
            run_visual_qa,
            job_id=job_id,
            shopify_url=body.shopify_url,
            figma_url=body.figma_url,
            pages=body.pages,
            shopify_password=body.shopify_password,
            diff_threshold=body.diff_threshold,
            exclude_sections=body.exclude_sections,
        )
        logger.info(f"Visual QA job started — id={job_id}, pages={body.pages}")

    return {"job_id": job_id, "status": "pending", "message": "Job started. Poll /visual-qa/{job_id} for results."}


@app.get("/visual-qa/{job_id}", tags=["Visual QA"])
async def visual_qa_status(job_id: str):
    """Poll job status. status: pending | running | complete | failed."""
    job = await run_in_threadpool(get_vqa_job, job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job


# ---------- History ----------

@app.get("/history", tags=["History"])
async def list_history(limit: int = 10):
    """Return the last N analyses, newest first."""
    try:
        records = await run_in_threadpool(get_history, min(limit, 100))
        return {"records": records, "count": len(records)}
    except Exception as e:
        logger.error(f"/history error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/history/{history_id}", tags=["History"])
async def get_history_record(history_id: str):
    """Fetch a single history record by ID (for reloading into the UI)."""
    record = await run_in_threadpool(get_history_item, history_id)
    if not record:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return record


@app.delete("/history/{history_id}", tags=["History"])
async def remove_history_record(history_id: str):
    """Delete a history record by ID."""
    deleted = await run_in_threadpool(delete_history_item, history_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return {"deleted": True}