import os
import logging
from datetime import datetime, timezone

from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")

# ── Lazy client — not constructed until first DB call ─────────────────────────
# mongodb+srv:// URIs do a DNS SRV lookup at MongoClient() time, which fails
# during import in environments without Atlas access (CI, dev without VPN, tests).
# Deferring construction means the import always succeeds; the error surfaces
# only when code actually tries to reach MongoDB.
_client: MongoClient | None = None
_indexes_ensured = False


def _get_db():
    global _client, _indexes_ensured
    if _client is None:
        _client = MongoClient(_MONGODB_URI)
    return _client["testrix"]


def _maybe_ensure_indexes() -> None:
    global _indexes_ensured
    if _indexes_ensured:
        return
    db = _get_db()
    try:
        db["history"].create_index([("timestamp", DESCENDING)], background=True)
        db["visual_qa_jobs"].create_index([("created_at", DESCENDING)], background=True)
        db["visual_qa_jobs"].create_index([("status", DESCENDING)], background=True)
        db["ai_crawl_jobs"].create_index([("created_at", DESCENDING)], background=True)
        db["ai_crawl_jobs"].create_index([("status", DESCENDING)], background=True)
        logger.info("MongoDB indexes ensured")
        _indexes_ensured = True
    except Exception as e:
        logger.warning(f"MongoDB index creation failed (non-fatal): {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col_history():
    return _get_db()["history"]

def _col_vqa():
    return _get_db()["visual_qa_jobs"]

def _col_crawl():
    return _get_db()["ai_crawl_jobs"]


def _serialize(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("timestamp"), datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()
    return doc


def _serialize_vqa(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])
    for key in ("created_at", "updated_at"):
        if isinstance(doc.get(key), datetime):
            doc[key] = doc[key].isoformat()
    return doc


# ── History ───────────────────────────────────────────────────────────────────

def save_history(input_text: str, bug_analysis: dict, test_cases: list) -> str:
    _maybe_ensure_indexes()
    doc = {
        "input_text": input_text,
        "bug_analysis": bug_analysis,
        "test_cases": test_cases,
        "timestamp": datetime.now(timezone.utc),
    }
    result = _col_history().insert_one(doc)
    logger.info(f"History saved — id={result.inserted_id}")
    return str(result.inserted_id)


def get_history(limit: int = 10) -> list[dict]:
    projection = {
        "input_text": 1,
        "timestamp": 1,
        "bug_analysis.bug": 1,
    }
    cursor = _col_history().find({}, projection).sort("timestamp", DESCENDING).limit(limit)
    return [_serialize(doc) for doc in cursor]


def get_history_item(history_id: str) -> dict | None:
    try:
        oid = ObjectId(history_id)
    except Exception:
        return None
    doc = _col_history().find_one({"_id": oid})
    return _serialize(doc) if doc else None


def delete_history_item(history_id: str) -> bool:
    try:
        oid = ObjectId(history_id)
    except Exception:
        return False
    result = _col_history().delete_one({"_id": oid})
    return result.deleted_count == 1


# ── Visual QA jobs ────────────────────────────────────────────────────────────

def create_vqa_job(shopify_url: str, figma_url: str, pages: list[str]) -> str:
    _maybe_ensure_indexes()
    doc = {
        "shopify_url": shopify_url,
        "figma_file_key": _extract_figma_key(figma_url),
        "figma_url": figma_url,
        "pages": pages,
        "status": "pending",
        "progress": "",
        "result": None,
        "error": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = _col_vqa().insert_one(doc)
    logger.info(f"VQA job created — id={result.inserted_id}")
    return str(result.inserted_id)


def update_vqa_job(job_id: str, **fields) -> None:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return
    fields["updated_at"] = datetime.now(timezone.utc)
    _col_vqa().update_one({"_id": oid}, {"$set": fields})


def get_vqa_job(job_id: str) -> dict | None:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return None
    doc = _col_vqa().find_one({"_id": oid})
    return _serialize_vqa(doc) if doc else None


# ── AI Crawl jobs ─────────────────────────────────────────────────────────────

def create_ai_crawl_job(seed_url: str, max_pages: int, max_depth: int) -> str:
    _maybe_ensure_indexes()
    doc = {
        "seed_url": seed_url,
        "max_pages": max_pages,
        "max_depth": max_depth,
        "status": "pending",
        "progress": "",
        "result": None,
        "error": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = _col_crawl().insert_one(doc)
    logger.info(f"AI crawl job created — id={result.inserted_id}")
    return str(result.inserted_id)


def update_ai_crawl_job(job_id: str, **fields) -> None:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return
    fields["updated_at"] = datetime.now(timezone.utc)
    _col_crawl().update_one({"_id": oid}, {"$set": fields})


def get_ai_crawl_job(job_id: str) -> dict | None:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return None
    doc = _col_crawl().find_one({"_id": oid})
    return _serialize_vqa(doc) if doc else None


# ── Utility ───────────────────────────────────────────────────────────────────

def _extract_figma_key(figma_url: str) -> str:
    parts = figma_url.split("/")
    for i, part in enumerate(parts):
        if part in ("design", "file", "board", "slides", "make") and i + 1 < len(parts):
            return parts[i + 1]
    return ""
