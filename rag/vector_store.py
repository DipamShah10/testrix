import os
import logging
import threading
from rag.data_loader import load_data

logger = logging.getLogger(__name__)

_db = None
_db_lock = threading.Lock()
_INDEX_PATH = "faiss_index"


def _get_db():
    # Imported here, not at module load — langchain/FAISS is a heavy,
    # optional dependency used only by the legacy /test-cases and
    # /bug-analysis endpoints. A module-level import means any environment
    # issue in this dependency chain (e.g. a blocked native DLL) takes down
    # the entire app at startup, including Visual QA, which never calls this.
    global _db
    if _db is not None:
        return _db
    with _db_lock:
        if _db is not None:
            return _db

        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embedding = HuggingFaceEmbeddings()

        if os.path.exists(_INDEX_PATH):
            logger.info("Loading FAISS index from disk")
            _db = FAISS.load_local(_INDEX_PATH, embedding, allow_dangerous_deserialization=True)
        else:
            logger.info("Building FAISS index from data files")
            docs = load_data()
            _db = FAISS.from_texts(docs, embedding)
            _db.save_local(_INDEX_PATH)
            logger.info(f"FAISS index saved to {_INDEX_PATH}/")

    return _db


def search_context(query: str) -> list[str]:
    db = _get_db()
    results = db.similarity_search(query, k=3)
    return [r.page_content for r in results]