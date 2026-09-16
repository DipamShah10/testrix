import logging
from fastapi.concurrency import run_in_threadpool
from ai_engine.llm import ask_ai
from ai_engine.prompts import test_case_prompt
from ai_engine.utils import extract_json_array
from rag.vector_store import search_context

logger = logging.getLogger(__name__)


async def generate_test_cases(feature: str):
    context = await run_in_threadpool(search_context, feature)
    prompt = test_case_prompt(feature, context)
    raw = await ask_ai(prompt)

    parsed = extract_json_array(raw)
    if parsed is not None:
        return parsed

    logger.warning("Test case service could not parse JSON array — returning raw")
    return {"error": "Invalid JSON from AI", "raw_output": raw}