from fastapi.concurrency import run_in_threadpool
from ai_engine.llm import ask_ai
from ai_engine.prompts import bug_analysis_prompt
from rag.vector_store import search_context


async def analyze_bug(bug: str) -> str:
    context = await run_in_threadpool(search_context, bug)
    prompt = bug_analysis_prompt(bug, context)
    return await ask_ai(prompt)