import asyncio
import os
import logging
import re

from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_SYSTEM_PROMPT = "You are a senior QA engineer and API testing expert."

_MAX_RETRIES = 3
# Groq's TPM rate-limit body includes "Please try again in {N}s" — parse it so
# we wait exactly as long as asked instead of guessing a fixed backoff.
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _retry_delay(exc: RateLimitError, attempt: int) -> float:
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 0.5
    return 2.0 * (attempt + 1)


async def ask_ai(prompt: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            logger.info(f"LLM call — model={MODEL}, prompt_chars={len(prompt)}")
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content
            logger.info(f"LLM response — chars={len(content)}")
            return content
        except RateLimitError as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                delay = _retry_delay(e, attempt)
                logger.warning(
                    f"LLM rate-limited (attempt {attempt + 1}/{_MAX_RETRIES}) — "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"LLM call failed after {_MAX_RETRIES} attempts: {e}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"AI service error: {str(e)}") from e

    raise RuntimeError(f"AI service error: {last_exc}") from last_exc