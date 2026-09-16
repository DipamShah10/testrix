import asyncio
import os
import logging
import re

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Explicit timeout — without one, a single stalled connection (not an error,
# just a hang) can block silently for the SDK's default (~10 min) with zero
# CPU usage and no exception raised, which looks identical to "the job is
# stuck" from the outside. 45s is generous for a single chat completion but
# short enough that a genuine stall surfaces as a normal, retryable error.
client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    timeout=45.0,
    max_retries=0,  # we do our own retry/fallback loop below — don't double up
)

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_SYSTEM_PROMPT = "You are a senior QA engineer and API testing expert."

# Each Groq model has its own separate TPM bucket, so switching models on a
# 429 gets a fresh quota immediately — no need to wait out the primary
# model's cooldown. Order matters: primary first, then fallbacks tried in
# order. Override with GROQ_MODEL_FALLBACKS (comma-separated) if needed;
# defaults to other text-capable models confirmed live on this account's
# GET /v1/models on 2026-09-16.
_DEFAULT_FALLBACKS = ["openai/gpt-oss-20b", "groq/compound-mini", "qwen/qwen3.8-27b"]
_raw_fallbacks = os.environ.get("GROQ_MODEL_FALLBACKS", "")
_configured_fallbacks = [m.strip() for m in _raw_fallbacks.split(",") if m.strip()] or _DEFAULT_FALLBACKS
# Primary first, de-duplicated, fallbacks never include the primary twice
MODEL_CHAIN = [MODEL] + [m for m in _configured_fallbacks if m != MODEL]

_MAX_RETRIES = 3
# Groq's rate-limit body includes "Please try again in {N}s" for per-minute
# caps, but "Please try again in {M}m{N}s" for per-day caps (e.g. "2m43.296s").
# The old regex only matched the trailing "{N}s" and silently dropped the
# minutes part, so a multi-minute wait got parsed as a few seconds.
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)
# A daily (TPD) or requests-per-day cap won't clear for potentially hours —
# retrying the same model on any fixed backoff is pointless. Skip straight
# to the next model in the chain instead of sleeping at all.
_DAILY_LIMIT_RE = re.compile(r"tokens per day|requests per day|\bTPD\b|\bRPD\b", re.IGNORECASE)


def _is_daily_limit(exc: RateLimitError) -> bool:
    return bool(_DAILY_LIMIT_RE.search(str(exc)))


def _retry_delay(exc: RateLimitError, attempt: int) -> float:
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        minutes = float(match.group(1)) if match.group(1) else 0.0
        seconds = float(match.group(2))
        return minutes * 60 + seconds + 0.5
    return 2.0 * (attempt + 1)


async def ask_ai(prompt: str) -> str:
    last_exc: Exception | None = None

    for model_index, model in enumerate(MODEL_CHAIN):
        is_last_model = model_index == len(MODEL_CHAIN) - 1
        # Only the primary model gets the full retry+backoff treatment (its
        # 429 might just be a momentary burst). Fallback models get a single
        # try each — if a fallback is also rate-limited, move on immediately
        # rather than sleeping on a model we're about to abandon anyway.
        attempts = _MAX_RETRIES if model_index == 0 else 1

        for attempt in range(attempts):
            try:
                logger.info(f"LLM call — model={model}, prompt_chars={len(prompt)}")
                response = await client.chat.completions.create(
                    model=model,
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
                daily = _is_daily_limit(e)
                if daily:
                    logger.warning(f"LLM daily quota exhausted on {model} — no point retrying this model today")
                if not daily and attempt < attempts - 1:
                    delay = _retry_delay(e, attempt)
                    logger.warning(
                        f"LLM rate-limited on {model} (attempt {attempt + 1}/{attempts}) — "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                elif not is_last_model:
                    logger.warning(
                        f"LLM rate-limited on {model} — switching to fallback "
                        f"model {MODEL_CHAIN[model_index + 1]}"
                    )
                    break  # stop retrying this model, move to the next one
                else:
                    logger.error(f"LLM call failed after exhausting model chain {MODEL_CHAIN}: {e}")
            except (APITimeoutError, APIConnectionError) as e:
                # A stall/connection drop is not a quota problem — a different
                # model is unlikely to be stuck on the same dead connection,
                # so treat it the same as a rate limit: short retry on the
                # primary, then move down the fallback chain.
                last_exc = e
                if attempt < attempts - 1:
                    delay = 2.0 * (attempt + 1)
                    logger.warning(
                        f"LLM call timed out/dropped on {model} (attempt {attempt + 1}/{attempts}) — "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                elif not is_last_model:
                    logger.warning(
                        f"LLM call timed out/dropped on {model} — switching to fallback "
                        f"model {MODEL_CHAIN[model_index + 1]}"
                    )
                    break
                else:
                    logger.error(f"LLM call failed after exhausting model chain {MODEL_CHAIN}: {e}")
            except Exception as e:
                logger.error(f"LLM call failed on {model}: {e}")
                raise RuntimeError(f"AI service error: {str(e)}") from e

    raise RuntimeError(f"AI service error: {last_exc}") from last_exc