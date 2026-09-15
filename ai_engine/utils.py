import json
import re
import logging

logger = logging.getLogger(__name__)


def _strip_markdown(content: str) -> str:
    content = content.strip()
    # Remove opening fence (```json, ```, etc.) anchored at the start
    content = re.sub(r"^```[a-zA-Z]*\s*\n?", "", content)
    # Remove closing fence anchored at the end
    content = re.sub(r"\n?```\s*$", "", content.strip())
    return content.strip()


def extract_json_object(content: str) -> dict | None:
    try:
        content = _strip_markdown(content)
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(content[start:end])
    except Exception as e:
        logger.warning(f"JSON object extraction failed: {e}")
        return None


def extract_json_array(content: str) -> list | None:
    try:
        content = _strip_markdown(content)
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(content[start:end])
    except Exception as e:
        logger.warning(f"JSON array extraction failed: {e}")
        return None