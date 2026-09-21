"""JSON extraction and structured chat helpers."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ChatClient(Protocol):
    def chat_text(
        self,
        model: str,
        messages: list[dict],
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> tuple[str, dict]:
        ...


def extract_json_blob(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_model(text: str, model: type[T]) -> T:
    blob = extract_json_blob(text)
    return model.model_validate_json(blob)


def chat_structured(
    client: ChatClient,
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    parser: Callable[[str], T],
    retries: int = 1,
) -> tuple[T, str, dict]:
    """Call chat API, parse JSON into Pydantic model, retry once on failure."""

    raw, usage = client.chat_text(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    total_usage = dict(usage)
    attempt = 0
    cur_messages = list(messages)
    cur_raw = raw
    while True:
        try:
            return parser(cur_raw), cur_raw, total_usage
        except Exception as err:  # noqa: BLE001
            if attempt >= retries:
                raise
            attempt += 1
            cur_messages = cur_messages + [
                {"role": "assistant", "content": cur_raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON for the schema. "
                        f"Error: {err}. Output ONLY a single JSON object, no prose."
                    ),
                },
            ]
            cur_raw, usage2 = client.chat_text(
                model=model_id,
                messages=cur_messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            total_usage = merge_usage(total_usage, usage2)


def merge_usage(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "input_tokens": int(a.get("input_tokens", 0)) + int(b.get("input_tokens", 0)),
        "output_tokens": int(a.get("output_tokens", 0)) + int(b.get("output_tokens", 0)),
        "cost": float(a.get("cost", 0) or 0) + float(b.get("cost", 0) or 0),
    }
    # Keep the most recent served model (the retry actually produced the result).
    served = b.get("served_model") or a.get("served_model")
    if served:
        merged["served_model"] = served
    return merged


def now_ms() -> int:
    return int(time.perf_counter() * 1000)
