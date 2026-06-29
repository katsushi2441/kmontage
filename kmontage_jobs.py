from __future__ import annotations

import json
import os
from typing import Any

import requests


def _generate_url(base_url: str) -> str:
    base_url = (base_url or "").rstrip("/")
    if base_url.endswith("/api/generate"):
        return base_url
    return f"{base_url}/api/generate"


def ollama_generate_job(
    prompt: str,
    ollama_url: str = "http://192.168.0.14:11434",
    model: str = "gemma4:12b-it-qat",
    temperature: float = 0.1,
    num_predict: int = 4096,
    request_timeout: int = 900,
    source: str = "rqdb4ai",
    **_: Any,
) -> dict[str, Any]:
    """RQDB4AI entrypoint for serialized kmontage Ollama generation."""
    prompt = str(prompt or "")
    if not prompt.strip():
        raise RuntimeError("prompt is required")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
        },
    }
    endpoint = _generate_url(ollama_url or os.environ.get("OLLAMA_URL", "http://192.168.0.14:11434"))
    res = requests.post(endpoint, json=payload, timeout=int(request_timeout))
    res.raise_for_status()
    data = res.json()
    response = str(data.get("response") or "")
    if not response.strip():
        raise RuntimeError(json.dumps({"error": "empty_ollama_response", "model": model, "source": source}, ensure_ascii=False))
    return {
        "ok": True,
        "response": response,
        "model": model,
        "ollama_url": ollama_url,
        "source": source,
        "response_chars": len(response),
    }
