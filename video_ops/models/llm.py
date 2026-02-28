from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LLMConfig:
    provider: str  # none | openai | ollama

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = ""

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = ""


def load_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
    return LLMConfig(
        provider=provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", ""),
    )


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def enabled(self) -> bool:
        if self.cfg.provider == "openai":
            return bool(self.cfg.openai_api_key.strip())
        if self.cfg.provider == "ollama":
            return bool(self.cfg.ollama_model.strip())
        return False

    def chat(self, messages: List[Dict[str, Any]], images_b64_jpeg: Optional[List[str]] = None) -> str:
        provider = self.cfg.provider
        if provider == "openai":
            return self._chat_openai(messages, images_b64_jpeg=images_b64_jpeg)
        if provider == "ollama":
            return self._chat_ollama(messages)
        raise RuntimeError("LLM provider disabled")

    def _chat_ollama(self, messages: List[Dict[str, Any]]) -> str:
        url = self.cfg.ollama_host.rstrip("/") + "/api/chat"
        payload = {
            "model": self.cfg.ollama_model,
            "messages": messages,
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")

    def _chat_openai(self, messages: List[Dict[str, Any]], images_b64_jpeg: Optional[List[str]] = None) -> str:
        # Uses official OpenAI python SDK.
        from openai import OpenAI

        client = OpenAI(api_key=self.cfg.openai_api_key)

        if images_b64_jpeg:
            # Attach images to the last user message (simple pattern)
            new_msgs = []
            for m in messages:
                new_msgs.append(m)
            # Find last user message
            for i in range(len(new_msgs) - 1, -1, -1):
                if new_msgs[i].get("role") == "user":
                    content = [{"type": "text", "text": str(new_msgs[i].get("content", ""))}]
                    for b64 in images_b64_jpeg:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            }
                        )
                    new_msgs[i] = {"role": "user", "content": content}
                    break
            messages = new_msgs

        resp = client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


def image_file_to_b64_jpeg(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
