from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any


class MissingAPIKeyError(RuntimeError):
    pass


class FatalAPIError(RuntimeError):
    pass


_PACING_LOCK = threading.Lock()
_LAST_REQUEST_STARTED = 0.0


@dataclass(frozen=True)
class APISettings:
    api_key_env: str
    base_url_env: str
    model_env: str
    default_base_url: str
    default_model: str
    timeout_sec: float
    max_retries: int
    retry_base_sec: float
    temperature: float
    max_tokens: int
    json_mode: bool
    extra_body: dict[str, Any]
    request_interval_sec: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "APISettings":
        return cls(
            api_key_env=config["env_api_key"],
            base_url_env=config["env_base_url"],
            model_env=config["env_model"],
            default_base_url=config["default_base_url"],
            default_model=config["default_model"],
            timeout_sec=float(config["timeout_sec"]),
            max_retries=int(config["max_retries"]),
            retry_base_sec=float(config.get("retry_base_sec", 2)),
            temperature=float(config["temperature"]),
            max_tokens=int(config["max_tokens"]),
            json_mode=bool(config.get("json_mode", False)),
            extra_body=dict(config.get("extra_body") or {}),
            request_interval_sec=max(
                0.0,
                float(config.get("request_interval_sec", 0.0)),
            ),
        )

    def resolved(self, *, require_key: bool = True) -> dict[str, Any]:
        values = {
            "api_key": os.environ.get(self.api_key_env, "").strip(),
            "base_url": os.environ.get(self.base_url_env, self.default_base_url).strip(),
            "model": os.environ.get(self.model_env, self.default_model).strip(),
        }
        if require_key and not values["api_key"]:
            raise MissingAPIKeyError(
                f"缺少 API key：请设置环境变量 {self.api_key_env}；仅验证流程可使用 --dry-run。"
            )
        return values


def chat_completion(messages: list[dict[str, Any]], settings: APISettings) -> dict[str, Any]:
    global _LAST_REQUEST_STARTED

    from openai import OpenAI

    resolved = settings.resolved()
    with _PACING_LOCK:
        remaining = settings.request_interval_sec - (
            time.monotonic() - _LAST_REQUEST_STARTED
        )
        if remaining > 0:
            time.sleep(remaining)
        _LAST_REQUEST_STARTED = time.monotonic()
    client = OpenAI(api_key=resolved["api_key"], base_url=resolved["base_url"], timeout=settings.timeout_sec)
    errors: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for attempt in range(1, settings.max_retries + 1):
        started = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": resolved["model"],
                "messages": messages,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
            }
            if settings.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if settings.extra_body:
                kwargs["extra_body"] = settings.extra_body
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            usage = response.usage
            return {
                "text": choice.message.content or "",
                "model": resolved["model"],
                "base_url": resolved["base_url"],
                "finish_reason": getattr(choice, "finish_reason", None),
                "attempt": attempt,
                "attempt_latency_sec": time.perf_counter() - started,
                "latency_sec": time.perf_counter() - total_started,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                },
                "retry_errors": errors,
            }
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            message = str(exc)
            errors.append(
                {
                    "attempt": attempt,
                    "status_code": status,
                    "error_type": type(exc).__name__,
                    "message": message[:500],
                    "latency_sec": time.perf_counter() - started,
                }
            )
            lowered = message.lower()
            if status in {401, 403, 404} or any(
                marker in lowered for marker in ("unauthorized", "invalid api key", "permission denied", "model not found")
            ):
                raise FatalAPIError(message) from exc
            if attempt == settings.max_retries:
                raise RuntimeError(f"API 调用重试 {attempt} 次后失败: {message}") from exc
            time.sleep(min(settings.retry_base_sec * (2 ** (attempt - 1)), 30.0))
    raise AssertionError("unreachable")
