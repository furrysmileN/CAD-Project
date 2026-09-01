"""连通性探测：文本已通，再测一张图。不打印密钥。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rq2_harness.api_client import APISettings, chat_completion
from rq2_harness.common import read_jsonl
from rq2_harness.prompting import RAW_CADQUERY_SYSTEM, build_messages
from rq2_harness.qwen_keyfile import apply_qwen_keyfile


def main() -> int:
    info = apply_qwen_keyfile()
    print("keyfile", info, flush=True)
    row = next(iter(read_jsonl(Path(__file__).resolve().parents[1] / "outputs" / "harness_vs_cadrille" / "manifest_n40.jsonl")))
    print("sample", row["sample_id"], flush=True)
    settings = APISettings(
        api_key_env="VLM_API_KEY",
        base_url_env="VLM_BASE_URL",
        model_env="VLM_MODEL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3.8-max",
        timeout_sec=90,
        max_retries=1,
        retry_base_sec=1,
        temperature=0,
        max_tokens=80,
        json_mode=False,
        extra_body={},
        request_interval_sec=0,
    )
    messages, _audit = build_messages(row, "I", image_max_edge=512, plan_version="v3")
    images = [item for item in messages[1]["content"] if item.get("type") == "image_url"]
    print("encoded_images", len(images), "first_url_chars", len(images[0]["image_url"]["url"]), flush=True)
    messages[0]["content"] = RAW_CADQUERY_SYSTEM
    messages[1]["content"] = images[:1] + [{"type": "text", "text": "Reply with one word: the part family."}]
    try:
        out = chat_completion(messages, settings)
    except Exception as exc:
        print("vision_fail", type(exc).__name__, str(exc)[:400], flush=True)
        return 1
    print(
        "vision_ok",
        {"chars": len(out.get("text") or ""), "latency": out.get("latency_sec"), "text": (out.get("text") or "")[:80]},
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
