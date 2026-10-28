"""Groq text service used by chat and optional disease explanations.

Image analysis is implemented separately in ``vision_service.py``. This module
only sends text to Groq; an optional disease explanation receives the short,
validated vision result plus any symptom text, never image pixels.
"""
from __future__ import annotations

import os
from typing import Any

import requests

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODELS = (
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
)

_SYSTEM_PROMPT = (
    "তুমি বাংলাদেশের কৃষকদের জন্য স্মার্ট কৃষি বুদ্ধিমত্তা নামের একটি বাংলা কৃষি সহায়ক। "
    "সহজ, সম্মানজনক ও সংক্ষিপ্ত বাংলায় উত্তর দাও। প্রয়োজনমতো Markdown শিরোনাম, **bold**, "
    "*italic* এবং ছোট bullet/numbered list ব্যবহার করো; অতিরিক্ত সাজসজ্জা কোরো না। "
    "সাধারণ চাষাবাদ, আবহাওয়া-সচেতন পরিকল্পনা এবং নিরাপদ পরবর্তী পদক্ষেপে সহায়তা করো। "
    "লাইভ বাজারদর, সরকারি তথ্য বা আবহাওয়া পাওয়া হয়েছে বলে দাবি কোরো না যদি সেগুলো "
    "অনুরোধে না থাকে। নমুনা বা সিমুলেটেড দামের কথা স্পষ্টভাবে নমুনা বলো। রোগের "
    "ছবি-ভিত্তিক শ্রেণিবিন্যাস নিশ্চিত রোগনির্ণয় নয়। কীটনাশক/সারের ব্র্যান্ড, মাত্রা, "
    "মিশ্রণ বা বিরতি অনুমান করে বলবে না; পণ্যের বর্তমান বাংলাদেশি label এবং নিবন্ধিত "
    "ফসল–বালাই ব্যবহার মিলিয়ে দেখতে বলো।"
)

def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def candidate_models() -> list[str]:
    """Use low-latency text models in order; POST fallback avoids a slow model-list lookup."""
    primary = os.getenv("GROQ_MODEL_PRIMARY", DEFAULT_MODELS[0])
    fallbacks = os.getenv("GROQ_MODEL_FALLBACKS", ",".join(DEFAULT_MODELS[1:])).split(",")
    return _unique([primary, *fallbacks, *DEFAULT_MODELS])[:3]


def _error_code(response: requests.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return str(error.get("code") or error.get("type") or "")[:60]
    except (ValueError, requests.RequestException):
        pass
    return ""


def complete(messages: list[dict[str, str]], max_tokens: int = 700) -> tuple[str | None, str | None, str | None]:
    """Return ``(answer, model, error_code)``; never expose a secret in errors."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None, None, "missing_key"

    safe_messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for message in messages[-10:]:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            safe_messages.append({"role": role, "content": content[:3500]})
    if len(safe_messages) == 1:
        return None, None, "empty_messages"

    try:
        token_limit = min(max(int(max_tokens), 64), 1200)
    except (TypeError, ValueError):
        token_limit = 700

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Smart-Krishi-Buddhimotta/1.0",
    }
    last_error = "unavailable"
    for model in candidate_models():
        payload: dict[str, Any] = {"model": model, "messages": safe_messages}
        if "gpt-oss" in model.casefold():
            payload["max_completion_tokens"] = token_limit
            payload["reasoning_effort"] = "low"
        else:
            payload["temperature"] = 0.35
            payload["max_tokens"] = token_limit

        try:
            response = requests.post(
                GROQ_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=(4, 18),
            )
        except requests.Timeout:
            last_error = "timeout"
            continue
        except requests.RequestException:
            last_error = "network"
            continue

        if response.status_code in {401, 403}:
            return None, None, "invalid_key"
        if response.status_code == 429:
            last_error = "rate_limit"
            continue
        if response.status_code in {400, 404, 422}:
            last_error = _error_code(response) or f"model_{response.status_code}"
            continue
        if response.status_code >= 500:
            last_error = "provider_unavailable"
            continue
        if response.status_code >= 400:
            last_error = _error_code(response) or f"http_{response.status_code}"
            continue

        try:
            body = response.json()
            choices = body.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            answer = str(content or "").strip()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError):
            answer = ""

        if answer:
            return answer, model, None
        last_error = "empty_reply"

    return None, None, last_error


def disease_summary(
    crop_label: str,
    disease_label: str,
    score_percent: float,
    alternatives: list[dict[str, Any]],
    symptom: str,
    visual_assessment: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Ask Groq for cautious text; the text model receives findings, not pixels."""
    scores = "; ".join(
        f"{item.get('label', 'অজানা')}: {float(item.get('score_percent', 0)):.1f}%"
        for item in alternatives[:3]
    )
    if visual_assessment:
        vision_context = (
            "ছবিটি Krittim AI Vision-এর প্রাথমিক বিশ্লেষণ পেয়েছে; তুমি ছবির pixel দেখছ না। "
            f"তার সংক্ষিপ্ত ফল: crop={visual_assessment.get('crop_bn') or visual_assessment.get('crop_en') or 'অনিশ্চিত'}; "
            f"disease={visual_assessment.get('disease_bn') or visual_assessment.get('disease_en') or 'অনিশ্চিত'}; "
            f"cause_type={visual_assessment.get('cause_type') or 'unknown'}; "
            f"visible signs={'; '.join(visual_assessment.get('visible_signs_bn') or []) or 'not stated'}. "
        )
        if alternatives:
            local = alternatives[0]
            model_context = (
                " স্থানীয় শ্রেণিবিন্যাসের সহায়ক অনুমান: "
                f"{local.get('crop_label', crop_label)} — {local.get('label', disease_label)} "
                f"({float(local.get('score_percent', score_percent)):.1f}%; calibrated নয়)।"
            )
        else:
            model_context = ""
    else:
        vision_context = "কোনো multimodal vision ফল নেই; তুমি ছবিটি দেখোনি। "
        model_context = (
            f"ছবির শ্রেণিবিন্যাস মডেলের fallback অনুমান: {crop_label} — {disease_label}; "
            f"স্কোর {score_percent:.1f}% (calibrated নয়); শীর্ষ বিকল্প {scores or 'নেই'}।"
        )
    user_text = (
        f"{vision_context}{model_context} ব্যবহারকারীর লেখা লক্ষণ: "
        f"{symptom.strip()[:500] or 'কোনো লক্ষণ লেখা হয়নি'}। "
        "বাংলায় Markdown-এ ২–৩টি সংক্ষিপ্ত ব্যাখ্যা/যাচাইয়ের ধাপ দাও। এটিকে নিশ্চিত "
        "রোগনির্ণয় বলবে না। নির্দিষ্ট কীটনাশক, ব্র্যান্ড, ওষুধ, সার, মিশ্রণ, মাত্রা, "
        "প্রয়োগপথ বা পানির পরিমাণ বলবে না; কেবল অ্যাপের আলাদা DAE-যাচাইকৃত লেবেল-তথ্য "
        "ব্যবহারযোগ্য। স্কোরকে calibrated probability বা নির্ভুলতা বলে উপস্থাপন করবে না।"
    )
    return complete([{"role": "user", "content": user_text}], max_tokens=420)
