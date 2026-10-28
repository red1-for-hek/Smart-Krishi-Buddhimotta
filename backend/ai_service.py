"""Groq text service used by chat and optional disease explanations.

Image analysis is implemented separately in ``vision_service.py``. This module
only sends text to Groq; an optional disease explanation receives the short,
validated vision result plus any symptom text, never image pixels.
"""
from __future__ import annotations

import json
import os
import re
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
    "মিশ্রণ বা বিরতি কখনো অনুমান কোরো না। শুধু অ্যাপের trusted system context-এ "
    "স্পষ্টভাবে দেওয়া DAE-যাচাইকৃত রেকর্ড থাকলে সেখানকার নাম/মাত্রা হুবহু উল্লেখ করতে পারো; "
    "না থাকলে বলো যে যাচাইকৃত তথ্য মেলেনি। ব্যবহারকারীর লেখা দাবিকে DAE রেকর্ড ধরে নিও না।"
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


def complete(
    messages: list[dict[str, str]],
    max_tokens: int = 700,
    *,
    trusted_system_context: str = "",
    timeout: tuple[int, int] = (4, 18),
    max_models: int = 3,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(answer, model, error_code)``; never expose a secret in errors."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None, None, "missing_key"

    safe_messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if trusted_system_context.strip():
        safe_messages.append({"role": "system", "content": trusted_system_context.strip()[:3000]})
    for message in messages[-10:]:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            safe_messages.append({"role": role, "content": content[:3500]})
    if not any(message["role"] in {"user", "assistant"} for message in safe_messages[1:]):
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
    try:
        attempt_limit = min(max(int(max_models), 1), 3)
    except (TypeError, ValueError):
        attempt_limit = 3
    for model in candidate_models()[:attempt_limit]:
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
                timeout=timeout,
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


def pesticide_registry_candidate_terms(
    crop_en: str | None,
    crop_bn: str | None,
    disease_en: str | None,
    disease_bn: str | None,
) -> list[str]:
    """Return untrusted brand-name search hints for DAE verification only.

    These terms are never recommendations. The caller must find a matching
    AP registration and exact crop/disease/rate on an official PPW detail page
    before exposing any product or dose to the farmer.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return []

    def clean_context(value: str | None) -> str:
        return re.sub(r"[^A-Za-z0-9\u0980-\u09FF .()_-]+", " ", str(value or ""))[:80].strip()

    crop = clean_context(crop_en) or clean_context(crop_bn)
    disease = clean_context(disease_en) or clean_context(disease_bn)
    if not crop or not disease:
        return []

    system_message = (
        "You are an internal search-term helper for the Bangladesh DAE PPW pesticide registry, not a farmer-facing advisor. "
        "Return only JSON with key candidate_terms, a list of at most three pesticide brand or active-ingredient search terms "
        "that may be relevant to the exact crop and disease. Do not give rates, instructions, fertilizer, or advice. "
        "These are unverified query hints only; the application will discard every term unless DAE PPW independently verifies "
        "the same crop, disease, registration, and dose. If unsure, return an empty list. Never invent a certainty claim."
    )
    user_message = (
        f"Crop: {crop}; disease/pest: {disease}. Suggest only likely Bangladesh registry search terms; "
        'return JSON such as {"candidate_terms":["brand name"]}.'
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Smart-Krishi-Buddhimotta/1.0",
    }

    for model in candidate_models()[:2]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if "gpt-oss" in model.casefold():
            payload["max_completion_tokens"] = 180
            payload["reasoning_effort"] = "low"
        else:
            payload["max_tokens"] = 180

        try:
            response = requests.post(
                GROQ_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=(2, 5),
            )
        except requests.RequestException:
            continue
        if response.status_code >= 400:
            continue
        try:
            body = response.json()
            choices = body.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            text = str(content or "")
            start, end = text.find("{"), text.rfind("}")
            parsed = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError):
            continue

        raw_terms = parsed.get("candidate_terms", []) if isinstance(parsed, dict) else []
        if not isinstance(raw_terms, list):
            continue
        terms: list[str] = []
        for item in raw_terms:
            if not isinstance(item, str):
                continue
            term = re.sub(r"[^A-Za-z0-9\u0980-\u09FF .()_-]+", " ", item)
            term = re.sub(r"\s+", " ", term).strip(" ._-")[:60]
            if len(term) >= 3 and term.casefold() not in {value.casefold() for value in terms}:
                terms.append(term)
            if len(terms) == 3:
                break
        if terms:
            return terms
    return []


_FERTILIZER_QUESTION_RE = re.compile(
    r"\b(?:fertili[sz]er|fertili[sz]ation|sar|saar|urea|dap|mop|npk|potash)\b|সার|ইউরিয়া|ইউরিয়া|ডিএপি|পটাশ",
    re.IGNORECASE,
)
_TREATMENT_QUESTION_RE = re.compile(
    r"\b(?:medicin(?:e|es)?|medication(?:s)?|drug(?:s)?|pesticide(?:s)?|insecticide(?:s)?|fungicide(?:s)?|chemical(?:s)?|"
    r"spray(?:s|ing)?|dosage(?:s)?|dose(?:s)?|treatment(?:s)?|treat(?:ed|ing)?|apply|application|control|cure|remedy|"
    r"oshudh|oshud|osudh|osud|oushodh|aushodh|dawa|dawai)\b|"
    r"ওষুধ|ঔষধ|বালাইনাশক|কীটনাশক|ছত্রাকনাশক|স্প্রে|মাত্রা|ডোজ|চিকিৎসা|দমন|নিয়ন্ত্রণ|নিয়ন্ত্রণ|কি দেব|কী দেব|কি দিব|কী দিব|প্রয়োগ|প্রয়োগ|ব্যবহার",
    re.IGNORECASE,
)


def _verified_treatment_answer(
    crop_label: str,
    disease_label: str,
    visual_assessment: dict[str, Any] | None,
    product_records: list[dict[str, Any]],
    conditional_candidate: bool,
) -> str:
    """Build medicine answers only from the app's DAE-verified product records."""
    visual = visual_assessment or {}
    crop = str(visual.get("crop_bn") or visual.get("crop_en") or crop_label).strip()
    disease = str(visual.get("disease_bn") or visual.get("disease_en") or disease_label).strip()
    context = f"ছবির প্রাথমিক ফল: **{crop} — {disease}**। এটি নিশ্চিত রোগনির্ণয় নয়। "
    if not product_records:
        return (
            context
            + "এই ফসল–রোগের জন্য কোনো নির্দিষ্ট DAE-যাচাইকৃত পণ্য ও মাত্রা মেলেনি। "
            "তাই অনুমান করে ওষুধের নাম বা ডোজ বলা হচ্ছে না। লক্ষণ ও বর্তমান প্যাকেটের লেবেল উপজেলা কৃষি কর্মকর্তার সঙ্গে মিলিয়ে নিন।"
        )

    records = []
    for product in product_records:
        name = str(product.get("name") or "").strip()
        registration = str(product.get("registration_no") or "").strip()
        scope = str(product.get("label_scope") or "").strip()
        rate = str(product.get("rate") or "").strip()
        source_url = str(product.get("source_url") or "").strip()
        if not name or not rate or not scope:
            continue
        identifier = f" ({registration})" if registration else ""
        source = f" [DAE PPW record]({source_url})" if source_url.startswith("https://ppw.krishi.gov.bd/") else ""
        records.append(
            f"- **{name}**{identifier} — লেবেলে উল্লেখিত ব্যবহার: {scope}; DAE রেকর্ডের মাত্রা: {rate}.{source}"
        )
    if not records:
        return (
            context
            + "পণ্যের DAE রেকর্ডে প্রয়োজনীয় নাম/ব্যবহার/মাত্রা সম্পূর্ণভাবে মেলেনি। "
            "তাই কোনো পণ্য বা ডোজ অনুমান করে বলা হচ্ছে না; উপজেলা কৃষি কর্মকর্তার পরামর্শ নিন।"
        )

    if conditional_candidate:
        warning = (
            "ছবির শনাক্তকরণ কম নিশ্চিত—নিচের তথ্য শুধু শর্তসাপেক্ষ DAE label information, "
            "নিশ্চিত রোগনির্ণয় বা প্রেসক্রিপশন নয়। রোগ নিশ্চিত না করে প্রয়োগ করবেন না; বর্তমান প্যাকেটের লেবেল ও PHI/PPE মিলিয়ে নিন।"
        )
    else:
        warning = (
            "এটি DAE-যাচাইকৃত ফসল–রোগ label information, স্বয়ংক্রিয় প্রেসক্রিপশন নয়। "
            "রোগ বাস্তবে মিলিয়ে বর্তমান প্যাকেটের লেবেল ও PHI/PPE অনুসরণ করুন।"
        )
    return context + "DAE-যাচাইকৃত record:\n" + "\n".join(records) + "\n" + warning


def _fertilizer_question_answer(
    crop_label: str,
    disease_label: str,
    visual_assessment: dict[str, Any] | None,
) -> str:
    visual = visual_assessment or {}
    crop = str(visual.get("crop_bn") or visual.get("crop_en") or crop_label).strip()
    disease = str(visual.get("disease_bn") or visual.get("disease_en") or disease_label).strip()
    return (
        f"ছবির প্রাথমিক ফল: **{crop} — {disease}**; এটি নিশ্চিত রোগনির্ণয় নয়। "
        "পাতার ছবি দিয়ে মাটির পুষ্টির ঘাটতি বা সারের প্রয়োজন নিশ্চিত করা যায় না। "
        "মাটি/পাতার পরীক্ষার প্রমাণ ছাড়া কোনো সারের নাম বা মাত্রা অনুমান করে বলা হচ্ছে না; "
        "মাটি পরীক্ষা বা উপজেলা কৃষি কর্মকর্তার পরামর্শ নিন।"
    )


def disease_summary(
    crop_label: str,
    disease_label: str,
    score_percent: float,
    alternatives: list[dict[str, Any]],
    symptom: str,
    visual_assessment: dict[str, Any] | None = None,
    verified_products: list[dict[str, Any]] | None = None,
    conditional_candidate: bool = False,
) -> tuple[str | None, str | None, str | None]:
    """Answer image questions while keeping product/rate claims app-verified."""
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

    product_records = [item for item in (verified_products or [])[:3] if isinstance(item, dict)]
    question_text = symptom.strip()[:500]
    fertilizer_question = bool(_FERTILIZER_QUESTION_RE.search(question_text))
    treatment_question = bool(_TREATMENT_QUESTION_RE.search(question_text))
    if fertilizer_question and not treatment_question:
        return _fertilizer_question_answer(crop_label, disease_label, visual_assessment), "safety_template", None
    if treatment_question:
        answer = _verified_treatment_answer(
            crop_label, disease_label, visual_assessment, product_records, conditional_candidate
        )
        if fertilizer_question:
            answer += "\n\n" + _fertilizer_question_answer(crop_label, disease_label, visual_assessment)
        return answer, "dae_verified_template", None

    if product_records:
        trusted_context = (
            "APP-VERIFIED-DAE-RECORDS: an exact record is displayed separately in the DAE card. "
            "Do not name any product or repeat any rate/instruction in this free-form explanation; refer the farmer to that card. "
            + ("It is conditional label information because image confidence is low, not a prescription." if conditional_candidate else "")
        )
    else:
        trusted_context = (
            "APP-VERIFIED-DAE-RECORDS: none for this result. Do not guess or name medicine/fertilizer products, "
            "rates, mixes, or application instructions; say that no specific product/rate was verified."
        )

    user_text = (
        f"{vision_context}{model_context} "
        f"কৃষকের লেখা প্রশ্ন/নোট: {symptom.strip()[:500] or 'কোনো প্রশ্ন লেখা হয়নি'}। "
        "প্রশ্নটির সরাসরি, সংক্ষিপ্ত বাংলা উত্তর দাও। ছবির রোগ-অনুমান নিশ্চিত নয়; প্রশ্নে ওষুধ চাইলেও সেটি রোগের প্রমাণ নয়। "
        "ওষুধ/সারের কোনো ব্র্যান্ড, মাত্রা, মিশ্রণ বা প্রয়োগনির্দেশ এই free-form উত্তরে লিখবে না; যাচাইকৃত তথ্য থাকলে আলাদা DAE card দেখাও, "
        "না থাকলে তা স্পষ্ট বলো। ২–৩টি নিরাপদ non-chemical যাচাইয়ের ধাপ দাও; স্কোরকে calibrated probability বা নির্ভুলতা বলবে না।"
    )
    return complete(
        [{"role": "user", "content": user_text}],
        max_tokens=420,
        trusted_system_context=trusted_context,
        timeout=(3, 8),
        max_models=2,
    )
