"""FastAPI backend for the Next.js Smart Krishi frontend.

The UI stays in ``ui-smart-krishi-buddhimotta``. This API provides Groq chat,
Qwen multimodal image triage (when configured), the local ResNet9 fallback,
prototype-price calculations from the bundled CSV, and an optional Open-Meteo
proxy. It does not retrain any model.
"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import io
import math
import os
import re
import sys
import threading
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ai_service import complete, disease_summary
from backend.dae_live_lookup import lookup_ppw_live_product
from backend.disease_guidance import (
    get_disease_guidance,
    get_unclassified_visual_guidance,
    get_visual_disease_guidance,
)
from backend.vision_service import DEFAULT_VISION_MODEL, analyze_leaf_image
from backend.data_mappings import (
    CROP_BN_TO_SAMPLE_EN,
    DEMO_BASE_PRICES,
    DISTRICT_ALIASES,
    DISTRICT_BN_TO_EN,
)
from backend.disease_labels import DISEASE_TRANSLATION, disease_only_label


app = FastAPI(
    title="স্মার্ট কৃষি বুদ্ধিমত্তা API",
    description="বাংলাদেশের কৃষকদের জন্য Smart Krishi Buddhimotta Python API; প্রোটোটাইপ ডেটা স্পষ্টভাবে চিহ্নিত।",
    version="1.0.0",
)

_frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
_allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
if _frontend_url:
    _allowed_origins.append(_frontend_url)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"https://[a-zA-Z0-9-]+\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

CSV_PATH = _REPO_ROOT / "bd_crop_price_data.csv"
_DISEASE_CHECKPOINT_FILES = (
    "plant_disease_weights.pth",
    "plant-disease-model-complete.pth",
)
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
BANGLA_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1800)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=10)
    context: str | None = Field(default=None, max_length=1000)


class DiseaseRequest(BaseModel):
    image_data_url: str = Field(min_length=50, max_length=3_000_000)
    symptom: str = Field(default="", max_length=500)
    include_ai_explanation: bool = False


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _canonical_district(value: str) -> str:
    raw = str(value or "").strip()
    if raw in DISTRICT_BN_TO_EN:
        raw = DISTRICT_BN_TO_EN[raw]
    if raw == "নেত্রকোণা":
        raw = "Netrokona"
    key = raw.casefold().strip()
    if key in DISTRICT_ALIASES:
        key = DISTRICT_ALIASES[key]
    # Preserve a stable, title-cased value for comparisons with the dataset.
    return key.title() if key else "Dhaka"


def _canonical_crop(value: str) -> str:
    raw = str(value or "").strip()
    if raw in CROP_BN_TO_SAMPLE_EN:
        raw = CROP_BN_TO_SAMPLE_EN[raw]
    key = _normalize_key(raw)
    aliases = {
        "chilli": "chilli", "chili": "chilli", "rice": "rice",
        "gram": "gram", "oilseeds": "oilseeds", "pulses": "pulses",
    }
    return aliases.get(key, key)


@lru_cache(maxsize=1)
def _read_price_rows() -> list[dict[str, Any]]:
    """Load the tracked prototype CSV once. Never imply it is current/live data."""
    if not CSV_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                crop = _canonical_crop(row.get("Crop_Name", ""))
                # Inherited sample data includes a non-food commodity; do not expose it.
                if crop == "opium":
                    continue
                try:
                    price = float(row.get("Price_Tk_kg", ""))
                    date = dt.date.fromisoformat(str(row.get("Price_Date", ""))[:10])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(price) or price <= 0:
                    continue
                rows.append({
                    "district": _canonical_district(row.get("District_Name", "")),
                    "crop": crop,
                    "date": date,
                    "price": price,
                })
    except OSError:
        return []
    return rows


def _bn_date(value: dt.date) -> str:
    return f"{value.day:02d}/{value.month:02d}".translate(BANGLA_DIGITS)


def _bn_full_date(value: dt.date) -> str:
    return f"{value.day:02d}/{value.month:02d}/{value.year}".translate(BANGLA_DIGITS)


def _forecast_from_sample(district: str, crop_bn: str, crop_key: str) -> dict[str, Any]:
    rows = _read_price_rows()
    district_key = _canonical_district(district)
    crop_rows = [row for row in rows if row["crop"] == crop_key]
    district_rows = [row for row in crop_rows if row["district"] == district_key]

    if district_rows:
        source_rows = district_rows
        source_label = "সংযুক্ত নমুনা CSV · নির্বাচিত জেলার রেকর্ড"
        scope = "district sample"
    elif crop_rows:
        source_rows = crop_rows
        source_label = "সংযুক্ত নমুনা CSV · অন্য জেলার নমুনা থেকে গড়"
        scope = "cross-district sample"
    else:
        source_rows = []
        source_label = "সার্ভার-সিমুলেটেড ডেমো · এই ফসলের CSV রেকর্ড নেই"
        scope = "synthetic demo"

    today = dt.datetime.now(ZoneInfo("Asia/Dhaka")).date()
    seed = zlib.crc32(f"{district_key}|{crop_key}".encode("utf-8"))
    if source_rows:
        source_rows = sorted(source_rows, key=lambda row: row["date"])
        latest_date = source_rows[-1]["date"]
        sample_as_of = latest_date.isoformat()
        if scope == "district sample":
            source_label = f"নমুনা CSV · নির্বাচিত জেলার রেকর্ড, সর্বশেষ {_bn_full_date(latest_date)}"
        else:
            source_label = f"নমুনা CSV · অন্য জেলার সম্মিলিত রেকর্ড, সর্বশেষ {_bn_full_date(latest_date)}"
        base = float(source_rows[-1]["price"])
        recent = source_rows[-14:]
        if len(recent) > 1:
            x_mean = (len(recent) - 1) / 2
            y_mean = sum(item["price"] for item in recent) / len(recent)
            denominator = sum((i - x_mean) ** 2 for i in range(len(recent)))
            slope = (
                sum((i - x_mean) * (item["price"] - y_mean) for i, item in enumerate(recent))
                / denominator if denominator else 0.0
            )
            trend = max(-0.003, min(0.003, slope / max(base, 1.0)))
        else:
            trend = 0.0
    else:
        latest_date = None
        sample_as_of = ""
        base = float(DEMO_BASE_PRICES.get(crop_bn, 50.0))
        trend = (((seed >> 8) % 7) - 3) * 0.00045

    phase = (seed % 29) / 29 * (2 * math.pi)

    def cycle(position: int) -> float:
        return math.sin(position / 5.8 + phase) * 0.012

    history: list[dict[str, Any]] = []
    for index in range(14):
        relative = index - 13
        value = max(1.0, base * (1 + trend * relative + cycle(relative)))
        history_date = today - dt.timedelta(days=13 - index)
        history.append({"day": _bn_date(history_date), "price": round(value, 2)})

    last_price = history[-1]["price"]
    future: list[dict[str, Any]] = []
    for day in range(1, 15):
        movement = trend * day + cycle(day) - cycle(0)
        value = max(1.0, last_price * (1 + movement))
        future_date = today + dt.timedelta(days=day)
        future.append({"day": _bn_date(future_date), "price": round(value, 2)})

    average = sum(item["price"] for item in future) / len(future)
    return {
        "district": district,
        "crop": crop_bn,
        "history": history,
        "forecast": future,
        "latest_price": last_price,
        "average_price": round(average, 2),
        "source": source_label,
        "scope": scope,
        "sample_data_as_of": sample_as_of,
        "data_kind": "prototype_non_live",
        "note": (
            "সব দাম নমুনা বা সিমুলেটেড; CSV-র তারিখ ২০২৫-০৫-১৮-এ শেষ। "
            "চার্টের তারিখ ও ভবিষ্যৎ রেখা প্রদর্শনী সিমুলেশন, বাস্তব বাজারদর বা যাচাইকৃত পূর্বাভাস নয়।"
        ),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "smart-krishi-api",
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "groq_vision_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "groq_vision_model": os.getenv("GROQ_VISION_MODEL", DEFAULT_VISION_MODEL).strip() or DEFAULT_VISION_MODEL,
        "disease_weights_present": any(
            (_REPO_ROOT / filename).is_file() for filename in _DISEASE_CHECKPOINT_FILES
        ),
        "disease_primary_weights_present": (_REPO_ROOT / _DISEASE_CHECKPOINT_FILES[0]).is_file(),
        "disease_fallback_checkpoint_present": (_REPO_ROOT / _DISEASE_CHECKPOINT_FILES[1]).is_file(),
    }


@app.get("/api/forecast")
def price_forecast(
    district: str = Query(default="ঢাকা", max_length=80),
    crop: str = Query(default="ধান", max_length=80),
) -> dict[str, Any]:
    return _forecast_from_sample(district, crop, _canonical_crop(crop))


@app.get("/api/weather")
def weather(
    lat: float = Query(...),
    lng: float = Query(...),
) -> dict[str, Any]:
    if not (math.isfinite(lat) and math.isfinite(lng)):
        raise HTTPException(status_code=400, detail="অবস্থানের স্থানাঙ্ক সঠিক নয়।")
    if not (20.3 <= lat <= 26.9 and 88.0 <= lng <= 93.7):
        raise HTTPException(status_code=400, detail="বাংলাদেশের একটি বৈধ অবস্থান নির্বাচন করুন।")
    query = {
        "latitude": lat,
        "longitude": lng,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max,sunrise,sunset",
        "timezone": "Asia/Dhaka",
        "forecast_days": 7,
    }
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=query,
            timeout=(4, 10),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "current" not in payload or "daily" not in payload:
            raise ValueError("Open-Meteo response is incomplete")
    except (requests.RequestException, ValueError):
        raise HTTPException(status_code=502, detail="Open-Meteo থেকে আবহাওয়ার তথ্য পাওয়া যায়নি।") from None
    return {
        "weather": payload,
        "source": "Open-Meteo",
        "fetchedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    if not os.getenv("GROQ_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Krittim-S1 সহায়কটি এখন কনফিগার করা নেই। অন্য ফিচার ব্যবহার করা যাবে।",
        )
    messages = [{"role": item.role, "content": item.content} for item in request.messages]
    if request.context:
        for index, message in enumerate(messages):
            if message["role"] == "user":
                messages[index] = {
                    "role": "user",
                    "content": f"প্রাসঙ্গিক অ্যাপ তথ্য: {request.context}\n\nপ্রশ্ন/বার্তা: {message['content']}",
                }
                break
    answer, _model, error = complete(messages, max_tokens=750)
    if not answer:
        if error == "invalid_key":
            raise HTTPException(status_code=503, detail="AI সেবার কনফিগারেশন যাচাই করা যায়নি।")
        if error == "rate_limit":
            raise HTTPException(status_code=503, detail="AI সেবার অনুরোধসীমা সাময়িকভাবে পূর্ণ। পরে আবার চেষ্টা করুন।")
        raise HTTPException(status_code=502, detail="AI সেবাটি এখন সাড়া দিচ্ছে না। পরে আবার চেষ্টা করুন।")
    return {"answer": answer, "assistant": "Krittim-S1"}


_DISEASE_RUNTIME: tuple[Any, Any] | None = None
_DISEASE_LOAD_LOCK = threading.Lock()


def _load_disease_model():
    """Load the first usable bundled checkpoint once; never retrain or double-load."""
    global _DISEASE_RUNTIME
    if _DISEASE_RUNTIME is not None:
        return _DISEASE_RUNTIME
    with _DISEASE_LOAD_LOCK:
        if _DISEASE_RUNTIME is not None:
            return _DISEASE_RUNTIME
        import torch  # imported lazily so health/chat/forecast can start without inference loaded
        from disease_model import load_resnet9

        present = [
            _REPO_ROOT / filename
            for filename in _DISEASE_CHECKPOINT_FILES
            if (_REPO_ROOT / filename).is_file()
        ]
        if not present:
            raise FileNotFoundError("No bundled PlantVillage .pth checkpoint is present")

        load_errors: list[Exception] = []
        for weights_path in present:
            try:
                model = load_resnet9(str(weights_path))
                model.eval()
            except Exception as exc:
                load_errors.append(exc)
                continue
            _DISEASE_RUNTIME = (model, torch)
            return _DISEASE_RUNTIME

        raise RuntimeError("All bundled PlantVillage .pth checkpoints failed to load") from load_errors[-1]


def _parse_image_data_url(image_data_url: str) -> bytes:
    match = re.fullmatch(r"data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/=\r\n]+)", image_data_url.strip())
    if not match:
        raise HTTPException(status_code=400, detail="JPG, PNG বা WebP ছবি যোগ করুন।")
    encoded = match.group(2)
    if len(encoded) > 2_900_000:
        raise HTTPException(status_code=413, detail="ছবির আকার কমিয়ে আবার চেষ্টা করুন।")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        raise HTTPException(status_code=400, detail="ছবির তথ্য পড়া যায়নি।") from None
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="ছবিটি ২ এমবি-র বেশি। ছোট ছবি দিয়ে আবার চেষ্টা করুন।")
    return image_bytes


def _display_class(class_label: str) -> tuple[str, str]:
    crop_key = class_label.split("___", 1)[0]
    crop_names = {
        "Apple": "আপেল", "Blueberry": "ব্লুবেরি", "Cherry_(including_sour)": "চেরি",
        "Corn_(maize)": "ভুট্টা", "Grape": "আঙুর", "Orange": "কমলা", "Peach": "পীচ",
        "Pepper,_bell": "ক্যাপসিকাম", "Potato": "আলু", "Raspberry": "রাস্পবেরি",
        "Soybean": "সয়াবিন", "Squash": "স্কোয়াশ", "Strawberry": "স্ট্রবেরি", "Tomato": "টমেটো",
    }
    crop_label = crop_names.get(crop_key, crop_key.replace("_", " "))
    return crop_label, disease_only_label(class_label, crop_labels=(crop_label,))


def _vision_image_data_url(image: Image.Image) -> str:
    """Downscale/encode a safe JPEG copy for the optional Groq vision request."""
    for dimension, quality in ((1280, 84), (1040, 76), (800, 68)):
        prepared = image.copy()
        prepared.thumbnail((dimension, dimension), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        prepared.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = buffer.getvalue()
        if encoded and len(encoded) <= MAX_IMAGE_BYTES:
            return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    raise ValueError("image could not be reduced within the vision request limit")


def _cause_label_bn(cause_type: str) -> str:
    return {
        "fungal": "সম্ভাব্য ছত্রাকজনিত",
        "bacterial": "সম্ভাব্য ব্যাকটেরিয়াজনিত",
        "viral": "সম্ভাব্য ভাইরাসজনিত",
        "insect_or_mite": "সম্ভাব্য পোকা/মাকড়ের আক্রমণ",
        "nutrient": "সম্ভাব্য পুষ্টিঘাটতি—নিশ্চিত নয়",
        "abiotic": "সম্ভাব্য আবহাওয়া/পরিবেশজনিত",
        "healthy": "স্পষ্ট রোগের লক্ষণ নেই",
    }.get(cause_type, "ছবি থেকে কারণ নির্ধারণ করা যায়নি")


def _visual_disease_label(visual: dict[str, Any]) -> str:
    class_label = visual.get("plantvillage_class_label")
    if class_label and class_label.endswith("___healthy") and visual.get("cause_type") != "healthy":
        class_label = None
    return disease_only_label(
        class_label=class_label,
        disease_label=visual.get("disease_bn") or visual.get("disease_en"),
        crop_labels=(visual.get("crop_bn"), visual.get("crop_en")),
    )


def _augment_with_live_ppw_guidance(guidance: dict[str, Any], visual: dict[str, Any]) -> dict[str, Any]:
    """Try the official PPW source only after the curated allow-list misses."""
    if guidance.get("products"):
        return guidance

    lookup = lookup_ppw_live_product(visual)
    status = lookup.get("status")
    product = lookup.get("product")
    if status == "matched" and isinstance(product, dict):
        guidance["category"] = "possible_disease_with_verified_label"
        guidance["products"] = [product]
        guidance["product_note"] = (
            "কিউরেটেড তালিকায় সরাসরি match না থাকায় AI-এর ফসল ও রোগের নাম দিয়ে সরকারি DAE PPW-এর লাইভ public search করা হয়েছে। "
            "ছবিটি PPW-তে পাঠানো হয়নি; একই official ব্যবহারবিধিতে ফসল–রোগ ও মাত্রা মিললেই নিচের পণ্য দেখানো হয়েছে। "
            "এটি সক্রিয় নিবন্ধন বা স্থানীয় দোকানে মজুতের নিশ্চয়তা নয়—ব্যবহারের আগে আসল প্যাকেটের বর্তমান লেবেল, PHI ও নিরাপত্তা নির্দেশনা মিলিয়ে নিন।"
        )
        checked_on = lookup.get("checked_on")
        if isinstance(checked_on, str):
            guidance["source_checked_on"] = checked_on
        sources = guidance.get("sources", [])
        if isinstance(sources, list) and not any(source.get("url") == product.get("source_url") for source in sources if isinstance(source, dict)):
            sources.append({"label": product.get("source_label", "DAE PPW live product record"), "url": product.get("source_url", "")})
        guidance["sources"] = sources
    elif status == "no_match":
        guidance["product_note"] = (
            "কিউরেটেড DAE তালিকায় এই ফসল–রোগের পণ্য ছিল না। সরকারি PPW live search-এও একই ব্যবহারবিধিতে ফসল, রোগ ও নির্দিষ্ট মাত্রা একসঙ্গে যাচাই করা যায়নি; "
            "তাই ব্র্যান্ড বা ডোজ অনুমান করে দেওয়া হচ্ছে না। পরে আবার চেষ্টা করুন বা স্থানীয় কৃষি কর্মকর্তার সঙ্গে মিলিয়ে নিন।"
        )
    elif status == "unavailable":
        guidance["product_note"] = (
            "কিউরেটেড DAE তালিকায় নির্দিষ্ট পণ্য মেলেনি এবং সরকারি PPW live search-এ এখন সংযোগ পাওয়া যায়নি। "
            "যাচাইহীন ব্র্যান্ড/ডোজ দেওয়া হচ্ছে না; পরে আবার চেষ্টা করুন বা স্থানীয় কৃষি কর্মকর্তার সঙ্গে মিলিয়ে নিন।"
        )
    return guidance


@app.post("/api/disease")
def disease(request: DiseaseRequest) -> dict[str, Any]:
    image_bytes = _parse_image_data_url(request.image_data_url)
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="ছবির রেজোলিউশন কমিয়ে আবার চেষ্টা করুন।")
            source.verify()
        with Image.open(io.BytesIO(image_bytes)) as source:
            vision_image = ImageOps.exif_transpose(source).convert("RGB")
            vision_data_url = _vision_image_data_url(vision_image)
            classifier_image = vision_image.resize((256, 256), Image.Resampling.LANCZOS)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="ছবিটি পড়া যায়নি। অন্য JPG, PNG বা WebP ছবি দিন।") from None

    allowed_classes = list(DISEASE_TRANSLATION.keys())
    # Qwen is the preferred visual path. Load the heavier local runtime only
    # when Qwen is not configured or cannot return a usable visual result.
    vision = analyze_leaf_image(
        vision_data_url,
        [],
        allowed_classes,
        request.symptom,
    )
    visual_result = vision.get("result") if vision.get("available") else None

    top_predictions: list[dict[str, Any]] = []
    local_model_error: str | None = None
    if not visual_result:
        try:
            import numpy as np
            from disease_model import CLASS_LABELS, predict_torch
            model, torch = _load_disease_model()
            pixels = np.asarray(classifier_image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
            probabilities = np.asarray(predict_torch(model, tensor), dtype=np.float64)
            if probabilities.ndim != 1 or len(probabilities) != len(CLASS_LABELS):
                raise ValueError("model class count does not match the checkpoint")
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
            probabilities = np.clip(probabilities, 0.0, 1.0)
            if probabilities.sum() <= 0:
                raise ValueError("model returned no valid scores")
            probabilities = probabilities / probabilities.sum()
            top_indices = np.argsort(probabilities)[::-1][:3]
            for index in top_indices:
                label = CLASS_LABELS[int(index)]
                crop_label, disease_label = _display_class(label)
                top_predictions.append({
                    "class_label": label,
                    "crop_label": crop_label,
                    "label": disease_label,
                    "score_percent": round(float(probabilities[int(index)] * 100), 1),
                })
        except FileNotFoundError:
            local_model_error = "weights_missing"
        except Exception:
            local_model_error = "inference_failed"

    if not top_predictions and not visual_result:
        if local_model_error == "weights_missing":
            detail = "ছবি বিশ্লেষণের জন্য কোনো মডেল এখন চালু নেই। পরে আবার চেষ্টা করুন বা কৃষি কর্মকর্তার সঙ্গে যোগাযোগ করুন।"
        elif vision.get("status") == "not_configured":
            detail = "অনলাইন ছবি-ভিত্তিক AI কনফিগার করা নেই এবং স্থানীয় মডেলটিও ব্যবহার করা যাচ্ছে না। পরে আবার চেষ্টা করুন।"
        else:
            detail = "ছবি বিশ্লেষণের সেবা এখন সাড়া দিচ্ছে না। পরে আবার চেষ্টা করুন।"
        raise HTTPException(status_code=503, detail=detail)

    if top_predictions:
        prediction = top_predictions[0]
    else:
        crop_fallback = visual_result.get("crop_bn") or visual_result.get("crop_en") or "ফসল অনিশ্চিত"
        disease_fallback = _visual_disease_label(visual_result) or "লক্ষণ স্পষ্ট নয়—রোগ শনাক্ত করা যায়নি"
        prediction = {
            "class_label": visual_result.get("plantvillage_class_label") or "VISION_ONLY",
            "crop_label": crop_fallback,
            "label": disease_fallback,
            "score_percent": 0.0,
        }

    if visual_result:
        leaf_status = visual_result.get("leaf_status", "uncertain")
        diagnosis_crop = visual_result.get("crop_bn") or visual_result.get("crop_en") or "ফসল অনিশ্চিত"
        if leaf_status == "not_leaf":
            diagnosis_disease = "পাতার ছবি নয়"
        elif leaf_status == "uncertain" and visual_result.get("cause_type") == "healthy":
            diagnosis_disease = "ছবি স্পষ্ট নয়—গাছ স্বাভাবিক কি না নিশ্চিত নয়"
        elif visual_result.get("cause_type") == "healthy":
            diagnosis_disease = "স্পষ্ট রোগের লক্ষণ নেই"
        else:
            diagnosis_disease = _visual_disease_label(visual_result) or "লক্ষণ স্পষ্ট নয়—রোগ শনাক্ত করা যায়নি"
        if leaf_status == "not_leaf":
            diagnosis_cause_label = "পাতার ছবি নয়"
        elif leaf_status == "uncertain":
            diagnosis_cause_label = "ছবি/লক্ষণ স্পষ্ট নয়—যাচাই প্রয়োজন"
        else:
            diagnosis_cause_label = _cause_label_bn(visual_result.get("cause_type", "unknown"))
        diagnosis = {
            "source": "krittim_vision",
            "leaf_status": leaf_status,
            "crop_label": diagnosis_crop,
            "disease_label": diagnosis_disease,
            "cause_type": visual_result.get("cause_type", "unknown"),
            "cause_label_bn": diagnosis_cause_label,
            "visible_signs_bn": visual_result.get("visible_signs_bn", []),
            "nonchemical_actions_bn": visual_result.get("nonchemical_actions_bn", []),
            "confidence_band": visual_result.get("confidence_band", "low"),
        }
        guidance = get_visual_disease_guidance(visual_result)
        if not guidance["products"]:
            guidance = _augment_with_live_ppw_guidance(guidance, visual_result)
        scope_warning = (
            "Krittim AI Vision-এর ছবি-ভিত্তিক ফল নিশ্চিত রোগনির্ণয় নয়—ভুল হতে পারে। "
            "দৃশ্যমান লক্ষণ ও প্যাকেটের বর্তমান লেবেল মিলিয়ে নিন; DAE পণ্য কেবল নির্দিষ্ট crop–disease record মেললে দেখানো হয়।"
        )
    else:
        diagnosis = {
            "source": "plantvillage_classifier",
            "leaf_status": "uncertain",
            "crop_label": prediction["crop_label"],
            "disease_label": prediction["label"],
            "cause_type": "unknown",
            "cause_label_bn": "স্থানীয় মডেলের অনুমান—কারণ নিশ্চিত নয়",
            "visible_signs_bn": [],
            "nonchemical_actions_bn": [],
            "confidence_band": "low",
        }
        guidance = get_disease_guidance(prediction["class_label"])
        if guidance["products"]:
            # Do not turn the less capable, 38-class fallback into a pesticide
            # recommendation without an independent visual crop/disease result.
            guidance = get_unclassified_visual_guidance()
            guidance["product_note"] = (
                "Krittim AI Vision পাওয়া যায়নি; স্থানীয় ৩৮-শ্রেণির শ্রেণিবিন্যাস একা ফসল–রোগ নিশ্চিত করার জন্য যথেষ্ট নয়। "
                "তাই সম্ভাব্য DAE product/rate দেখানো হচ্ছে না; নতুন ছবি বা স্থানীয় কৃষি কর্মকর্তার মাধ্যমে যাচাই করুন।"
            )
        scope_warning = (
            "Krittim AI Vision পাওয়া যায়নি; এটি স্থানীয় ৩৮-শ্রেণির PlantVillage মডেলের fallback অনুমান, calibrated সম্ভাবনা নয়। "
            "বাংলাদেশের সব ফসল/রোগ এতে নেই—শুধু স্কোর দেখে চিকিৎসা শুরু করবেন না।"
        )

    answer = None
    ai_error = None
    if request.include_ai_explanation:
        answer, _model_name, ai_error = disease_summary(
            diagnosis["crop_label"], diagnosis["disease_label"], prediction["score_percent"],
            top_predictions, request.symptom,
            visual_assessment=visual_result,
        )

    return {
        "prediction": prediction,
        "diagnosis": diagnosis,
        "top_predictions": top_predictions,
        "scope_warning": scope_warning,
        "guidance": guidance,
        "vision_analysis": {
            "available": bool(visual_result),
            "status": vision.get("status", "provider_unavailable"),
        },
        "local_model_available": bool(top_predictions),
        "image_sent_to_vision_service": bool(vision.get("image_sent_to_groq")),
        "summary": answer,
        "summary_requested": request.include_ai_explanation,
        "summary_available": answer is not None,
        "summary_error_code": ai_error,
        "model_scope": {
            "name": "Krittim AI Vision + ResNet9 / PlantVillage fallback",
            "class_count": 38,
            "all_bangladesh_crops_supported": False,
        },
        "data_kind": "multimodal_visual_estimate_with_local_dae_allow_list",
    }
