"""Conservative live fallback to the official DAE PPW pesticide repository.

The public search endpoint returns candidate product identities, not use
recommendations. A candidate is returned only when its official detail page's
"use instructions" section contains the crop, disease, and one unambiguous
rate together. Search hits, product names, and active ingredients alone are
never treated as a crop/disease recommendation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
import os
import re
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import requests

PPW_ORIGIN = "https://ppw.krishi.gov.bd"
PPW_SEARCH_URL = f"{PPW_ORIGIN}/pesticide-search"

# Keep live traffic bounded: at most four keyword searches and eight detail
# pages for a single diagnosis. Responses (including misses) are short-cached.
_SEARCH_TIMEOUT = (1.5, 3.0)
_DETAIL_TIMEOUT = (1.5, 3.0)
_MAX_SEARCH_TERMS = 4
_MAX_CANDIDATES = 8
_MAX_DETAIL_WORKERS = 4
_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ENTRIES = 256

_HEADERS = {
    "User-Agent": "SmartKrishiBuddhimotta/1.0 (official DAE PPW product-use verification)",
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{PPW_ORIGIN}/pesticide-list",
}

_CROP_ALIASES: dict[str, tuple[str, ...]] = {
    "rice": ("rice", "paddy", "ধান", "ধানের"),
    "tomato": ("tomato", "টমেটো", "টমেটোর"),
    "potato": ("potato", "আলু", "আলুর"),
    "mango": ("mango", "আম", "আমের"),
    "chili": ("chili", "chilli", "chili pepper", "chilli pepper", "মরিচ", "লঙ্কা"),
    "capsicum": ("capsicum", "bell pepper", "sweet pepper", "ক্যাপসিকাম"),
    "brinjal": ("brinjal", "eggplant", "aubergine", "বেগুন", "বেগুনের"),
    "maize": ("maize", "corn", "ভুট্টা", "ভুট্টার"),
    "jute": ("jute", "পাট", "পাটের"),
    "tea": ("tea", "চা", "চায়ের", "চায়ের"),
    "groundnut": ("groundnut", "peanut", "বাদাম", "বাদামের"),
    "onion": ("onion", "পেঁয়াজ", "পেঁয়াজ", "পেঁয়াজের", "পেঁয়াজের"),
    "banana": ("banana", "কলা", "কলার"),
    "cucumber": ("cucumber", "শসা", "শসার"),
    "pumpkin": ("pumpkin", "sweet gourd", "squash", "কুমড়া", "কুমড়া", "লাউ", "করলা", "তরমুজ"),
    "citrus": ("citrus", "orange", "lemon", "লেবু", "কমলা"),
    "apple": ("apple", "আপেল"),
    "grape": ("grape", "আঙ্গুর", "আঙুর"),
    "bean": ("bean", "শিম", "বরবটি"),
    "cotton": ("cotton", "তুলা"),
}

# Specific phrases come before broad ones so, for example, "sheath blight"
# is not reduced to the generic "blight" term.
_DISEASE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sheath blight", ("sheath blight", "শীথ ব্লাইট", "সিথ ব্লাইট")),
    ("early blight", ("early blight", "আর্লি ব্লাইট")),
    ("late blight", ("late blight", "লেট ব্লাইট")),
    ("powdery mildew", ("powdery mildew", "পাউডারি মিলডিউ")),
    ("purple blotch", ("purple blotch", "পার্পল ব্লচ")),
    ("leaf spot", ("leaf spot", "পাতার দাগ")),
    ("anthracnose", ("anthracnose", "অ্যানথ্রাকনোজ", "এ্যানথ্রাকনোজ")),
    ("bacterial spot", ("bacterial spot", "ব্যাকটেরিয়াল দাগ", "ব্যাকটেরিয়াল দাগ")),
    ("tikka", ("tikka", "টিক্কা")),
    ("blast", ("blast", "ব্লাস্ট", "ব্লাষ্ট")),
    ("rust", ("rust", "মরিচা")),
    ("mosaic virus", ("mosaic virus", "মোজাইক ভাইরাস")),
    ("leaf mold", ("leaf mold", "লিফ মোল্ড")),
    ("septoria leaf spot", ("septoria leaf spot", "সেপটোরিয়া পাতার দাগ")),
)

_CLASS_CROP_ALIASES: dict[str, tuple[str, ...]] = {
    "Apple": _CROP_ALIASES["apple"],
    "Cherry_(including_sour)": ("cherry", "চেরি"),
    "Corn_(maize)": _CROP_ALIASES["maize"],
    "Pepper,_bell": ("capsicum", "bell pepper", "pepper", "ক্যাপসিকাম"),
    "Potato": _CROP_ALIASES["potato"],
    "Squash": ("squash", "pumpkin", "কুমড়া", "কুমড়া"),
    "Tomato": _CROP_ALIASES["tomato"],
    "Grape": _CROP_ALIASES["grape"],
    "Orange": ("orange", "citrus", "কমলা"),
    "Peach": ("peach", "পীচ"),
    "Strawberry": ("strawberry", "স্ট্রবেরি"),
}

_AMOUNT = r"\d+(?:[.,]\d+)?\s*(?:m\s*l|ml|millilit(?:er|re)s?|মি\.?\s*লি\.?|মিলি|g|gm|gram|grams|গ্রাম|গ্রা|kg|কেজি)"
_DENOMINATOR = r"(?:\d+\s*)?(?:লিটার|litre|liter|l|হেক্টর|hectare|ha|একর|acre|শতক|decimal|বিঘা|bigha)"
_DOSE_PATTERNS = (
    re.compile(
        rf"(?P<rate>{_AMOUNT}\s*(?:/|per|প্রতি)\s*{_DENOMINATOR}(?:\s*(?:পানি|water|of\s+water))?)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<rate>(?:প্রতি|per)\s*{_DENOMINATOR}(?:\s*(?:পানি|water))?(?:তে)?\s*{_AMOUNT})",
        re.IGNORECASE,
    ),
)

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


class _VisibleTextParser(HTMLParser):
    """Extract visible text while preserving boundaries between HTML blocks."""

    _BLOCK_TAGS = {
        "address", "article", "br", "dd", "div", "dl", "dt", "h1", "h2", "h3",
        "h4", "h5", "h6", "li", "ol", "p", "section", "table", "td", "th", "tr", "ul",
    }
    _SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        text = "".join(self.parts).replace("\xa0", " ")
        text = re.sub(r"[\t\r\f\v ]+", " ", text)
        return re.sub(r"\n\s*\n+", "\n", text).strip()


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u0980-\u09ff]+", " ", str(value or "").casefold()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {_normalize(text)} "
    normalized_phrase = _normalize(phrase)
    return bool(normalized_phrase and f" {normalized_phrase} " in normalized_text)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in alias.strip().split() if part]
    if not parts:
        return re.compile(r"(?!)")
    body = r"[\s\-–—,:()]+".join(parts)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _positions(text: str, aliases: tuple[str, ...]) -> list[tuple[int, int]]:
    found: set[tuple[int, int]] = set()
    for alias in aliases:
        for match in _alias_pattern(alias).finditer(text):
            found.add((match.start(), match.end()))
    return sorted(found)


def _crop_aliases(crop_en: str | None, crop_bn: str | None) -> tuple[str, ...]:
    text = f"{crop_en or ''} {crop_bn or ''}"
    for aliases in _CROP_ALIASES.values():
        if any(_contains_phrase(text, alias) for alias in aliases):
            return aliases
    return tuple(dict.fromkeys(value.strip() for value in (crop_en, crop_bn) if value and value.strip()))


def _disease_aliases(disease_en: str | None, disease_bn: str | None) -> tuple[tuple[str, ...], str | None]:
    text = f"{disease_en or ''} {disease_bn or ''}"
    for canonical, aliases in _DISEASE_ALIASES:
        if any(_contains_phrase(text, alias) for alias in aliases):
            return aliases, canonical

    fallback: list[str] = []
    for value in (disease_en, disease_bn):
        if value and value.strip():
            cleaned = value.strip()
            # The vision prompt asks for disease-only labels, but strip an
            # accidental leading English crop name if one is still present.
            crop = ("rice", "tomato", "potato", "mango", "chili", "brinjal", "maize", "jute", "tea")
            normalized = _normalize(cleaned)
            for crop_name in crop:
                if normalized.startswith(f"{crop_name} "):
                    cleaned = cleaned[len(crop_name):].strip(" _-—:,")
                    break
            if cleaned:
                fallback.append(cleaned)
    return tuple(dict.fromkeys(fallback)), None


def _class_label_matches_visual(visual: dict[str, Any]) -> bool:
    class_label = visual.get("plantvillage_class_label")
    if not class_label:
        return True
    if not isinstance(class_label, str) or "___" not in class_label:
        return False
    class_crop, class_disease = class_label.split("___", 1)
    crop_en = str(visual.get("crop_en") or "")
    crop_bn = str(visual.get("crop_bn") or "")
    disease_en = str(visual.get("disease_en") or "")
    disease_bn = str(visual.get("disease_bn") or "")
    class_aliases = _CLASS_CROP_ALIASES.get(class_crop, (class_crop.replace("_", " "),))
    if not any(_contains_phrase(f"{crop_en} {crop_bn}", alias) for alias in class_aliases):
        return False
    visual_disease_aliases, visual_canonical = _disease_aliases(disease_en, disease_bn)
    class_disease = class_disease.replace("_", " ")
    if visual_canonical:
        return any(_contains_phrase(class_disease, alias) for alias in visual_disease_aliases)
    visual_norm = _normalize(" ".join(visual_disease_aliases))
    class_norm = _normalize(class_disease)
    return bool(visual_norm and class_norm and (visual_norm in class_norm or class_norm in visual_norm))


def _is_eligible(visual: dict[str, Any]) -> bool:
    if visual.get("leaf_status") != "leaf":
        return False
    if visual.get("confidence_band") not in {"low", "medium", "high"}:
        return False
    if visual.get("cause_type", "unknown") not in {"fungal", "bacterial", "insect_or_mite", "unknown"}:
        return False
    if not (visual.get("crop_en") or visual.get("crop_bn")):
        return False
    if not (visual.get("disease_en") or visual.get("disease_bn")):
        return False
    return _class_label_matches_visual(visual)


def _search_terms(visual: dict[str, Any], extra_search_terms: tuple[str, ...] | list[str] = ()) -> list[str]:
    disease_en = str(visual.get("disease_en") or "").strip()
    disease_bn = str(visual.get("disease_bn") or "").strip()
    _disease_aliases_found, canonical = _disease_aliases(disease_en, disease_bn)
    terms: list[str] = []
    if canonical:
        terms.append(canonical)
        last_word = canonical.split()[-1]
        if last_word != canonical and len(last_word) >= 4:
            terms.append(last_word)
    else:
        # Prefer the English diagnostic field because PPW product names are
        # predominantly English; Bengali is a fallback for local-only names.
        candidate = disease_en or disease_bn
        normalized = _normalize(candidate)
        if normalized and len(normalized) >= 3:
            terms.append(normalized)
            words = normalized.split()
            if len(words) > 1 and len(words[-1]) >= 4:
                terms.append(words[-1])

    safe_extra_terms = [
        re.sub(r"\s+", " ", str(term or "")).strip()[:60]
        for term in extra_search_terms[:2]
        if isinstance(term, str) and len(_normalize(term)) >= 3
    ]
    terms.extend(safe_extra_terms)
    if len(terms) < _MAX_SEARCH_TERMS and not safe_extra_terms:
        crop_term = _normalize(visual.get("crop_en") or visual.get("crop_bn"))
        if crop_term and len(crop_term) >= 3:
            terms.append(crop_term)
    return list(dict.fromkeys(term[:60] for term in terms if term))[:_MAX_SEARCH_TERMS]


def _allowed_response_url(response: requests.Response) -> bool:
    try:
        parsed = urlsplit(response.url)
        return parsed.scheme == "https" and parsed.hostname == "ppw.krishi.gov.bd"
    except (TypeError, ValueError):
        return False


def _search_candidates(keyword: str) -> list[dict[str, Any]]:
    response = requests.get(
        PPW_SEARCH_URL,
        params={"page": 1, "ajax_keyword": keyword},
        headers=_HEADERS,
        timeout=_SEARCH_TIMEOUT,
        allow_redirects=False,
    )
    response.raise_for_status()
    if not _allowed_response_url(response):
        return []
    payload = response.json()
    records = payload.get("data", []) if isinstance(payload, dict) else []
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _detail_candidate(record: dict[str, Any]) -> dict[str, str] | None:
    raw_id = record.get("id")
    product_id = str(raw_id or "")
    if not re.fullmatch(r"\d{1,12}", product_id):
        return None
    registration = str(record.get("app_php_no") or "").strip()
    registration_match = re.fullmatch(r"AP\s*-\s*(\d{1,8})", registration, re.IGNORECASE)
    if not registration_match:
        return None
    registration_no = f"AP-{registration_match.group(1)}"
    brand_name = re.sub(r"\s+", " ", str(record.get("brand_name") or "")).strip()
    if not brand_name:
        return None
    chemical = record.get("chemical")
    active_ingredients = ""
    if isinstance(chemical, dict):
        active_ingredients = re.sub(r"\s+", " ", str(chemical.get("chemical_name") or "")).strip()
    return {
        "id": product_id,
        "registration_no": registration_no,
        "brand_name": brand_name[:100],
        "active_ingredients": active_ingredients[:160],
    }


def _extract_use_section(html_body: str) -> str:
    parser = _VisibleTextParser()
    try:
        parser.feed(html_body)
        parser.close()
    except Exception:
        return ""
    page_text = parser.text()
    start_pattern = re.compile(
        r"(?:^|\n)\s*(?:ব্যবহার\s*বিধি|ব্যবহারবিধি|ব্যবহার\s*পদ্ধতি|directions?\s+for\s+use|method\s+of\s+use)\s*(?:\n|$)",
        re.IGNORECASE,
    )
    start = start_pattern.search(page_text)
    if start:
        tail = page_text[start.end():]
    else:
        fallback = re.search(r"(?:ব্যবহার\s*বিধি|ব্যবহারবিধি|ব্যবহার\s*পদ্ধতি|directions?\s+for\s+use|method\s+of\s+use)", page_text, re.IGNORECASE)
        if not fallback:
            return ""
        tail = page_text[fallback.end():]
    end_pattern = re.search(
        r"(?:^|\n)\s*(?:নির্দেশনা|সতর্কতা|একই\s+ধরনের|same\s+type|warnings?)\s*(?:\n|$)",
        tail,
        re.IGNORECASE,
    )
    section = tail[:end_pattern.start()] if end_pattern else tail[:4000]
    return re.sub(r"\s+", " ", section).strip()


def _dose_matches(text: str) -> list[re.Match[str]]:
    found: dict[tuple[int, int], re.Match[str]] = {}
    for pattern in _DOSE_PATTERNS:
        for match in pattern.finditer(text):
            found[(match.start("rate"), match.end("rate"))] = match
    return [found[key] for key in sorted(found)]


def _find_exact_rate(
    use_text: str,
    crop_aliases: tuple[str, ...],
    disease_aliases: tuple[str, ...],
) -> str | None:
    crop_positions = _positions(use_text, crop_aliases)
    disease_positions = _positions(use_text, disease_aliases)
    if not crop_positions or not disease_positions:
        return None

    dose_matches = _dose_matches(use_text)
    exact_matches: dict[str, tuple[int, int]] = {}
    for crop_start, crop_end in crop_positions:
        for disease_start, disease_end in disease_positions:
            if crop_end <= disease_start:
                gap = disease_start - crop_end
                pair_end = disease_end
                pair_start = crop_start
            elif disease_end <= crop_start:
                gap = crop_start - disease_end
                pair_end = crop_end
                pair_start = disease_start
            else:
                gap = 0
                pair_start = min(crop_start, disease_start)
                pair_end = max(crop_end, disease_end)
            # Crop and disease must be written as one local use pair, not as
            # distant generic lists on the same product page.
            if gap > 60:
                continue
            gap_text = use_text[min(crop_end, disease_end):max(crop_start, disease_start)]
            gap_words = set(_normalize(gap_text).split())
            allowed_pair_words = {"crop", "disease", "pest", "name", "for", "of", "on", "ফসল", "রোগ", "রোগের", "নাম", "পোকা", "পোকার", "বালাই"}
            if gap_words - allowed_pair_words or re.search(r"[,;।.!?\n]", gap_text):
                continue
            for dose in dose_matches:
                dose_start, dose_end = dose.span("rate")
                if dose_start < pair_end or dose_start - pair_end > 110:
                    continue
                between = use_text[pair_end:dose_start]
                if re.search(r"[;।.!?]", between):
                    continue
                # A different crop beginning between this pair and dose makes
                # the association ambiguous (e.g. a combined multi-crop label).
                other_crop_between = any(
                    start > pair_start and start < dose_start and (start, end) != (crop_start, crop_end)
                    for start, end in crop_positions
                )
                if other_crop_between:
                    continue
                rate = re.sub(r"\s+", " ", dose.group("rate")).strip(" ,:;।.")
                exact_matches[rate] = (pair_start, dose_end)
    if len(exact_matches) != 1:
        return None
    return next(iter(exact_matches))


def _fetch_detail(candidate: dict[str, str]) -> dict[str, str] | None:
    detail_url = f"{PPW_ORIGIN}/pesticide-details/{candidate['id']}"
    response = requests.get(detail_url, headers=_HEADERS, timeout=_DETAIL_TIMEOUT, allow_redirects=False)
    response.raise_for_status()
    if not _allowed_response_url(response):
        return None
    html_body = response.text
    parser = _VisibleTextParser()
    try:
        parser.feed(html_body)
        parser.close()
    except Exception:
        return None
    visible_text = parser.text()
    registration_match = re.search(r"\bAP\s*-\s*(\d{1,8})\b", visible_text, re.IGNORECASE)
    if not registration_match or f"AP-{registration_match.group(1)}" != candidate["registration_no"]:
        return {**candidate, "use_text": "", "source_url": detail_url}
    return {**candidate, "use_text": _extract_use_section(html_body), "source_url": detail_url}


def _cache_key(visual: dict[str, Any], extra_search_terms: tuple[str, ...] | list[str] = ()) -> str:
    pieces = [
        _normalize(visual.get("crop_en")),
        _normalize(visual.get("crop_bn")),
        _normalize(visual.get("disease_en")),
        _normalize(visual.get("disease_bn")),
    ]
    pieces.extend(_normalize(term) for term in extra_search_terms[:2])
    return "|".join(pieces)


def _cache_get(key: str) -> dict[str, Any] | None:
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if not entry:
            return None
        created, payload = entry
        if now - created > _CACHE_TTL_SECONDS:
            _CACHE.pop(key, None)
            return None
        return dict(payload)


def _cache_put(key: str, payload: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX_ENTRIES:
            oldest = min(_CACHE, key=lambda item: _CACHE[item][0])
            _CACHE.pop(oldest, None)
        _CACHE[key] = (time.monotonic(), dict(payload))


def _product_from_detail(detail: dict[str, str], visual: dict[str, Any], rate: str) -> dict[str, Any]:
    crop_label = str(visual.get("crop_bn") or visual.get("crop_en") or "")[:60]
    disease_label = str(visual.get("disease_bn") or visual.get("disease_en") or "")[:80]
    return {
        "name": detail["brand_name"],
        "registration_no": detail["registration_no"],
        "active_ingredients": detail["active_ingredients"] or "DAE PPW live search record-এ পাওয়া যায়নি",
        "label_scope": f"{crop_label} — {disease_label}",
        "rate": rate,
        "application_method": "DAE PPW লাইভ detail page-এর একই ব্যবহারবিধিতে এই ফসল–রোগ ও মাত্রা মিলেছে; প্রয়োগপদ্ধতি, PPE ও PHI প্যাকেটের বর্তমান লেবেল থেকে নিশ্চিত করুন।",
        "directions": [
            "এই পণ্যটি DAE PPW লাইভ detail page-এ একই ফসল–রোগের ব্যবহারবিধি ও মাত্রা একসঙ্গে মিলিয়ে পাওয়া গেছে।",
            "রেকর্ড মানেই সক্রিয় নিবন্ধন বা স্থানীয় দোকানে মজুতের নিশ্চয়তা নয়; ব্যবহারের আগে আসল প্যাকেটের বর্তমান লেবেল ও স্থানীয় কৃষি কর্মকর্তার পরামর্শ মিলিয়ে নিন।",
        ],
        "source_label": f"DAE PPW live product record — {detail['brand_name']}",
        "source_url": detail["source_url"],
    }


def _search_live(
    visual: dict[str, Any],
    extra_search_terms: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    crop_aliases = _crop_aliases(visual.get("crop_en"), visual.get("crop_bn"))
    disease_aliases, _canonical = _disease_aliases(visual.get("disease_en"), visual.get("disease_bn"))
    if not crop_aliases or not disease_aliases:
        return {"status": "skipped", "product": None}

    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    had_search_response = False
    for keyword in _search_terms(visual, extra_search_terms):
        try:
            records = _search_candidates(keyword)
            had_search_response = True
        except (requests.RequestException, ValueError, TypeError):
            continue
        for record in records:
            candidate = _detail_candidate(record)
            if candidate and candidate["id"] not in seen_ids:
                seen_ids.add(candidate["id"])
                candidates.append(candidate)
                if len(candidates) >= _MAX_CANDIDATES:
                    break
        if len(candidates) >= _MAX_CANDIDATES:
            break

    if not candidates:
        return {"status": "no_match" if had_search_response else "unavailable", "product": None}

    details_by_id: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_DETAIL_WORKERS) as executor:
        futures = {executor.submit(_fetch_detail, candidate): candidate for candidate in candidates}
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                detail = future.result()
            except (requests.RequestException, ValueError, TypeError):
                continue
            if detail:
                details_by_id[candidate["id"]] = detail

    if not details_by_id:
        return {"status": "unavailable", "product": None}
    details = [details_by_id[candidate["id"]] for candidate in candidates if candidate["id"] in details_by_id]
    for detail in details:
        rate = _find_exact_rate(detail["use_text"], crop_aliases, disease_aliases)
        if rate:
            return {
                "status": "matched",
                "product": _product_from_detail(detail, visual, rate),
                "checked_on": datetime.now(timezone.utc).date().isoformat(),
            }
    return {"status": "no_match", "product": None}


def lookup_ppw_live_product(
    visual: dict[str, Any],
    extra_search_terms: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Best-effort exact-use lookup; failure never creates a guessed product."""
    if os.getenv("DAE_LIVE_LOOKUP_ENABLED", "true").strip().casefold() in {"0", "false", "no", "off"}:
        return {"status": "disabled", "product": None}
    if not _is_eligible(visual):
        return {"status": "skipped", "product": None}

    safe_terms = tuple(
        re.sub(r"[^A-Za-z0-9\u0980-\u09FF .()_-]+", " ", term).strip()[:60]
        for term in extra_search_terms[:2]
        if isinstance(term, str) and len(_normalize(term)) >= 3
    )
    key = _cache_key(visual, safe_terms)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        result = _search_live(visual, safe_terms)
    except Exception:
        # Keep provider/HTML errors from disrupting the disease endpoint.
        result = {"status": "unavailable", "product": None}
    if safe_terms and result.get("status") == "matched":
        result["ai_candidate_search"] = True
    _cache_put(key, result)
    return result
