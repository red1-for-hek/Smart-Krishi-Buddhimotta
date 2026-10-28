"""Bengali labels for the inherited 38-class disease checkpoint.

Inherited label mapping retained from the Streamlit reference UI; this module
does not change or retrain the model.
"""

import re

_CROP_TRANSLATIONS = {
    "Apple": "আপেল", "Blueberry": "ব্লুবেরি", "Cherry_(including_sour)": "চেরি",
    "Corn_(maize)": "ভুট্টা", "Grape": "আঙ্গুর", "Orange": "কমলা", "Peach": "পীচ",
    "Pepper,_bell": "ক্যাপসিকাম", "Potato": "আলু", "Raspberry": "রাস্পবেরি",
    "Soybean": "সয়াবিন", "Squash": "স্কোয়াশ", "Strawberry": "স্ট্রবেরি", "Tomato": "টমেটো",
}

DISEASE_TRANSLATION = {'Apple___Apple_scab': 'আপেল স্ক্যাব রোগ',
 'Apple___Black_rot': 'আপেলের কালো পচন রোগ',
 'Apple___Cedar_apple_rust': 'আপেলের সিডার মরিচা রোগ',
 'Apple___healthy': 'আপেল গাছ সুস্থ',
 'Blueberry___healthy': 'ব্লুবেরি গাছ সুস্থ',
 'Cherry_(including_sour)___Powdery_mildew': 'চেরি পাউডারি মিলডিউ রোগ',
 'Cherry_(including_sour)___healthy': 'চেরি গাছ সুস্থ',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'ভুট্টার সারকোস্পোরা পাতার দাগ রোগ',
 'Corn_(maize)___Common_rust_': 'ভুট্টার সাধারণ মরিচা রোগ',
 'Corn_(maize)___Northern_Leaf_Blight': 'ভুট্টার নর্দান লিফ ব্লাইট রোগ',
 'Corn_(maize)___healthy': 'ভুট্টা গাছ সুস্থ',
 'Grape___Black_rot': 'আঙ্গুরের কালো পচন রোগ',
 'Grape___Esca_(Black_Measles)': 'আঙ্গুরের এসকা (কালো দাগ) রোগ',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'আঙ্গুরের পাতাঝলসানো রোগ',
 'Grape___healthy': 'আঙ্গুর গাছ সুস্থ',
 'Orange___Haunglongbing_(Citrus_greening)': 'কমলার হুয়াংলংবিং (গ্রিনিং) রোগ',
 'Peach___Bacterial_spot': 'পীচ ব্যাকটেরিয়াল দাগ রোগ',
 'Peach___healthy': 'পীচ গাছ সুস্থ',
 'Pepper,_bell___Bacterial_spot': 'ক্যাপসিকাম ব্যাকটেরিয়াল দাগ রোগ',
 'Pepper,_bell___healthy': 'ক্যাপসিকাম গাছ সুস্থ',
 'Potato___Early_blight': 'আলুর আর্লি ব্লাইট রোগ',
 'Potato___Late_blight': 'আলুর লেট ব্লাইট রোগ',
 'Potato___healthy': 'আলু গাছ সুস্থ',
 'Raspberry___healthy': 'রাস্পবেরি গাছ সুস্থ',
 'Soybean___healthy': 'সয়াবিন গাছ সুস্থ',
 'Squash___Powdery_mildew': 'স্কোয়াশ পাউডারি মিলডিউ রোগ',
 'Strawberry___Leaf_scorch': 'স্ট্রবেরির পাতাঝলসানো রোগ',
 'Strawberry___healthy': 'স্ট্রবেরি গাছ সুস্থ',
 'Tomato___Bacterial_spot': 'টমেটো ব্যাকটেরিয়াল দাগ রোগ',
 'Tomato___Early_blight': 'টমেটো আর্লি ব্লাইট রোগ',
 'Tomato___Late_blight': 'টমেটো লেট ব্লাইট রোগ',
 'Tomato___Leaf_Mold': 'টমেটো লিফ মোল্ড রোগ',
 'Tomato___Septoria_leaf_spot': 'টমেটো সেপটোরিয়া পাতার দাগ রোগ',
 'Tomato___Spider_mites Two-spotted_spider_mite': 'টমেটো স্পাইডার মাইট আক্রমণ',
 'Tomato___Target_Spot': 'টমেটো টার্গেট স্পট রোগ',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'টমেটো ইয়েলো লিফ কার্ল ভাইরাস',
 'Tomato___Tomato_mosaic_virus': 'টমেটো মোজাইক ভাইরাস',
 'Tomato___healthy': 'টমেটো গাছ সুস্থ'}


def disease_only_label(
    class_label: str | None = None,
    disease_label: str | None = None,
    crop_labels: tuple[str | None, ...] | list[str | None] = (),
) -> str:
    """Return a disease-only Bengali label, removing a leading crop name.

    Crop and class metadata remain available to callers for matching curated
    crop–disease guidance; this function only shapes the visible diagnosis.
    """
    if class_label and class_label.endswith("___healthy"):
        return "স্পষ্ট রোগের লক্ষণ নেই"

    label = (disease_label or DISEASE_TRANSLATION.get(class_label or "") or "").strip()
    if not label and class_label:
        label = class_label.split("___", 1)[-1].replace("_", " ").strip()
    if not label:
        return ""

    candidates = [value.strip() for value in crop_labels if isinstance(value, str) and value.strip()]
    if class_label:
        crop_key = class_label.split("___", 1)[0]
        candidates.append(_CROP_TRANSLATIONS.get(crop_key, crop_key.replace("_", " ")))
    for crop in sorted(set(candidates), key=len, reverse=True):
        prefix = re.compile(
            rf"^\s*{re.escape(crop)}(?:(?:'s|’s)|(?:এর|ের|র|ে))?(?=\s|[:—–·,]|$)\s*(?:[:—–·,]\s*)?",
            re.IGNORECASE,
        )
        trimmed = prefix.sub("", label, count=1).strip()
        if trimmed != label:
            label = trimmed
            break

    return re.sub(r"^[\s,;:—–·-]+|[\s,;:—–·-]+$", "", label).strip()
