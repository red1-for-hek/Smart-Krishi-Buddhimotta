"""Rule-based, Bangladesh-relevant guidance for crop-image assessments.

Chemical recommendations are deliberately allow-listed: a product is shown only
when the DAE Pesticide Information Repository has a registration number and an
explicit crop/pest-use record. The repository is not a live stock feed. No AI
output is used to invent pesticide brands, doses, or water amounts.
"""
from __future__ import annotations

import re
from typing import Any

from backend.disease_labels import disease_only_label

CHECKED_ON = "2026-10-28"
DAE_REGISTRY_URL = "https://dae.gov.bd/site/files/4ea8e032-77df-4e6b-952b-5eab3004113c/-Registered-Pesticide"
PPW_REPOSITORY_URL = "https://ppw.krishi.gov.bd/"
PRISM_LOCAL_LIST_URL = "https://prism.dae.gov.bd/pesticide-list"
DAE_LABEL_LAW_URL = "https://dae.gov.bd/pages/notices/6922f1ecdbfbab28ce0d6c95"

_BASE_SOURCES = [
    {"label": "DAE Pesticide Information Repository (PPW)", "url": PPW_REPOSITORY_URL},
    {"label": "DAE PRISM: উপজেলা-ভিত্তিক বাজারজাত পণ্যের তালিকা", "url": PRISM_LOCAL_LIST_URL},
    {"label": "DAE: বালাইনাশকের লেবেল ব্যবহারের নির্দেশনা", "url": DAE_LABEL_LAW_URL},
]

# These are labels to compare against the plant, not symptoms the classifier
# claims it has actually observed in the uploaded photo.
SIGNS_TO_CHECK: dict[str, str] = {
    "Apple___Apple_scab": "পাতা বা ফলে জলপাই-বাদামি, খসখসে দাগ আছে কি দেখুন।",
    "Apple___Black_rot": "পাতা বা ফলে বাদামি/কালচে পচা দাগ এবং ফল শুকিয়ে যাওয়ার লক্ষণ মিলিয়ে দেখুন।",
    "Apple___Cedar_apple_rust": "পাতার ওপর কমলা-হলুদ দাগ বা নিচে গঠন আছে কি দেখুন।",
    "Cherry_(including_sour)___Powdery_mildew": "পাতা বা কচি ডগায় সাদা গুঁড়ার মতো আস্তরণ আছে কি দেখুন।",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "ভুট্টার পাতায় শিরার সঙ্গে লম্বা ধূসর/বাদামি দাগ আছে কি দেখুন।",
    "Corn_(maize)___Common_rust_": "পাতায় ছোট, উঁচু, লালচে-বাদামি গুঁড়ার ফোস্কার মতো দাগ আছে কি দেখুন।",
    "Corn_(maize)___Northern_Leaf_Blight": "পাতায় লম্বা, ধূসর-সবুজ, পরে বাদামি হওয়া দাগ আছে কি দেখুন।",
    "Grape___Black_rot": "আঙুরের পাতা বা ফলে গাঢ় দাগ ও ফলের কালো পচন মিলিয়ে দেখুন।",
    "Grape___Esca_(Black_Measles)": "পাতায় শিরার মাঝখানে হলুদ/লালচে অংশ, শুকনো ডাল বা ফলে কালো দাগ আছে কি দেখুন।",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "পাতায় অনিয়মিত বাদামি দাগ এবং হলদে কিনারা আছে কি দেখুন।",
    "Orange___Haunglongbing_(Citrus_greening)": "নতুন পাতায় অসমান হলদে দাগ, ছোট/বিকৃত ফল বা এক পাশের ডাল শুকানো আছে কি দেখুন।",
    "Peach___Bacterial_spot": "পাতা/ফলে ছোট গাঢ় দাগ, হলুদ কিনারা বা ফলের ক্ষত আছে কি দেখুন।",
    "Pepper,_bell___Bacterial_spot": "ক্যাপসিকামের পাতা বা ফলে ছোট জলভেজা/গাঢ় দাগ আছে কি দেখুন।",
    "Rice___Blast": "ধানের পাতায় ধূসর কেন্দ্র ও বাদামি কিনারাযুক্ত লম্বাটে দাগ বা শীষের গোড়ায় কালচে ক্ষত আছে কি মিলিয়ে দেখুন।",
    "Rice___Sheath_blight": "ধানের পাতার খাপে অনিয়মিত ধূসর/সাদা কেন্দ্র ও বাদামি কিনারাযুক্ত দাগ আছে কি দেখুন।",
    "Potato___Early_blight": "আলুর পুরোনো পাতায় গোল বাদামি দাগের ভেতর বলয় আছে কি দেখুন।",
    "Potato___Late_blight": "আলুর পাতায় দ্রুত বাড়তে থাকা জলভেজা কালচে দাগ, বিশেষত পাতার নিচে সাদা আবরণ আছে কি দেখুন।",
    "Squash___Powdery_mildew": "স্কোয়াশের পাতায় মুছে না-যাওয়া সাদা গুঁড়ার মতো দাগ আছে কি দেখুন।",
    "Strawberry___Leaf_scorch": "স্ট্রবেরির পাতায় ছোট বেগুনি/লালচে দাগ, পরে বাদামি কেন্দ্র আছে কি দেখুন।",
    "Tomato___Bacterial_spot": "টমেটোর পাতা/ফলে ছোট জলভেজা থেকে গাঢ় দাগ আছে কি দেখুন।",
    "Tomato___Early_blight": "টমেটোর পুরোনো পাতায় গোল বাদামি দাগের ভেতর বলয় আছে কি দেখুন।",
    "Tomato___Late_blight": "টমেটোর পাতায় দ্রুত বাড়তে থাকা জলভেজা কালচে দাগ, বিশেষত পাতার নিচে সাদা আবরণ আছে কি দেখুন।",
    "Tomato___Leaf_Mold": "টমেটো পাতার ওপর ফ্যাকাশে দাগ এবং নিচে জলপাই-সবুজ ছাঁচ আছে কি দেখুন।",
    "Tomato___Septoria_leaf_spot": "পাতায় অনেক ছোট গোল দাগ, গাঢ় কিনারা ও ফিকে কেন্দ্র আছে কি দেখুন।",
    "Tomato___Spider_mites Two-spotted_spider_mite": "পাতার নিচে সূক্ষ্ম জাল, চলমান ক্ষুদ্র মাকড় বা ছোট ফ্যাকাশে বিন্দু আছে কি দেখুন।",
    "Tomato___Target_Spot": "টমেটোর পাতায় গোল বাদামি দাগ ও একাধিক বলয় আছে কি দেখুন।",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "নতুন পাতা কুঁকড়ে/হলদে হওয়া ও গাছ খাটো হয়ে যাওয়ার লক্ষণ মিলিয়ে দেখুন।",
    "Tomato___Tomato_mosaic_virus": "পাতায় হালকা-গাঢ় সবুজ মোজাইক ছোপ বা বিকৃতি আছে কি দেখুন।",
}

# Product/label information transcribed from DAE's Pesticide Information
# Repository records. Keep rates and crop/pest scope attached to the exact AP
# registration; never generalize a rate to another product or crop.
ACROBAT_MZ = {
    "name": "Acrobat MZ",
    "registration_no": "AP-353",
    "active_ingredients": "Mancozeb + Dimethomorph",
    "label_scope": "",
    "rate": "৪ গ্রাম / ১ লিটার পানি",
    "application_method": "স্প্রে মেশিনে মিশ্রণ তৈরি করে গাছে প্রয়োগ; গোড়ায় ঢালার নির্দেশ নয়।",
    "directions": [
        "DAE PPW record-এ প্রাথমিক সংক্রমণের পর্যায়ে ব্যবহারের কথা আছে; প্রয়োজন হলে ২ সপ্তাহ পর পুনঃপ্রয়োগের নির্দেশও আছে।",
        "রেকর্ডে স্প্রে-পরবর্তী ফসল তোলার বিরতি ৭–১৪ দিন বলা আছে; প্যাকেটের বর্তমান লেবেলই চূড়ান্ত।",
    ],
    "source_label": "DAE PPW product record — Acrobat MZ",
    "source_url": "https://ppw.krishi.gov.bd/pesticide-details/11633",
}

CABRIO_TOP = {
    "name": "Cabrio Top",
    "registration_no": "AP-1770",
    "active_ingredients": "Pyraclostrobin + Metiram",
    "label_scope": "",
    "rate": "৩ গ্রাম / ১ লিটার পানি",
    "application_method": "DAE PPW record-এ স্প্রে করার নির্দেশ আছে; গোড়ায় ঢালার নির্দেশ নয়।",
    "directions": [
        "DAE PPW record-এ রোগের আগেই বা রোগের প্রাথমিক পর্যায়ে প্রয়োগের কথা আছে; ব্যবহারের আগে প্যাকেটের লেবেলে আপনার নির্দিষ্ট ফসলটি আছে কি না মিলিয়ে নিন।",
        "রেকর্ডে প্রয়োগের পর ৭–১৪ দিন ফসল না তোলার কথা আছে; প্যাকেটের বর্তমান লেবেলই চূড়ান্ত।",
    ],
    "source_label": "DAE PPW product record — Cabrio Top",
    "source_url": "https://ppw.krishi.gov.bd/pesticide-details/4522",
}

COMPANION = {
    "name": "Companion",
    "registration_no": "AP-664",
    "active_ingredients": "Carbendazim + Mancozeb",
    "label_scope": "",
    "rate": "২ গ্রাম / ১ লিটার পানি",
    "application_method": "এই ফসল–রোগ সারিতে প্রয়োগপথ স্পষ্ট নয়; প্যাকেটের বর্তমান লেবেলে স্প্রে/গোড়ার নির্দেশ না মিলিয়ে প্রয়োগ করবেন না।",
    "directions": [
        "DAE PPW record-এ এই নির্দিষ্ট ফসল–রোগের জন্য ২ গ্রাম/লিটার মাত্রা আছে। প্যাকেটের বর্তমান লেবেলই চূড়ান্ত।",
        "প্যাকেটের লেবেলে ফসল, রোগ, প্রয়োগপদ্ধতি ও ফসল তোলার বিরতি না মিললে ব্যবহার করবেন না।",
    ],
    "source_label": "DAE PPW product record — Companion",
    "source_url": "https://ppw.krishi.gov.bd/pesticide-details/13610",
}

M_ZEB = {
    "name": "M-Zeb 80 WP",
    "registration_no": "AP-2491",
    "active_ingredients": "Mancozeb",
    "label_scope": "",
    "rate": "২ গ্রাম / ১ লিটার পানি",
    "application_method": "DAE PPW record-এ শুধু পানিতে মিশ্রণের মাত্রা দেওয়া আছে; প্রয়োগপথ বর্তমান প্যাকেটের লেবেল থেকে নিশ্চিত করুন।",
    "directions": [
        "DAE PPW record-এ টমেটো — Early blight-এর নির্দিষ্ট ব্যবহার আছে।",
        "রেকর্ডে ৭–১৪ দিনের মধ্যে গবাদিপশু/হাঁস-মুরগির প্রবেশ এবং ফসল বিক্রি/খাওয়ার বিষয়ে সতর্কতা আছে; সঠিক বিরতি বর্তমান প্যাকেটের লেবেলে মিলিয়ে নিন।",
    ],
    "source_label": "DAE PPW product record — M-Zeb 80 WP",
    "source_url": "https://ppw.krishi.gov.bd/pesticide-details/2021",
}

SELTIMA = {
    "name": "Seltima",
    "registration_no": "AP-5112",
    "active_ingredients": "Pyraclostrobin",
    "label_scope": "",
    "rate": "২ মিলি/লিটার পানি; ১০০০ মিলি/হেক্টর",
    "application_method": "DAE PPW record: বুটিং পর্যায়ে প্রথম প্রয়োগ, ১০ দিন পর দ্বিতীয়; ভোরে/বিকেলে প্রয়োগ। প্রয়োগপদ্ধতি বর্তমান প্যাকেটের লেবেল থেকে নিশ্চিত করুন।",
    "directions": [
        "DAE PPW record-এ ধান — Blast এবং Sheath blight, দুই রোগের ব্যবহারই নির্দিষ্ট করা আছে।",
        "রেকর্ডে প্রয়োগের পর ৭–১৪ দিনের মধ্যে ফসল কাটা/গবাদিপশু প্রবেশে নিষেধ আছে; সঠিক বিরতি প্যাকেটের বর্তমান লেবেলে মিলিয়ে নিন।",
        "গ্লাভস ও লেবেলে বলা PPE ব্যবহার করুন; গায়ে বা বাতাসের বিপরীতে স্প্রে করবেন না এবং পুকুর/জলাশয় দূষিত করবেন না।",
    ],
    "source_label": "DAE PPW product record — Seltima",
    "source_url": "https://ppw.krishi.gov.bd/pesticide-details/3563",
}

CABRIO_TOP_MANGO = {
    **CABRIO_TOP,
    "rate": "১ গ্রাম / ১ লিটার পানি",
}

_PRODUCT_BY_CLASS: dict[str, list[dict[str, Any]]] = {
    "Potato___Early_blight": [ACROBAT_MZ],
    "Potato___Late_blight": [ACROBAT_MZ],
    "Squash___Powdery_mildew": [CABRIO_TOP],
    "Tomato___Early_blight": [COMPANION, M_ZEB],
    "Rice___Blast": [SELTIMA],
    "Rice___Sheath_blight": [SELTIMA],
    "Chili___Anthracnose": [ACROBAT_MZ],
    "Groundnut___Tikka": [ACROBAT_MZ],
    "Onion___Purple_blotch": [ACROBAT_MZ],
    "Banana___Leaf_spot": [ACROBAT_MZ],
    "Cucurbits___Powdery_mildew": [CABRIO_TOP],
    "Mango___Anthracnose": [CABRIO_TOP_MANGO],
}

_PRODUCT_SCOPE_BY_CLASS = {
    "Potato___Early_blight": "আলু — Early blight",
    "Potato___Late_blight": "আলু — Late blight",
    "Squash___Powdery_mildew": "স্কোয়াশ/কুমড়াজাতীয় — Powdery mildew",
    "Tomato___Early_blight": "টমেটো — Early blight",
    "Rice___Blast": "ধান — Blast",
    "Rice___Sheath_blight": "ধান — Sheath blight",
    "Chili___Anthracnose": "মরিচ — Anthracnose",
    "Groundnut___Tikka": "বাদাম — Tikka disease",
    "Onion___Purple_blotch": "পেঁয়াজ — Purple blotch",
    "Banana___Leaf_spot": "কলা — Leaf spot",
    "Cucurbits___Powdery_mildew": "কুমড়াজাতীয় (Cucurbits) — Powdery mildew",
    "Mango___Anthracnose": "আম — Anthracnose",
}

# If Qwen supplies a PlantVillage label alongside an exact DAE crop–disease
# pair, contradictory labels suppress the product rather than choosing one.
_COMPATIBLE_VISION_LABELS = {
    "Potato___Early_blight": {"Potato___Early_blight"},
    "Potato___Late_blight": {"Potato___Late_blight"},
    "Tomato___Early_blight": {"Tomato___Early_blight"},
    "Rice___Blast": set(),
    "Rice___Sheath_blight": set(),
    "Chili___Anthracnose": set(),
    "Groundnut___Tikka": set(),
    "Onion___Purple_blotch": set(),
    "Banana___Leaf_spot": set(),
    "Cucurbits___Powdery_mildew": {"Squash___Powdery_mildew"},
    "Mango___Anthracnose": set(),
}


def _watering_guidance(class_label: str) -> dict[str, str]:
    crop = class_label.split("___", 1)[0]
    if crop in {"Potato", "Tomato", "Corn_(maize)"}:
        care = (
            "মাটির আর্দ্রতা দেখে গোড়ায় সেচ দিন; পাতার ওপর পানি ছিটানো ও জলাবদ্ধতা এড়ান, "
            "পানি নিষ্কাশনের পথ খোলা রাখুন। ভেজা/কুয়াশাচ্ছন্ন আবহাওয়ায় অতিরিক্ত সেচ দেবেন না।"
        )
    else:
        care = (
            "মাটির আর্দ্রতা দেখে গোড়ায় প্রয়োজনমতো সেচ দিন; পাতায় পানি জমিয়ে রাখা ও "
            "জলাবদ্ধতা এড়ান।"
        )
    return {
        "care": care,
        "amount_note": (
            "এই ছবি থেকে জমির আয়তন, মাটির ধরন/আর্দ্রতা, গাছের বয়স ও আবহাওয়া জানা যায় না—"
            "তাই নিরাপদে নির্দিষ্ট লিটার/বিঘা বলা সম্ভব নয়।"
        ),
    }


def _general_actions(class_label: str) -> tuple[str, list[str], str]:
    """Return (category, practical steps, product note) for uncurated classes."""
    if "___healthy" in class_label:
        return (
            "healthy_class",
            [
                "পাতার দুই পাশ ও নতুন পাতায় লক্ষণ আছে কি নিয়মিত দেখুন; ছবির ‘healthy’ শ্রেণি রোগমুক্তির নিশ্চয়তা নয়।",
                "লক্ষণ না মিললে শুধু মডেল-স্কোরের কারণে স্প্রে/ওষুধ দেবেন না।",
            ],
            "কোনো ওষুধ প্রয়োজন বলে এই ফল প্রমাণ করে না; প্রতিরোধমূলকভাবে বালাইনাশক দেবেন না।",
        )

    if "Virus" in class_label or "Citrus_greening" in class_label:
        return (
            "possible_viral_or_greening",
            [
                "নতুন পাতার কুঁকড়ানো, মোজাইক/অসমান হলদে ভাব এবং আক্রান্ত গাছের বিস্তার হাতে মিলিয়ে দেখুন।",
                "ভাইরাসে ছত্রাকনাশক সারায় না। লক্ষণ নিশ্চিত হলে আক্রান্ত গাছ/চারা আলাদা রাখুন; বাহক পোকা চোখে পড়লে ফসল-লেবেলমাফিক ব্যবস্থা নিন।",
                "সরঞ্জাম ও হাত পরিষ্কার রাখুন; রোগাক্রান্ত গাছের বংশবিস্তার/চারা তৈরি করবেন না।",
            ],
            "ভাইরাসের জন্য নিরাময়কারী স্প্রে নেই; বাহক পোকা নিশ্চিত না হলে কীটনাশক দেবেন না।",
        )

    if "Bacterial" in class_label:
        return (
            "possible_bacterial",
            [
                "পাতার দুই পাশ, কাণ্ড ও ফলেও একই রকম দাগ আছে কি মিলিয়ে দেখুন; অন্য গাছেও ছড়াচ্ছে কি লক্ষ্য করুন।",
                "ভেজা গাছে হাত দেওয়া/ছাঁটাই এড়ান; ব্যবহারের পর সরঞ্জাম পরিষ্কার করুন এবং আক্রান্ত অবশেষ জমিতে ফেলে রাখবেন না।",
                "রোগের লক্ষণ নিশ্চিত হওয়ার আগে অ্যান্টিবায়োটিক বা তামাযুক্ত স্প্রে অনুমান করে ব্যবহার করবেন না।",
            ],
            "এই ছবির মডেল-লেবেল একা ব্যাকটেরিয়াল রোগ নিশ্চিত করে না; ফসল ও রোগের নির্দিষ্ট DAE label match না থাকায় পণ্য/মাত্রা দেখানো হচ্ছে না।",
        )

    if "Spider_mites" in class_label:
        return (
            "possible_mite",
            [
                "পাতার নিচে মাকড়, সূক্ষ্ম জাল ও ক্ষুদ্র চলমান দাগ হাতে/আতশকাচে নিশ্চিত করুন।",
                "আক্রান্ত গাছ আলাদা পর্যবেক্ষণ করুন; অল্প আক্রান্ত পাতা জমিতে না ফেলে সরিয়ে ফেলুন।",
                "মাকড় নিশ্চিত না হলে কীটনাশক/মাকড়নাশক দেবেন না; ভুল পণ্য উপকারী পোকাও মেরে ফেলতে পারে।",
            ],
            "টমেটো ও Spider mite—এই দুইয়ের জন্য বর্তমান, স্পষ্ট DAE label match যাচাই না হওয়ায় নির্দিষ্ট পণ্য/মাত্রা দেওয়া হচ্ছে না।",
        )

    if class_label == "Tomato___Early_blight":
        return (
            "possible_leaf_disease",
            [
                "টমেটোর পুরোনো পাতায় গোল বাদামি বলয়যুক্ত দাগ আছে কি হাতে মিলিয়ে দেখুন; অন্য রোগ/পুষ্টিঘাটতির লক্ষণও বাদ দিন।",
                "আক্রান্ত পাতা জমিতে না ফেলে সরিয়ে দিন, পাতায় দীর্ঘসময় পানি লাগা কমান এবং গাছের মাঝে বাতাস চলাচল রাখুন।",
                "স্প্রে করার আগে ফসল ও রোগের সঙ্গে পণ্যের বর্তমান লেবেল মিলিয়ে উপজেলা কৃষি অফিসারের পরামর্শ নিন।",
            ],
            "এই ছবির মডেল-লেবেল একা রোগ নিশ্চিত করে না; দৃশ্যমান লক্ষণ ও ফসল মিলিয়ে DAE লেবেল যাচাই করুন।",
        )

    if class_label in {
        "Chili___Anthracnose", "Groundnut___Tikka", "Onion___Purple_blotch",
        "Banana___Leaf_spot", "Cucurbits___Powdery_mildew", "Mango___Anthracnose",
    }:
        return (
            "possible_leaf_disease",
            [
                "পাতার দুই পাশ, কাণ্ড ও ফলেও একই লক্ষণ আছে কি হাতে মিলিয়ে দেখুন; পাশের সুস্থ গাছের সঙ্গে তুলনা করুন।",
                "আক্রান্ত অবশেষ জমিতে ফেলে রাখবেন না; গাছের মাঝে বাতাস চলাচল ও পানি নিষ্কাশন বজায় রাখুন।",
                "ফসল–রোগ মিলে গেলেও প্যাকেটের বর্তমান লেবেল, প্রয়োগপথ ও ফসল তোলার বিরতি না মিলিয়ে পণ্য প্রয়োগ করবেন না।",
            ],
            "এই ছবির AI-অনুমান নিশ্চিত রোগনির্ণয় নয়; DAE-র নির্দিষ্ট ফসল–রোগ রেকর্ড ও বর্তমান প্যাকেটের লেবেল মিলিয়ে নিন।",
        )

    return (
        "possible_leaf_disease",
        [
            "পাতার ওপর-নিচ, কাণ্ড এবং গাছের একাধিক অংশে লক্ষণ মিলিয়ে দেখুন; পাশের সুস্থ গাছের সঙ্গে তুলনা করুন।",
            "আক্রান্ত/ঝরা পাতা জমিতে ফেলে রাখবেন না; গাছের মাঝে বাতাস চলাচল ও পানি নিষ্কাশন বজায় রাখুন।",
            "ছবির শ্রেণি নিশ্চিত রোগ নয়—ফসল, লক্ষণ ও পণ্যের label একসাথে না মিললে বালাইনাশক প্রয়োগ করবেন না।",
        ],
        "এই crop–disease জুটির জন্য স্পষ্ট, যাচাইযোগ্য DAE নিবন্ধিত product-use match না থাকায় কোনো পণ্য বা ডোজ সুপারিশ করা হচ্ছে না।",
    )


def get_disease_guidance(class_label: str) -> dict[str, Any]:
    """Return deterministic advice for one exact crop–disease class/pair."""
    healthy = "___healthy" in class_label
    category, actions, product_note = _general_actions(class_label)
    products = [
        {**product, "label_scope": _PRODUCT_SCOPE_BY_CLASS.get(class_label, product.get("label_scope", ""))}
        for product in _PRODUCT_BY_CLASS.get(class_label, [])
    ]
    if products:
        category = "possible_disease_with_verified_label"
        product_note = (
            "নীচে DAE PPW repository-তে AP নম্বরসহ নির্দিষ্ট ফসল–বালাই ব্যবহারের রেকর্ড আছে। "
            "এটি সক্রিয় নিবন্ধন বা আপনার এলাকার দোকানের মজুতের নিশ্চয়তা নয়। কেনার আগে আসল প্যাকেটের বর্তমান "
            "লেবেলে একই ফসল ও রোগ, মাত্রা, ফসল তোলার বিরতি (PHI) এবং নিরাপত্তা নির্দেশনা মিলিয়ে নিন।"
        )
        if len(products) > 1:
            product_note += " একাধিক বিকল্প একসাথে মেশাবেন/প্রয়োগ করবেন না; লেবেল ও স্থানীয় কৃষি কর্মকর্তার পরামর্শে একটি বেছে নিন।"
    elif not healthy:
        product_note += " স্থানীয়ভাবে অনুমোদিত পণ্য বাছতে DAE PRISM-এর উপজেলা-ভিত্তিক তালিকা দেখুন।"

    signs = SIGNS_TO_CHECK.get(
        class_label,
        "এই শ্রেণির পরিচিত লক্ষণ পাতার দুই পাশ, কাণ্ড ও নতুন পাতায় হাতে মিলিয়ে দেখুন।",
    )
    if healthy:
        reason = (
            "মডেলের শীর্ষ শ্রেণি ‘সুস্থ’; এটি প্রশিক্ষণ-শ্রেণির অনুমান, "
            "পরীক্ষিত রোগমুক্তির নিশ্চয়তা নয়।"
        )
    else:
        display_label = disease_only_label(class_label)
        reason = (
            f"মডেল এই ছবির সম্ভাব্য রোগ-শ্রেণি হিসেবে ‘{display_label}’ দিয়েছে। "
            "এটি নিশ্চিত রোগনির্ণয় নয়; নিচের লক্ষণগুলো হাতে মিলিয়ে দেখুন।"
        )

    sources = list(_BASE_SOURCES)
    for product in products:
        sources.append({"label": product["source_label"], "url": product["source_url"]})

    return {
        "category": category,
        "reason": reason,
        "signs_to_check": signs,
        "actions": actions,
        "product_note": product_note,
        "products": products,
        "watering": _watering_guidance(class_label),
        "source_checked_on": CHECKED_ON,
        "sources": sources,
    }


def _normalise_vision_text(value: Any) -> str:
    text = str(value or "").casefold()
    return re.sub(r"[^a-z0-9\u0980-\u09ff]+", " ", text).strip()


def _has_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = f" {_normalise_vision_text(text)} "
    return any(f" {_normalise_vision_text(phrase)} " in normalized for phrase in phrases)


def _visual_product_label(crop_en: str | None, crop_bn: str | None, disease_en: str | None, disease_bn: str | None) -> str | None:
    crop_text = f"{crop_en or ''} {crop_bn or ''}"
    disease_text = f"{disease_en or ''} {disease_bn or ''}"

    if _has_phrase(crop_text, ("rice", "ধান")):
        if _has_phrase(disease_text, ("sheath blight", "শীথ ব্লাইট", "সিথ ব্লাইট")):
            return "Rice___Sheath_blight"
        if _has_phrase(disease_text, ("leaf blast", "neck blast", "rice blast", "blast", "লিফ ব্লাস্ট", "নেক ব্লাস্ট", "ব্লাস্ট")):
            return "Rice___Blast"
    if _has_phrase(crop_text, ("potato", "আলু")):
        if _has_phrase(disease_text, ("early blight", "আর্লি ব্লাইট")):
            return "Potato___Early_blight"
        if _has_phrase(disease_text, ("late blight", "লেট ব্লাইট")):
            return "Potato___Late_blight"
    if _has_phrase(crop_text, ("tomato", "টমেটো")) and _has_phrase(disease_text, ("early blight", "আর্লি ব্লাইট")):
        return "Tomato___Early_blight"
    # DAE's label says Chili; do not equate this with capsicum/bell pepper.
    if _has_phrase(crop_text, ("chili", "chilli", "chili pepper", "chilli pepper", "মরিচ", "লঙ্কা")) and _has_phrase(disease_text, ("anthracnose", "অ্যানথ্রাকনোজ", "এ্যানথ্রাকনোজ")):
        return "Chili___Anthracnose"
    if _has_phrase(crop_text, ("groundnut", "peanut", "বাদাম")) and _has_phrase(disease_text, ("tikka", "টিক্কা")):
        return "Groundnut___Tikka"
    if _has_phrase(crop_text, ("onion", "পেঁয়াজ", "পেঁয়াজ")) and _has_phrase(disease_text, ("purple blotch", "পার্পল ব্লচ")):
        return "Onion___Purple_blotch"
    if _has_phrase(crop_text, ("banana", "কলা")) and _has_phrase(disease_text, ("leaf spot", "পাতার দাগ")):
        return "Banana___Leaf_spot"
    cucurbits = (
        "cucurbits", "squash", "pumpkin", "cucumber", "watermelon", "bottle gourd",
        "bitter gourd", "ridge gourd", "snake gourd", "শসা", "কুমড়া", "কুমড়া",
        "লাউ", "করলা", "স্কোয়াশ", "স্কোয়াশ", "তরমুজ", "কুমড়াজাতীয়", "কুমড়াজাতীয়",
    )
    if _has_phrase(crop_text, cucurbits) and _has_phrase(disease_text, ("powdery mildew", "পাউডারি মিলডিউ")):
        return "Cucurbits___Powdery_mildew"
    if _has_phrase(crop_text, ("mango", "আম")) and _has_phrase(disease_text, ("anthracnose", "অ্যানথ্রাকনোজ", "এ্যানথ্রাকনোজ")):
        return "Mango___Anthracnose"
    return None


def get_unclassified_visual_guidance(cause_type: str = "unknown") -> dict[str, Any]:
    """Safe next steps when the image result has no exact local rule match."""
    actions = [
        "একই গাছের নতুন ও পুরোনো পাতা, পাতার দুই পাশ এবং গোটা গাছের পরিষ্কার ছবি তুলে লক্ষণ মিলিয়ে দেখুন।",
        "আক্রান্ত পাতা/অবশেষ জমিতে ফেলে রাখবেন না; পাতায় পানি জমা ও জলাবদ্ধতা এড়িয়ে মাটির আর্দ্রতা দেখে গোড়ায় সেচ দিন।",
        "পাতার ছবি থেকে সার-ঘাটতি নিশ্চিত করা যায় না; মাটি পরীক্ষা বা উপজেলা কৃষি অফিসারের পরামর্শ ছাড়া বাড়তি সার দেবেন না।",
    ]
    if cause_type == "nutrient":
        actions = [
            "পুষ্টিঘাটতি নিশ্চিত করতে নতুন ও পুরোনো পাতার লক্ষণ এবং জমির মাটি পরীক্ষা করুন।",
            "শুধু ছবির ভিত্তিতে ইউরিয়া/NPK বা অন্য সার দেবেন না; মাটি পরীক্ষার ফল ও স্থানীয় সুপারিশ অনুসরণ করুন।",
        ]
    elif cause_type in {"viral", "bacterial"}:
        actions = [
            "একই লক্ষণ অন্য গাছে ছড়াচ্ছে কি দেখুন; আক্রান্ত গাছ/চারা সাময়িক আলাদা রাখুন।",
            "ভেজা গাছে কাজ/ছাঁটাই এড়িয়ে ব্যবহার-পরবর্তী সরঞ্জাম পরিষ্কার করুন; কারণ নিশ্চিত না করে ওষুধ দেবেন না।",
        ]
    return {
        "category": "visual_unclassified",
        "reason": "ছবিভিত্তিক AI ফলের সঙ্গে নির্দিষ্ট, যাচাইকৃত ফসল–রোগ নির্দেশনা মেলেনি।",
        "signs_to_check": "পাতার দুই পাশ, নতুন ও পুরোনো পাতা, কাণ্ড এবং ফলের লক্ষণ হাতে মিলিয়ে দেখুন।",
        "actions": actions,
        "product_note": "এই ফসল–রোগ জুটির জন্য যাচাইকৃত DAE label-use match পাওয়া যায়নি; কোনো ব্র্যান্ড, মাত্রা বা প্রয়োগপথ অনুমান করে দেওয়া হচ্ছে না।",
        "products": [],
        "watering": {
            "care": "মাটির আর্দ্রতা দেখে গোড়ায় প্রয়োজনমতো সেচ দিন; পাতায় পানি জমা ও জলাবদ্ধতা এড়ান।",
            "amount_note": "ছবি থেকে জমির আয়তন, মাটি, গাছের বয়স বা আবহাওয়া জানা যায় না—তাই নির্দিষ্ট লিটার বলা নিরাপদ নয়।",
        },
        "source_checked_on": CHECKED_ON,
        "sources": list(_BASE_SOURCES),
    }


def get_visual_disease_guidance(visual: dict[str, Any]) -> dict[str, Any]:
    """Match visual results to DAE use records, even when showing a conditional candidate."""
    cause_type = str(visual.get("cause_type", "unknown"))
    if visual.get("leaf_status") != "leaf":
        return get_unclassified_visual_guidance(cause_type)

    exact_label = _visual_product_label(
        visual.get("crop_en"), visual.get("crop_bn"),
        visual.get("disease_en"), visual.get("disease_bn"),
    )
    if exact_label:
        qwen_class = visual.get("plantvillage_class_label")
        compatible_classes = _COMPATIBLE_VISION_LABELS.get(exact_label, set())
        if qwen_class and qwen_class not in compatible_classes:
            guidance = get_unclassified_visual_guidance(cause_type)
            guidance["product_note"] = (
                "ছবির crop–disease বর্ণনা ও মডেলের শ্রেণি একে অপরের সঙ্গে মেলেনি; ভুল চিকিৎসা এড়াতে DAE পণ্য/মাত্রা দেখানো হচ্ছে না। "
                "নতুন ছবি বা স্থানীয় কৃষি কর্মকর্তার সাহায্যে যাচাই করুন।"
            )
            return guidance
        if cause_type not in {"fungal", "unknown"}:
            guidance = get_unclassified_visual_guidance(cause_type)
            guidance["product_note"] = (
                "AI-এর দেওয়া রোগের কারণ-ধরন ও যাচাইকৃত DAE label-এর ছত্রাকজনিত ব্যবহারের মধ্যে অমিল আছে; "
                "ভুল পণ্য প্রয়োগ এড়াতে নাম/মাত্রা দেখানো হচ্ছে না। নতুন ছবি বা স্থানীয় কৃষি কর্মকর্তার সাহায্যে যাচাই করুন।"
            )
            return guidance

        guidance = get_disease_guidance(exact_label)
        if visual.get("confidence_band") not in {"medium", "high"}:
            if not guidance["products"]:
                guidance = get_unclassified_visual_guidance(cause_type)
                guidance["product_note"] = (
                    "ছবিভিত্তিক শনাক্তকরণ কম নিশ্চিত এবং এই নির্দিষ্ট ফসল–রোগের জন্য যাচাইকৃত পণ্য-রেকর্ড মেলেনি; "
                    "তাই কোনো ব্র্যান্ড/মাত্রা অনুমান করা হচ্ছে না।"
                )
                return guidance
            guidance["category"] = "possible_disease_with_verified_label_candidate"
            guidance["product_note"] = (
                "ছবিভিত্তিক AI শনাক্তকরণ কম নিশ্চিত/অনুপস্থিত—এটি রোগনির্ণয় নয়। নিচের পণ্য ও মাত্রা DAE-র "
                "একই ফসল–রোগের label-use রেকর্ড থেকে দেখানো শর্তসাপেক্ষ তথ্য; কেবল রোগটি বাস্তবে নিশ্চিত হলে বিবেচনা করুন। "
                "প্রয়োগের আগে আসল প্যাকেটের বর্তমান লেবেল, PHI ও PPE এবং উপজেলা কৃষি কর্মকর্তার পরামর্শ মিলিয়ে নিন।"
            )
        return guidance

    guidance = get_unclassified_visual_guidance(cause_type)
    if visual.get("confidence_band") not in {"medium", "high"}:
        guidance["product_note"] = (
            "ছবিভিত্তিক AI-এর শনাক্তকরণ কম নিশ্চিত বা নিশ্চিততার তথ্য অনুপস্থিত, এবং DAE-তে একই ফসল–রোগের "
            "label-use যাচাই হয়নি; তাই কোনো ব্র্যান্ড/মাত্রা অনুমান করা হচ্ছে না। ভালো আলোতে নতুন ছবি তুলুন বা উপজেলা কৃষি অফিসে মিলিয়ে নিন।"
        )
    return guidance
