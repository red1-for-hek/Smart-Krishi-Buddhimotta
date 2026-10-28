"""Safety checks for the deterministic disease-treatment allow-list."""
from __future__ import annotations

import unittest

from backend.disease_guidance import get_disease_guidance, get_visual_disease_guidance


class DiseaseGuidanceTests(unittest.TestCase):
    def test_acrobat_is_only_listed_for_supported_potato_blight_classes(self) -> None:
        for class_label in ("Potato___Early_blight", "Potato___Late_blight"):
            with self.subTest(class_label=class_label):
                guidance = get_disease_guidance(class_label)
                product = guidance["products"][0]
                self.assertEqual([item["registration_no"] for item in guidance["products"]], ["AP-353"])
                self.assertEqual(product["rate"], "৪ গ্রাম / ১ লিটার পানি")
                self.assertIn("আলু", product["label_scope"])
                self.assertIn("স্প্রে", product["application_method"])

    def test_cabrio_cucurbit_powdery_mildew_mapping_matches_record(self) -> None:
        guidance = get_disease_guidance("Squash___Powdery_mildew")
        self.assertEqual([item["registration_no"] for item in guidance["products"]], ["AP-1770"])
        product = guidance["products"][0]
        self.assertEqual(product["label_scope"], "স্কোয়াশ/কুমড়াজাতীয় — Powdery mildew")
        self.assertEqual(product["rate"], "৩ গ্রাম / ১ লিটার পানি")
        self.assertEqual(product["source_url"], "https://ppw.krishi.gov.bd/pesticide-details/4522")

    def test_verified_companion_record_is_narrowly_mapped_to_tomato_early_blight(self) -> None:
        guidance = get_disease_guidance("Tomato___Early_blight")
        product = guidance["products"][0]
        self.assertEqual(product["registration_no"], "AP-664")
        self.assertEqual(product["rate"], "২ গ্রাম / ১ লিটার পানি")
        self.assertIn("টমেটো", product["label_scope"])
        self.assertIn("application_method", product)
        self.assertEqual(product["source_url"], "https://ppw.krishi.gov.bd/pesticide-details/13610")
        self.assertNotIn("টমেটো", guidance["reason"])
        self.assertIn("আর্লি ব্লাইট", guidance["reason"])

    def test_visual_cucurbit_pair_uses_the_exact_recorded_rate(self) -> None:
        guidance = get_visual_disease_guidance({
            "leaf_status": "leaf",
            "crop_en": "cucumber",
            "crop_bn": "শসা",
            "disease_en": "powdery mildew",
            "disease_bn": "পাউডারি মিলডিউ",
            "cause_type": "fungal",
        })
        product = guidance["products"][0]
        self.assertEqual(product["registration_no"], "AP-1770")
        self.assertEqual(product["rate"], "৩ গ্রাম / ১ লিটার পানি")
        self.assertIn("কুমড়াজাতীয়", product["label_scope"])

    def test_mango_cabrio_rate_is_not_generalised_from_cucurbits(self) -> None:
        guidance = get_disease_guidance("Mango___Anthracnose")
        product = guidance["products"][0]
        self.assertEqual(product["registration_no"], "AP-1770")
        self.assertEqual(product["rate"], "১ গ্রাম / ১ লিটার পানি")
        self.assertIn("আম", product["label_scope"])

    def test_conflicting_plantvillage_label_withholds_visual_product(self) -> None:
        guidance = get_visual_disease_guidance({
            "leaf_status": "leaf",
            "crop_en": "tomato",
            "crop_bn": "টমেটো",
            "disease_en": "early blight",
            "disease_bn": "আর্লি ব্লাইট",
            "cause_type": "fungal",
            "confidence_band": "medium",
            "plantvillage_class_label": "Pepper,_bell___Bacterial_spot",
        })
        self.assertEqual(guidance["products"], [])
        self.assertIn("মেলেনি", guidance["product_note"])

    def test_low_confidence_visual_pair_withholds_product_even_if_a_dae_record_exists(self) -> None:
        guidance = get_visual_disease_guidance({
            "leaf_status": "leaf",
            "crop_en": "tomato",
            "crop_bn": "টমেটো",
            "disease_en": "early blight",
            "cause_type": "fungal",
            "confidence_band": "low",
        })
        self.assertEqual(guidance["products"], [])
        self.assertIn("কম নিশ্চিত", guidance["product_note"])

    def test_visual_exact_pair_can_reach_verified_use_outside_plantvillage_classes(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "Chili pepper",
            "crop_bn": "মরিচ",
            "disease_en": "Anthracnose",
            "disease_bn": "অ্যানথ্রাকনোজ",
            "cause_type": "fungal",
        }
        guidance = get_visual_disease_guidance(visual)
        self.assertEqual([item["registration_no"] for item in guidance["products"]], ["AP-353"])
        self.assertEqual(guidance["products"][0]["label_scope"], "মরিচ — Anthracnose")

    def test_capsicum_bacterial_spot_does_not_match_a_chili_product(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "Capsicum (bell pepper)",
            "crop_bn": "ক্যাপসিকাম",
            "disease_en": "Bacterial spot",
            "disease_bn": "ব্যাকটেরিয়াল দাগ",
            "cause_type": "bacterial",
        }
        guidance = get_visual_disease_guidance(visual)
        self.assertEqual(guidance["products"], [])
        self.assertIn("যাচাইকৃত DAE", guidance["product_note"])

    def test_unverified_crop_disease_and_non_leaf_images_get_no_product(self) -> None:
        cases = [
            {"leaf_status": "leaf", "crop_en": "Tomato", "disease_en": "Bacterial spot", "cause_type": "bacterial"},
            {"leaf_status": "not_leaf", "crop_en": "Potato", "disease_en": "Late blight", "cause_type": "fungal"},
            {"leaf_status": "uncertain", "crop_en": None, "disease_en": None, "cause_type": "unknown"},
        ]
        for visual in cases:
            with self.subTest(visual=visual):
                self.assertEqual(get_visual_disease_guidance(visual)["products"], [])

    def test_no_product_is_inferred_for_unverified_local_model_classes(self) -> None:
        for class_label in (
            "Tomato___Late_blight",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Corn_(maize)___Common_rust_",
        ):
            with self.subTest(class_label=class_label):
                self.assertEqual(get_disease_guidance(class_label)["products"], [])

    def test_healthy_class_never_gets_a_pesticide_recommendation(self) -> None:
        guidance = get_disease_guidance("Potato___healthy")
        self.assertEqual(guidance["category"], "healthy_class")
        self.assertEqual(guidance["products"], [])
        self.assertIn("সুস্থ", guidance["reason"])


if __name__ == "__main__":
    unittest.main()
