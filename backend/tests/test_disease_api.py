"""Disease endpoint contract tests with model providers mocked."""
from __future__ import annotations

import base64
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy  # ensure the API's lazy import remains stable across mocked requests
from PIL import Image

from backend import main
from backend.disease_labels import DISEASE_TRANSLATION
from backend.main import DiseaseRequest


def _image_data_url() -> str:
    image = Image.new("RGB", (64, 64), color=(50, 135, 72))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class DiseaseApiTests(unittest.TestCase):
    def test_local_loader_tries_original_full_checkpoint_if_compact_weights_fail(self) -> None:
        model = type("FakeModel", (), {"eval": lambda self: self})()
        torch_module = types.ModuleType("torch")
        model_module = types.ModuleType("disease_model")
        attempted: list[str] = []

        def load_resnet9(path: str):
            checkpoint_name = Path(path).name
            attempted.append(checkpoint_name)
            if checkpoint_name == "plant_disease_weights.pth":
                raise ValueError("invalid compact checkpoint")
            return model

        model_module.load_resnet9 = load_resnet9
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "plant_disease_weights.pth").touch()
            (repo_root / "plant-disease-model-complete.pth").touch()
            with patch.object(main, "_REPO_ROOT", repo_root), patch.object(
                main, "_DISEASE_RUNTIME", None
            ), patch.dict(sys.modules, {"torch": torch_module, "disease_model": model_module}):
                runtime = main._load_disease_model()

        self.assertEqual(
            attempted,
            ["plant_disease_weights.pth", "plant-disease-model-complete.pth"],
        )
        self.assertIs(runtime[0], model)
        self.assertIs(runtime[1], torch_module)

    def _fake_model_module(self):
        module = types.ModuleType("disease_model")
        module.CLASS_LABELS = list(DISEASE_TRANSLATION)
        module.predict_torch = lambda _model, _tensor: []
        return module

    def test_qwen_result_drives_short_verified_dae_guidance_when_local_model_missing(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "tomato",
            "crop_bn": "টমেটো",
            "disease_en": "early blight",
            "disease_bn": "টমেটো আর্লি ব্লাইট",
            "cause_type": "fungal",
            "pathogen_en": None,
            "visible_signs_bn": ["পুরোনো পাতায় বলয়যুক্ত বাদামি দাগ"],
            "nonchemical_actions_bn": ["আক্রান্ত পাতা সরান"],
            "confidence_band": "medium",
            "plantvillage_class_label": "Tomato___Early_blight",
        }
        vision_response = {
            "available": True,
            "status": "ok",
            "model": "qwen/qwen3.8-27b",
            "result": visual,
            "image_sent_to_groq": True,
        }
        fake_module = self._fake_model_module()
        with patch.dict(sys.modules, {"disease_model": fake_module}), patch.object(
            main, "_load_disease_model", side_effect=FileNotFoundError()
        ) as local_load, patch.object(
            main, "analyze_leaf_image", return_value=vision_response
        ) as analyze_mock:
            result = main.disease(DiseaseRequest(image_data_url=_image_data_url(), symptom="পাতায় দাগ"))

        local_load.assert_not_called()
        self.assertEqual(analyze_mock.call_args.args[1], [])
        self.assertTrue(analyze_mock.call_args.args[0].startswith("data:image/jpeg;base64,"))
        self.assertEqual(analyze_mock.call_args.args[3], "পাতায় দাগ")
        self.assertEqual(result["diagnosis"]["source"], "krittim_vision")
        self.assertEqual(result["diagnosis"]["crop_label"], "টমেটো")
        self.assertEqual(result["diagnosis"]["disease_label"], "আর্লি ব্লাইট")
        self.assertTrue(result["image_sent_to_vision_service"])
        self.assertFalse(result["local_model_available"])
        self.assertEqual(result["guidance"]["products"][0]["registration_no"], "AP-664")
        self.assertEqual(result["guidance"]["products"][0]["rate"], "২ গ্রাম / ১ লিটার পানি")
        self.assertEqual(result["top_predictions"], [])

    def test_qwen_rice_result_returns_automatic_verified_blast_product(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "rice",
            "crop_bn": "ধান",
            "disease_en": "rice blast",
            "disease_bn": "ধানের ব্লাস্ট",
            "cause_type": "fungal",
            "pathogen_en": None,
            "visible_signs_bn": ["পাতায় লম্বাটে দাগ"],
            "nonchemical_actions_bn": [],
            "confidence_band": "medium",
            "plantvillage_class_label": None,
        }
        vision_response = {
            "available": True,
            "status": "ok",
            "model": "qwen/qwen3.8-27b",
            "result": visual,
            "image_sent_to_groq": True,
        }
        fake_module = self._fake_model_module()
        with patch.dict(sys.modules, {"disease_model": fake_module}), patch.object(
            main, "_load_disease_model", side_effect=FileNotFoundError()
        ), patch.object(main, "analyze_leaf_image", return_value=vision_response):
            result = main.disease(DiseaseRequest(image_data_url=_image_data_url()))

        self.assertEqual(result["diagnosis"]["crop_label"], "ধান")
        self.assertEqual(result["diagnosis"]["disease_label"], "ব্লাস্ট")
        self.assertEqual(result["guidance"]["products"][0]["registration_no"], "AP-5112")
        self.assertEqual(result["guidance"]["products"][0]["rate"], "২ মিলি/লিটার পানি; ১০০০ মিলি/হেক্টর")

    def test_low_confidence_rice_blast_still_shows_conditional_official_use_candidate(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "rice",
            "crop_bn": "ধান",
            "disease_en": "rice blast",
            "disease_bn": "ধানের ব্লাস্ট",
            "cause_type": "fungal",
            "pathogen_en": None,
            "visible_signs_bn": [],
            "nonchemical_actions_bn": [],
            "confidence_band": "low",
            "plantvillage_class_label": None,
        }
        vision_response = {
            "available": True,
            "status": "ok",
            "model": "qwen/qwen3.8-27b",
            "result": visual,
            "image_sent_to_groq": True,
        }
        fake_module = self._fake_model_module()
        with patch.dict(sys.modules, {"disease_model": fake_module}), patch.object(
            main, "_load_disease_model", side_effect=FileNotFoundError()
        ), patch.object(main, "analyze_leaf_image", return_value=vision_response), patch.object(
            main, "lookup_ppw_live_product"
        ) as live_lookup:
            result = main.disease(DiseaseRequest(image_data_url=_image_data_url()))

        live_lookup.assert_not_called()
        self.assertEqual(result["diagnosis"]["confidence_band"], "low")
        self.assertEqual(result["guidance"]["category"], "possible_disease_with_verified_label_candidate")
        self.assertEqual(result["guidance"]["products"][0]["registration_no"], "AP-5112")
        self.assertEqual(result["guidance"]["products"][0]["rate"], "২ মিলি/লিটার পানি; ১০০০ মিলি/হেক্টর")
        self.assertIn("রোগনির্ণয় নয়", result["guidance"]["product_note"])

    def test_missing_curated_match_uses_live_ppw_result_without_sending_photo_to_ppw(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "eggplant",
            "crop_bn": "বেগুন",
            "disease_en": "fruit rot",
            "disease_bn": "ফল পচা",
            "cause_type": "fungal",
            "pathogen_en": None,
            "visible_signs_bn": ["ফলে পচা দাগ"],
            "nonchemical_actions_bn": [],
            "confidence_band": "high",
            "plantvillage_class_label": None,
        }
        live_product = {
            "name": "Example PPW Product",
            "registration_no": "AP-1234",
            "active_ingredients": "Example active",
            "label_scope": "বেগুন — ফল পচা",
            "rate": "২ গ্রাম/লিটার পানি",
            "application_method": "বর্তমান লেবেল দেখুন।",
            "directions": ["Exact DAE use verified."],
            "source_label": "DAE PPW live product record — Example PPW Product",
            "source_url": "https://ppw.krishi.gov.bd/pesticide-details/1234",
        }
        vision_response = {
            "available": True,
            "status": "ok",
            "model": "qwen/qwen3.8-27b",
            "result": visual,
            "image_sent_to_groq": True,
        }
        fake_module = self._fake_model_module()
        with patch.dict(sys.modules, {"disease_model": fake_module}), patch.object(
            main, "_load_disease_model", side_effect=AssertionError("Qwen result should not load local model")
        ), patch.object(main, "analyze_leaf_image", return_value=vision_response), patch.object(
            main, "lookup_ppw_live_product", return_value={"status": "matched", "product": live_product, "checked_on": "2026-10-28"}
        ) as ppw_search:
            result = main.disease(DiseaseRequest(image_data_url=_image_data_url()))

        ppw_search.assert_called_once_with(visual)
        self.assertEqual(result["guidance"]["products"], [live_product])
        self.assertEqual(result["guidance"]["source_checked_on"], "2026-10-28")
        self.assertIn("ছবিটি PPW-তে পাঠানো হয়নি", result["guidance"]["product_note"])
        self.assertIn(live_product["source_url"], [source["url"] for source in result["guidance"]["sources"]])

    def test_default_text_model_candidate_is_verified_before_becoming_a_product(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "rice",
            "crop_bn": "ধান",
            "disease_en": "rice blast",
            "disease_bn": "ধানের ব্লাস্ট",
            "cause_type": "fungal",
            "confidence_band": "medium",
            "plantvillage_class_label": None,
        }
        product = {
            "name": "Noblast 38SC",
            "registration_no": "AP-5797",
            "active_ingredients": "Active ingredient",
            "label_scope": "ধান — ধানের ব্লাস্ট",
            "rate": "প্রতি লিটার পানিতে 1 মিলি",
            "application_method": "DAE detail page verified; current label applies.",
            "directions": ["Exact official use matched."],
            "source_label": "DAE PPW live product record — Noblast 38SC",
            "source_url": "https://ppw.krishi.gov.bd/pesticide-details/13821",
        }
        with patch.object(
            main,
            "lookup_ppw_live_product",
            side_effect=[{"status": "no_match", "product": None}, {"status": "matched", "product": product, "checked_on": "2026-10-28", "ai_candidate_search": True}],
        ) as ppw_search, patch.object(
            main, "pesticide_registry_candidate_terms", return_value=["Noblast 38SC"]
        ) as candidate_model:
            guidance = main._augment_with_live_ppw_guidance(main.get_unclassified_visual_guidance(), visual)

        candidate_model.assert_called_once_with("rice", "ধান", "rice blast", "ধানের ব্লাস্ট")
        self.assertEqual(ppw_search.call_args_list[0].args, (visual,))
        self.assertEqual(ppw_search.call_args_list[1].kwargs, {"extra_search_terms": ["Noblast 38SC"]})
        self.assertEqual(guidance["products"][0]["registration_no"], "AP-5797")
        self.assertEqual(guidance["products"][0]["rate"], "প্রতি লিটার পানিতে 1 মিলি")
        self.assertIn("DAE detail page-এ যাচাই", guidance["product_note"])

    def test_unverified_live_search_results_never_create_a_product_recommendation(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "jute",
            "crop_bn": "পাট",
            "disease_en": "leaf spot",
            "disease_bn": "পাতার দাগ",
            "cause_type": "fungal",
            "confidence_band": "medium",
            "plantvillage_class_label": None,
        }
        for status, expected_copy in (
            ("no_match", "যাচাই করা যায়নি"),
            ("unavailable", "সংযোগ পাওয়া যায়নি"),
        ):
            with self.subTest(status=status), patch.object(
                main, "lookup_ppw_live_product", return_value={"status": status, "product": None}
            ), patch.object(main, "pesticide_registry_candidate_terms", return_value=[]):
                guidance = main._augment_with_live_ppw_guidance(main.get_unclassified_visual_guidance(), visual)
            self.assertEqual(guidance["products"], [])
            self.assertIn(expected_copy, guidance["product_note"])

    def test_local_classifier_remains_the_fallback_when_qwen_is_not_configured(self) -> None:
        labels = list(DISEASE_TRANSLATION)
        probabilities = numpy.zeros(len(labels), dtype=numpy.float64)
        probabilities[labels.index("Potato___Late_blight")] = 1.0

        class FakeTensor:
            def permute(self, *_args):
                return self

            def unsqueeze(self, *_args):
                return self

        class FakeTorch:
            @staticmethod
            def from_numpy(_pixels):
                return FakeTensor()

        module = types.ModuleType("disease_model")
        module.CLASS_LABELS = labels
        module.predict_torch = lambda _model, _tensor: probabilities
        vision_response = {
            "available": False,
            "status": "not_configured",
            "model": "qwen/qwen3.8-27b",
            "result": None,
            "image_sent_to_groq": False,
        }
        with patch.dict(sys.modules, {"disease_model": module}), patch.object(
            main, "_load_disease_model", return_value=(object(), FakeTorch)
        ), patch.object(main, "analyze_leaf_image", return_value=vision_response):
            result = main.disease(DiseaseRequest(image_data_url=_image_data_url()))

        self.assertEqual(result["diagnosis"]["source"], "plantvillage_classifier")
        self.assertFalse(result["image_sent_to_vision_service"])
        self.assertTrue(result["local_model_available"])
        self.assertEqual(result["prediction"]["class_label"], "Potato___Late_blight")
        self.assertEqual(result["prediction"]["label"], "লেট ব্লাইট রোগ")
        self.assertEqual(result["top_predictions"][0]["label"], "লেট ব্লাইট রোগ")
        self.assertEqual(result["guidance"]["products"], [])
        self.assertIn("দেখানো হচ্ছে না", result["guidance"]["product_note"])

    def test_capsicum_vision_result_does_not_receive_chili_label_product(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "capsicum",
            "crop_bn": "ক্যাপসিকাম",
            "disease_en": "bacterial spot",
            "disease_bn": "ব্যাকটেরিয়াল দাগ",
            "cause_type": "bacterial",
            "pathogen_en": None,
            "visible_signs_bn": [],
            "nonchemical_actions_bn": [],
            "confidence_band": "medium",
            "plantvillage_class_label": None,
        }
        vision_response = {"available": True, "status": "ok", "model": "qwen/qwen3.8-27b", "result": visual, "image_sent_to_groq": True}
        fake_module = self._fake_model_module()
        with patch.dict(sys.modules, {"disease_model": fake_module}), patch.object(
            main, "_load_disease_model", side_effect=FileNotFoundError()
        ), patch.object(main, "analyze_leaf_image", return_value=vision_response), patch.object(
            main, "lookup_ppw_live_product", return_value={"status": "no_match", "product": None}
        ) as ppw_search, patch.object(main, "pesticide_registry_candidate_terms", return_value=[]):
            result = main.disease(DiseaseRequest(image_data_url=_image_data_url()))

        ppw_search.assert_called_once_with(visual)
        self.assertEqual(result["diagnosis"]["crop_label"], "ক্যাপসিকাম")
        self.assertEqual(result["guidance"]["products"], [])
        self.assertIn("যাচাই করা যায়নি", result["guidance"]["product_note"])

    def test_missing_vision_disease_text_uses_a_valid_local_class_label(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "tomato",
            "crop_bn": "টমেটো",
            "disease_en": None,
            "disease_bn": None,
            "cause_type": "fungal",
            "visible_signs_bn": ["পাতায় বলয়যুক্ত বাদামি দাগ"],
            "nonchemical_actions_bn": [],
            "confidence_band": "medium",
            "plantvillage_class_label": "Tomato___Early_blight",
        }
        vision_response = {"available": True, "status": "ok", "result": visual, "image_sent_to_groq": True}
        fake_module = self._fake_model_module()
        with patch.dict(sys.modules, {"disease_model": fake_module}), patch.object(
            main, "_load_disease_model", side_effect=AssertionError("vision result should not load local model")
        ), patch.object(main, "analyze_leaf_image", return_value=vision_response):
            result = main.disease(DiseaseRequest(image_data_url=_image_data_url()))

        self.assertEqual(result["diagnosis"]["disease_label"], "আর্লি ব্লাইট রোগ")
        self.assertEqual(result["prediction"]["label"], "আর্লি ব্লাইট রোগ")
        self.assertEqual(result["guidance"]["products"], [])

    def test_unclear_leaf_has_plain_uncertainty_instead_of_repeated_unknown(self) -> None:
        visual = {
            "leaf_status": "leaf",
            "crop_en": "rice",
            "crop_bn": "ধান",
            "disease_en": None,
            "disease_bn": None,
            "cause_type": "unknown",
            "visible_signs_bn": [],
            "nonchemical_actions_bn": [],
            "confidence_band": "low",
            "plantvillage_class_label": None,
        }
        vision_response = {"available": True, "status": "ok", "result": visual, "image_sent_to_groq": True}
        fake_module = self._fake_model_module()
        with patch.dict(sys.modules, {"disease_model": fake_module}), patch.object(
            main, "_load_disease_model", side_effect=AssertionError("vision result should not load local model")
        ), patch.object(main, "analyze_leaf_image", return_value=vision_response):
            result = main.disease(DiseaseRequest(image_data_url=_image_data_url()))

        self.assertEqual(result["diagnosis"]["disease_label"], "লক্ষণ স্পষ্ট নয়—রোগ শনাক্ত করা যায়নি")
        self.assertEqual(result["guidance"]["products"], [])


if __name__ == "__main__":
    unittest.main()
