"""Unit tests for Qwen response validation and fallback behavior."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import Mock, patch

from backend import vision_service


class VisionServiceTests(unittest.TestCase):
    def test_prompt_covers_bangladesh_crops_outside_local_classes(self) -> None:
        self.assertIn("rice/paddy", vision_service._SYSTEM_PROMPT)
        self.assertIn("outside the local model's 38 classes", vision_service._SYSTEM_PROMPT)
        self.assertIn("without a crop name", vision_service._SYSTEM_PROMPT)

    def test_missing_api_key_does_not_send_image(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False), patch.object(
            vision_service.requests, "post"
        ) as post:
            result = vision_service.analyze_leaf_image("data:image/jpeg;base64,abc", [], ["Tomato___Early_blight"])
        post.assert_not_called()
        self.assertFalse(result["available"])
        self.assertEqual(result["status"], "not_configured")
        self.assertFalse(result["image_sent_to_groq"])

    def test_vision_result_is_normalised_and_unsafe_action_is_removed(self) -> None:
        model_payload = {
            "leaf_status": "leaf",
            "crop_en": "tomato",
            "crop_bn": "টমেটো",
            "disease_en": "early blight",
            "disease_bn": "আর্লি ব্লাইট",
            "cause_type": "fungal",
            "pathogen_en": "Alternaria",
            "visible_signs_bn": ["পাতায় বলয়যুক্ত দাগ"],
            "nonchemical_actions_bn": ["আক্রান্ত পাতা সরান", "স্প্রে করুন"],
            "confidence_band": "high",
            "plantvillage_class_label": "Tomato___Early_blight",
        }
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps(model_payload, ensure_ascii=False)}}]}
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False), patch.object(
            vision_service.requests, "post", return_value=response
        ) as post:
            result = vision_service.analyze_leaf_image(
                "data:image/jpeg;base64,abc", [], ["Tomato___Early_blight"]
            )
        self.assertTrue(result["available"])
        self.assertTrue(result["image_sent_to_groq"])
        self.assertEqual(result["result"]["crop_bn"], "টমেটো")
        self.assertEqual(result["result"]["plantvillage_class_label"], "Tomato___Early_blight")
        self.assertEqual(result["result"]["nonchemical_actions_bn"], ["আক্রান্ত পাতা সরান"])
        self.assertEqual(post.call_args.kwargs["json"]["model"], vision_service.DEFAULT_VISION_MODEL)
        content = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(post.call_args.kwargs["json"]["response_format"], {"type": "json_object"})

    def test_pesticide_instructions_cannot_leak_into_visible_diagnosis_labels(self) -> None:
        payload = {
            "leaf_status": "leaf",
            "crop_en": "tomato",
            "crop_bn": "টমেটো",
            "disease_en": "Apply Kocide at 2 ml per litre",
            "disease_bn": "কপার স্প্রে ব্যবহার করুন",
            "cause_type": "bacterial",
            "visible_signs_bn": ["পাতায় গোল দাগ"],
            "nonchemical_actions_bn": [],
            "confidence_band": "high",
            "plantvillage_class_label": None,
        }
        result = vision_service._normalise_payload(payload, set())
        self.assertIsNotNone(result)
        self.assertIsNone(result["disease_en"])
        self.assertIsNone(result["disease_bn"])

    def test_unsupported_model_class_is_discarded(self) -> None:
        payload = {
            "leaf_status": "leaf",
            "crop_en": "tomato",
            "disease_en": "early blight",
            "cause_type": "fungal",
            "visible_signs_bn": [],
            "nonchemical_actions_bn": [],
            "confidence_band": "medium",
            "plantvillage_class_label": "../../unsafe",
        }
        result = vision_service._normalise_payload(payload, {"Tomato___Early_blight"})
        self.assertIsNotNone(result)
        self.assertIsNone(result["plantvillage_class_label"])

    def test_provider_error_is_safe_and_marks_transmission_attempt(self) -> None:
        response = Mock(status_code=429)
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False), patch.object(
            vision_service.requests, "post", return_value=response
        ):
            result = vision_service.analyze_leaf_image("data:image/jpeg;base64,abc", [], [])
        self.assertFalse(result["available"])
        self.assertEqual(result["status"], "rate_limit")
        self.assertTrue(result["image_sent_to_groq"])
        self.assertNotIn("test-key", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
