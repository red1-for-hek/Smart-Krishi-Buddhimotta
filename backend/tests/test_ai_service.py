"""Unit tests for the fast text model routing used by chat and explanations."""
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from backend import ai_service


class AIServiceTests(unittest.TestCase):
    def test_defaults_prefer_gpt_oss_20b_then_fast_llama(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                ai_service.candidate_models(),
                ["openai/gpt-oss-20b", "llama-3.1-8b-instant"],
            )

    def test_configured_models_keep_order_and_append_safe_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GROQ_MODEL_PRIMARY": "custom/text-model",
                "GROQ_MODEL_FALLBACKS": "llama-3.1-8b-instant,custom/backup",
            },
            clear=True,
        ):
            self.assertEqual(
                ai_service.candidate_models(),
                [
                    "custom/text-model",
                    "llama-3.1-8b-instant",
                    "custom/backup",
                ],
            )

    def test_registry_product_names_are_only_unverified_search_terms(self) -> None:
        success = Mock(status_code=200)
        success.json.return_value = {
            "choices": [{"message": {"content": '{"candidate_terms":["Noblast 38SC", "Seltima", "Noblast 38SC"]}'}}]
        }
        with patch.dict(os.environ, {"GROQ_API_KEY": "server-secret"}, clear=True), patch.object(
            ai_service.requests, "post", return_value=success
        ) as post:
            terms = ai_service.pesticide_registry_candidate_terms("rice", "ধান", "rice blast", "ধানের ব্লাস্ট")

        self.assertEqual(terms, ["Noblast 38SC", "Seltima"])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "openai/gpt-oss-20b")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("image_url", repr(payload))
        self.assertNotIn("server-secret", repr(payload))

    def test_registry_candidates_are_disabled_without_server_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(ai_service.requests, "post") as post:
            terms = ai_service.pesticide_registry_candidate_terms("rice", "ধান", "blast", "ব্লাস্ট")
        self.assertEqual(terms, [])
        post.assert_not_called()

    def test_completion_falls_back_without_a_model_list_request(self) -> None:
        unavailable = Mock(status_code=404)
        unavailable.json.return_value = {"error": {"code": "model_not_found"}}
        success = Mock(status_code=200)
        success.json.return_value = {
            "choices": [{"message": {"content": "পাতার লক্ষণ পর্যবেক্ষণ করুন।"}}]
        }
        messages = [{"role": "user", "content": "পাতায় দাগ দেখা যাচ্ছে"}]
        with patch.dict(os.environ, {"GROQ_API_KEY": "server-secret"}, clear=True), patch.object(
            ai_service.requests, "post", side_effect=[unavailable, success]
        ) as post, patch.object(ai_service.requests, "get") as get:
            answer, model, error = ai_service.complete(messages)

        self.assertEqual(answer, "পাতার লক্ষণ পর্যবেক্ষণ করুন।")
        self.assertEqual(model, "llama-3.1-8b-instant")
        self.assertIsNone(error)
        self.assertEqual(post.call_count, 2)
        get.assert_not_called()
        first_payload = post.call_args_list[0].kwargs["json"]
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(first_payload["model"], "openai/gpt-oss-20b")
        self.assertEqual(first_payload["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", second_payload)
        self.assertNotIn("server-secret", repr((answer, model, error)))


if __name__ == "__main__":
    unittest.main()
