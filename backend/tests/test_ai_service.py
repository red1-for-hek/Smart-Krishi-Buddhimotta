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

    def test_question_completion_sends_dae_context_as_system_data_and_uses_default_fallback(self) -> None:
        unavailable = Mock(status_code=404)
        unavailable.json.return_value = {"error": {"code": "model_not_found"}}
        success = Mock(status_code=200)
        success.json.return_value = {
            "choices": [{"message": {"content": "রেকর্ডের তথ্য মিলিয়ে নিন।"}}]
        }
        context = "APP-VERIFIED-DAE-RECORDS: Seltima, AP-5112, rice blast, ২ মিলি/লিটার।"
        with patch.dict(os.environ, {"GROQ_API_KEY": "server-secret"}, clear=True), patch.object(
            ai_service.requests, "post", side_effect=[unavailable, success]
        ) as post:
            answer, model, error = ai_service.complete(
                [{"role": "user", "content": "Ei roger jonne kon medicine use korbo?"}],
                trusted_system_context=context,
                timeout=(3, 8),
                max_models=2,
            )

        self.assertEqual(answer, "রেকর্ডের তথ্য মিলিয়ে নিন।")
        self.assertEqual(model, "llama-3.1-8b-instant")
        self.assertIsNone(error)
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            self.assertEqual(call.kwargs["timeout"], (3, 8))
            payload = call.kwargs["json"]
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][1], {"role": "system", "content": context})
            self.assertEqual(payload["messages"][2]["role"], "user")
            self.assertIn("Ei roger jonne kon medicine use korbo?", payload["messages"][2]["content"])

    def test_transliterated_medicine_question_answers_from_exact_conditional_dae_data(self) -> None:
        product = {
            "name": "Seltima",
            "registration_no": "AP-5112",
            "label_scope": "ধান — ধানের ব্লাস্ট",
            "rate": "২ মিলি/লিটার পানি; ১০০০ মিলি/হেক্টর",
            "application_method": "DAE label details; current package label applies.",
            "source_url": "https://ppw.krishi.gov.bd/pesticide-details/3563",
        }
        visual = {"crop_bn": "ধান", "disease_bn": "ব্লাস্ট", "visible_signs_bn": ["পাতায় লম্বাটে দাগ"]}
        with patch.object(ai_service, "complete") as complete:
            answer, model, error = ai_service.disease_summary(
                "ধান", "ব্লাস্ট", 0, [], "Ei roger jonne kon medicine use korbo?",
                visual_assessment=visual,
                verified_products=[product],
                conditional_candidate=True,
            )

        complete.assert_not_called()
        self.assertEqual(model, "dae_verified_template")
        self.assertIsNone(error)
        for verified_fact in (product["name"], product["registration_no"], product["label_scope"], product["rate"], product["source_url"]):
            self.assertIn(verified_fact, answer)
        self.assertIn("শর্তসাপেক্ষ", answer)
        self.assertIn("নিশ্চিত রোগনির্ণয় বা প্রেসক্রিপশন নয়", answer)

    def test_transliterated_medicine_question_without_dae_match_names_no_product(self) -> None:
        with patch.object(ai_service, "complete") as complete:
            answer, model, error = ai_service.disease_summary(
                "পাট", "লক্ষণ অনিশ্চিত", 0, [], "Ei roger jonne kon medicine use korbo?",
                visual_assessment={"crop_bn": "পাট", "disease_bn": None, "visible_signs_bn": []},
                verified_products=[],
            )

        complete.assert_not_called()
        self.assertEqual(model, "dae_verified_template")
        self.assertIsNone(error)
        self.assertIn("কোনো নির্দিষ্ট DAE-যাচাইকৃত পণ্য ও মাত্রা মেলেনি", answer)
        self.assertIn("অনুমান করে ওষুধের নাম বা ডোজ বলা হচ্ছে না", answer)
        self.assertNotIn("Noblast", answer)
        self.assertNotIn("Seltima", answer)
        self.assertNotIn("২ মিলি", answer)

    def test_fertilizer_question_does_not_reuse_a_pesticide_record(self) -> None:
        pesticide = {
            "name": "Seltima", "registration_no": "AP-5112",
            "label_scope": "ধান — ধানের ব্লাস্ট", "rate": "২ মিলি/লিটার পানি",
            "source_url": "https://ppw.krishi.gov.bd/pesticide-details/3563",
        }
        with patch.object(ai_service, "complete") as complete:
            answer, model, error = ai_service.disease_summary(
                "ধান", "ব্লাস্ট", 0, [], "Ei rog e ki sar debo?",
                visual_assessment={"crop_bn": "ধান", "disease_bn": "ব্লাস্ট", "cause_type": "fungal"},
                verified_products=[pesticide],
            )

        complete.assert_not_called()
        self.assertEqual(model, "safety_template")
        self.assertIsNone(error)
        self.assertIn("মাটির পুষ্টির ঘাটতি বা সারের প্রয়োজন নিশ্চিত করা যায় না", answer)
        self.assertNotIn("Seltima", answer)
        self.assertNotIn("AP-5112", answer)
        self.assertNotIn("২ মিলি", answer)
        self.assertIn("মাটি পরীক্ষা", answer)

    def test_non_treatment_disease_question_hides_product_details_from_freeform_model(self) -> None:
        product = {
            "name": "Seltima", "registration_no": "AP-5112",
            "label_scope": "ধান — ধানের ব্লাস্ট", "rate": "২ মিলি/লিটার পানি",
            "source_url": "https://ppw.krishi.gov.bd/pesticide-details/3563",
        }
        with patch.object(ai_service, "complete", return_value=("লক্ষণ মিলিয়ে দেখুন।", "model", None)) as complete:
            ai_service.disease_summary(
                "ধান", "ব্লাস্ট", 0, [], "পাতায় দাগ কেন হচ্ছে?",
                visual_assessment={"crop_bn": "ধান", "disease_bn": "ব্লাস্ট", "visible_signs_bn": []},
                verified_products=[product],
            )

        context = complete.call_args.kwargs["trusted_system_context"]
        self.assertIn("exact record is displayed separately", context)
        self.assertIn("Do not name any product", context)
        self.assertNotIn("Seltima", context)
        self.assertNotIn("AP-5112", context)
        self.assertNotIn("২ মিলি", context)
        self.assertIn("free-form উত্তরে লিখবে না", complete.call_args.args[0][0]["content"])


if __name__ == "__main__":
    unittest.main()
