"""Safety tests for automatic DAE PPW live-product lookup."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from backend import dae_live_lookup


class _FakeResponse:
    def __init__(self, *, url: str, payload=None, text: str = "", status_code: int = 200):
        self.url = url
        self._payload = payload
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise dae_live_lookup.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


def _visual(**overrides):
    value = {
        "leaf_status": "leaf",
        "crop_en": "rice",
        "crop_bn": "ধান",
        "disease_en": "rice blast",
        "disease_bn": "ধানের ব্লাস্ট",
        "cause_type": "fungal",
        "confidence_band": "high",
        "plantvillage_class_label": None,
    }
    value.update(overrides)
    return value


def _noblast_detail_html(registration: str = "AP-5797") -> str:
    return f"""
    <html><body>
      <h4>Noblast 38SC</h4>
      <h6>এপি/পিএইচপি নম্বর</h6><p>{registration}</p>
      <h6>ব্যবহারবিধি</h6>
      <p>Noblast 38 SC চায়ের রেড রাস্ট দমনে হেক্টর প্রতি 750 মিলি, আলুর লেইট ব্লাইট দমনে হেক্টর প্রতি 750 মিলি এবং ধানের ব্লাস্ট রোগে প্রতি লিটার পানিতে 1 মিলি স্প্রে করুন।</p>
      <h6>নির্দেশনা</h6><p>প্রয়োজনীয় নিরাপত্তা সরঞ্জাম ব্যবহার করুন।</p>
    </body></html>
    """


def _search_payload():
    return {
        "data": [
            {
                "id": 13821,
                "app_php_no": "AP-5797",
                "brand_name": "Noblast 38SC",
                "chemical": {"chemical_name": "Active ingredient not parsed"},
            }
        ]
    }


class DaeLiveLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        dae_live_lookup._CACHE.clear()

    def test_live_lookup_requires_exact_crop_disease_and_one_local_rate(self) -> None:
        search_url = dae_live_lookup.PPW_SEARCH_URL
        detail_url = f"{dae_live_lookup.PPW_ORIGIN}/pesticide-details/13821"

        def fake_get(url, **kwargs):
            if url == search_url:
                keyword = kwargs["params"]["ajax_keyword"]
                if keyword == "blast":
                    return _FakeResponse(url=f"{search_url}?ajax_keyword=blast", payload=_search_payload())
                return _FakeResponse(url=f"{search_url}?ajax_keyword={keyword}", payload={"data": []})
            if url == detail_url:
                return _FakeResponse(url=detail_url, text=_noblast_detail_html())
            raise AssertionError(f"Unexpected PPW URL: {url}")

        with patch.object(dae_live_lookup.requests, "get", side_effect=fake_get) as request:
            result = dae_live_lookup.lookup_ppw_live_product(_visual())

        self.assertEqual(result["status"], "matched")
        product = result["product"]
        self.assertEqual(product["name"], "Noblast 38SC")
        self.assertEqual(product["registration_no"], "AP-5797")
        self.assertEqual(product["rate"], "প্রতি লিটার পানিতে 1 মিলি")
        self.assertIn("ধান", product["label_scope"])
        self.assertEqual(product["source_url"], detail_url)
        self.assertIn("বর্তমান লেবেল", product["application_method"])
        self.assertGreaterEqual(request.call_count, 2)

    def test_generic_separate_crop_and_disease_lists_are_not_treated_as_a_pair(self) -> None:
        use_text = "ফসল: টমেটো, আলু, মরিচ। রোগ: Early blight, Late blight, Anthracnose। মাত্রা 2 g/L।"
        crop_aliases = dae_live_lookup._crop_aliases("tomato", "টমেটো")
        disease_aliases, _ = dae_live_lookup._disease_aliases("early blight", "আর্লি ব্লাইট")
        self.assertIsNone(dae_live_lookup._find_exact_rate(use_text, crop_aliases, disease_aliases))

    def test_multiple_neighboring_rates_are_rejected_as_ambiguous(self) -> None:
        use_text = "ফসল:ধান রোগের নাম:লিফ ব্লাস্ট অনুমোদিত মাত্রা:৪০০ গ্রাম/হেক্টর একর প্রতি:১৮০ গ্রাম ৫ শতকে ৬ গ্রাম।"
        crop_aliases = dae_live_lookup._crop_aliases("rice", "ধান")
        disease_aliases, _ = dae_live_lookup._disease_aliases("rice blast", "ধানের ব্লাস্ট")
        self.assertIsNone(dae_live_lookup._find_exact_rate(use_text, crop_aliases, disease_aliases))

    def test_plantvillage_crop_disease_conflict_prevents_any_network_search(self) -> None:
        visual = _visual(plantvillage_class_label="Tomato___Late_blight")
        with patch.object(dae_live_lookup.requests, "get") as request:
            result = dae_live_lookup.lookup_ppw_live_product(visual)
        request.assert_not_called()
        self.assertEqual(result["status"], "skipped")

    def test_low_confidence_viral_and_uncertain_results_do_not_trigger_search(self) -> None:
        cases = [
            _visual(confidence_band="low"),
            _visual(cause_type="viral"),
            _visual(cause_type="nutrient"),
            _visual(leaf_status="uncertain"),
        ]
        with patch.object(dae_live_lookup.requests, "get") as request:
            for visual in cases:
                self.assertEqual(dae_live_lookup.lookup_ppw_live_product(visual)["status"], "skipped")
        request.assert_not_called()

    def test_exact_bacterial_use_can_match_the_same_official_verification_gate(self) -> None:
        visual = _visual(
            disease_en="bacterial blight",
            disease_bn="ব্যাকটেরিয়াল ব্লাইট",
            cause_type="bacterial",
        )
        detail_url = f"{dae_live_lookup.PPW_ORIGIN}/pesticide-details/13821"
        html = """
        <html><body><h4>Example AP product</h4><h6>AP/PHP number</h6><p>AP-5797</p>
        <h6>ব্যবহারবিধি</h6><p>ধানের bacterial blight রোগে প্রতি লিটার পানিতে 1 মিলি ব্যবহার করুন।</p>
        <h6>নির্দেশনা</h6></body></html>
        """

        def fake_get(url, **kwargs):
            if url == dae_live_lookup.PPW_SEARCH_URL:
                keyword = kwargs["params"]["ajax_keyword"]
                payload = _search_payload() if keyword == "bacterial blight" else {"data": []}
                return _FakeResponse(url=f"{url}?ajax_keyword={keyword}", payload=payload)
            if url == detail_url:
                return _FakeResponse(url=detail_url, text=html)
            raise AssertionError(f"Unexpected PPW URL: {url}")

        with patch.object(dae_live_lookup.requests, "get", side_effect=fake_get):
            result = dae_live_lookup.lookup_ppw_live_product(visual)

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["product"]["rate"], "প্রতি লিটার পানিতে 1 মিলি")

    def test_detail_registration_must_match_search_result(self) -> None:
        search_url = dae_live_lookup.PPW_SEARCH_URL
        detail_url = f"{dae_live_lookup.PPW_ORIGIN}/pesticide-details/13821"

        def fake_get(url, **kwargs):
            if url == search_url:
                keyword = kwargs["params"]["ajax_keyword"]
                return _FakeResponse(url=f"{search_url}?ajax_keyword={keyword}", payload=_search_payload() if keyword == "blast" else {"data": []})
            if url == detail_url:
                return _FakeResponse(url=detail_url, text=_noblast_detail_html("AP-1234"))
            raise AssertionError(f"Unexpected PPW URL: {url}")

        with patch.object(dae_live_lookup.requests, "get", side_effect=fake_get):
            result = dae_live_lookup.lookup_ppw_live_product(_visual())
        self.assertEqual(result["status"], "no_match")
        self.assertIsNone(result["product"])

    def test_upstream_failure_is_safe_and_does_not_invent_a_product(self) -> None:
        with patch.object(dae_live_lookup.requests, "get", side_effect=dae_live_lookup.requests.Timeout):
            result = dae_live_lookup.lookup_ppw_live_product(_visual(crop_en="rice-timeout"))
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["product"])
