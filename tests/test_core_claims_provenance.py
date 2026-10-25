import unittest

import _util  # noqa: F401
from helm.memory import SpanTable, citation_metrics, cited_recall_loop, extract_claims, span_hash, validate_cited_memory
from helm.memory.claims import ClaimRegistry
from helm.types import Claim, Status


class TestClaims(unittest.TestCase):
    def test_json_entity_attr_value(self):
        t = '{"entity": "Widget-7", "attr": "stock", "value": 40, "code": 200}'
        self.assertEqual(extract_claims(t, "lookup"), [("widget-7.stock", "40")])

    def test_json_generic_fields_skip_meta(self):
        got = dict(extract_claims('{"id": "u1", "plan": "pro", "code": 200, "ts": 5}'))
        self.assertEqual(got, {"u1.plan": "pro"})

    def test_regex_fallback_and_noise_keys(self):
        got = dict(extract_claims("deploy ok: build=1.5 code=200 region=eu-west"))
        self.assertEqual(got, {"build": "1.5", "region": "eu-west"})

    def test_numeric_normalisation(self):
        self.assertEqual(extract_claims("total = 40.0")[0][1], "40")

    def test_registry_contradiction_needs_newer_success(self):
        r = ClaimRegistry()
        old = Claim("k", "1", 1, Status.SUCCESS, "a")
        r.add(old)
        self.assertIsNone(r.contradicting(old))
        r.add(Claim("k", "2", 5, Status.FAILED, "b"))  # failed claims never contradict
        self.assertIsNone(r.contradicting(old))
        r.add(Claim("k", "2", 6, Status.SUCCESS, "c"))
        self.assertEqual(r.contradicting(old).item_id, "c")
        self.assertIsNone(r.contradicting(old, upto_step=5))  # not yet visible at step 5


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.t = SpanTable()
        self.t.add_log("lookup", "L1", 1, {"text": "stock of widget-7 is 40 units"})
        self.p = self.t.pointer("lookup", "L1", 1, "text", 0, 12)

    def test_resolve_and_hash(self):
        self.assertEqual(self.t.resolve(self.p), "stock of wid")
        bad = type(self.p)(self.p.tool, self.p.log_id, self.p.step_id, self.p.field, 0, 12, "deadbeef0000")
        self.assertIsNone(self.t.resolve(bad))
        self.assertEqual(self.p.hash, span_hash("stock of wid"))

    def cite(self, s=0, e=12, step=1):
        return f"[[CITE tool=lookup log=L1 step={step} field=text start={s} end={e}]]"

    def test_validator_error_codes(self):
        ok = f"<memory>Widget stock is 40 {self.cite()}.</memory>"
        self.assertEqual(validate_cited_memory(ok, self.t)[1], "OK")
        self.assertEqual(validate_cited_memory("no block", self.t)[1], "MissingMemoryBlock")
        self.assertEqual(validate_cited_memory("<memory>Uncited claim.</memory>", self.t)[1], "MissingCite")
        self.assertEqual(validate_cited_memory("<memory>x [[CITE tool=lookup]].</memory>", self.t)[1], "MalformedCite")
        self.assertEqual(validate_cited_memory(f"<memory>x {self.cite(step=9)}.</memory>", self.t)[1], "UnresolvedPointer")
        self.assertEqual(validate_cited_memory(f"<memory>x {self.cite(0, 99)}.</memory>", self.t)[1], "UnresolvedPointer")
        wrong = {("lookup", "L1", 1, "text", 0, 12): "0" * 12}
        self.assertEqual(validate_cited_memory(f"<memory>x {self.cite()}.</memory>", self.t, wrong)[1], "HashMismatch")

    def test_regeneration_appends_diagnostics_then_succeeds(self):
        seen = []

        def gen(req):
            seen.append(req)
            return "<memory>Uncited.</memory>" if len(seen) == 1 else f"<memory>Cited {self.cite()}.</memory>"

        out, info = cited_recall_loop(gen, "req", self.t, 3, fallback=lambda r: "FALLBACK")
        self.assertIn("CITE", out)
        self.assertEqual(info["attempts"], 2)
        self.assertIn("MissingCite", seen[1])

    def test_persistent_failure_falls_back(self):
        out, info = cited_recall_loop(lambda r: "<memory>No cite.</memory>", "req", self.t, 2, fallback=lambda r: "FALLBACK")
        self.assertEqual(out, "FALLBACK")
        self.assertTrue(info["fallback"])
        self.assertEqual(info["attempts"], 2)

    def test_citation_metrics(self):
        y = f"<memory>Stock {self.cite()}. Price is high.</memory>"
        m = citation_metrics(y, self.t, support_fn=lambda s, span: True)
        self.assertAlmostEqual(m["coverage"], 0.5)
        self.assertAlmostEqual(m["valid"], 1.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertNotEqual(citation_metrics(y, self.t)["precision"], citation_metrics(y, self.t)["precision"])  # NaN w/o judge


if __name__ == "__main__":
    unittest.main()
