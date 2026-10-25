import unittest

import numpy as np

import _util  # noqa: F401
from helm.index import FlatIndex, HNSWIndex


def _unit(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestHNSW(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = _unit(rng, 1500, 24)
        self.Q = _unit(rng, 60, 24)
        self.h = HNSWIndex(24, M=16, ef_construction=200, ef_search=64)
        self.f = FlatIndex(24)
        for i, v in enumerate(self.X):
            self.h.add(str(i), v)
            self.f.add(str(i), v)

    def _recall(self):
        r = []
        for q in self.Q:
            a = {e for e, _ in self.h.search(q, 10)}
            b = {e for e, _ in self.f.search(q, 10)}
            r.append(len(a & b) / 10)
        return float(np.mean(r))

    def test_recall_against_exact(self):
        self.assertGreaterEqual(self._recall(), 0.95)

    def test_delete_and_compaction(self):
        for i in range(0, 1500, 2):
            self.h.remove(str(i))
            self.f.remove(str(i))
        self.assertEqual(len(self.h), 750)
        self.assertGreaterEqual(self._recall(), 0.95)
        for e, _ in self.h.search(self.Q[0], 20):
            self.assertEqual(int(e) % 2, 1)  # no deleted id is ever returned

    def test_allowed_filter(self):
        allow = {str(i) for i in range(1, 1500, 7)}
        got = {e for e, _ in self.h.search(self.Q[1], 10, allowed=allow)}
        want = {e for e, _ in self.f.search(self.Q[1], 10, allowed=allow)}
        self.assertTrue(got <= allow)
        self.assertGreaterEqual(len(got & want), 9)

    def test_upsert_and_empty(self):
        idx = HNSWIndex(4)
        self.assertEqual(idx.search(np.ones(4, dtype=np.float32), 3), [])
        idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
        idx.add("a", np.array([0, 1, 0, 0], dtype=np.float32))
        self.assertEqual(len(idx), 1)
        self.assertAlmostEqual(idx.search(np.array([0, 1, 0, 0], dtype=np.float32), 1)[0][1], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
