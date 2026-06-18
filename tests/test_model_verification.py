"""Tests for the SHA-256 model-integrity gate in frameshift.utils.detection.

The detector loads PyTorch .pt files, which are pickle archives -> loading them
is arbitrary code execution. ensure_verified_model() must therefore NEVER return
a path whose contents don't match the pinned digest. These tests pin that
guarantee.

The heavy runtime deps (cv2, numpy, mediapipe, ultralytics) are stubbed so the
test runs without a full install; only the pure verification logic is exercised.
"""
import sys
import types
import hashlib
import tempfile
import unittest
from pathlib import Path


def _install_stubs():
    """Register minimal fake modules so detection.py imports without heavy deps."""
    for name in ("cv2", "numpy", "requests", "tqdm"):
        sys.modules.setdefault(name, types.ModuleType(name))

    # tqdm.tqdm callable
    sys.modules["tqdm"].tqdm = lambda *a, **k: None

    # numpy.ndarray is referenced in type annotations evaluated at import time.
    sys.modules["numpy"].ndarray = object

    mp = types.ModuleType("mediapipe")
    mp.solutions = types.SimpleNamespace(face_detection=types.SimpleNamespace(FaceDetection=object))
    sys.modules.setdefault("mediapipe", mp)

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = object
    sys.modules.setdefault("ultralytics", ultra)

    # scenedetect is imported transitively via frameshift/__init__ -> main.
    sd = types.ModuleType("scenedetect")
    sd.open_video = lambda *a, **k: None
    sd.SceneManager = object
    sys.modules.setdefault("scenedetect", sd)
    sd_det = types.ModuleType("scenedetect.detectors")
    sd_det.ContentDetector = object
    sys.modules.setdefault("scenedetect.detectors", sd_det)


_install_stubs()

# Import the real module under test (repo root must be on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frameshift.utils import detection  # noqa: E402


class Sha256OfFileTests(unittest.TestCase):
    def test_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "blob.bin"
            data = b"frameshift-test-payload" * 1000
            p.write_bytes(data)
            self.assertEqual(detection._sha256_of_file(p), hashlib.sha256(data).hexdigest())


class EnsureVerifiedModelTests(unittest.TestCase):
    def setUp(self):
        self._saved_registry = detection.MODEL_REGISTRY.copy()

    def tearDown(self):
        detection.MODEL_REGISTRY.clear()
        detection.MODEL_REGISTRY.update(self._saved_registry)

    def test_unknown_model_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(detection.ensure_verified_model("not-registered.pt", Path(d)))

    def test_matching_existing_file_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            model_dir = Path(d)
            data = b"a legitimate model file"
            digest = hashlib.sha256(data).hexdigest()
            (model_dir / "fake.pt").write_bytes(data)
            detection.MODEL_REGISTRY["fake.pt"] = {
                "url": "https://example.invalid/fake.pt",
                "sha256": digest,
            }
            result = detection.ensure_verified_model("fake.pt", model_dir)
            self.assertEqual(result, model_dir / "fake.pt")

    def test_tampered_existing_file_rejected(self):
        # File present but hash mismatched: it must be deleted and (since the
        # download URL is bogus) the call must fail closed with None.
        with tempfile.TemporaryDirectory() as d:
            model_dir = Path(d)
            (model_dir / "fake.pt").write_bytes(b"MALICIOUS PAYLOAD")
            detection.MODEL_REGISTRY["fake.pt"] = {
                "url": "https://example.invalid/fake.pt",
                "sha256": hashlib.sha256(b"the real file").hexdigest(),
            }

            # Make the download path deterministically fail (no network in test).
            def _boom(*a, **k):
                raise RuntimeError("network disabled in test")

            detection.requests.get = _boom  # type: ignore[attr-defined]

            result = detection.ensure_verified_model("fake.pt", model_dir)
            self.assertIsNone(result)
            # The tampered file must not survive to be loaded later.
            self.assertFalse((model_dir / "fake.pt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
