from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
import torch

from backend.app.services.nlp.language_detector import (
    LANGUAGE_NAMES,
    MAX_INPUT_CHARS,
    LanguageDetectionResult,
    LanguageDetector,
)


class FakeTokenizer:
    def __init__(self, token_count=12):
        self.texts = []
        self.token_count = token_count

    def __call__(self, text, **kwargs):
        self.texts.append((text, kwargs))
        return {"input_ids": torch.ones((1, self.token_count), dtype=torch.long)}


class FakeModel:
    def __init__(self, logits, labels=None):
        self.logits = torch.tensor(logits) if isinstance(logits, list) else logits
        self.config = SimpleNamespace(
            id2label=labels or {0: "en", 1: "hi", 2: "fr"},
            _commit_hash="local-test-revision",
        )
        self.calls = 0

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, **_inputs):
        self.calls += 1
        if isinstance(self.logits, Exception):
            raise self.logits
        return SimpleNamespace(logits=self.logits)


def ready_detector(logits, labels=None, token_count=12):
    detector = LanguageDetector()
    detector._loaded = True
    detector._device = torch.device("cpu")
    detector._tokenizer = FakeTokenizer(token_count)
    detector._model = FakeModel(logits, labels)
    return detector


ENGLISH = "This is a sufficiently long English sentence for detection."
HINDI = "यह एक पर्याप्त लंबा हिंदी वाक्य है भाषा पहचान के लिए।"


def test_confident_english_prediction_and_metadata():
    detector = ready_detector([[5.0, 0.1, 0.1]])
    result = detector.detect(ENGLISH)
    payload = result.to_dict()
    assert result.status == "completed" and result.outcome == "success"
    assert payload["language_code"] == "en" and payload["language_name"] == "English"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["confidence_semantics"].startswith("Maximum softmax probability")
    assert payload["model_name"] == "papluca/xlm-roberta-base-language-detection"
    assert payload["model_revision"] == "local-test-revision"


def test_confident_hindi_prediction():
    result = ready_detector([[0.1, 5.0, 0.1]]).detect(HINDI)
    assert result.detected_language == "hi"
    assert result.language_name == "Hindi"
    assert result.outcome == "success"


def test_another_published_supported_language_prediction():
    result = ready_detector([[0.1, 0.1, 5.0]]).detect("Bonjour, ceci est une phrase française suffisamment longue.")
    assert result.detected_language == "fr"
    assert result.language_name == "French"
    assert len(LANGUAGE_NAMES) == 20


def test_unknown_or_unsupported_model_label_is_undetermined():
    result = ready_detector([[6.0, 0.1]], {0: "xx", 1: "en"}).detect(ENGLISH)
    assert result.status == "completed"
    assert result.outcome == "undetermined"
    assert result.detected_language == "unknown"
    assert any("unsupported" in warning for warning in result.warnings)


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_empty_and_whitespace_input_complete_as_no_text_without_loading(text):
    detector = LanguageDetector()
    detector.load = lambda **_kwargs: pytest.fail("empty input must not load the model")
    result = detector.detect(text)
    assert result.status == "completed" and result.outcome == "no_text"


def test_short_input_is_insufficient_without_prediction():
    detector = LanguageDetector()
    detector.load = lambda **_kwargs: pytest.fail("short input must not load the model")
    result = detector.detect("Hi!")
    assert result.status == "completed" and result.outcome == "insufficient_input"


def test_malformed_input_is_insufficient_and_safe():
    result = LanguageDetector().detect(None)  # type: ignore[arg-type]
    assert result.status == "completed" and result.outcome == "insufficient_input"
    assert "Language identification requires text input." in result.warnings


def test_noisy_ocr_text_is_insufficient():
    noisy = "!!!!!!!!!!!! ??????? ####### $$$$$ !!!! #######"
    result = LanguageDetector().detect(noisy)
    assert result.outcome == "insufficient_input"
    assert LanguageDetector().detect("1234567890 1234567890 1234567890").outcome == "insufficient_input"


def test_low_confidence_is_undetermined_and_score_stays_bounded():
    result = ready_detector([[1.0, 1.0, 1.0]]).detect(ENGLISH)
    assert result.status == "completed" and result.outcome == "undetermined"
    assert result.detected_language == "unknown"
    assert result.language_name is None
    assert 0.0 <= result.confidence <= 1.0


def test_missing_weights_are_unavailable_and_not_retried(monkeypatch):
    calls = []

    class MissingTokenizer:
        @classmethod
        def from_pretrained(cls, _model_id, **kwargs):
            calls.append(kwargs)
            raise FileNotFoundError("private local path")

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = MissingTokenizer
    transformers.AutoModelForSequenceClassification = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    detector = LanguageDetector()
    first = detector.detect(ENGLISH)
    second = detector.detect(ENGLISH)
    assert first.status == second.status == "unavailable"
    assert first.detected_language == "unknown"
    assert first.to_dict()["language_code"] is None
    assert first.confidence is None
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert "private local path" not in str(first.to_dict())


def test_missing_transformers_dependency_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = LanguageDetector().detect(ENGLISH)
    assert result.status == "unavailable" and result.outcome == "unavailable"


def test_model_initialization_failure_is_unavailable(monkeypatch):
    class Tokenizer:
        @classmethod
        def from_pretrained(cls, _model_id, **_kwargs):
            return FakeTokenizer()

    class BrokenModel:
        @classmethod
        def from_pretrained(cls, _model_id, **kwargs):
            assert kwargs["local_files_only"] is True
            raise RuntimeError("initialization detail with path C:/private")

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForSequenceClassification = BrokenModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    result = LanguageDetector().detect(ENGLISH)
    assert result.status == "unavailable"
    assert "C:/private" not in str(result.to_dict())


def test_local_loader_uses_local_files_only_and_model_label_mapping(monkeypatch):
    calls = []

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, _model_id, **kwargs):
            calls.append(kwargs)
            return FakeTokenizer()

    class Model(FakeModel):
        @classmethod
        def from_pretrained(cls, _model_id, **kwargs):
            calls.append(kwargs)
            return cls(torch.tensor([[5.0, 0.1, 0.1]]), {0: "en", 1: "hi", 2: "fr"})

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForSequenceClassification = Model
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    result = LanguageDetector().detect(ENGLISH)
    assert result.detected_language == "en"
    assert len(calls) == 2 and all(call["local_files_only"] is True for call in calls)


@pytest.mark.parametrize("mapping", [None, {}, {"bad-index": "en"}, {0: ""}, {0: None}])
def test_invalid_checkpoint_label_mappings_are_rejected(mapping):
    with pytest.raises(ValueError):
        LanguageDetector._validate_label_mapping(mapping)


@pytest.mark.asyncio
async def test_language_identification_timeout_maps_to_e1_failed_status():
    from backend.app.services.orchestrator import _run_stage

    class DB:
        async def commit(self): pass
        async def rollback(self): pass
        def add(self, _event): pass

    analysis = SimpleNamespace(id="language-timeout", current_step=None, completed_steps="[]")
    stages = {}
    result = await _run_stage(
        "language_identification",
        lambda: time.sleep(.05),
        analysis,
        DB(),
        stages,
        required=False,
        timeout_seconds=.001,
    )
    assert result is None
    assert stages["language_identification"].status == "failed"
    assert stages["language_identification"].outcome is None
    assert stages["language_identification"].error_category == "STAGE_TIMEOUT"


def test_inference_exception_is_failed_with_safe_message():
    result = ready_detector(RuntimeError("secret C:/private/model")).detect(ENGLISH)
    assert result.status == "failed" and result.outcome == "failed"
    serialized = str(result.to_dict())
    assert "secret" not in serialized and "C:/private" not in serialized


@pytest.mark.parametrize("logits", [None, torch.tensor([1.0, 2.0]), torch.tensor([[float("nan"), 0.0]])])
def test_malformed_or_nonfinite_model_output_is_failed(logits):
    result = ready_detector(logits).detect(ENGLISH)
    assert result.status == "failed" and result.outcome == "failed"


def test_long_input_is_bounded_and_truncation_is_reported():
    raw = "Long OCR text with meaning. " * 500
    detector = ready_detector([[5.0, 0.1, 0.1]])
    result = detector.detect(raw)
    passed_text, tokenizer_options = detector._tokenizer.texts[0]
    assert len(passed_text) == MAX_INPUT_CHARS
    assert tokenizer_options["truncation"] is True
    assert tokenizer_options["max_length"] == 256
    assert result.input_truncated is True
    assert any("truncated" in warning for warning in result.warnings)


def test_input_text_is_passed_without_rewriting():
    raw = "  This is OCR text, unchanged!\n"
    detector = ready_detector([[5.0, 0.1, 0.1]])
    detector.detect(raw)
    assert detector._tokenizer.texts[0][0] == raw


def test_concurrent_inference_is_serialized():
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    class ConcurrentModel(FakeModel):
        def __call__(self, **_inputs):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with guard:
                active -= 1
            return SimpleNamespace(logits=torch.tensor([[5.0, 0.1, 0.1]]))

    detector = ready_detector([[5.0, 0.1, 0.1]])
    detector._model = ConcurrentModel(torch.tensor([[5.0, 0.1, 0.1]]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(detector.detect, [ENGLISH] * 8))
    assert all(result.outcome == "success" for result in results)
    assert maximum_active == 1


def test_result_serialization_remains_backward_compatible_and_nullable():
    result = LanguageDetectionResult()
    payload = result.to_dict()
    assert payload["detected_language"] == "unknown"
    assert payload["top_languages"] == []
    assert payload["model_name"] == "papluca/xlm-roberta-base-language-detection"
    assert payload["model_version"] is None
    assert payload["confidence"] is None
