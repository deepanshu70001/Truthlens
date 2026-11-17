"""Controlled evidence-fusion behavior checks; not model-accuracy evaluation."""
from backend.app.services.evidence.schema import EvidenceCollection, EvidenceItem
from backend.app.services.fusion.engine import fuse_evidence


def _item(kind="model_evidence", source="detector", direction="synthetic", status="completed", description="signal"):
    return EvidenceItem(type=kind, source=source, strength="high", direction=direction,
                        description=description, status=status, confidence=0.9)


def test_unavailable_directional_evidence_does_not_vote():
    collection = EvidenceCollection(items=[_item(status="unavailable")])
    assessment = fuse_evidence(collection)
    assert assessment.label == "inconclusive"
    assert assessment.overall_score is None
    assert "Ignored unavailable evidence" in assessment.reasoning[0]


def test_failed_and_skipped_evidence_do_not_change_completed_result():
    base = fuse_evidence(EvidenceCollection(items=[_item()]))
    with_unavailable = fuse_evidence(EvidenceCollection(items=[
        _item(), _item(source="failed_detector", status="failed", description="failure"),
        _item(source="not_run", status="skipped", description="skip"),
    ]))
    assert with_unavailable.label == base.label
    assert with_unavailable.overall_score == base.overall_score


def test_completed_forensics_are_informational_not_authenticity_votes():
    model_only = fuse_evidence(EvidenceCollection(items=[_item()]))
    with_forensics = fuse_evidence(EvidenceCollection(items=[
        _item(), _item(kind="forensic_evidence", source="ela", direction="authentic", description="forensic cue"),
    ]))
    assert with_forensics.label == model_only.label
    assert with_forensics.overall_score == model_only.overall_score


def test_duplicate_completed_evidence_is_counted_once():
    one = _item()
    collection = EvidenceCollection(items=[one, _item()])
    assert fuse_evidence(collection).overall_score == fuse_evidence(EvidenceCollection(items=[one])).overall_score
