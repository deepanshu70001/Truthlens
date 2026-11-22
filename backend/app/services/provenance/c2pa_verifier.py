"""Optional verification adapter for the official C2PA Python SDK.

This module does not parse JUMBF/CBOR itself. It trusts only the verification
diagnostics returned by the SDK and reports signature, content binding, and
signer trust as separate properties.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from backend.app.config import settings
from backend.app.core.logging import get_logger

log = get_logger("c2pa")
IMPLEMENTATION = "c2pa-python"


@dataclass
class C2PAResult:
    status: str = "unavailable"
    outcome: Optional[str] = "unavailable"
    verification_outcome: str = "unavailable"
    manifest_present: Optional[bool] = None
    manifest_readable: Optional[bool] = None
    signature_status: str = "unverified"
    content_binding_status: str = "unverified"
    signer_identity: Optional[Dict[str, str]] = None
    signer_trust_status: str = "unverified"
    trust_check_configured: bool = False
    assertions: list[str] = field(default_factory=list)
    ingredient_count: Optional[int] = None
    referenced_resources_status: str = "not_checked"
    referenced_resources_available: Optional[int] = None
    referenced_resources_checked: int = 0
    referenced_resource_scope: str = "active_manifest_thumbnail_only"
    validation_codes: list[str] = field(default_factory=list)
    implementation: str = IMPLEMENTATION
    implementation_version: Optional[str] = None
    runtime_ms: int = 0
    error_category: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "outcome": self.outcome,
            "verification_outcome": self.verification_outcome,
            "manifest_present": self.manifest_present,
            "manifest_readable": self.manifest_readable,
            "signature_status": self.signature_status,
            "content_binding_status": self.content_binding_status,
            "signer_identity": self.signer_identity,
            "signer_trust_status": self.signer_trust_status,
            "trust_check_configured": self.trust_check_configured,
            "assertions": self.assertions,
            "ingredient_count": self.ingredient_count,
            "referenced_resources_status": self.referenced_resources_status,
            "referenced_resources_available": self.referenced_resources_available,
            "referenced_resources_checked": self.referenced_resources_checked,
            "referenced_resource_scope": self.referenced_resource_scope,
            "validation_codes": self.validation_codes,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "runtime_ms": self.runtime_ms,
            "error_category": self.error_category,
            "warnings": self.warnings,
            "interpretation": (
                "Cryptographic credential verification describes the signed asset and signer only. "
                "It does not establish whether the depicted event is true."
            ),
        }


def _mime_type(image_data: bytes) -> str:
    with Image.open(io.BytesIO(image_data)) as image:
        fmt = (image.format or "").upper()
    return Image.MIME.get(fmt, "application/octet-stream")


def _safe_text(value: Any, limit: int = 300) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value))
    text = re.sub(r"(?i)(?:[a-z]:\\|/)(?:[^\s:/\\]+[\\/])+[^\s:]*", "[redacted path]", text)
    text = re.sub(r"(?i)(api[_ -]?key|key|token|secret|password)\s*[:=]\s*\S+", r"\1=[redacted]", text)
    return text[:limit]


def _verification_entries(store: Dict[str, Any]) -> list[Dict[str, Any]]:
    results = store.get("validation_results", {})
    if not isinstance(results, dict):
        return []
    entries = []
    active = results.get("activeManifest", {})
    if isinstance(active, dict):
        for kind in ("success", "informational", "failure"):
            values = active.get(kind, [])
            if isinstance(values, list):
                entries.extend(item for item in values if isinstance(item, dict))
    return entries


def verify_content_credentials(image_data: bytes, sdk_module=None) -> C2PAResult:
    """Read and verify a manifest from the original image byte stream."""
    started = time.perf_counter()
    try:
        if not isinstance(image_data, bytes) or not image_data:
            return C2PAResult(status="failed", outcome="failed", verification_outcome="failed", error_category="INVALID_IMAGE_BYTES",
                warnings=["Original image bytes were empty or invalid."])
        sdk = sdk_module or importlib.import_module("c2pa")
        version = importlib.metadata.version("c2pa-python") if sdk_module is None else getattr(sdk_module, "__version__", "mock")
        settings_data: Dict[str, Any] = {"verify": {
            "verify_after_reading": True,
            "remote_manifest_fetch": False,
            "verify_trust": False,
            "verify_timestamp_trust": False,
        }}
        trust_path = str(getattr(settings, "C2PA_TRUST_ANCHORS_PATH", "") or "").strip()
        trust_configured = bool(trust_path and Path(trust_path).is_file())
        if trust_configured:
            settings_data["verify"]["verify_trust"] = True
            settings_data["trust"] = {"trust_anchors": Path(trust_path).read_text(encoding="utf-8")}
        sdk_settings = sdk.Settings.from_dict(settings_data)
        context = sdk.Context(sdk_settings)
        with context as ctx:
            with sdk.Reader(_mime_type(image_data), io.BytesIO(image_data), context=ctx) as reader:
                raw_store = reader.json()
                store = json.loads(raw_store) if isinstance(raw_store, str) else raw_store
                if not isinstance(store, dict):
                    raise ValueError("SDK returned a malformed manifest store")
                active_label = store.get("active_manifest")
                manifests = store.get("manifests", {})
                if not active_label and not manifests:
                    return C2PAResult(status="completed", outcome="no_results", verification_outcome="no_manifest", manifest_present=False,
                        manifest_readable=None, implementation_version=version,
                        runtime_ms=int((time.perf_counter()-started)*1000),
                        warnings=["No embedded C2PA manifest was found; this is not evidence of manipulation."])
                if not isinstance(manifests, dict) or not isinstance(active_label, str) or not isinstance(manifests.get(active_label), dict):
                    return C2PAResult(status="completed", outcome="success", verification_outcome="verification_failed", manifest_present=True,
                        manifest_readable=False, implementation_version=version,
                        runtime_ms=int((time.perf_counter()-started)*1000),
                        warnings=["The C2PA manifest store is present but structurally incomplete."])
                manifest = manifests[active_label]
                assertions = manifest.get("assertions", [])
                if not isinstance(assertions, list):
                    return C2PAResult(status="completed", outcome="success", verification_outcome="verification_failed", manifest_present=True,
                        manifest_readable=False, implementation_version=version,
                        runtime_ms=int((time.perf_counter()-started)*1000),
                        warnings=["The active C2PA manifest has malformed assertion data."])

                entries = _verification_entries(store)
                codes = list(dict.fromkeys(_safe_text(e.get("code", ""), 120) for e in entries if e.get("code")))[:100]
                active_validation = store.get("validation_results", {}).get("activeManifest", {})
                successful = active_validation.get("success", []) if isinstance(active_validation, dict) else []
                success_codes = {str(entry.get("code", "")) for entry in successful if isinstance(entry, dict)} if isinstance(successful, list) else set()
                failed_codes = {str(entry.get("code", "")) for entry in _verification_entries_failure(store)}
                signature_fail = any(code.startswith("claimSignature.") and code not in {"claimSignature.validated", "claimSignature.insideValidity"} for code in failed_codes)
                signature_valid = "claimSignature.validated" in success_codes
                signature_status = "invalid" if signature_fail else "valid" if signature_valid else "unverified"
                hash_failures = {"assertion.dataHash.mismatch", "assertion.bmffHash.mismatch", "assertion.boxesHash.mismatch"}
                hash_matches = {"assertion.dataHash.match", "assertion.bmffHash.match", "assertion.boxesHash.match"}
                binding_status = "mismatch" if failed_codes & hash_failures else "valid" if success_codes & hash_matches else "unverified"
                trust_codes = set(codes)
                trust_state = str(store.get("validation_state", "")).casefold()
                if not trust_configured:
                    signer_trust = "unverified"
                elif "signingCredential.untrusted" in trust_codes:
                    signer_trust = "untrusted"
                elif "signingCredential.invalid" in trust_codes:
                    signer_trust = "invalid"
                elif "signingCredential.revoked" in trust_codes:
                    signer_trust = "revoked"
                elif "signingCredential.expired" in trust_codes:
                    signer_trust = "expired"
                elif trust_state == "trusted":
                    signer_trust = "trusted"
                else:
                    signer_trust = "unverified"

                signature_info = manifest.get("signature_info", {})
                signer = None
                if isinstance(signature_info, dict):
                    signer = {key: _safe_text(signature_info[key]) for key in ("common_name", "issuer", "alg", "time") if signature_info.get(key) is not None}
                labels = [_safe_text(item.get("label", ""), 160) for item in assertions if isinstance(item, dict) and item.get("label")]
                ingredients = manifest.get("ingredients")
                ingredient_count = len(ingredients) if isinstance(ingredients, list) else sum(label.startswith("c2pa.ingredient") for label in labels)
                refs = []
                thumbnail = manifest.get("thumbnail")
                if isinstance(thumbnail, dict) and isinstance(thumbnail.get("identifier"), str):
                    refs.append(thumbnail["identifier"])
                available = 0
                checked = 0
                if refs and hasattr(reader, "resource_to_stream"):
                    for ref in refs[:20]:
                        checked += 1
                        try:
                            resource = reader.resource_to_stream(ref)
                            if resource is not None:
                                resource.read(1)
                                available += 1
                                if hasattr(resource, "close"):
                                    resource.close()
                        except Exception:
                            continue
                resource_status = "available" if checked and available == checked else "partial" if available else "unavailable" if checked else "not_checked"
                verification_outcome = (
                    "verification_failed" if signature_status == "invalid" or binding_status == "mismatch"
                    else "verified" if signature_status == "valid" and binding_status == "valid"
                    else "unverified"
                )
                return C2PAResult(status="completed", outcome="success", verification_outcome=verification_outcome, manifest_present=True,
                    manifest_readable=True, signature_status=signature_status,
                    content_binding_status=binding_status, signer_identity=signer,
                    signer_trust_status=signer_trust, trust_check_configured=trust_configured, assertions=labels[:100],
                    ingredient_count=ingredient_count, referenced_resources_status=resource_status,
                    referenced_resources_available=available if checked else None,
                    referenced_resources_checked=checked, validation_codes=codes,
                    implementation_version=version, runtime_ms=int((time.perf_counter()-started)*1000),
                    warnings=([] if verification_outcome == "verified" and signer_trust == "trusted" else [
                        "Signature, content binding, and signer trust are reported separately; one or more properties remain invalid or unverified."
                    ]))
    except ModuleNotFoundError:
        return C2PAResult(status="unavailable", outcome="unavailable", verification_outcome="unavailable", error_category="VERIFIER_DEPENDENCY_MISSING",
            runtime_ms=int((time.perf_counter()-started)*1000), warnings=["C2PA verification library is not installed."])
    except Exception as exc:
        log.warning("C2PA verification could not complete (%s)", type(exc).__name__)
        return C2PAResult(status="failed", outcome="failed", verification_outcome="failed", error_category="C2PA_VERIFICATION_ERROR",
            runtime_ms=int((time.perf_counter()-started)*1000), warnings=["C2PA manifest could not be read or verified."])


def _verification_entries_failure(store: Dict[str, Any]) -> list[Dict[str, Any]]:
    results = store.get("validation_results", {})
    active = results.get("activeManifest", {}) if isinstance(results, dict) else {}
    failures = active.get("failure", []) if isinstance(active, dict) else []
    return failures if isinstance(failures, list) else []
