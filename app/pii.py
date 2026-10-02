"""Personal-data redaction with Microsoft Presidio, used on memory facts, logs and traces.

Algorithm: Presidio runs a set of recognisers over the text. Each one reports (entity type, where,
confidence). We keep only the entity types in REDACTED and replace each hit with its label:

    "Email me at ann.lee@acme.com, card 4111 1111 1111 1111"
 -> "Email me at <EMAIL_ADDRESS>, card <CREDIT_CARD>"

Only PATTERN-based recognisers are used (email, phone, card, IBAN, IP, SSN, ...): they are precise.
Name / place / organisation detection ("NER") is NOT used: measured with the small spaCy model it
tags "Email" as a person and "Priya" as an organisation (benchmark B-016), so it would both miss
real names and mangle normal sentences. That is a documented limit, not an oversight.

A cheap pre-check (every pattern we redact contains a digit or an '@') skips Presidio for most
text, because the first call is slow (about 1 s model load) and logs are written constantly.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

REDACTED = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "US_SSN",
    "US_ITIN",
    "US_PASSPORT",
    "US_BANK_NUMBER",
    "UK_NHS",
    "CRYPTO",
    "MEDICAL_LICENSE",
)
MIN_SCORE = 0.4  # Presidio gives a bare phone-like number 0.4; lower values are mostly noise
_MAYBE_PII = re.compile(r"[@\d]")
# Identifiers we generate ourselves (job ids, org ids) are full of digits but are not personal data,
# and redacting them would make logs useless for debugging.
_OWN_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


@dataclass(frozen=True)
class Redaction:
    text: str
    found: tuple[str, ...]  # entity types that were replaced, in order of appearance


@lru_cache
def _engines() -> tuple[Any, Any]:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    nlp = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    ).create_engine()
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"]), AnonymizerEngine()


def redact(text: str) -> Redaction:
    if not text or not _MAYBE_PII.search(text) or _OWN_ID.fullmatch(text.strip()):
        return Redaction(text, ())
    analyzer, anonymizer = _engines()
    hits = analyzer.analyze(
        text=text, language="en", entities=list(REDACTED), score_threshold=MIN_SCORE
    )
    if not hits:
        return Redaction(text, ())
    result = anonymizer.anonymize(text=text, analyzer_results=hits)
    return Redaction(result.text, tuple(h.entity_type for h in sorted(hits, key=lambda h: h.start)))


def redact_any(data: Any) -> Any:
    """Redact every string inside nested dicts/lists (what a trace or a log record holds)."""
    if isinstance(data, str):
        return redact(data).text
    if isinstance(data, dict):
        return {k: redact_any(v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return type(data)(redact_any(v) for v in data)
    return data
