"""Generate the shipped synthetic PII corpus (offsets computed, never hand-written)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# (pii_type, value) planted into a template at {}. Values are reserved/documented
# example values only — see the corpus README block in redaction_suite.py.
Doc = tuple[str, str, str, list[tuple[str, str]]]

EN: list[Doc] = [
    (
        "en-onboarding",
        "en",
        "Onboarding note: the new joiner {} can be reached at {} or on {} while the "
        "laptop is prepared. Home address on file: {}.",
        [
            ("person_name", "John Q. Public"),
            ("email", "j.q.public@example.com"),
            ("phone", "+1 (415) 555-0142"),
            ("postal_address", "1600 Example Road, Springfield"),
        ],
    ),
    (
        "en-payroll",
        "en",
        "Payroll export row: SSN {} , card on file {} , born {} . Escalate mismatches "
        "to the finance desk.",
        [
            ("ssn", "123-45-6789"),
            ("credit_card", "4111 1111 1111 1111"),
            ("date_of_birth", "1984-03-02"),
        ],
    ),
    (
        "en-incident",
        "en",
        "Incident 4412: the gateway at {} rejected the callback after the on-call "
        "engineer {} rotated credentials.",
        [
            ("ip", "192.0.2.44"),
            ("person_name", "Dana Example"),
        ],
    ),
    (
        "en-leak",
        "en",
        "The pasted config contained api_key={} and the line 'Password: {}' in plain "
        "text, plus the deploy key {} .",
        [
            ("api_key", "sk-test0000000000000000000000dead"),
            ("secret", "hunter2example"),
            ("aws_key", "AKIAIOSFODNN7EXAMPLE"),
        ],
    ),
    (
        "en-support",
        "en",
        "Support ticket: customer wired the refund to {} and asked us to confirm by "
        "email at {} .",
        [
            ("iban", "GB33BUKB20201555555555"),
            ("email", "support.case@example.org"),
        ],
    ),
    (
        "en-session",
        "en",
        "Debug dump from the browser session: token: {} — please scrub before "
        "attaching this to the ticket.",
        [
            (
                "jwt",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
                ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            ),
        ],
    ),
    (
        "en-vendor",
        "en",
        "Vendor contact {} confirmed the shipment; reach the warehouse line on {} or "
        "the shared inbox {} .",
        [
            ("person_name", "Sam Placeholder"),
            ("phone", "+1 202 555 0188"),
            ("email", "warehouse@example.net"),
        ],
    ),
    (
        "en-clean-1",
        "en",
        "The quarterly report summarises revenue growth across three regions and "
        "flags two supply risks for the next planning cycle.",
        [],
    ),
    (
        "en-clean-2",
        "en",
        "Release 1.4 shipped on schedule. No regressions were observed in the "
        "integration suite, and the rollback plan was exercised once.",
        [],
    ),
    (
        "en-clean-3",
        "en",
        "Meeting notes: agenda, action items, and the follow-up date agreed for next "
        "week. Owners were assigned to each item.",
        [],
    ),
]

DE: list[Doc] = [
    (
        "de-personalakte",
        "de",
        "Personalakte: {} ist per E-Mail unter {} sowie telefonisch unter {} "
        "erreichbar. Wohnanschrift: {} .",
        [
            ("person_name", "Erika Mustermann"),
            ("email", "erika.mustermann@example.de"),
            ("phone", "+49 30 555 01234"),
            ("postal_address", "Musterstraße 12, 10115 Berlin"),
        ],
    ),
    (
        "de-lohnbuchhaltung",
        "de",
        "Lohnbuchhaltung: Überweisung auf {} , Kreditkarte {} hinterlegt, geboren am "
        "{} . Rückfragen an die Buchhaltung.",
        [
            ("iban", "DE89370400440532013000"),
            ("credit_card", "5555 5555 5555 4444"),
            ("date_of_birth", "1979-11-30"),
        ],
    ),
    (
        "de-steuer",
        "de",
        "Steuerliche Identifikationsnummer {} wurde im Antrag von {} angegeben.",
        [
            ("national_id", "12345678901"),
            ("person_name", "Max Beispiel"),
        ],
    ),
    (
        "de-vorfall",
        "de",
        "Vorfallbericht: Das Gateway unter {} wies die Anfrage ab, nachdem der "
        "Zugriffsschlüssel {} rotiert wurde.",
        [
            ("ip", "203.0.113.7"),
            ("aws_key", "AKIAIOSFODNN7EXAMPLE"),
        ],
    ),
    (
        "de-konfiguration",
        "de",
        "In der Konfigurationsdatei standen api_key={} und die Zeile 'Password: {}' "
        "im Klartext.",
        [
            ("api_key", "sk-test1111111111111111111111beef"),
            ("secret", "geheim2example"),
        ],
    ),
    (
        "de-kundenanfrage",
        "de",
        "Kundenanfrage von {} : Rückerstattung bitte auf {} , Bestätigung an {} "
        "senden.",
        [
            ("person_name", "Lena Musterfrau"),
            ("iban", "AT611904300234573201"),
            ("email", "kundenservice@example.de"),
        ],
    ),
    (
        "de-sitzung",
        "de",
        "Aus dem Sitzungsprotokoll kopiert: token: {} — vor dem Anhängen entfernen.",
        [
            (
                "jwt",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                ".eyJzdWIiOiI5ODc2NTQzMjEwIiwibmFtZSI6IkVyaWthIn0"
                ".7cVQ0GBxJ4bJcVQCPMPmJ2yD0k0jvxKcQzp9y0dqk5Q",
            ),
        ],
    ),
    (
        "de-sauber-1",
        "de",
        "Der Quartalsbericht fasst das Umsatzwachstum in drei Regionen zusammen und "
        "benennt zwei Lieferrisiken für die nächste Planungsrunde.",
        [],
    ),
    (
        "de-sauber-2",
        "de",
        "Version 1.4 wurde planmäßig ausgeliefert. In der Integrationssuite traten "
        "keine Regressionen auf.",
        [],
    ),
    (
        "de-sauber-3",
        "de",
        "Besprechungsnotiz: Tagesordnung, Aufgaben und der Folgetermin in der "
        "nächsten Woche. Für jede Aufgabe wurde eine verantwortliche Person benannt.",
        [],
    ),
]


def build(docs: list[Doc]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for doc_id, language, template, planted in docs:
        parts = template.split("{}")
        if len(parts) != len(planted) + 1:
            msg = f"{doc_id}: {len(parts) - 1} placeholders for {len(planted)} values"
            raise SystemExit(msg)
        text = ""
        spans: list[dict[str, object]] = []
        for index, part in enumerate(parts):
            text += part
            if index < len(planted):
                pii_type, value = planted[index]
                start = len(text)
                text += value
                spans.append(
                    {"pii_type": pii_type, "value": value, "start": start, "end": len(text)}
                )
        for span in spans:
            assert text[span["start"] : span["end"]] == span["value"], doc_id
        rows.append(
            {
                "doc_id": doc_id,
                "language": language,
                "sensitivity": "internal",
                "text": text,
                "pii": spans,
            }
        )
    return rows


def main() -> None:
    out = Path(sys.argv[1])
    rows = build(EN) + build(DE)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"{len(rows)} rows, {sum(len(r['pii']) for r in rows)} gold spans")
    print("sha256", digest)
    print("revision (first 40 hex)", digest[:40])


if __name__ == "__main__":
    main()
