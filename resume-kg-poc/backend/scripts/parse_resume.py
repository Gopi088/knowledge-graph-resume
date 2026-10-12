#!/usr/bin/env python3
"""Parse one resume and write its canonical structured JSON plus pipeline artifacts.

Example:
    python scripts/parse_resume.py "/path/to/resume.pdf"
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.pipeline.pipeline import run_pipeline  # noqa: E402
from scripts.batch_pipeline_test import extract_resume  # noqa: E402


def readable_resume(canonical: dict) -> dict:
    """Project the provenance-rich canonical model into a concise JSON view."""
    personal = {
        key: value for key, value in (canonical.get("personal_information") or {}).items()
        if key != "source_blocks" and value not in (None, "", [])
    }
    for key, value in (canonical.get("personal_details") or {}).items():
        if key != "source_blocks" and value not in (None, "", []) and key not in personal:
            personal[key] = value

    summaries = canonical.get("profile_snapshot") or []
    if isinstance(summaries, str):
        summaries = [summaries]
    summary = " ".join(str(item).strip() for item in summaries if str(item).strip())
    experience_match = re.search(r"\b\d+(?:\+|\s*\+)?\s*(?:years?|yrs?)\b", summary, re.I)

    work = []
    for employer in canonical.get("work_history") or []:
        roles = employer.get("roles") or []
        if not roles:
            roles = [{}]
        for role in roles:
            item = {
                "company": employer.get("company", ""),
                "designation": role.get("designation", ""),
                "start_date": role.get("start_date", ""),
                "end_date": role.get("end_date", ""),
                "responsibilities": role.get("role_and_responsibilities", []) or [],
            }
            if role.get("duration"):
                item["duration"] = role["duration"]
            if role.get("location"):
                item["location"] = role["location"]
            if role.get("clients"):
                item["clients"] = role["clients"]
            work.append(item)

    projects = []
    for project in canonical.get("projects") or []:
        item = {}
        for key in ("name", "organization", "technologies", "responsibilities", "description", "client",
                    "start_date", "end_date", "duration", "date"):
            if project.get(key):
                item[key] = project[key]
        projects.append(item)

    education = []
    for record in canonical.get("education") or []:
        item = {key: record[key] for key in ("degree", "institution", "university_or_board", "year", "score")
                if record.get(key) not in (None, "", [])}
        education.append(item)

    output = {
        "personal_details": personal,
        "professional_summary": summary,
        "total_experience": experience_match.group(0) if experience_match else None,
        "work_experience": work,
        "skills": [item.get("name", "") if isinstance(item, dict) else str(item)
                   for item in canonical.get("skills") or []
                   if (item.get("name") if isinstance(item, dict) else item)],
        "technical_skills": canonical.get("technical_skills") or {},
        "projects": projects,
        "education": education,
        "key_strengths": (canonical.get("source_sections") or {}).get("key_strengths", []),
    }
    for field in ("certifications", "awards", "achievements", "publications", "domain_experience"):
        if canonical.get(field):
            output[field] = canonical[field]
    unresolved = canonical.get("unresolved_blocks") or []
    if unresolved:
        output["unresolved_blocks"] = unresolved
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resume_path", type=Path, help="PDF, DOC/DOCX, text, RTF, or image resume")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Directory for canonical_resume.json and intermediate artifacts; defaults to backend/single_resume_output/<filename>.",
    )
    parser.add_argument("--candidate-name", default="Candidate", help="Graph label for the resume subject")
    args = parser.parse_args(argv)

    source = args.resume_path.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Resume file does not exist: {source}")

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "resume"
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  BACKEND_DIR / "single_resume_output" / safe_stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Some document libraries emit optional-dependency notices to stdout;
    # keep those out of the JSON stream as well.
    with contextlib.redirect_stdout(sys.stderr):
        text, method = extract_resume(source)
    if not text.strip():
        parser.error(f"No text could be extracted from {source}; the PDF may be scanned or the file may be empty.")

    # Keep stdout as machine-readable, pretty JSON. Pipeline progress remains
    # visible in the terminal on stderr and will not corrupt `> output.json`.
    with contextlib.redirect_stdout(sys.stderr):
        artifacts = run_pipeline(text, safe_stem, str(output_dir), candidate_name=args.candidate_name)
    canonical_path = output_dir / "canonical_resume.json"
    readable = readable_resume(artifacts["canonical_resume.json"])
    readable_path = output_dir / "parsed_resume_readable.json"
    readable_path.write_text(json.dumps(readable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Extracted with: {method}", file=sys.stderr)
    print(f"Canonical JSON artifact: {canonical_path}", file=sys.stderr)
    print(f"Readable JSON artifact: {readable_path}", file=sys.stderr)
    print(json.dumps(readable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
