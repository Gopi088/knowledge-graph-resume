#!/usr/bin/env python3
"""Hybrid, ground-truth-based batch evaluator for resume block extraction.

The evaluator runs the existing extraction/pipeline code without modifying it,
saves every returned JSON artifact, and compares canonical section content with
manually verified JSON. Cosine similarity only creates review suggestions; the
reported precision/recall/F1 scores use exact normalized item matches.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.pipeline.pipeline import (  # noqa: E402
    extract_docx_text,
    extract_image_text,
    extract_pdf_text,
    run_pipeline,
)


DEFAULT_INPUT_DIR = Path(
    r"C:\Users\Gopi Gedar\Eragina Digital Solutions Private Limited\Eragina Digital - Documents\RESUMES\resume\RESUMES_FLAT"
)
SUPPORTED_SUFFIXES = {
    ".pdf", ".doc", ".docx", ".docm", ".txt", ".rtf",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EVALUATED_SECTIONS = ("experience", "projects", "skills", "education")
SECTION_ALIASES = {
    "experience": {"work_history", "work_experience", "experience", "professional_experience", "employment"},
    "projects": {"projects", "project", "project_highlights"},
    "skills": {"skills", "technical_skills", "key_skills", "competencies"},
    "education": {"education", "educations", "academic_background", "qualifications"},
}
IGNORED_KEYS = {
    "source_blocks", "source_block_ids", "block_ids", "source_sentences", "source_evidence",
    "source_path", "source_paths", "source_line_indices", "sentence_indices", "section_id",
    "confidence", "method", "id", "entity_id", "relationship_id", "page", "bbox",
    "source_audit", "schema_version",
}
TOKEN_RE = re.compile(r"[\w+#.]+", re.UNICODE)


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def rtf_to_text(path: Path) -> str:
    text = read_text(path)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", lambda m: bytes([int(m.group(0)[2:], 16)]).decode("cp1252", errors="replace"), text)
    text = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), text)
    text = re.sub(r"\\(?:par|line)\b ?", "\n", text, flags=re.I)
    text = re.sub(r"\\tab\b ?", "\t", text, flags=re.I)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = re.sub(r"\\[{}\\]", "", text).replace("{", "").replace("}", "")
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def extract_legacy_doc(path: Path) -> tuple[str, str]:
    if os.name == "nt":
        try:
            from win32com.client import DispatchEx

            word = DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            document = None
            try:
                document = word.Documents.Open(str(path.resolve()), ReadOnly=True,
                                               ConfirmConversions=False, AddToRecentFiles=False,
                                               Visible=False)
                text = document.Content.Text
                if text.strip():
                    return text, "microsoft_word_com"
            finally:
                if document is not None:
                    document.Close(False)
                word.Quit()
        except ImportError:
            pass
        except Exception:
            # Try command-line converters next; failures are reported if none work.
            pass
    for executable, label, args in (
        ("antiword", "antiword", lambda exe: [exe, str(path)]),
        ("catdoc", "catdoc", lambda exe: [exe, str(path)]),
    ):
        exe = shutil.which(executable)
        if exe:
            result = subprocess.run(args(exe), capture_output=True, text=True, timeout=90)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout, label
    office = shutil.which("libreoffice") or shutil.which("soffice")
    if office:
        with tempfile.TemporaryDirectory(prefix="resume-eval-doc-") as temp_dir:
            result = subprocess.run([office, "--headless", "--convert-to", "txt:Text", "--outdir", temp_dir, str(path)],
                                    capture_output=True, text=True, timeout=120)
            converted = Path(temp_dir) / f"{path.stem}.txt"
            if result.returncode == 0 and converted.is_file():
                return read_text(converted), "libreoffice"
    raise RuntimeError("Cannot read legacy .doc: install antiword, catdoc, or LibreOffice.")


def extract_resume(path: Path) -> tuple[str, str]:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return extract_pdf_text(str(path)), "pymupdf/tesseract"
    if suffix in {".docx", ".docm"}:
        return extract_docx_text(str(path)), "ooxml"
    if suffix == ".doc":
        return extract_legacy_doc(path)
    if suffix in IMAGE_SUFFIXES:
        return extract_image_text(str(path)), "tesseract"
    if suffix == ".txt":
        return read_text(path), "plain_text"
    if suffix == ".rtf":
        return rtf_to_text(path), "rtf"
    raise ValueError(f"Unsupported file type: {suffix or '(none)'}")


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "resume"


def normalize(value: Any) -> str:
    text = str(value).casefold().replace("&", " and ")
    return " ".join(TOKEN_RE.findall(text))


def _content_items(value: Any, path: str = "", association: str = "") -> list[dict[str, str]]:
    """Flatten meaningful leaf values while retaining their field paths."""
    items: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in IGNORED_KEYS or str(key).startswith("_"):
                continue
            child_path = f"{path}.{key}" if path else str(key)
            items.extend(_content_items(child, child_path, association))
    elif isinstance(value, list):
        for child in value:
            items.extend(_content_items(child, path, association))
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        if text:
            items.append({"path": path, "text": text, "normalized": normalize(text),
                          "association": normalize(association)})
    return items


def _record_anchor(section: str, record: dict[str, Any]) -> str:
    preferred = {
        "experience": ("company", "employer", "organization", "organization_name"),
        "projects": ("name", "project_name", "title"),
    }.get(section, ())
    lower = {str(key).casefold(): value for key, value in record.items()}
    values = []
    for key in preferred:
        value = lower.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
            break
    if section == "education":
        degree = lower.get("degree") or lower.get("qualification") or lower.get("program")
        if isinstance(degree, str) and degree.strip():
            values.append(degree.strip())
        institution = lower.get("institution") or lower.get("school")
        if isinstance(institution, str) and institution.strip():
            values.append(institution.strip())
    return " | ".join(values)


def canonical_sections(document: Any) -> dict[str, list[dict[str, str]]]:
    result = {section: [] for section in EVALUATED_SECTIONS}
    if not isinstance(document, dict):
        return result
    for key, value in document.items():
        canonical = next((section for section, aliases in SECTION_ALIASES.items()
                          if str(key).casefold() in aliases), None)
        if canonical:
            if canonical in {"experience", "projects", "education"} and isinstance(value, list):
                for record in value:
                    if isinstance(record, dict):
                        result[canonical].extend(_content_items(
                            record, str(key), _record_anchor(canonical, record)
                        ))
                    else:
                        result[canonical].extend(_content_items(record, str(key)))
            else:
                result[canonical].extend(_content_items(value, str(key)))
    return result


def _tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    """Build deterministic TF-IDF vectors once per section, without ML dependencies."""
    counts = [Counter(normalize(text).split()) for text in texts]
    document_frequency = Counter(term for doc in counts for term in doc.keys())
    total = len(counts)
    result = []
    for doc in counts:
        result.append({term: (1 + math.log(tf)) * (1 + math.log((total + 1) / (document_frequency[term] + 1)))
                       for term, tf in doc.items()})
    return result


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    norm_a = math.sqrt(sum(value * value for value in left.values()))
    norm_b = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _pair_suggestions(predicted: list[dict[str, str]], expected: list[dict[str, str]], threshold: float,
                      other_expected: dict[str, list[dict[str, str]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact TP/FP/FN are computed elsewhere; return similarity review pairs/errors."""
    def identity(item: dict[str, str]) -> tuple[str, str]:
        return item["normalized"], item.get("association", "")
    pred_counts = Counter(identity(item) for item in predicted)
    truth_counts = Counter(identity(item) for item in expected)
    unmatched_pred = []
    unmatched_truth = []
    used: Counter[str] = Counter()
    for item in predicted:
        key = identity(item)
        if used[key] < truth_counts[key]:
            used[key] += 1
        else:
            unmatched_pred.append(item)
    used.clear()
    for item in expected:
        key = identity(item)
        if used[key] < pred_counts[key]:
            used[key] += 1
        else:
            unmatched_truth.append(item)

    suggestions: list[dict[str, Any]] = []
    classified: list[dict[str, Any]] = []
    available_truth = set(range(len(unmatched_truth)))
    vectors = _tfidf_vectors([item["text"] for item in unmatched_pred + unmatched_truth])
    truth_offset = len(unmatched_pred)
    for pred_index, pred in enumerate(unmatched_pred):
        wrongly_associated_index = next((idx for idx in sorted(available_truth)
                                         if unmatched_truth[idx]["normalized"] == pred["normalized"]
                                         and unmatched_truth[idx].get("association", "") != pred.get("association", "")), None)
        if wrongly_associated_index is not None:
            truth = unmatched_truth[wrongly_associated_index]
            available_truth.remove(wrongly_associated_index)
            classified.append({"category": "wrongly_associated_block", "predicted": pred["text"],
                               "predicted_record": pred.get("association", ""),
                               "expected": truth["text"], "expected_record": truth.get("association", "")})
            continue
        best_index, best_score = None, -1.0
        for idx in sorted(available_truth):
            score = _cosine(vectors[pred_index], vectors[truth_offset + idx])
            if score > best_score:
                best_index, best_score = idx, score
        if best_index is not None and best_score >= threshold:
            truth = unmatched_truth[best_index]
            available_truth.remove(best_index)
            suggestions.append({"predicted": pred["text"], "expected": truth["text"],
                                "cosine_similarity": round(best_score, 4),
                                "review_flag": "semantic_difference_or_corruption"})
            classified.append({"category": "corrupted_or_incorrect_value", "predicted": pred["text"],
                               "expected": truth["text"], "cosine_similarity": round(best_score, 4)})
            continue
        if best_index is not None:
            truth = unmatched_truth[best_index]
            suggestions.append({"predicted": pred["text"], "expected": truth["text"],
                                "cosine_similarity": round(best_score, 4),
                                "review_flag": "low_semantic_similarity"})
        other_match = None
        for other_section, other_items in other_expected.items():
            if other_section and any(item["normalized"] == pred["normalized"] for item in other_items):
                other_match = other_section
                break
        if other_match:
            classified.append({"category": "wrongly_associated_section", "predicted": pred["text"],
                               "expected_section": other_match})
        else:
            classified.append({"category": "extra_content", "predicted": pred["text"]})
    for idx in sorted(available_truth):
        truth = unmatched_truth[idx]
        classified.append({"category": "missing_content", "expected": truth["text"], "path": truth["path"]})
    return suggestions, classified


def compare_documents(predicted_doc: Any, truth_doc: Any, threshold: float) -> dict[str, Any]:
    predicted = canonical_sections(predicted_doc)
    expected = canonical_sections(truth_doc)
    section_results: dict[str, Any] = {}
    overall_counts = Counter()
    for section in EVALUATED_SECTIONS:
        pred_items = predicted[section]
        truth_items = expected[section]
        truth_has_section = isinstance(truth_doc, dict) and any(
            str(key).casefold() in SECTION_ALIASES[section] for key in truth_doc
        )
        if not truth_items:
            # An absent/empty reference section is not a positive labeled sample;
            # do not turn it into an artificial 100% score (or assume omission).
            empty_reference_errors = ([{"category": "extra_content", "predicted": item["text"]}
                                       for item in pred_items] if truth_has_section else [])
            section_results[section] = {
                "status": "no_reference_items" if truth_has_section else "not_annotated",
                "precision": None, "recall": None, "f1": None,
                "true_positive": 0, "false_positive": 0, "false_negative": 0,
                "predicted_item_count": len(pred_items), "ground_truth_item_count": 0,
                "missing_content": [], "error_categories": empty_reference_errors, "cosine_review_flags": [],
            }
            continue
        def identity(item: dict[str, str]) -> tuple[str, str]:
            return item["normalized"], item.get("association", "")
        pred_counts = Counter(identity(item) for item in pred_items if item["normalized"])
        truth_counts = Counter(identity(item) for item in truth_items if item["normalized"])
        tp = sum((pred_counts & truth_counts).values())
        fp = sum((pred_counts - truth_counts).values())
        fn = sum((truth_counts - pred_counts).values())
        precision = tp / (tp + fp) if tp + fp else (1.0 if not truth_items else 0.0)
        recall = tp / (tp + fn) if tp + fn else (1.0 if not pred_items else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        other = {name: items for name, items in expected.items() if name != section}
        similarity_flags, errors = _pair_suggestions(pred_items, truth_items, threshold, other)
        # Explicitly list exact missing content, preserving source field paths.
        missing = []
        available = pred_counts.copy()
        for item in truth_items:
            key = identity(item)
            if available[key]:
                available[key] -= 1
            else:
                missing.append({"path": item["path"], "text": item["text"]})
        section_results[section] = {
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "predicted_item_count": len(pred_items), "ground_truth_item_count": len(truth_items),
            "missing_content": missing, "error_categories": errors,
            "cosine_review_flags": similarity_flags,
        }
        overall_counts.update({"tp": tp, "fp": fp, "fn": fn, "evaluated_sections": 1})

    tp, fp, fn = overall_counts["tp"], overall_counts["fp"], overall_counts["fn"]
    if overall_counts["evaluated_sections"]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        precision = recall = f1 = None
    def other_values(document: Any) -> list[dict[str, str]]:
        if not isinstance(document, dict):
            return []
        evaluated_keys = set().union(*SECTION_ALIASES.values())
        remainder = {key: value for key, value in document.items()
                     if str(key).casefold() not in evaluated_keys and str(key).casefold() != "entities"}
        return _content_items(remainder, "root")
    pred_other, truth_other = other_values(predicted_doc), other_values(truth_doc)
    pred_other_counts = Counter(item["normalized"] for item in pred_other)
    truth_other_counts = Counter(item["normalized"] for item in truth_other)
    other_missing = list((truth_other_counts - pred_other_counts).elements())
    other_extra = list((pred_other_counts - truth_other_counts).elements())
    return {
        "status": "scored" if f1 is not None else "no_reference_labels",
        "overall": {"precision": round(precision, 4) if precision is not None else None,
                    "recall": round(recall, 4) if recall is not None else None,
                    "f1": round(f1, 4) if f1 is not None else None, "true_positive": tp,
                                         "false_positive": fp, "false_negative": fn},
        "sections": section_results,
        "structure": compare_structure(predicted_doc, truth_doc),
        "entities": compare_entity_lists(predicted_doc, truth_doc),
        "other_values": {"missing_count": len(other_missing), "extra_count": len(other_extra),
                         "missing_values": other_missing, "extra_values": other_extra},
    }


def compare_structure(predicted: Any, truth: Any) -> dict[str, Any]:
    def shape(node: Any, prefix: str = "$") -> dict[str, str]:
        found = {prefix: ("object" if isinstance(node, dict) else "array" if isinstance(node, list)
                          else "null" if node is None else "boolean" if isinstance(node, bool)
                          else "number" if isinstance(node, (int, float)) else "string")}
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).casefold() in IGNORED_KEYS or str(key).startswith("_"):
                    continue
                path = f"{prefix}.{key}" if prefix else str(key)
                found.update(shape(value, path))
        elif isinstance(node, list):
            for value in node:
                found.update(shape(value, prefix + "[]"))
        return found
    pred_shape, truth_shape = shape(predicted), shape(truth)
    pred_keys, truth_keys = set(pred_shape), set(truth_shape)
    type_mismatches = [{"path": path, "predicted_type": pred_shape[path], "expected_type": truth_shape[path]}
                       for path in sorted(pred_keys & truth_keys) if pred_shape[path] != truth_shape[path]]
    return {"missing_paths": sorted(truth_keys - pred_keys), "extra_paths": sorted(pred_keys - truth_keys),
            "missing_path_count": len(truth_keys - pred_keys), "extra_path_count": len(pred_keys - truth_keys),
            "type_mismatches": type_mismatches, "type_mismatch_count": len(type_mismatches)}


def _find_entity_values(document: Any) -> list[str] | None:
    if not isinstance(document, dict) or "entities" not in document:
        return None
    entities = document.get("entities")
    if isinstance(entities, dict):
        entities = entities.get("entities", [])
    if not isinstance(entities, list):
        return None
    values = []
    for entity in entities:
        if isinstance(entity, str):
            values.append(entity)
        elif isinstance(entity, dict):
            value = entity.get("text") or entity.get("label") or entity.get("name")
            if value:
                values.append(str(value))
    return values


def compare_entity_lists(predicted: Any, truth: Any) -> dict[str, Any]:
    predicted_entities, truth_entities = _find_entity_values(predicted), _find_entity_values(truth)
    if predicted_entities is None or truth_entities is None:
        return {"status": "unavailable", "reason": "Both JSON documents must include an entities array to score entities."}
    if not truth_entities:
        return {"status": "no_reference_entities", "reason": "The reference entities array has no labeled entities.",
                "extra_entities": predicted_entities}
    pred = Counter(normalize(item) for item in predicted_entities)
    expected = Counter(normalize(item) for item in truth_entities)
    tp, fp, fn = sum((pred & expected).values()), sum((pred - expected).values()), sum((expected - pred).values())
    precision = tp / (tp + fp) if tp + fp else (1.0 if not truth_entities else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not predicted_entities else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"status": "scored", "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "extra_entities": list((pred - expected).elements()), "missing_entities": list((expected - pred).elements())}


def ground_truth_path(root: Path, resume_path: Path, input_root: Path) -> Path:
    relative = resume_path.relative_to(input_root)
    # Ground truth mirrors source directories; only the extension changes to .json.
    candidate = (root / relative).with_suffix(".json")
    if candidate.is_file():
        return candidate
    # Also accept a flat JSON directory with the source basename.
    flat = root / f"{resume_path.stem}.json"
    return flat if flat.is_file() else candidate


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def evaluate_resume(path: Path, input_root: Path, output_dir: Path, truth_root: Path | None,
                    similarity_threshold: float) -> dict[str, Any]:
    started = time.perf_counter()
    relative = path.relative_to(input_root)
    artifact_dir = output_dir / "extracted" / relative.parent / safe_stem(f"{path.stem}_{path.suffix.lstrip('.').casefold()}")
    result: dict[str, Any] = {"filename": path.name, "relative_path": relative.as_posix(),
                              "status": "failed", "failure_stage": "extraction", "failure_reason": "",
                              "ground_truth_status": "not_configured", "overall_f1": None,
                              "experience_f1": None, "projects_f1": None, "skills_f1": None, "education_f1": None,
                              "entity_f1": None, "embedding_method": "not_reached",
                              "artifact_directory": str(artifact_dir),
                              "elapsed_seconds": 0.0}
    try:
        text, extraction_method = extract_resume(path)
        if not text.strip():
            raise ValueError("No text extracted from resume.")
        artifacts = run_pipeline(text, safe_stem(path.stem), str(artifact_dir), candidate_name="Candidate", debug=False)
        for filename, payload in artifacts.items():
            save_json(artifact_dir / filename, payload)
        result["embedding_method"] = artifacts.get("embeddings.json", {}).get("model", "unknown")
        result.update({"status": "processed", "failure_stage": "", "extraction_method": extraction_method,
                       "text_char_count": len(text), "artifact_directory": str(artifact_dir)})
        canonical = artifacts.get("canonical_resume.json")
        if not isinstance(canonical, dict):
            raise ValueError("Pipeline did not return canonical_resume.json as an object.")
        if truth_root is None:
            result["ground_truth_status"] = "not_configured"
        else:
            reference_path = ground_truth_path(truth_root, path, input_root)
            if not reference_path.is_file():
                result["ground_truth_status"] = "missing_file"
                result["failure_reason"] = f"Ground-truth JSON not found: {reference_path}"
            else:
                truth = None
                try:
                    truth = json.loads(reference_path.read_text(encoding="utf-8-sig"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    result["ground_truth_status"] = "invalid_json"
                    result["failure_reason"] = f"Invalid ground-truth JSON: {type(exc).__name__}: {exc}"
                else:
                    if not isinstance(truth, dict):
                        result["ground_truth_status"] = "invalid_schema"
                        result["failure_reason"] = "Ground-truth JSON root must be an object."
                        truth = None
                if isinstance(truth, dict):
                    try:
                        prediction_for_scoring = dict(canonical)
                        extracted_entities = artifacts.get("entities.json", {}).get("entities", [])
                        prediction_for_scoring["entities"] = extracted_entities
                        scores = compare_documents(prediction_for_scoring, truth, similarity_threshold)
                        result.update({"ground_truth_status": scores["status"], "ground_truth_file": str(reference_path),
                                       "overall_f1": scores["overall"]["f1"]})
                        result.update({f"{section}_f1": scores["sections"][section]["f1"] for section in EVALUATED_SECTIONS})
                        if scores["entities"].get("status") == "scored":
                            result["entity_f1"] = scores["entities"]["f1"]
                        save_json(artifact_dir / "evaluation.json", scores)
                    except Exception as exc:
                        result["ground_truth_status"] = "evaluation_failed"
                        result["failure_reason"] = f"Comparison error: {type(exc).__name__}: {exc}"
    except Exception as exc:  # Each resume is isolated so a bad file does not abort the batch.
        result["failure_stage"] = result.get("failure_stage") or "pipeline"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "failed"
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    save_json(artifact_dir / "processing.json", result)
    return result


def write_report(output_dir: Path, rows: list[dict[str, Any]], truth_configured: bool,
                 similarity_threshold: float, total_files: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resume_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    save_json(output_dir / "resume_results.json", rows)

    scored = [row for row in rows if row.get("ground_truth_status") == "scored"]
    status_counts = Counter(row.get("status", "unknown") for row in rows)
    gt_counts = Counter(row.get("ground_truth_status", "unknown") for row in rows)
    embedding_methods = Counter(row.get("embedding_method", "unknown") for row in rows)
    section_totals = {section: Counter() for section in EVALUATED_SECTIONS}
    structure_totals = Counter()
    entity_totals = Counter()
    error_totals = Counter()
    other_value_totals = Counter()
    similarity_flags = 0
    for row in scored:
        eval_path = Path(row["artifact_directory"]) / "evaluation.json"
        try:
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for section in EVALUATED_SECTIONS:
            metrics = evaluation["sections"][section]
            section_totals[section].update({"tp": metrics["true_positive"], "fp": metrics["false_positive"], "fn": metrics["false_negative"]})
            if metrics.get("status") == "scored":
                section_totals[section]["evaluated"] += 1
            similarity_flags += len(metrics["cosine_review_flags"])
            error_totals.update(item["category"] for item in metrics["error_categories"])
        structure_totals.update({"missing_paths": evaluation["structure"]["missing_path_count"],
                                 "extra_paths": evaluation["structure"]["extra_path_count"],
                                 "type_mismatches": evaluation["structure"]["type_mismatch_count"]})
        other_value_totals.update({"missing": evaluation["other_values"]["missing_count"],
                                   "extra": evaluation["other_values"]["extra_count"]})
        if evaluation["entities"].get("status") == "scored":
            entity_totals.update({"tp": evaluation["entities"]["true_positive"],
                                  "fp": evaluation["entities"]["false_positive"],
                                  "fn": evaluation["entities"]["false_negative"]})

    def metrics(counts: Counter, require_evaluated: bool = False) -> tuple[float | None, float | None, float | None]:
        if require_evaluated and not counts["evaluated"]:
            return None, None, None
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else (1.0 if tp + fp == 0 else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    lines = ["# Hybrid Resume Block Evaluation", "", f"- Resumes discovered: **{total_files}**",
             f"- Resumes processed: **{len(rows)}**", f"- Processing failures: **{status_counts['failed']}**",
             f"- Ground-truth files scored: **{gt_counts['scored']}**",
             f"- Ground-truth missing: **{gt_counts['missing_file']}**",
             f"- Ground-truth invalid: **{gt_counts['invalid_json']}**",
             f"- Ground-truth schema errors: **{gt_counts['invalid_schema']}**",
             f"- Comparison errors: **{gt_counts['evaluation_failed']}**",
             f"- Embedding methods: **{'; '.join(f'{name}: {count}' for name, count in sorted(embedding_methods.items()))}**",
             f"- Valid references with no labeled block items: **{gt_counts['no_reference_labels']}**",
             f"- Similarity threshold for review flags: **{similarity_threshold:.2f}**", ""]
    if not truth_configured:
        lines.extend(["## Accuracy status", "", "**Not measured:** no `--ground-truth-dir` was supplied. The script does not treat prior parser outputs as ground truth.",
                      "Run again with a directory of manually verified JSON files to calculate precision, recall, and F1.", ""])
    elif not scored:
        if gt_counts["no_reference_labels"]:
            reason = "Valid reference JSON was found, but none contains labeled items for the scored blocks."
        else:
            reason = "No valid matching ground-truth JSON files were available."
        lines.extend(["## Accuracy status", "", f"**Not measured:** {reason}", ""])
    else:
        lines.extend(["## Section-level block metrics", "", "Scores aggregate exact normalized content-item matches across scored resumes. Cosine similarity is only a review flag and does not change these counts.",
                      "", "| Block/section | Precision | Recall | F1 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"])
        def metric_text(value: float | None) -> str:
            return f"{value:.1%}" if value is not None else "N/A"
        for section in EVALUATED_SECTIONS:
            p, r, f = metrics(section_totals[section], require_evaluated=True)
            c = section_totals[section]
            lines.append(f"| {section.title()} | {metric_text(p)} | {metric_text(r)} | {metric_text(f)} | {c['tp']} | {c['fp']} | {c['fn']} |")
        p, r, f = metrics(Counter({"tp": sum(c["tp"] for c in section_totals.values()),
                                   "fp": sum(c["fp"] for c in section_totals.values()),
                                   "fn": sum(c["fn"] for c in section_totals.values())}))
        lines.extend([f"| **Overall** | **{metric_text(p)}** | **{metric_text(r)}** | **{metric_text(f)}** | **{sum(c['tp'] for c in section_totals.values())}** | **{sum(c['fp'] for c in section_totals.values())}** | **{sum(c['fn'] for c in section_totals.values())}** |",
                      "", f"Section cosine similarity review flags: **{similarity_flags}**. Entity scores are calculated only when both reference and extracted JSON contain an `entities` array.", ""])
        if entity_totals:
            ep, er, ef = metrics(entity_totals)
            lines.append(f"Entity precision/recall/F1: **{metric_text(ep)} / {metric_text(er)} / {metric_text(ef)}** (aggregated exact normalized entity matches).")
            lines.append("")
        lines.extend(["## Error categories", ""])
        lines.extend([f"- {name.replace('_', ' ')}: **{count}**" for name, count in error_totals.most_common()] or ["No section content discrepancies found."])
        lines.extend([f"- Missing structural paths: **{structure_totals['missing_paths']}**",
                      f"- Extra structural paths: **{structure_totals['extra_paths']}**",
                      f"- Structural type mismatches: **{structure_totals['type_mismatches']}**",
                      f"- Missing non-section values: **{other_value_totals['missing']}**",
                      f"- Extra non-section values: **{other_value_totals['extra']}**", ""])
    lines.extend(["## Per-resume results", "", "| Resume | Processing | Reference | Overall F1 | Experience F1 | Projects F1 | Skills F1 | Education F1 | Entity F1 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        def pct(value: Any) -> str:
            return f"{float(value):.1%}" if value is not None else "N/A"
        lines.append(f"| {row.get('relative_path', row.get('filename'))} | {row.get('status')} | {row.get('ground_truth_status')} | {pct(row.get('overall_f1'))} | {pct(row.get('experience_f1'))} | {pct(row.get('projects_f1'))} | {pct(row.get('skills_f1'))} | {pct(row.get('education_f1'))} | {pct(row.get('entity_f1'))} |")
    lines.extend(["", "## Artifact layout", "", "- `extracted/<relative path>/<resume stem>/`: every JSON artifact returned by the existing pipeline.",
                  "- `evaluation.json` in each scored resume folder: section metrics, structural differences, entity metrics, missing/extra/corrupted/wrong-section content, and cosine review suggestions.",
                  "- `resume_results.csv` and `resume_results.json`: consolidated per-resume results.",
                  "- Cosine similarity is advisory only; exact normalized content matches determine TP/FP/FN.",
                  "- Only manually verified reference JSON is accepted as ground truth. No reference means accuracy is reported as not measured.", ""])
    (output_dir / "hybrid_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Resume folder to evaluate.")
    parser.add_argument("--ground-truth-dir", type=Path, help="Manually verified JSON directory. Mirror resume subfolders; e.g. Name.pdf -> Name.json.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "hybrid_evaluation", help="Evaluation outputs and extracted artifacts.")
    parser.add_argument("--similarity-threshold", type=float, default=0.72, help="Cosine threshold for human review suggestions (0-1).")
    parser.add_argument("--max-files", type=int, default=0, help="Optional limit for a quick run; 0 evaluates all resumes.")
    args = parser.parse_args(argv)
    input_root = args.input_dir.expanduser().resolve()
    if not input_root.is_dir():
        parser.error(f"Input directory does not exist: {input_root}")
    truth_root = args.ground_truth_dir.expanduser().resolve() if args.ground_truth_dir else None
    if truth_root is not None and not truth_root.is_dir():
        parser.error(f"Ground-truth directory does not exist: {truth_root}")
    if not 0 <= args.similarity_threshold <= 1:
        parser.error("--similarity-threshold must be between 0 and 1")
    if args.max_files < 0:
        parser.error("--max-files must be zero or positive")

    output_base = args.output_dir.expanduser().resolve()
    output_dir = output_base
    if output_dir.exists() and any(output_dir.iterdir()):
        from datetime import datetime, timezone
        suffix = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        output_dir = output_base.with_name(f"{output_base.name}_{suffix}")
        counter = 1
        while output_dir.exists():
            output_dir = output_base.with_name(f"{output_base.name}_{suffix}_{counter}")
            counter += 1
    resumes = sorted((path for path in input_root.rglob("*") if path.is_file()
                      and path.suffix.casefold() in SUPPORTED_SUFFIXES and not path.name.startswith("~$")),
                     key=lambda path: path.relative_to(input_root).as_posix().casefold())
    if output_dir == input_root or input_root in output_dir.parents:
        resumes = [path for path in resumes if output_dir not in path.resolve().parents]
    total_files = len(resumes)
    if args.max_files:
        resumes = resumes[:args.max_files]
    if not resumes:
        parser.error(f"No supported resume files found in {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for index, path in enumerate(resumes, 1):
        print(f"[{index}/{len(resumes)}] {path.relative_to(input_root)}", flush=True)
        row = evaluate_resume(path, input_root, output_dir, truth_root, args.similarity_threshold)
        rows.append(row)
        status = row["status"]
        reference = row["ground_truth_status"]
        print(f"    {status} | reference={reference} | embedding={row.get('embedding_method')} | {row['elapsed_seconds']:.2f}s")
        if index % 25 == 0 or index == len(resumes):
            write_report(output_dir, rows, truth_root is not None, args.similarity_threshold, total_files)

    write_report(output_dir, rows, truth_root is not None, args.similarity_threshold, total_files)
    failures = sum(row["status"] != "processed" for row in rows)
    scored = sum(row["ground_truth_status"] == "scored" for row in rows)
    print(f"\nHybrid evaluation complete: {len(rows)} processed; {failures} processing failure(s); {scored} reference comparison(s).")
    if truth_root is None:
        print("Accuracy: NOT MEASURED (no manually verified ground-truth directory supplied).")
    elif not scored:
        print("Accuracy: NOT MEASURED (no valid matching ground-truth JSON files found).")
    else:
        report = output_dir / "hybrid_evaluation_report.md"
        print(f"Metrics and review details: {report}")
    print(f"Extracted JSON artifacts: {output_dir / 'extracted'}")
    print(f"Per-resume results: {output_dir / 'resume_results.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
