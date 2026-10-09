"""Batch evaluation harness for the resume-processing pipeline.

Evaluates (does NOT modify) the actual project flow:

    RESUME FILE -> text extraction -> run_pipeline() -> artifacts
    run_pipeline = TextRepository -> preprocess -> detect_blocks
                   -> extract_entities -> extract_relationships
                   -> embed_entities -> cosine_similarity_matrix
                   -> cluster -> build_graph

Usage:
    python scripts/batch_pipeline_test.py                  # full dataset
    python scripts/batch_pipeline_test.py --limit 10       # smoke test
    python scripts/batch_pipeline_test.py --resume-id foo  # single resume
    python scripts/batch_pipeline_test.py --start 0 --end 100

NOTE on extraction: the project itself only implements PDF text extraction
(extract_pdf_text via PyMuPDF). DOCX extraction (python-docx) and legacy-DOC
fallback handling live in THIS evaluation script so the .docx/.doc resumes in
the dataset can reach the actual run_pipeline() flow. The parsing pipeline
itself is called unmodified.
"""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# Resolve project imports (script lives in <repo>/scripts/)
# --------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
sys.path.insert(0, os.path.join(BACKEND_DIR, "app"))
sys.path.insert(0, BACKEND_DIR)

from pipeline.pipeline import run_pipeline  # noqa: E402  (actual project flow)

SCRIPT_VERSION = "1.0.0"

# Heuristic diagnostic cutoffs (NOT accuracy measurements; labelled as such).
MIN_CHARS_OK = 300          # below this, extraction is "suspiciously short"
EMPTY_STRUCTURE_SECTIONS = 1  # <=1 section means only Header/Other present

FAILURE_STAGES = [
    "EXTRACTION", "PREPROCESSING", "SECTION_DETECTION", "BLOCK_SEGMENTATION",
    "SEMANTIC_CLASSIFICATION", "ENTITY_EXTRACTION", "RELATIONSHIP_EXTRACTION",
    "STRUCTURED_OUTPUT", "CANONICALIZATION", "EMBEDDING", "SIMILARITY",
    "CLUSTERING", "GRAPH_BUILDING", "OUTPUT_VALIDATION", "UNKNOWN",
]


# --------------------------------------------------------------------------
# Extraction (evaluation-side; project natively supports PDF via PyMuPDF)
# --------------------------------------------------------------------------
def extract_pdf_text(path):
    """Same logic as pipeline.extract_pdf_text (PyMuPDF page.get_text)."""
    import fitz
    doc = fitz.open(path)
    pages = doc.page_count
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text, {"pages": pages, "method": "pymupdf"}


def extract_docx_text(path):
    import docx
    document = docx.Document(path)
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts), {"pages": None, "method": "python-docx"}


def extract_legacy_doc_text(path):
    """Best-effort fallback for OLE .doc files (no antiword/libreoffice here).

    Tries python-docx first (works if mislabelled), then scans the raw bytes
    for UTF-16LE text runs. Raises if nothing usable is recovered.
    """
    try:
        return extract_docx_text(path)
    except Exception:
        pass
    with open(path, "rb") as f:
        raw = f.read()
    # Scan for UTF-16LE runs of printable ASCII (typical WordDocument content).
    runs = re.findall(
        rb"(?:[\x20-\x7e]\x00){4,}", raw)
    texts = []
    for run in runs:
        try:
            s = run.decode("utf-16-le").strip()
        except Exception:
            continue
        # skip binary/table-control garbage: require some lowercase/vowel mix
        if len(s) >= 4 and re.search(r"[a-z]{2,}", s):
            texts.append(s)
    text = "\n".join(texts)
    # de-duplicate repeated header/footer fragments while keeping order
    seen, deduped = set(), []
    for line in text.split("\n"):
        key = line.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(line)
    text = "\n".join(deduped)
    if len(text.strip()) < 50:
        raise ValueError(
            "legacy .doc: no usable text recovered (ole-text-scan yielded "
            f"{len(text.strip())} chars; antiword/libreoffice not installed)")
    return text, {"pages": None, "method": "ole-text-scan-fallback"}


def extract_text(path, ext):
    if ext == ".pdf":
        return extract_pdf_text(path)
    if ext == ".docx":
        return extract_docx_text(path)
    if ext == ".doc":
        return extract_legacy_doc_text(path)
    if ext in (".txt", ".md", ".json"):
        with open(path, "r", errors="replace") as f:
            return f.read(), {"pages": None, "method": "plain-text"}
    raise ValueError(f"unsupported file format '{ext}'")


# --------------------------------------------------------------------------
# Failure-stage inference from tracebacks
# --------------------------------------------------------------------------
TRACEBACK_STAGE_MAP = [
    ("preprocessing", "PREPROCESSING"),
    ("segmentation", "BLOCK_SEGMENTATION"),
    ("detect_blocks", "BLOCK_SEGMENTATION"),
    ("entity_extraction", "ENTITY_EXTRACTION"),
    ("extract_entities", "ENTITY_EXTRACTION"),
    ("relation_extraction", "RELATIONSHIP_EXTRACTION"),
    ("extract_relationships", "RELATIONSHIP_EXTRACTION"),
    ("embeddings", "EMBEDDING"),
    ("embed_entities", "EMBEDDING"),
    ("similarity", "SIMILARITY"),
    ("clustering", "CLUSTERING"),
    ("graph_builder", "GRAPH_BUILDING"),
    ("build_graph", "GRAPH_BUILDING"),
    ("_build_block_audit", "STRUCTURED_OUTPUT"),
    ("_build_section_keyed", "CANONICALIZATION"),
    ("text_repository", "PREPROCESSING"),
]


def infer_stage_from_traceback(tb_str):
    lowered = tb_str.lower()
    for needle, stage in TRACEBACK_STAGE_MAP:
        if needle in lowered:
            return stage
    return "UNKNOWN"


# --------------------------------------------------------------------------
# Single-resume processing
# --------------------------------------------------------------------------
def process_one(path, resume_id, out_tmp_base, debug=False):
    t0 = time.time()
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower()
    size = os.path.getsize(path)
    row = {
        "resume_id": resume_id, "filename": filename, "extension": ext,
        "file_size": size,
    }

    # ---- A. extraction ----
    try:
        raw_text, ext_info = extract_text(path, ext)
        extraction_success = True
        error_message = ""
        tb_str = ""
    except Exception as e:  # noqa: BLE001
        raw_text, ext_info = "", {"pages": None, "method": "none"}
        extraction_success = False
        error_message = f"{type(e).__name__}: {e}"
        tb_str = traceback.format_exc()
    chars = len(raw_text)
    lines = raw_text.count("\n") + 1 if raw_text else 0
    row.update({
        "extraction_success": int(extraction_success),
        "extraction_method": ext_info.get("method", ""),
        "extracted_characters": chars,
        "extracted_lines": lines,
        "extracted_pages": ext_info.get("pages") if ext_info.get("pages") is not None else "",
    })
    if debug:
        print(f"  [extract] {filename}: success={extraction_success} "
              f"chars={chars} method={ext_info.get('method')}")

    if not extraction_success or not raw_text.strip() or chars < 50:
        row.update({
            "status": "EXTRACTION_FAILED", "failure_stage": "EXTRACTION",
            "failure_type": "exception" if not extraction_success else "empty_text",
            "error_message": error_message or "extracted text empty (<50 chars)",
            "traceback": tb_str,
            "processing_time_seconds": round(time.time() - t0, 3),
        })
        row.update(empty_metrics())
        return row, None, raw_text

    # ---- B..H actual pipeline ----
    doc_id = os.path.splitext(filename)[0][:80] or resume_id
    tmp_dir = os.path.join(out_tmp_base, resume_id)
    try:
        artifacts = run_pipeline(raw_text, doc_id, tmp_dir,
                                 candidate_name=doc_id)
        error_message, tb_str = "", ""
        pipeline_ok = True
    except Exception as e:  # noqa: BLE001
        artifacts = None
        error_message = f"{type(e).__name__}: {e}"
        tb_str = traceback.format_exc()
        pipeline_ok = False
    duration = round(time.time() - t0, 3)
    row["processing_time_seconds"] = duration

    if not pipeline_ok:
        row.update({
            "status": "FAILED",
            "failure_stage": infer_stage_from_traceback(tb_str),
            "failure_type": "exception",
            "error_message": error_message[:500],
            "traceback": tb_str,
        })
        row.update(empty_metrics())
        return row, tmp_dir, raw_text

    # ---- metrics from actual artifacts ----
    try:
        m = collect_metrics(artifacts, raw_text)
    except Exception as e:  # noqa: BLE001
        row.update({
            "status": "FAILED", "failure_stage": "OUTPUT_VALIDATION",
            "failure_type": "metric_collection_error",
            "error_message": f"{type(e).__name__}: {e}"[:500],
            "traceback": traceback.format_exc(),
        })
        row.update(empty_metrics())
        return row, tmp_dir, raw_text
    row.update(m)
    row["error_message"] = ""
    row["traceback"] = ""

    # ---- classification ----
    status, stage, warnings = classify(row)
    row["status"] = status
    row["failure_stage"] = stage
    row["failure_type"] = "warning" if status == "PASS_WITH_WARNINGS" else (
        "partial" if status == "PARTIAL" else "")
    row["warnings"] = "; ".join(warnings)
    if debug:
        print(f"  [result] {filename}: {status} stage={stage} "
              f"blocks={row['block_count']} ent={row['entity_count']} "
              f"rel={row['relationship_count']} edges={row['graph_edge_count']} "
              f"{duration}s")
    return row, tmp_dir, raw_text


def empty_metrics():
    return {
        "preprocessing_success": 0, "sentence_count": 0,
        "section_count": 0, "block_count": 0, "classified_block_count": 0,
        "unknown_block_count": 0, "entity_count": 0, "relationship_count": 0,
        "assigned_sentence_count": 0, "assigned_source_line_count": 0,
        "source_line_count": 0, "assigned_entity_count": 0,
        "assigned_relationship_count": 0, "unassigned_sentence_count": 0,
        "unassigned_source_line_count": 0, "sentence_coverage": "",
        "source_line_coverage": "", "canonical_output_success": 0,
        "canonical_section_keys": "", "graph_output_success": 0,
        "graph_node_count": 0, "graph_edge_count": 0,
        "embedding_model": "", "cluster_count": "",
        "warnings": "", "output_directory": "",
    }


def collect_metrics(artifacts, raw_text):
    raw = artifacts.get("raw_text.json", {})
    blocks = artifacts.get("blocks.json", {})
    cov = blocks.get("coverage", {})
    audit = artifacts.get("block_audit.json", {})
    asum = audit.get("summary", {})
    entities_doc = artifacts.get("entities.json", {})
    rels_doc = artifacts.get("relationships.json", {})
    graph = artifacts.get("graph.json", {})
    gstats = graph.get("stats", {})
    resume_blocks = artifacts.get("resume_blocks.json", {})
    emb = artifacts.get("embeddings.json", {})
    clusters = artifacts.get("clusters.json", {})

    pre = entities_doc.get("preprocessing", {})
    sentences = pre.get("sentence_count", cov.get("sentence_count", 0))

    block_list = blocks.get("blocks", [])
    # entry_type 'source_line_context' = fallback block with no structural role
    classified = sum(1 for b in block_list
                     if b.get("entry_type") != "source_line_context")
    unknown = len(block_list) - classified

    section_keys = sorted(k for k in resume_blocks.keys() if not k.startswith("_"))
    n_nodes = gstats.get("node_count", len(graph.get("nodes", [])))
    n_edges = gstats.get("edge_count", len(graph.get("edges", [])))

    sent_total = cov.get("sentence_count", 0) or 0
    sent_assigned = cov.get("assigned_sentence_count", 0) or 0
    line_total = asum.get("source_line_count", cov.get("source_line_count", 0)) or 0
    line_assigned = asum.get("assigned_source_line_count",
                             cov.get("assigned_source_line_count", 0)) or 0
    n_ent = entities_doc.get("count", 0)
    n_rel = rels_doc.get("count", 0)

    cl = clusters
    if isinstance(cl, dict):
        n_clusters = cl.get("n_clusters", cl.get("k", ""))
    else:
        n_clusters = ""

    return {
        "preprocessing_success": int(sentences > 0),
        "sentence_count": sentences,
        "section_count": asum.get("section_count", len(blocks.get("sections", []))),
        "block_count": blocks.get("block_count", len(block_list)),
        "classified_block_count": classified,
        "unknown_block_count": unknown,
        "entity_count": n_ent,
        "relationship_count": n_rel,
        "assigned_sentence_count": sent_assigned,
        "assigned_source_line_count": line_assigned,
        "source_line_count": line_total,
        "assigned_entity_count": cov.get("assigned_entity_count", ""),
        "assigned_relationship_count": cov.get("assigned_relationship_count", ""),
        "unassigned_sentence_count": sent_total - sent_assigned,
        "unassigned_source_line_count": line_total - line_assigned,
        "sentence_coverage": round(sent_assigned / sent_total, 4) if sent_total else "",
        "source_line_coverage": round(line_assigned / line_total, 4) if line_total else "",
        "canonical_output_success": int(bool(section_keys)),
        "canonical_section_keys": "|".join(section_keys),
        "graph_output_success": int(bool(n_nodes)),
        "graph_node_count": n_nodes,
        "graph_edge_count": n_edges,
        "embedding_model": emb.get("model", ""),
        "cluster_count": n_clusters,
        "output_directory": "",
    }


def classify(r):
    """Status rules (documented in report):
    FAILED/EXTRACTION_FAILED: exception or unusable text.
    PARTIAL: completed but structurally empty (no sections/blocks/sentences,
             or zero entities AND zero relationships with content present).
    PASS_WITH_WARNINGS: completed, but >=1 heuristic diagnostic fires.
    PASS: completed with no diagnostics.
    failure_stage = earliest stage showing evidence of a problem.
    """
    warnings = []
    if r["extracted_characters"] < MIN_CHARS_OK:
        warnings.append(f"short_extraction(<{MIN_CHARS_OK}chars)")
    if r["sentence_count"] == 0:
        warnings.append("no_sentences(preprocessing_empty)")
    if r["section_count"] <= EMPTY_STRUCTURE_SECTIONS:
        warnings.append("weak_section_detection(<=1section)")
    if r["block_count"] == 0:
        warnings.append("no_blocks")
    if r["block_count"] and r["unknown_block_count"] == r["block_count"]:
        warnings.append("all_blocks_unclassified_fallback")
    try:
        slc = float(r["source_line_coverage"])
        if r["source_line_count"] and slc < 0.8:
            warnings.append(f"low_source_line_coverage({slc})")
    except (TypeError, ValueError):
        pass
    if r["entity_count"] <= 1:
        warnings.append("no_entities_beyond_person")
    if r["relationship_count"] == 0:
        warnings.append("no_relationships")
    if not r["canonical_output_success"]:
        warnings.append("empty_structured_output")
    if not r["graph_output_success"]:
        warnings.append("empty_graph")
    elif r["graph_edge_count"] == 0:
        warnings.append("graph_has_nodes_but_no_edges")

    # earliest failure stage supported by evidence
    if r["sentence_count"] == 0:
        stage = "PREPROCESSING"
    elif r["section_count"] <= EMPTY_STRUCTURE_SECTIONS or r["block_count"] == 0:
        stage = "SECTION_DETECTION"
    elif r["entity_count"] <= 1:
        stage = "ENTITY_EXTRACTION"
    elif r["relationship_count"] == 0:
        stage = "RELATIONSHIP_EXTRACTION"
    elif not r["canonical_output_success"]:
        stage = "STRUCTURED_OUTPUT"
    elif not r["graph_output_success"] or r["graph_edge_count"] == 0:
        stage = "GRAPH_BUILDING"
    else:
        stage = "—"
    # warnings that do not trigger the hard conditions above still point at
    # the earliest suspicious stage (heuristic diagnostics, not accuracy)
    if stage == "—" and warnings:
        wtxt = " ".join(warnings)
        if "coverage" in wtxt or "unclassified" in wtxt or "no_blocks" in wtxt:
            stage = "BLOCK_SEGMENTATION"
        elif "short_extraction" in wtxt:
            stage = "EXTRACTION"

    if r["sentence_count"] == 0 or r["block_count"] == 0 \
            or not r["canonical_output_success"] or not r["graph_output_success"]:
        return "PARTIAL", (stage if stage != "—" else "UNKNOWN"), warnings
    if warnings:
        return "PASS_WITH_WARNINGS", stage, warnings
    return "PASS", "—", warnings


# --------------------------------------------------------------------------
# Main batch driver
# --------------------------------------------------------------------------
CSV_COLUMNS = [
    "resume_id", "filename", "extension", "file_size", "status",
    "failure_stage", "failure_type", "error_message",
    "processing_time_seconds", "extraction_success", "extraction_method",
    "extracted_characters", "extracted_lines", "extracted_pages",
    "preprocessing_success", "section_count", "block_count",
    "classified_block_count", "unknown_block_count", "entity_count",
    "relationship_count", "assigned_sentence_count", "sentence_count",
    "assigned_source_line_count", "source_line_count",
    "assigned_entity_count", "assigned_relationship_count",
    "unassigned_sentence_count", "unassigned_source_line_count",
    "sentence_coverage", "source_line_coverage",
    "canonical_output_success", "canonical_section_keys",
    "graph_output_success", "graph_node_count", "graph_edge_count",
    "embedding_model", "cluster_count", "warnings", "output_directory",
]


def discover_resumes(input_dir):
    files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)))
    return files


def save_failure_artifacts(fail_dir, row, tmp_dir, raw_text):
    os.makedirs(fail_dir, exist_ok=True)
    with open(os.path.join(fail_dir, "error.txt"), "w") as f:
        f.write(f"resume_id: {row['resume_id']}\nfilename: {row['filename']}\n"
                f"status: {row['status']}\nfailure_stage: {row['failure_stage']}\n"
                f"failure_type: {row['failure_type']}\n"
                f"error: {row['error_message']}\nwarnings: {row.get('warnings', '')}\n")
    with open(os.path.join(fail_dir, "traceback.txt"), "w") as f:
        f.write(row.get("traceback") or "(no exception traceback)")
    meta = {k: v for k, v in row.items() if k != "traceback"}
    with open(os.path.join(fail_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    preview = (raw_text or "")[:4000]
    with open(os.path.join(fail_dir, "extracted_text_preview.txt"), "w") as f:
        f.write(preview)
    # keep pipeline artifacts when they exist (needed to reproduce/debug)
    if tmp_dir and os.path.isdir(tmp_dir):
        for fname in ("raw_text.json", "entities.json", "relationships.json",
                      "blocks.json", "block_audit.json", "resume_blocks.json",
                      "similarity.json", "clusters.json", "graph.json"):
            src = os.path.join(tmp_dir, fname)
            if os.path.exists(src) and os.path.getsize(src) < 5_000_000:
                shutil.copy2(src, os.path.join(fail_dir, fname))


def load_checkpoint(path):
    """Reload previously completed rows so a killed run can resume."""
    rows, done = [], set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add(r["resume_id"])
    return rows, done


def coerce_row(r):
    """Convert checkpoint CSV strings back to numeric types used by analysis."""
    for k in ("file_size", "extracted_characters", "extracted_lines",
              "section_count", "block_count", "classified_block_count",
              "unknown_block_count", "entity_count", "relationship_count",
              "assigned_sentence_count", "sentence_count",
              "assigned_source_line_count", "source_line_count",
              "unassigned_sentence_count", "unassigned_source_line_count",
              "graph_node_count", "graph_edge_count",
              "extraction_success", "preprocessing_success",
              "canonical_output_success", "graph_output_success"):
        if k in r and r[k] not in ("", None):
            try:
                r[k] = int(float(r[k]))
            except (TypeError, ValueError):
                pass
    for k in ("processing_time_seconds", "sentence_coverage",
              "source_line_coverage"):
        if k in r and r[k] not in ("", None):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
    for k in ("extracted_pages", "assigned_entity_count",
              "assigned_relationship_count", "cluster_count"):
        if k in r and r[k] not in ("", None):
            try:
                r[k] = int(float(r[k]))
            except (TypeError, ValueError):
                pass
    return r


def build_failure_summary(rows):
    total = len(rows)
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    stage_counts = {}
    for r in rows:
        if r["status"] in ("FAILED", "EXTRACTION_FAILED", "PARTIAL",
                            "PASS_WITH_WARNINGS"):
            st = r["failure_stage"]
            stage_counts[st] = stage_counts.get(st, 0) + 1
    err_counts = {}
    for r in rows:
        if r["error_message"]:
            key = r["error_message"][:160]
            err_counts[key] = err_counts.get(key, 0) + 1
    common_errors = sorted(err_counts.items(), key=lambda kv: -kv[1])[:15]
    times = [r["processing_time_seconds"] for r in rows]
    return {
        "total_resumes": total,
        "successful": by_status.get("PASS", 0),
        "pass_with_warnings": by_status.get("PASS_WITH_WARNINGS", 0),
        "partial": by_status.get("PARTIAL", 0),
        "failed": by_status.get("FAILED", 0),
        "extraction_failed": by_status.get("EXTRACTION_FAILED", 0),
        "extraction_failures": sum(1 for r in rows if not r["extraction_success"]),
        "structural_failures": sum(1 for r in rows if r["failure_stage"] in (
            "SECTION_DETECTION", "BLOCK_SEGMENTATION", "PREPROCESSING")),
        "entity_failures": sum(1 for r in rows if r["failure_stage"] == "ENTITY_EXTRACTION"),
        "relationship_failures": sum(1 for r in rows if r["failure_stage"] == "RELATIONSHIP_EXTRACTION"),
        "structured_output_failures": sum(1 for r in rows if r["failure_stage"] in (
            "STRUCTURED_OUTPUT", "CANONICALIZATION")),
        "graph_failures": sum(1 for r in rows if r["failure_stage"] == "GRAPH_BUILDING"),
        "unknown_failures": sum(1 for r in rows if r["failure_stage"] == "UNKNOWN"),
        "failure_stage_counts": stage_counts,
        "status_counts": by_status,
        "common_error_messages": [{"message": m, "count": c} for m, c in common_errors],
        "average_processing_time": round(statistics.mean(times), 3) if times else 0,
        "median_processing_time": round(statistics.median(times), 3) if times else 0,
        "min_processing_time": round(min(times), 3) if times else 0,
        "max_processing_time": round(max(times), 3) if times else 0,
        "total_processing_time": round(sum(times), 1) if times else 0,
    }


def get_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_report(report_path, rows, summary, meta):
    total = summary["total_resumes"]
    slowest = sorted(rows, key=lambda r: -r["processing_time_seconds"])[:10]
    weakest_cov = sorted(
        [r for r in rows if r["status"] not in ("FAILED", "EXTRACTION_FAILED")
         and r["source_line_coverage"] != ""],
        key=lambda r: (float(r["source_line_coverage"] or 0)))[:10]
    most_unknown = sorted(
        [r for r in rows if r["block_count"]],
        key=lambda r: -(r["unknown_block_count"] / max(r["block_count"], 1)))[:10]
    no_ent = [r for r in rows if r["status"] not in ("FAILED", "EXTRACTION_FAILED")
              and r["entity_count"] <= 1][:10]
    no_rel = [r for r in rows if r["status"] not in ("FAILED", "EXTRACTION_FAILED")
              and r["relationship_count"] == 0][:10]

    def pct(n):
        return f"{100.0 * n / total:.1f}%" if total else "—"

    L = []
    L.append("# Resume Batch Evaluation Report\n")
    L.append("**Scope: pipeline execution + structural/output completeness. "
             "No manually labelled ground truth exists for this dataset, so "
             "this report does NOT establish field-level parsing accuracy.**\n")
    L.append("## 1. Executive Summary\n")
    L.append(f"Total resumes tested: **{total}**  \n"
             f"PASS: **{summary['successful']}** ({pct(summary['successful'])})  \n"
             f"PASS_WITH_WARNINGS: **{summary['pass_with_warnings']}**  \n"
             f"PARTIAL: **{summary['partial']}**  \n"
             f"FAILED: **{summary['failed']}**  \n"
             f"EXTRACTION_FAILED: **{summary['extraction_failed']}**  \n")
    L.append("## 2. Dataset\n")
    L.append(f"- Dataset path: `{meta['dataset_path']}`\n"
             f"- Total files: {total}\n"
             f"- File types: {meta['file_type_distribution']}\n"
             f"- Commit: {meta['commit']} | Python: {meta['python_version']} | "
             f"Command: `{meta['command']}`\n")
    L.append("## 3. Overall Results\n")
    L.append("| Status | Count | % |\n|---|---|---|")
    for st in ("PASS", "PASS_WITH_WARNINGS", "PARTIAL", "FAILED", "EXTRACTION_FAILED"):
        n = summary["status_counts"].get(st, 0)
        L.append(f"| {st} | {n} | {pct(n)} |")
    L.append("\n## 4. Stage-wise Results (resumes passing each measurable stage)\n")
    L.append(stage_table(rows))
    L.append("\n## 5. Failure Breakdown (by earliest failure stage)\n")
    L.append("| Failure stage | Count | % |\n|---|---|---|")
    for st, n in sorted(summary["failure_stage_counts"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {st} | {n} | {pct(n)} |")
    L.append("\n## 6. Structural/Parsing Quality Indicators\n")
    L.append(quality_lines(rows))
    L.append("\n## 7. Top Failure Patterns\n")
    for item in summary["common_error_messages"][:10]:
        L.append(f"- ({item['count']}x) {item['message']}")
    for st, n in sorted(summary["failure_stage_counts"].items(), key=lambda kv: -kv[1])[:5]:
        L.append(f"- Stage {st}: {n} resumes")
    L.append("\n## 8. Worst/Most Problematic Resumes\n")
    L.append("| filename | resume_id | failure stage | reason | metrics |")
    L.append("|---|---|---|---|---|")
    bad = [r for r in rows if r["status"] in ("FAILED", "EXTRACTION_FAILED", "PARTIAL")]
    bad += weakest_cov[:5]
    seen = set()
    for r in bad[:30]:
        if r["resume_id"] in seen:
            continue
        seen.add(r["resume_id"])
        L.append(f"| {r['filename']} | {r['resume_id']} | {r['failure_stage']} | "
                 f"{(r['error_message'] or r.get('warnings', ''))[:100]} | "
                 f"chars={r['extracted_characters']} blocks={r['block_count']} "
                 f"ent={r['entity_count']} rel={r['relationship_count']} |")
    L.append("\n## 9. Slowest Resumes\n")
    L.append("| filename | seconds | status |")
    L.append("|---|---|---|")
    for r in slowest:
        L.append(f"| {r['filename']} | {r['processing_time_seconds']} | {r['status']} |")
    L.append("\n## 10. Detailed Failure Analysis\n")
    L.append(f"- Extraction failures: {summary['extraction_failures']} "
             "(incl. legacy .doc without converter, corrupt/unreadable PDFs).  \n"
             f"- Structural (preprocessing/section/block): {summary['structural_failures']}.  \n"
             f"- Entity: {summary['entity_failures']}; Relationship: "
             f"{summary['relationship_failures']}.  \n"
             f"- Structured output: {summary['structured_output_failures']}; "
             f"Graph: {summary['graph_failures']}; Unknown: {summary['unknown_failures']}.")
    L.append("\nWeakest source-line coverage (completed resumes):")
    for r in weakest_cov:
        L.append(f"- {r['filename']} coverage={r['source_line_coverage']} "
                 f"unassigned_lines={r['unassigned_source_line_count']}/{r['source_line_count']}")
    L.append("\nMost unclassified (fallback-type) blocks:")
    for r in most_unknown:
        L.append(f"- {r['filename']} unknown={r['unknown_block_count']}/{r['block_count']}")
    L.append("\n## 11. Files Requiring Investigation (inspect first)\n")
    first = [r for r in rows if r["status"] in ("FAILED", "EXTRACTION_FAILED")][:20]
    for r in first:
        L.append(f"- {r['filename']} ({r['resume_id']}) stage={r['failure_stage']} "
                 f"err={r['error_message'][:120]} artifacts=failures/{r['resume_id']}/")
    L.append("\n## 12. Conclusion\n")
    dom = max(summary["failure_stage_counts"].items(),
              key=lambda kv: kv[1])[0] if summary["failure_stage_counts"] else "none"
    L.append(f"- Successfully processed (PASS): {summary['successful']}/{total}.  \n"
             f"- Need investigation (warnings/partial/failed): "
             f"{total - summary['successful']}/{total}.  \n"
             f"- Dominant failure stage: **{dom}**.  \n"
             "- This evaluation measures pipeline execution and "
             "structural/output completeness. It does not establish "
             "field-level parsing accuracy against manually labelled ground truth.")
    L.append("\n## Appendix: status rules\n")
    L.append("- EXTRACTION_FAILED: file unreadable/unsupported or <50 chars extracted.\n"
             "- FAILED: run_pipeline raised (stage inferred from traceback).\n"
             "- PARTIAL: completed but no sentences/blocks or empty structured/graph output.\n"
             "- PASS_WITH_WARNINGS: completed with >=1 heuristic diagnostic "
             "(short text, <=1 section, low line coverage <0.8, <=1 entity, "
             "0 relationships, empty structured output, graph without edges).\n"
             "- PASS: completed with no diagnostics.")
    with open(report_path, "w") as f:
        f.write("\n".join(L) + "\n")


def stage_table(rows):
    done = [r for r in rows if r["status"] not in ("FAILED", "EXTRACTION_FAILED")]
    total = len(rows) or 1
    checks = [
        ("extraction_success", lambda r: bool(r["extraction_success"])),
        ("preprocessing (sentences>0)", lambda r: r["sentence_count"] > 0),
        ("section_detection (>1 section)", lambda r: r["section_count"] > 1),
        ("block_segmentation (blocks>0)", lambda r: r["block_count"] > 0),
        ("entity_extraction (>1 entity)", lambda r: r["entity_count"] > 1),
        ("relationship_extraction (>0)", lambda r: r["relationship_count"] > 0),
        ("structured_output (canonical keys)", lambda r: bool(r["canonical_output_success"])),
        ("graph_output (nodes>0)", lambda r: bool(r["graph_output_success"])),
        ("graph_edges (>0)", lambda r: (r["graph_edge_count"] or 0) > 0),
    ]
    lines = ["| Stage | Passed | % |", "|---|---|---|"]
    for name, fn in checks:
        n = sum(1 for r in rows if fn(r))
        lines.append(f"| {name} | {n} | {100.0*n/total:.1f}% |")
    comp = [r for r in rows if r["source_line_coverage"] != ""]
    if comp:
        import statistics as st
        avg = st.mean(float(r["source_line_coverage"]) for r in comp)
        lines.append(f"| avg source-line coverage (completed) | {avg:.3f} | — |")
    return "\n".join(lines)


def quality_lines(rows):
    done = [r for r in rows if r["status"] not in ("FAILED", "EXTRACTION_FAILED")]
    n = len(done) or 1
    import statistics as st
    def avg(key):
        vals = [r[key] for r in done if isinstance(r[key], (int, float))]
        return round(st.mean(vals), 2) if vals else "—"
    covs = [float(r["source_line_coverage"]) for r in done
            if r["source_line_coverage"] != ""]
    return (
        f"- Resumes with pipeline output: {len(done)}\n"
        f"- Avg sections: {avg('section_count')} | Avg blocks: {avg('block_count')} "
        f"| Avg entities: {avg('entity_count')} | Avg relationships: {avg('relationship_count')}\n"
        f"- Avg graph nodes: {avg('graph_node_count')} | Avg graph edges: {avg('graph_edge_count')}\n"
        f"- Avg source-line coverage: {round(st.mean(covs), 3) if covs else '—'}\n"
        f"- Resumes with 0 relationships: {sum(1 for r in done if r['relationship_count']==0)}\n"
        f"- Resumes with graph nodes but 0 edges: "
        f"{sum(1 for r in done if r['graph_output_success'] and not r['graph_edge_count'])}\n"
    )


def main():
    ap = argparse.ArgumentParser(description="Batch evaluation of resume pipeline")
    ap.add_argument("--input", default=os.path.join(REPO_ROOT, "resumes"))
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "batch_test_output"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume-id", default=None,
                    help="process only files whose name contains this substring")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    files = discover_resumes(args.input)
    if args.resume_id:
        needle = args.resume_id.lower()
        files = [f for f in files if needle in os.path.basename(f).lower()]
    files = files[args.start:args.end]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("No resumes matched.")
        return

    # dataset inventory
    from collections import Counter
    exts = Counter(os.path.splitext(f)[1].lower() for f in files)
    print(f"Dataset: {args.input} | files selected: {len(files)} | ext: {dict(exts)}")

    os.makedirs(args.output, exist_ok=True)
    tmp_base = tempfile.mkdtemp(prefix="batch_pipe_")
    run_ts = datetime.now(timezone.utc).isoformat()
    # crash-safe checkpoint: every completed resume is appended immediately,
    # so a killed run loses at most the in-flight resume on restart.
    ckpt_path = os.path.join(args.output, "batch_results_checkpoint.csv")
    prev_rows, done_ids = load_checkpoint(ckpt_path)
    prev_rows = [coerce_row(r) for r in prev_rows]
    ckpt_file = open(ckpt_path, "a" if done_ids else "w", newline="")
    ckpt_writer = csv.DictWriter(ckpt_file, fieldnames=CSV_COLUMNS,
                                 extrasaction="ignore")
    if not done_ids:
        ckpt_writer.writeheader()
    rows = list(prev_rows)
    if done_ids:
        print(f"Checkpoint: resuming, {len(done_ids)} resumes already done, "
              f"{len(files)} selected this run.")
    t_start = time.time()
    for i, path in enumerate(files):
        rid = f"R{args.start + i:04d}"
        if rid in done_ids:
            continue
        print(f"[{len(rows)+1}] {os.path.basename(path)} ...", flush=True)
        row, tmp_dir, raw_text = process_one(path, rid, tmp_base, debug=args.debug)
        if row["status"] in ("FAILED", "EXTRACTION_FAILED", "PARTIAL"):
            fail_dir = os.path.join(args.output, "failures", rid)
            try:
                save_failure_artifacts(fail_dir, row, tmp_dir, raw_text)
                row["output_directory"] = os.path.relpath(fail_dir, args.output)
            except Exception as e:  # never let artifact-saving break the batch
                print(f"  WARNING: could not save failure artifacts: {e}")
                row["output_directory"] = ""
        else:
            row["output_directory"] = ""
        row.pop("traceback", None)
        rows.append(row)
        try:
            ckpt_writer.writerow(row)
            ckpt_file.flush()
            os.fsync(ckpt_file.fileno())
            done_ids.add(rid)
        except Exception as e:
            print(f"  WARNING: checkpoint write failed: {e}", flush=True)
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        ckpt_file.close()
    except Exception:
        pass
    rows = sorted(rows, key=lambda r: r["resume_id"])
    # CSVs
    csv_path = os.path.join(args.output, "batch_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    failed = [r for r in rows if r["status"] in ("FAILED", "EXTRACTION_FAILED")]
    with open(os.path.join(args.output, "failed_resumes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(failed)

    summary = build_failure_summary(rows)
    with open(os.path.join(args.output, "failure_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import fitz, spacy, sklearn, numpy  # noqa
        import sentence_transformers
        dep_versions = {
            "pymupdf": getattr(fitz, "__doc__", "").split()[0] if fitz.__doc__ else "unknown",
            "spacy": spacy.__version__, "scikit-learn": sklearn.__version__,
            "numpy": numpy.__version__,
            "sentence-transformers": sentence_transformers.__version__,
        }
    except Exception:
        dep_versions = {}
    meta = {
        "timestamp": run_ts, "commit": get_commit(),
        "python_version": platform.python_version(),
        "dependency_versions": dep_versions,
        "dataset_path": args.input, "num_files": len(rows),
        "script_version": SCRIPT_VERSION,
        "command": " ".join(sys.argv),
        "file_type_distribution": dict(exts),
    }
    with open(os.path.join(args.output, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    write_report(os.path.join(args.output, "batch_pipeline_report.md"),
                 rows, summary, meta)
    shutil.rmtree(tmp_base, ignore_errors=True)

    total_min = (time.time() - t_start) / 60
    print("=" * 50)
    print("BATCH RESUME EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Total resumes: {summary['total_resumes']}")
    print(f"PASS: {summary['successful']}")
    print(f"PASS WITH WARNINGS: {summary['pass_with_warnings']}")
    print(f"PARTIAL: {summary['partial']}")
    print(f"FAILED: {summary['failed']}")
    print(f"Extraction failures: {summary['extraction_failures']}")
    print(f"Structural failures: {summary['structural_failures']}")
    print(f"Entity failures: {summary['entity_failures']}")
    print(f"Relationship failures: {summary['relationship_failures']}")
    print(f"Structured output failures: {summary['structured_output_failures']}")
    print(f"Graph failures: {summary['graph_failures']}")
    print(f"Average time/resume: {summary['average_processing_time']} sec")
    print(f"Total runtime: {total_min:.2f} min")
    top = max(summary["failure_stage_counts"].items(),
              key=lambda kv: kv[1])[0] if summary["failure_stage_counts"] else "none"
    print(f"Top failure stage: {top}")
    print(f"Detailed report: {os.path.join(args.output, 'batch_pipeline_report.md')}")
    print(f"Failed resumes: {os.path.join(args.output, 'failed_resumes.csv')}")
    print(f"CSV results: {csv_path}")


if __name__ == "__main__":
    main()
