"""Resume processing pipeline.

Preserves extracted source text, creates contextual blocks, derives a
canonical resume hierarchy, validates structural graph edges, and writes
auditable NLP and graph artifacts.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from copy import deepcopy

from .text_repository import TextRepository, TextDocument
from .preprocessing import preprocess
from .segmentation import detect_blocks
from .entity_extraction import extract_entities
from .relation_extraction import extract_relationships
from .embeddings import embed_entities
from .similarity import cosine_similarity_matrix
from .clustering import cluster
from .graph_builder import build_resume_graph
from .canonical_resume import canonicalize_resume
from .document_blocks import extract_pdf_blocks
from .semantic_blocks import prepare_blocks, group_sections, coverage_report, validate_semantic_employment


def _build_block_audit(line_meta: list[dict], blocks: dict, source_format: str) -> dict:
    """Create a loss-checkable source-line → section → block report."""
    block_by_id = {block["id"]: block for block in blocks.get("blocks", [])}
    line_to_block = blocks.get("line_blocks", {})
    source_by_index: dict[int, dict] = {}
    section_order: list[str] = []
    for meta in line_meta:
        line_index = meta.get("line_index")
        text = meta.get("source_line", meta.get("line", "")).strip()
        if line_index is None or not text:
            continue
        section_id = meta.get("section_id", f"s0:{meta.get('section', 'Other')}")
        if section_id not in section_order:
            section_order.append(section_id)
        row = source_by_index.setdefault(line_index, {
            "line_index": line_index,
            "text": text,
            "section_id": section_id,
            "section": meta.get("section", "Other"),
            "source_path": meta.get("source_path"),
            "is_section_heading": False,
            "is_bullet": bool(meta.get("is_bullet")),
            "list_marker": meta.get("list_marker", ""),
            "inline_body": "",
            "is_page_number": bool(meta.get("is_page_number")),
        })
        row["is_section_heading"] = row["is_section_heading"] or bool(meta.get("is_header"))
        if not meta.get("is_header") and meta.get("line", "").strip() != text:
            row["inline_body"] = meta.get("line", "").strip()

    # PDF extractors often put each bullet glyph on its own source row. It is
    # structural punctuation for the next text row, so associate it with that
    # row's block instead of reporting it as lost content.
    marker_blocks = {}
    ordered_source = sorted(source_by_index.items())
    for position, (line_index, row) in enumerate(ordered_source):
        if not re.fullmatch(r"[•▪●○◦*-]+", row["text"].strip()):
            continue
        for next_index, next_row in ordered_source[position + 1:]:
            if next_row["section_id"] != row["section_id"]:
                break
            next_block = line_to_block.get(str(next_index))
            if next_block:
                marker_blocks[line_index] = next_block
                break

    assignments = []
    for line_index in sorted(source_by_index):
        row = source_by_index[line_index]
        block_id = line_to_block.get(str(line_index)) or marker_blocks.get(line_index)
        row["block_id"] = block_id
        if row.get("is_page_number"):
            row["assignment"] = "page_number_marker"
            row["assigned"] = True
        elif line_index in marker_blocks:
            row["assignment"] = "bullet_marker_for_following_content"
            row["assigned"] = True
        elif re.fullmatch(r"[•▪●○◦*-]+", row["text"].strip()):
            row["assignment"] = "decorative_bullet_marker"
            row["assigned"] = True
        elif row["is_section_heading"] and block_id:
            row["assignment"] = "section_heading_and_block_content"
            row["assigned"] = True
        elif row["is_section_heading"]:
            row["assignment"] = "section_heading"
            row["assigned"] = bool(row["section_id"])
        elif block_id in block_by_id:
            row["assignment"] = "context_block"
            row["assigned"] = True
        else:
            row["assignment"] = "unassigned_content"
            row["assigned"] = False
            row["reason"] = "No sentence or context block was produced for this non-empty source line."
        assignments.append(row)

    sections = []
    for section_id in section_order:
        section_lines = [row for row in assignments if row["section_id"] == section_id]
        section_block_ids = list(dict.fromkeys(
            row["block_id"] for row in section_lines if row.get("block_id")
        ))
        sections.append({
            "id": section_id,
            "name": section_lines[0]["section"],
            "heading_lines": [row["text"] for row in section_lines if row["is_section_heading"]],
            "source_line_indices": [row["line_index"] for row in section_lines],
            "block_ids": section_block_ids,
            "blocks": [{
                "id": block_id,
                "entry_type": block_by_id[block_id]["entry_type"],
                "source_line_indices": block_by_id[block_id]["source_line_indices"],
                "source_lines": block_by_id[block_id]["source_lines"],
                "source_paths": block_by_id[block_id].get("source_paths", []),
                "sentences": block_by_id[block_id]["sentences"],
                "entity_ids": block_by_id[block_id]["entity_ids"],
                "relationship_ids": block_by_id[block_id]["relationship_ids"],
            } for block_id in section_block_ids],
        })

    unassigned = [row["line_index"] for row in assignments if not row["assigned"]]
    return {
        "format": "resume-block-audit/v1",
        "source_format": source_format,
        "summary": {
            "source_line_count": len(assignments),
            "assigned_source_line_count": len(assignments) - len(unassigned),
            "unassigned_source_line_indices": unassigned,
            "section_count": len(sections),
            "block_count": blocks.get("block_count", 0),
            "sentence_count": blocks.get("coverage", {}).get("sentence_count", 0),
            "entity_count": blocks.get("coverage", {}).get("entity_count", 0),
            "relationship_count": blocks.get("coverage", {}).get("relationship_count", 0),
            "complete": not unassigned and blocks.get("coverage", {}).get("complete", False),
        },
        "coverage": blocks.get("coverage", {}),
        "sections": sections,
        "source_line_assignments": assignments,
    }


def _normalize_structured_resume(data: dict) -> dict:
    """Canonicalize common resume keys without discarding source fields."""
    aliases = {
        "personal_details": "personal_details", "personal_detail": "personal_details",
        "contact": "personal_information", "contact_details": "personal_information",
        "summary": "profile_snapshot", "professional_summary": "profile_snapshot",
        "career_summary": "profile_snapshot", "employment_history": "work_history",
        "professional_experience": "work_history", "experience": "work_history",
        "project_highlights": "projects", "key_projects": "projects",
        "core_banking_and_domain_expertise": "domain_experience",
        "domain_expertise": "domain_experience", "industry_domain_experience": "domain_experience",
        "technical_skill": "technical_skills", "technical skills": "technical_skills",
        "dob": "date_of_birth", "birth_date": "date_of_birth",
    }

    def slug(value) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")

    result = {}
    for original_key, original_value in data.items():
        key = aliases.get(slug(original_key), slug(original_key))
        value = deepcopy(original_value)
        if key in {"work_history", "education", "projects"} and isinstance(value, dict):
            value = [value]
        if key == "work_history" and isinstance(value, list):
            normalized_jobs = []
            for job in value:
                if not isinstance(job, dict):
                    normalized_jobs.append(job)
                    continue
                record = deepcopy(job)
                for canonical, candidates in {
                    "company": ("employer", "company_name", "organization"),
                    "designation": ("job_title", "title", "role", "position"),
                    "duration": ("dates", "date_range", "period", "employment_period"),
                    "role_and_responsibilities": ("responsibilities", "duties"),
                }.items():
                    if canonical not in record:
                        for candidate in candidates:
                            if candidate in record:
                                record[canonical] = deepcopy(record[candidate])
                                record.pop(candidate, None)
                                break
                    else:
                        for candidate in candidates:
                            if candidate in record and record[candidate] == record[canonical]:
                                record.pop(candidate, None)
                roles = record.get("roles")
                if isinstance(roles, list):
                    normalized_roles = []
                    for role in roles:
                        if isinstance(role, dict):
                            role_record = deepcopy(role)
                            if "designation" not in role_record:
                                for candidate in ("title", "job_title", "role", "position"):
                                    if candidate in role_record:
                                        role_record["designation"] = deepcopy(role_record[candidate])
                                        role_record.pop(candidate, None)
                                        break
                            else:
                                for candidate in ("title", "job_title", "role", "position"):
                                    if candidate in role_record and role_record[candidate] == role_record["designation"]:
                                        role_record.pop(candidate, None)
                            if "role_and_responsibilities" not in role_record:
                                for candidate in ("responsibilities", "duties"):
                                    if candidate in role_record:
                                        role_record["role_and_responsibilities"] = deepcopy(role_record[candidate])
                                        role_record.pop(candidate, None)
                                        break
                            else:
                                for candidate in ("responsibilities", "duties"):
                                    if (candidate in role_record
                                            and role_record[candidate] == role_record["role_and_responsibilities"]):
                                        role_record.pop(candidate, None)
                            if isinstance(role_record.get("role_and_responsibilities"), str):
                                role_record["role_and_responsibilities"] = [role_record["role_and_responsibilities"]]
                            normalized_roles.append(role_record)
                        else:
                            normalized_roles.append(role)
                    record["roles"] = normalized_roles
                elif record.get("designation") or record.get("role_and_responsibilities"):
                    role_record = {}
                    if record.get("designation"):
                        role_record["designation"] = record["designation"]
                    role_record["role_and_responsibilities"] = deepcopy(
                        record.get("role_and_responsibilities", [])
                    )
                    if isinstance(role_record["role_and_responsibilities"], str):
                        role_record["role_and_responsibilities"] = [role_record["role_and_responsibilities"]]
                    record["roles"] = [role_record]
                    record.pop("designation", None)
                    record.pop("role_and_responsibilities", None)
                normalized_jobs.append(record)
            value = normalized_jobs
        if key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key].extend(value)
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key].update(value)
        else:
            result[key] = value
    # Keep explicitly supplied personal attributes together even when a
    # structured parser emitted Languages, DOB, or Nationality as root fields.
    personal_keys = ("date_of_birth", "nationality", "languages")
    if any(key in result for key in personal_keys):
        personal = result.get("personal_details")
        if not isinstance(personal, dict):
            personal = {"additional_details": personal} if personal is not None else {}
            result["personal_details"] = personal
        for key in personal_keys:
            if key in result:
                root_value = result.pop(key)
                if key not in personal:
                    personal[key] = root_value
                elif personal[key] != root_value:
                    existing = personal[key] if isinstance(personal[key], list) else [personal[key]]
                    incoming = root_value if isinstance(root_value, list) else [root_value]
                    personal[key] = existing + [item for item in incoming if item not in existing]
    # Trailing location/signature values in structured resume JSON are still
    # personal data; keep them with the identity record instead of as orphan
    # top-level sections.
    if "place" in result or "name_at_end" in result:
        personal = result.setdefault("personal_information", {})
        if not isinstance(personal, dict):
            personal = {"additional_details": personal}
            result["personal_information"] = personal
        place = result.pop("place", None)
        if place and not personal.get("location"):
            personal["location"] = place
        signature = result.pop("name_at_end", None)
        if signature and signature != personal.get("name"):
            personal.setdefault("additional_details", []).append(signature)
    return result


def _build_section_keyed_output(block_audit: dict, structured_data=None, semantic=None) -> dict:
    """Render resume content cleanly by section; keep provenance in _audit."""
    if isinstance(structured_data, dict):
        # A structured source is already the most reliable representation of
        # company → role → responsibilities, education records, and personal
        # details. Preserve its full nested values and unfamiliar resume
        # sections instead of flattening and rebuilding them from NLP blocks.
        output = _normalize_structured_resume(structured_data)
        output.pop("_audit", None)
        output["_audit"] = {
            "format": "section-keyed-resume-blocks/v1",
            "source_format": block_audit.get("source_format"),
            "summary": block_audit.get("summary", {}),
            "unassigned_source_line_indices": block_audit.get("summary", {}).get("unassigned_source_line_indices", []),
            "block_map": [],
            "source_line_assignments": block_audit.get("source_line_assignments", []),
            "preserved_structured_source": True,
        }
        return output
    aliases = {
        "header": "personal_information",
        "personal information": "personal_information",
        "personal details": "personal_details",
        "contact": "personal_information",
        "summary": "profile_snapshot",
        "professional summary": "profile_snapshot",
        "experience": "work_history",
        "projects": "projects",
        "project highlights": "projects",
        "education": "education",
        "skills": "skills",
        "core competencies": "skills",
        "education qualification": "education",
        "educational qualification": "education",
        "technical skills": "technical_skills",
        "domain experience": "domain_experience",
        "domain expertise": "domain_experience",
        "core banking and domain expertise": "domain_experience",
        "languages": "personal_information",
        "certifications": "certifications",
        "achievements": "achievements",
        "achievements and certifications": "achievements_and_certifications",
        "tools and technology": "tools_and_technology",
        "technical skills": "technical_skills",
        "education qualification": "education",
        "professional experience": "work_history",
        "work history": "work_history",
        "employment history": "work_history",
    }
    # Common misspellings in resume headings should not become their own
    # output sections after preprocessing has already recognized the section.
    aliases.update({
        "pofessional experience": "work_history",
        "professional experience": "work_history",
        "eeducation": "education",
        "core competencies": "skills",
        "professional summary": "profile_snapshot",
    })
    output: dict = {}
    block_map = []
    top_level_extras = {}
    handled_semantic_keys = set()

    def clean(lines: list[str]) -> list[str]:
        return [re.sub(r"^[\s•▪●*-]+", "", str(line)).strip()
                for line in lines if str(line).strip()]

    def merge_nested(target: dict, incoming: dict) -> None:
        """Merge repeated semantic sections without replacing nested field groups."""
        for field, value in incoming.items():
            if isinstance(value, dict) and isinstance(target.get(field), dict):
                merge_nested(target[field], value)
            elif isinstance(value, list) and isinstance(target.get(field), list):
                target[field].extend(item for item in value if item not in target[field])
            else:
                target[field] = deepcopy(value)

    def complete_source_lines(section: dict) -> list[str]:
        """Return every source row in order, retaining bullets and wrapped text."""
        section_id = section["id"]
        rows = [row for row in block_audit.get("source_line_assignments", [])
                if row.get("section_id") == section_id
                and (not row.get("is_section_heading") or row.get("inline_body"))
                and str(row.get("text", "")).strip()]
        expanded = []
        for row in rows:
            if row.get("inline_body"):
                expanded.append({**row, "text": row["inline_body"], "is_bullet": False,
                                 "is_section_heading": False})
            elif not row.get("is_section_heading"):
                expanded.append(row)
        return expanded

    def reflow_bullets(rows: list[dict]) -> list[str]:
        """Join PDF-wrapped continuation rows to their originating bullet."""
        result: list[str] = []
        pending_bullet = False
        active_bullet = False
        for row in rows:
            if row.get("is_page_number"):
                continue
            raw = str(row.get("text", "")).strip()
            if re.fullmatch(r"[•▪●○◦*-]+", raw):
                pending_bullet = True
                continue
            starts_bullet = bool(row.get("is_bullet")) or pending_bullet or bool(re.match(r"^[•▪●○◦*-]\s*", raw))
            text = re.sub(r"^[\s•▪●○◦*-]+", "", raw).strip()
            pending_bullet = False
            if not text:
                continue
            # A new employer/role heading is a boundary even if PDF extraction
            # omitted its bullet marker.
            role_heading = len(text) < 120 and bool(re.search(
                r"\b(analyst|underwriter|engineer|manager|consultant|developer|designer|architect|"
                r"scientist|officer|specialist|associate|executive|intern|director|product owner)\b",
                text, re.I,
            )) and not text.endswith((",", ";"))
            is_job_heading = (not starts_bullet and len(text) < 220 and (
                bool(re.search(r"\b(?:19|20)\d{2}\b", text, re.I)) or role_heading
            ))
            if starts_bullet or is_job_heading or not result:
                result.append(text)
                active_bullet = starts_bullet
            else:
                if active_bullet:
                    result[-1] += " " + text
                else:
                    result.append(text)
        return result

    def build_education_records(rows: list[dict]) -> list[dict]:
        filtered_rows = []
        for row in rows:
            text = str(row.get("text", "")).strip()
            place_match = re.match(r"^Place:\s*([^|]+?)(?:\s{2,}.*)?$", text, re.I)
            if place_match:
                top_level_extras.setdefault("place", place_match.group(1).strip())
                continue
            filtered_rows.append(row)
        lines = reflow_bullets(filtered_rows)
        header_names = {"degree", "college/school", "college school", "university/board",
                        "university board", "year of passing", "percentage & pointer", "percentage pointer"}
        normalized_lines = [re.sub(r"^[\s•▪●*-]+", "", line).strip() for line in lines]
        if any(line.casefold().rstrip(":") == "degree" for line in normalized_lines) and any(
            line.casefold().rstrip(":") in {"college/school", "college school"} for line in normalized_lines
        ):
            table_lines = []
            for row in filtered_rows:
                if row.get("is_page_number"):
                    continue
                text = re.sub(r"^[\s•▪●○◦*-]+", "", str(row.get("text", ""))).strip()
                if not text or re.fullmatch(r"[•▪●○◦*-]+", text):
                    continue
                if table_lines and (table_lines[-1].rstrip().endswith(("&", "of", "in"))
                                    or text.casefold() in {"engineering", "road", "pointer"}):
                    table_lines[-1] += " " + text
                else:
                    table_lines.append(text)
            data_lines = [line for line in table_lines if line.casefold().rstrip(":") not in header_names
                          and line.casefold().strip() not in {"percentage & pointer", "year of passing percentage & pointer"}]
            records = []
            current = None
            degree_start = re.compile(r"^(?:b\.?e\.?\b|bachelor\b|diploma\b|ssc\b|hsc\b|class\s+\d+|(?:first|second|third|final)\s+year\b)", re.I)
            for line in data_lines:
                if degree_start.search(line):
                    if current:
                        records.append(current)
                    current = {"degree": line}
                    continue
                if current is None:
                    current = {"details": []}
                if re.fullmatch(r"(?:19|20)\d{2}", line):
                    current["year"] = line
                elif re.fullmatch(r"\d{1,3}(?:\.\d+)?\s*(?:/\s*\d+(?:\.\d+)?)?%?", line):
                    current["percentage_and_pointer"] = line
                elif re.search(r"\b(university|board|cbse|icse)\b", line, re.I):
                    current["university_or_board"] = (current.get("university_or_board", "") + " " + line).strip()
                elif "college_school" not in current:
                    current["college_school"] = line
                elif "year" not in current and "percentage_and_pointer" not in current:
                    current["college_school"] += " " + line
                else:
                    current.setdefault("details", []).append(line)
            if current:
                records.append(current)
            return records
        # If PDF extraction yields parallel columns rather than table rows,
        # reconstruct rows by the repeated education-field cadence.
        labels = []
        for line in normalized_lines:
            folded = line.casefold().rstrip(":")
            if folded in {"degree", "college/school", "college school", "university/board",
                          "university board", "year of passing", "percentage & pointer", "pointer"}:
                labels.append(folded)
        if len(labels) >= 5:
            data = [line for line in normalized_lines if line and line.casefold().rstrip(":") not in {
                "degree", "college/school", "college school", "university/board", "university board",
                "year of passing", "percentage & pointer", "pointer"}]
            degree_start = re.compile(r"^(?:b\.?e\.?\b|bachelor\b|diploma\b|ssc\b|hsc\b|class\s+\d+|(?:first|second|third|final)\s+year\b)", re.I)
            degrees = [line for line in data if degree_start.search(line)]
            if len(degrees) >= 2:
                chunks, current = [], None
                for line in data:
                    if degree_start.search(line):
                        if current:
                            chunks.append(current)
                        current = [line]
                    elif current is not None:
                        current.append(line)
                if current:
                    chunks.append(current)
                records = []
                for chunk in chunks:
                    record = {"degree": chunk[0]}
                    for line in chunk[1:]:
                        if re.fullmatch(r"(?:19|20)\d{2}", line):
                            record["year"] = line
                        elif re.fullmatch(r"\d{1,3}(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?%?", line):
                            record["percentage_and_pointer"] = line
                        elif re.search(r"\b(?:university|board|cbse|icse)\b", line, re.I):
                            record["university_or_board"] = line
                        elif "college_school" not in record:
                            record["college_school"] = line
                        else:
                            record["college_school"] += " " + line
                    records.append(record)
                return records
        year_only = re.compile(r"^(?:19|20)\d{2}$")
        year_range = re.compile(r"\b((?:19|20)\d{2})(?:\s*[-–—/]\s*((?:19|20)\d{2}|present|current))?\b", re.I)
        degree_terms = re.compile(
            r"\b(bachelor|master|ph\.?d|mba|b\.?\s?(?:tech|e|sc|com|ba|ca|arch|pharm)|"
            r"m\.?\s?(?:tech|e|sc|com|ba|ca|arch)|b\.?\s?s\.?|m\.?\s?s\.?|"
            r"b\.?\s?a\.?|m\.?\s?a\.?|diploma|certificate|doctorate|"
            r"associate degree|high school|secondary school|ssc|hsc|ged)\b", re.I
        )
        records: list[dict] = []
        current: list[str] = []
        current_has_qualification = False
        for line in lines:
            is_year = bool(year_only.fullmatch(line.strip()))
            has_qualification = bool(degree_terms.search(line))
            if current and (is_year or (has_qualification and current_has_qualification)):
                records.append(_education_record(current, year_range, degree_terms))
                current = []
                current_has_qualification = False
            if (current and not is_year and not has_qualification
                    and not re.search(r"\b(university|college|institute|school)\b", line, re.I)
                    and current_has_qualification
                    and not re.search(r"\b(university|college|institute|school)\b", current[-1], re.I)):
                current[-1] += " " + line
                continue
            current.append(line)
            current_has_qualification = current_has_qualification or has_qualification
        if current:
            records.append(_education_record(current, year_range, degree_terms))
        return records

    def build_tool_groups(rows: list[dict]) -> dict:
        groups: dict[str, list[str]] = {}
        active = "general"
        category_labels = {
            "professional": "professional", "beginner": "beginner",
            "intermediate": "intermediate", "advanced": "advanced",
            "technical": "technical", "tools": "tools",
        }
        content: dict[str, list[str]] = {}
        for row in rows:
            text = re.sub(r"^[\s•▪●*-]+", "", str(row.get("text", ""))).strip()
            if not text:
                continue
            normalized = text.rstrip(":").casefold().strip()
            if normalized in category_labels:
                active = category_labels[normalized]
                content.setdefault(active, [])
                continue
            content.setdefault(active, []).append(text)
        for category, lines in content.items():
            flattened = " ".join(lines)
            groups[category] = [item.strip(" .") for item in re.split(r"\s*,\s*", flattened) if item.strip(" .")]
        return groups

    def build_technical_skills(rows: list[dict]) -> dict:
        groups: dict[str, list[str]] = {}
        active_category = "general"
        for row in rows:
            text = re.sub(r"^[\s•▪●*-]+", "", str(row.get("text", ""))).strip()
            if not text:
                continue
            match = re.match(r"^(Tools?|Databases?|Operating Systems?)\s*:\s*(.*)$", text, re.I)
            if match:
                label = match.group(1).casefold().replace(" ", "_")
                active_category = {"tool": "tools", "tools": "tools", "database": "databases",
                                   "databases": "databases", "operating_system": "operating_systems",
                                   "operating_systems": "operating_systems"}[label]
                groups.setdefault(active_category, [])
                text = match.group(2)
            values = [item.strip() for item in re.split(r"[,;]", text) if item.strip()]
            groups.setdefault(active_category, []).extend(values)
        return groups

    def build_skill_items(rows: list[dict]) -> list[str]:
        lines = reflow_bullets(rows)
        has_bullets = any(row.get("is_bullet") for row in rows)
        if has_bullets:
            return lines
        # PDF exports commonly wrap a comma-separated skills list across rows.
        joined = " ".join(lines).strip()
        if "," in joined and len(joined) < 3000:
            return [item.strip(" .;,") for item in joined.split(",") if item.strip(" .;,")]
        return lines

    def _education_record(lines: list[str], year_range, degree_terms) -> dict:
        record = {"details": []}
        qualifications = []
        institutions = []
        for line in lines:
            year_match = year_range.search(line)
            if year_match and "year" not in record:
                record["year"] = year_match.group(0).strip()
            body = year_range.sub("", line).strip(" ()|,;–—-")
            body = re.sub(r"\(\s*\)|\[\s*\]", "", body).strip(" ()|,;–—-")
            split_qualification = re.split(r"\s+(?:from|at)\s+", body, maxsplit=1, flags=re.I)
            if len(split_qualification) == 2 and degree_terms.search(split_qualification[0]):
                qualifications.append(split_qualification[0].strip(" ,;"))
                institutions.append(split_qualification[1].strip(" ,;."))
            elif degree_terms.search(body) and re.search(r"\b(university|college|institute|school)\b", body, re.I):
                # Keep a combined education row together while separating the
                # qualification from the institution phrase when possible.
                match = re.search(r"\b(?:from|at)\b", body, re.I)
                if match:
                    qualifications.append(body[:match.start()].strip(" ,;"))
                    institutions.append(body[match.end():].strip(" ,;."))
                else:
                    qualifications.append(body)
                    institutions.append(body)
            elif re.search(r"\b(university|college|institute|school)\b", body, re.I):
                institutions.append(body)
            elif degree_terms.search(body):
                qualifications.append(body)
            elif not year_only_match(line):
                record["details"].append(line)
        if qualifications:
            record["qualification"] = " ".join(qualifications)
        if institutions:
            record["institution"] = " ".join(institutions)
        if not record["details"]:
            record.pop("details")
        return record

    def year_only_match(line: str) -> bool:
        return bool(re.fullmatch(r"(?:19|20)\d{2}", line.strip()))

    def labeled_fields(lines: list[str]) -> dict:
        fields = {}
        known_keys = {
            "name", "title", "location", "phone", "email", "place", "name at end",
            "year", "qualification", "university", "institution", "degree", "gpa",
            "cgpa", "percentage", "company", "duration", "designation", "role",
            "dob", "date of birth", "nationality", "citizenship", "languages", "e-mail",
            "linkedin", "github", "website", "portfolio", "url",
        }
        aliases = {"dob": "date_of_birth", "date of birth": "date_of_birth",
                   "e-mail": "email", "mail": "email", "language": "languages",
                   "url": "website"}
        normalized_lines = [str(line).strip() for line in lines if str(line).strip()]
        index = 0
        while index < len(normalized_lines):
            line = normalized_lines[index]
            standalone = re.sub(r"\s*:\s*$", "", line).casefold().strip()
            if standalone in known_keys and index + 1 < len(normalized_lines):
                next_line = normalized_lines[index + 1]
                if next_line.casefold().strip().rstrip(":") not in known_keys:
                    field_key = aliases.get(standalone, "_".join(standalone.replace("&", "and").split()))
                    fields[field_key] = next_line
                    index += 2
                    continue
            current_key = None
            for part in re.split(r"\s*\|\s*", line):
                match = re.match(r"^([^:]{2,40}):\s*(.+)$", part.strip())
                label = match.group(1).casefold().strip() if match else ""
                if match and label in known_keys:
                    field_key = aliases.get(label, "_".join(label.replace("&", "and").split()))
                    fields[field_key] = match.group(2).strip()
                    current_key = field_key
                elif current_key and current_key == "email" and re.fullmatch(r"\+?[\d\s().-]{7,}", part.strip()):
                    fields["phone"] = part.strip()
                    current_key = None
                elif current_key:
                    fields[current_key] += " | " + part.strip()
            index += 1
        return fields

    def infer_header_identity(lines: list[str], fields: dict) -> dict:
        """Recover common unlabeled name/contact/title lines in the header."""
        email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
        phone_pattern = re.compile(r"(?<!\w)(?:\+?\d[\d().\s-]{7,}\d)(?!\w)")
        link_patterns = {
            "linkedin": re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s|,;]+", re.I),
            "github": re.compile(r"(?:https?://)?(?:www\.)?github\.com/[^\s|,;]+", re.I),
            "website": re.compile(r"https?://[^\s|,;]+|www\.[^\s|,;]+", re.I),
        }
        for line in lines:
            email_match = email_pattern.search(line)
            if email_match:
                fields.setdefault("email", email_match.group(0))
            for field, pattern in link_patterns.items():
                match = pattern.search(line)
                if match:
                    fields.setdefault(field, match.group(0).rstrip(".)"))
            for phone_match in phone_pattern.finditer(line):
                candidate = phone_match.group(0).strip()
                digits = re.sub(r"\D", "", candidate)
                if 7 <= len(digits) <= 15:
                    fields.setdefault("phone", candidate)
                    break
            location = re.match(r"^(?:location|address|based in|city)\s*[:|–—-]\s*(.+)$", line, re.I)
            if location:
                fields.setdefault("location", location.group(1).strip())
        if not fields.get("name"):
            for line in lines:
                candidate = re.sub(r"\s*\|\s*.*$", "", line).strip()
                if (candidate and not email_pattern.search(candidate)
                        and not re.search(r"\b(phone|mobile|email|location|address)\s*[:|]", candidate, re.I)
                        and len(candidate.split()) <= 6 and not re.search(r"\d", candidate)):
                    fields["name"] = candidate
                    break
        if not fields.get("title"):
            for line in lines:
                parts = [part.strip() for part in re.split(r"\s*[|•]\s*", line) if part.strip()]
                title_parts = [part for part in parts if not email_pattern.search(part)
                               and not phone_pattern.search(part)
                               and not re.match(r"^(?:location|address|phone|mobile|email)\s*:", part, re.I)
                               and part != fields.get("name")]
                if title_parts and any(re.search(r"\b(developer|analyst|engineer|scientist|manager|consultant|designer|specialist|student|intern|architect|executive|director|teacher|nurse)\b", p, re.I) for p in title_parts):
                    fields["title"] = title_parts[0]
                    break
        return fields

    for section in block_audit.get("sections", []):
        name = section["name"].strip()
        key = aliases.get(name.casefold(), "_".join(name.casefold().replace("&", "and").split()))
        heading_text = " ".join(section.get("heading_lines", [])).casefold()
        if "technical skills" in heading_text or name.casefold() == "technical skills":
            key = "technical_skills"
        if name.casefold() == "core competencies":
            key = "skills"
        if key == "header":
            key = "personal_information"
        source_rows = complete_source_lines(section)
        # Build the readable section content from source lines rather than
        # relying on entity-bearing blocks. This preserves plain text such as
        # coursework, long responsibility bullets, and skill details that may
        # not produce any entities or relationships.
        grouped = (semantic or {}).get("groups", {}).get(section["id"])
        if grouped:
            key, value = grouped["key"], grouped["value"]
            handled_semantic_keys.add(key)
            if isinstance(value, dict):
                merge_nested(output.setdefault(key, {}), value)
                base_index = 0
            elif key == "profile_snapshot":
                output.setdefault(key, []).extend(item.get("text", "") if isinstance(item, dict) else item
                                                 for item in value)
                base_index = len(output[key]) - len(value)
            else:
                base_index = len(output.get(key, []))
                output.setdefault(key, []).extend(value)
            for source_block in (semantic or {}).get("blocks", []):
                if source_block.get("section_id") != section["id"] or not source_block.get("destination"):
                    continue
                match = re.match(rf"^\$.{re.escape(key)}\[(\d+)\]", source_block["destination"])
                if not match:
                    continue
                item_index = int(match.group(1))
                block_id = next((block["id"] for block in section.get("blocks", [])
                                 if source_block["line_index"] in block.get("source_line_indices", [])), None)
                if block_id:
                    existing = next((entry for entry in block_map if entry["block_id"] == block_id
                                     and entry["section_key"] == key), None)
                    if existing:
                        if item_index not in existing["item_indices"]:
                            existing["item_indices"].append(item_index)
                    else:
                        block_map.append({"block_id": block_id, "section_id": section["id"],
                                          "section_key": key, "item_indices": [item_index],
                                          "source_line_indices": [source_block["line_index"]],
                                          "source_paths": [source_block["destination"]]})
            continue
        source_text = [str(row.get("text", "")).strip() for row in source_rows]
        if key == "profile_snapshot":
            source_text = reflow_bullets(source_rows)
            # PDF extraction wraps a prose summary at visual line boundaries.
            # Keep it as one readable summary unless the source has real bullets.
            has_summary_bullets = any(
                row.get("is_bullet") or re.match(r"^[\s•▪●○◦*-]+\S", str(row.get("text", "")))
                for row in source_rows if not row.get("is_page_number")
            )
            if not has_summary_bullets and source_text:
                source_text = [" ".join(source_text)]
        elif key == "education":
            source_text = build_education_records(source_rows)
        elif key == "tools_and_technology":
            tool_groups = build_tool_groups(source_rows)
            output.setdefault(key, {}).update(tool_groups)
            for block in section.get("blocks", []):
                block_map.append({"block_id": block["id"], "section_id": section["id"],
                                  "section_key": key, "item_indices": [],
                                  "source_line_indices": block.get("source_line_indices", []),
                                  "source_paths": block.get("source_paths", [])})
            continue
        elif key == "skills":
            source_text = build_skill_items(source_rows)
        elif key == "profile_snapshot":
            source_text = (semantic or {}).get("groups", {}).get(section["id"], {}).get("value", source_text)
        elif key in {"achievements_and_certifications", "certifications", "domain_experience"}:
            source_text = reflow_bullets(source_rows)
        elif key == "projects":
            source_text = reflow_bullets(source_rows)
        elif key == "technical_skills":
            source_text = build_technical_skills(source_rows)
            output.setdefault(key, {}).update(source_text)
            for block in section.get("blocks", []):
                block_map.append({"block_id": block["id"], "section_id": section["id"],
                                  "section_key": key, "item_indices": [],
                                  "source_line_indices": block.get("source_line_indices", []),
                                  "source_paths": block.get("source_paths", [])})
            continue
        elif key == "work_history":
            # Parent grouping is completed before serialization. Copy these
            # records into the section-keyed artifact rather than rebuilding
            # one details-only record per rendered block.
            work_records = []
            if semantic and key not in handled_semantic_keys:
                for semantic_group in semantic.get("groups", {}).values():
                    if semantic_group.get("key") == "work_history":
                        work_records.extend(deepcopy(semantic_group.get("value", [])))
                handled_semantic_keys.add(key)
            output.setdefault(key, []).extend(work_records)
            first_index = len(output[key]) - len(work_records)
            for block in section.get("blocks", []):
                block_map.append({"block_id": block["id"], "section_id": section["id"],
                                  "section_key": key, "item_indices": list(range(first_index, len(output[key]))),
                                  "source_line_indices": block.get("source_line_indices", []),
                                  "source_paths": block.get("source_paths", [])})
            continue
        if key == "personal_information":
            if isinstance(output.get(key), list):
                output[key] = {}
            lines = clean([str(row.get("text", "")) for row in source_rows])
            fields = labeled_fields(lines)
            if name.casefold() == "header" and lines:
                fields = infer_header_identity(lines, fields)
                if not fields.get("name"):
                    fields["name"] = lines[0]
                if len(lines) > 1 and not fields.get("title") and "@" not in lines[1]:
                    fields["title"] = lines[1]
            if name.casefold() == "languages":
                language_values = []
                for line in lines:
                    match = re.match(r"^languages?\s*:\s*(.+)$", line, re.I)
                    value = match.group(1) if match else line
                    if not re.match(r"^nationality\s*:", value, re.I):
                        language_values.extend(item.strip() for item in re.split(r"[,;]", value) if item.strip())
                if language_values:
                    fields["languages"] = language_values
            contact = output.setdefault(key, {})
            contact.update(fields)
            if not fields and lines:
                contact.setdefault("additional_details", []).extend(lines)
            for block in section.get("blocks", []):
                block_map.append({"block_id": block["id"], "section_id": section["id"],
                                  "section_key": key, "item_indices": [],
                                  "source_line_indices": block.get("source_line_indices", []),
                                  "source_paths": block.get("source_paths", [])})
            continue
        if key in {"profile_snapshot", "skills", "work_history", "education",
                   "domain_experience", "projects", "technical_skills",
                   "certifications", "achievements", "achievements_and_certifications"}:
            output.setdefault(key, []).extend(source_text)
            for block in section.get("blocks", []):
                block_map.append({
                    "block_id": block["id"], "section_id": section["id"],
                    "section_key": key, "item_indices": list(range(len(output[key]) - len(source_text), len(output[key]))),
                    "source_line_indices": block.get("source_line_indices", []),
                    "source_paths": block.get("source_paths", []),
                })
            continue
        for block in section.get("blocks", []):
            lines = clean(block.get("source_lines") or block.get("sentences", []))
            item_indices = []
            value = lines
            if key == "certifications":
                category_map = {}
                for line in lines:
                    if ":" in line:
                        category, items = line.split(":", 1)
                        category_key = "_".join(category.casefold().replace("&", "and").split())
                        category_map[category_key] = [item.strip() for item in items.split(";") if item.strip()]
                value = category_map or lines
            elif key == "work_history":
                raise ValueError(
                    "Work history cannot be serialized from an isolated source block; "
                    "semantic employer grouping must produce the parent record first."
                )
            elif key == "education":
                value = labeled_fields(lines) or lines
            elif key == "personal_information":
                value = labeled_fields(lines) or lines
                extra_fields = {}
                for path, out_key in (("$.place", "place"), ("$.name_at_end", "name_at_end")):
                    for source_path in block.get("source_paths", []):
                        if source_path == path:
                            line = next((line for line in lines if line.casefold().startswith(out_key.replace("_", " ") + ":")), None)
                            if line and isinstance(value, dict):
                                extra_fields[out_key] = line.split(":", 1)[1].strip()
                for extra_key, extra_value in extra_fields.items():
                    top_level_extras[extra_key] = extra_value
                    if isinstance(value, dict):
                        value.pop(extra_key, None)
            elif key == "tools_and_technology":
                nested = {}
                for line in lines:
                    if ":" in line:
                        category, items = line.split(":", 1)
                        category_key = "_".join(category.casefold().split())
                        nested[category_key] = [item.strip() for item in items.split(";") if item.strip()]
                value = nested or lines

            if key == "personal_information" and isinstance(value, dict):
                output.setdefault(key, {}).update(value)
            elif key == "personal_information":
                current = output.setdefault(key, {})
                if isinstance(current, list):
                    current = {"additional_details": current}
                    output[key] = current
                current.setdefault("additional_details", []).extend(lines)
            elif key in {"certifications", "tools_and_technology"} and isinstance(value, dict):
                if key not in output or not isinstance(output[key], dict):
                    output[key] = {}
                for category, category_items in value.items():
                    if isinstance(category_items, list) and isinstance(output[key].get(category), list):
                        output[key][category].extend(category_items)
                    else:
                        output[key][category] = category_items
            elif key == "education":
                output.setdefault(key, []).append(value)
                item_indices.append(len(output[key]) - 1)
            elif key == "work_history":
                output.setdefault(key, []).append(value)
                item_indices.append(len(output[key]) - 1)
            else:
                output.setdefault(key, []).extend(lines)
                item_indices.extend(range(len(output[key]) - len(lines), len(output[key])))
            block_map.append({
                "block_id": block["id"], "section_id": section["id"],
                "section_key": key, "item_indices": item_indices,
                "source_line_indices": block.get("source_line_indices", []),
                "source_paths": block.get("source_paths", []),
            })
    output.update(top_level_extras)
    inferred_profiles = [item for group in (semantic or {}).get("groups", {}).values()
                         if group.get("key") == "profile_snapshot" and group.get("inferred")
                         for item in group.get("value", [])]
    if inferred_profiles:
        output.setdefault("profile_snapshot", []).extend(inferred_profiles)
    output["_audit"] = {
        "format": "section-keyed-resume-blocks/v1",
        "source_format": block_audit.get("source_format"),
        "summary": block_audit.get("summary", {}),
        "unassigned_source_line_indices": block_audit.get("summary", {}).get("unassigned_source_line_indices", []),
        "block_map": block_map,
        "source_line_assignments": block_audit.get("source_line_assignments", []),
    }
    return output


def extract_pdf_text(pdf_path: str) -> str:
    return extract_pdf_blocks(pdf_path, _ocr_image_bytes)


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Use the local Tesseract executable when an image has no text layer."""
    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError("OCR is required for this scanned resume, but Tesseract is not installed.")
    with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
        image_file.write(image_bytes)
        image_file.flush()
        result = subprocess.run(
            [executable, image_file.name, "stdout"], check=True, capture_output=True, text=True
        )
    return result.stdout.strip()


def extract_image_text(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        text = _ocr_image_bytes(image_file.read())
    from .document_blocks import DocumentText
    rows = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        rows.append({"id": f"d{index}", "text": line.strip(), "page": 1, "bbox": None,
                     "block_index": index, "reading_order": len(rows), "font_size": None,
                     "bold": None, "extraction": "ocr", "artifact": None})
    return DocumentText(rows)


def extract_docx_text(docx_path: str) -> str:
    """Read DOCX paragraphs and tables in document order without adding a parser dependency."""
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(docx_path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    body = document.find(f"{namespace}body")
    if body is None:
        return ""

    def paragraph_text(paragraph) -> str:
        return "".join(node.text or "" for node in paragraph.iter(f"{namespace}t")).strip()

    lines = []
    for element in body:
        if element.tag == f"{namespace}p":
            text = paragraph_text(element)
            if text:
                lines.append(text)
        elif element.tag == f"{namespace}tbl":
            for row in element.findall(f"{namespace}tr"):
                cells = []
                for cell in row.findall(f"{namespace}tc"):
                    text = " ".join(filter(None, (paragraph_text(p) for p in cell.findall(f"{namespace}p"))))
                    cells.append(text)
                if any(cells):
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def run_pipeline(raw_text: str, doc_id: str, out_dir: str, candidate_name: str = "Candidate", debug: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Text Repository
    repo = TextRepository()
    doc = repo.add(TextDocument(doc_id=doc_id, raw_text=raw_text))
    raw_json = {"doc_id": doc.doc_id, "source": doc.source,
                "char_count": doc.char_count, "created_at": doc.created_at,
                "raw_text": doc.raw_text}

    # 2. Data preprocessing: structure-preserving normalization and segmentation
    pre = preprocess(raw_text)
    semantic_rows = prepare_blocks(pre, getattr(raw_text, "blocks", None))
    pre_group_blocks = [{
        "index": index,
        "id": row["id"],
        "raw_text": row.get("raw_text", row.get("text", "")),
        "normalized_text": row.get("normalized_text", row.get("text", "")),
        "section": row.get("section", "Other"),
        "semantic_classification": row.get("semantic_classification", row.get("block_type", "unknown")),
        "semantic_labels": row.get("semantic_labels", [row.get("semantic_classification", "unknown")]),
        "block_type": row.get("block_type", "UNKNOWN"),
        "page_number": row.get("page_number"),
        "is_page_marker": row.get("block_type") == "PAGE_ARTIFACT",
    } for index, row in enumerate(semantic_rows, 1)]
    semantic = (group_sections(semantic_rows) if not pre.get("structured_data") else
                {"groups": {}, "blocks": semantic_rows, "unresolved_blocks": []})
    semantic["pre_group_blocks"] = pre_group_blocks
    if not pre.get("structured_data"):
        validate_semantic_employment(semantic)
    sentences: list[str] = pre["sentences"]
    sentence_meta: list[dict] = pre.get("sentence_meta", [])
    # 3. Context blocks from source structure only; no embedding/keyword merge.
    blocks = detect_blocks(sentences, sentence_meta)
    # 4. Entity occurrences stay scoped to their source block. Repeated labels
    #    in separate contexts receive separate stable node IDs.
    entities = extract_entities(
        sentences, candidate_name=candidate_name,
        sentence_blocks=blocks["sentence_blocks"],
    )
    _sent_section = {m["index"]: m.get("section", "Other") for m in sentence_meta}
    _sent_section_id = {m["index"]: m.get("section_id", f"s0:{m.get('section', 'Other')}")
                        for m in sentence_meta}
    for e in entities:
        e["sections"] = sorted({_sent_section[i] for i in e.get("sentence_indices", [])
                                 if i in _sent_section})
        e["section_ids"] = sorted({_sent_section_id[i] for i in e.get("sentence_indices", [])
                                    if i in _sent_section_id})
    for block in blocks["blocks"]:
        block_entities = [entity for entity in entities if block["id"] in entity.get("block_ids", [])]
        block["entities"] = sorted({entity["text"] for entity in block_entities})
        block["entity_ids"] = [entity["id"] for entity in block_entities]
    # 5. Relationships require source-backed context/structural evidence.
    relationships = extract_relationships(sentences, entities, sentence_meta=sentence_meta,
                                          sentence_blocks=blocks["sentence_blocks"])
    block_by_id = {block["id"]: block for block in blocks["blocks"]}
    entity_by_id = {entity["id"]: entity for entity in entities}
    sentence_layer_by_index = {layer["sentence_index"]: layer for layer in blocks["sentence_layers"]}
    section_relationships: dict[str, list[str]] = {}
    relationship_layers = []
    for relationship_index, relationship in enumerate(relationships):
        relationship_id = f"r{relationship_index}"
        relationship["id"] = relationship_id
        edge_block_id = relationship.get("block_id")
        if edge_block_id in block_by_id:
            block_by_id[edge_block_id]["relationship_ids"].append(relationship_id)
            section_relationships.setdefault(block_by_id[edge_block_id]["section_id"], []).append(relationship_id)
        for sentence_index in relationship.get("evidence_sentence_indices", []):
            sentence_layer = sentence_layer_by_index.get(sentence_index)
            if sentence_layer and relationship_id not in sentence_layer.setdefault("relationship_ids", []):
                sentence_layer["relationship_ids"].append(relationship_id)
        relationship_layers.append({
            "relationship_id": relationship_id,
            "source_entity_id": relationship.get("source_id"),
            "target_entity_id": relationship.get("target_id"),
            "section_id": block_by_id.get(edge_block_id, {}).get("section_id"),
            "block_id": edge_block_id,
            "evidence_sentence_indices": relationship.get("evidence_sentence_indices", []),
        })
        source = entity_by_id.get(relationship.get("source_id"), {})
        target = entity_by_id.get(relationship.get("target_id"), {})
        source_blocks = set(source.get("block_ids", [])) if source.get("type") != "PERSON" else set()
        target_blocks = set(target.get("block_ids", [])) if target.get("type") != "PERSON" else set()
        for source_block_id in source_blocks:
            for target_block_id in target_blocks:
                if source_block_id == target_block_id:
                    continue
                left = block_by_id.get(source_block_id)
                right = block_by_id.get(target_block_id)
                if not left or not right or left["section_id"] != right["section_id"]:
                    continue
                if target_block_id not in left["related_block_ids"]:
                    left["related_block_ids"].append(target_block_id)
                if source_block_id not in right["related_block_ids"]:
                    right["related_block_ids"].append(source_block_id)
    for section in blocks["sections"]:
        section["entity_ids"] = [entity["id"] for entity in entities
                                 if section["id"] in entity.get("section_ids", [])]
        section["relationship_ids"] = section_relationships.get(section["id"], [])
    blocks["entity_layers"] = [
        {"entity_id": entity["id"], "section_ids": entity.get("section_ids", []),
         "block_ids": entity.get("block_ids", []), "source_sentence_indices": entity.get("sentence_indices", [])}
        for entity in entities
    ]
    blocks["relationship_layers"] = relationship_layers
    blocks["coverage"].update({
        "entity_count": len(entities),
        "assigned_entity_count": sum(bool(entity.get("block_ids")) for entity in entities),
        "relationship_count": len(relationships),
        "assigned_relationship_count": sum(
            relationship.get("block_id") in block_by_id for relationship in relationships
        ),
        "unassigned_sentence_indices": [
            index for index, block_id in enumerate(blocks["sentence_blocks"]) if not block_id
        ],
        "unassigned_source_line_indices": sorted(
            {meta.get("line_index", meta.get("index")) for meta in sentence_meta}
            - {int(line_index) for line_index in blocks["line_blocks"]}
        ),
    })
    blocks["coverage"]["complete"] = (
        blocks["coverage"]["assigned_sentence_count"] == blocks["coverage"]["sentence_count"]
        and blocks["coverage"]["assigned_source_line_count"] == blocks["coverage"]["source_line_count"]
        and blocks["coverage"]["assigned_entity_count"] == len(entities)
        and blocks["coverage"]["assigned_relationship_count"] == len(relationships)
    )
    block_audit = _build_block_audit(pre.get("line_meta", []), blocks, pre.get("source_format", "resume_text"))
    section_keyed_output = _build_section_keyed_output(
        block_audit, structured_data=pre.get("structured_data"), semantic=semantic
    )
    # 6. Embeddings (Sentence-BERT)
    emb = embed_entities(entities, sentences)
    # 7. Cosine similarity (analysis only; does not create graph edges)
    sim = cosine_similarity_matrix(emb["vectors"])
    # enrich top pairs with labels for UI
    texts = [e["text"] for e in entities]
    for p in sim["top_pairs"]:
        p["a"] = texts[p["i"]]
        p["b"] = texts[p["j"]]
    # 8. Clustering (analysis only; does not create graph edges)
    clusters = cluster(emb["vectors"], emb["entity_texts"])
    # 9. Graph (verified, source-grounded relationship edges only)
    # Canonical structured data is the only source used for the displayed KG.
    # Raw NLP mentions and contextual relation candidates remain available in
    # the block/entity analysis artifacts, but cannot create graph edges.
    section_keyed_output["_audit"]["semantic_block_map"] = {
        row["id"]: {k: row.get(k) for k in ("text", "raw_text", "normalized_text", "section", "page",
                                              "page_number", "bbox", "block_type", "semantic_classification",
                                              "semantic_labels", "destination", "reason")}
        for row in semantic_rows
    }
    canonical_resume = canonicalize_resume(section_keyed_output)
    # Contact labels may share a line (email | phone) or omit labels. Merge
    # the identity fields recovered from source Header rows into the stable
    # canonical contact record without changing the original source text.
    if not pre.get("structured_data"):
        header_sections = [section for section in block_audit.get("sections", [])
                           if section.get("name", "").casefold() == "header"]
        for section in header_sections:
            header_lines = [re.sub(r"^[\s•▪●*-]+", "", str(row.get("text", ""))).strip()
                            for row in block_audit.get("source_line_assignments", [])
                            if row.get("section_id") == section["id"] and str(row.get("text", "")).strip()]
            identity = {}
            for line in header_lines:
                email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", line, re.I)
                phone_match = re.search(r"(?<!\w)\+?\d[\d().\s-]{7,}\d(?!\w)", line)
                if email_match:
                    identity["email"] = email_match.group(0)
                if phone_match:
                    identity["phone"] = phone_match.group(0).strip()
                for key, pattern in (("linkedin", r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s|,;]+"),
                                     ("github", r"(?:https?://)?(?:www\.)?github\.com/[^\s|,;]+")):
                    match = re.search(pattern, line, re.I)
                    if match:
                        identity[key] = match.group(0)
                location = re.match(r"^(?:location|address|based in|city)\s*[:|–—-]\s*(.+)$", line, re.I)
                if location:
                    identity["location"] = location.group(1).strip()
            email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", " ".join(header_lines), re.I)
            phone_match = re.search(r"(?<!\w)\+?\d[\d().\s-]{7,}\d(?!\w)", " ".join(header_lines))
            location_match = next((re.match(r"^(?:location|address|based in|city)\s*[:|–—-]\s*(.+)$", line, re.I)
                                   for line in header_lines if re.match(r"^(?:location|address|based in|city)\s*[:|–—-]", line, re.I)), None)
            if email_match:
                identity["email"] = email_match.group(0)
            if phone_match:
                identity["phone"] = phone_match.group(0).strip()
            if location_match:
                identity["location"] = location_match.group(1).strip()
            if header_lines and not identity.get("name"):
                identity["name"] = header_lines[0]
            canonical_resume["personal_information"].update(identity)
    canonical_resume["unresolved_blocks"] = semantic["unresolved_blocks"]
    # Resolve remaining rendered section blocks to their canonical destination.
    section_paths = {"Header": "personal_information", "Personal Information": "personal_information",
                     "Languages": "personal_information", "Summary": "profile_snapshot", "Skills": "skills",
                     "Certifications": "certifications", "Tools and Technology": "technical_skills"}
    for row in semantic_rows:
        if not row.get("destination") and row["block_type"] not in {"PAGE_ARTIFACT", "UNKNOWN"}:
            key = section_paths.get(row["section"])
            if key:
                row["destination"] = f"$.{key}"
            elif pre.get("structured_data"):
                row["destination"] = row.get("source_path")
    semantic["coverage"] = coverage_report(semantic)
    block_audit["semantic_coverage"] = semantic["coverage"]
    graph = build_resume_graph(canonical_resume, candidate_name=candidate_name)

    canonical_relationships = graph["edges"]
    # Keep NLP context candidates distinguishable from validated graph edges
    # in coverage summaries so the UI never reports one count as the other.
    source_context_relationship_count = blocks["coverage"].get("relationship_count", 0)
    blocks["coverage"]["source_context_relationship_count"] = source_context_relationship_count
    blocks["coverage"]["canonical_relationship_count"] = len(canonical_relationships)
    blocks["coverage"]["canonical_relationships_validated"] = True
    block_audit["coverage"] = blocks["coverage"]
    block_audit["summary"]["source_context_relationship_count"] = source_context_relationship_count
    block_audit["summary"]["relationship_count"] = len(canonical_relationships)
    block_audit["summary"]["canonical_relationship_count"] = len(canonical_relationships)
    section_keyed_output["_audit"]["summary"] = block_audit["summary"]
    canonical_resume["source_audit"] = section_keyed_output["_audit"]

    artifacts = {
        "raw_text.json": raw_json,
        "entities.json": {"count": len(graph["nodes"]), "entities": [
                              {**node, "text": node["label"]} for node in graph["nodes"]
                          ], "source_mentions_count": len(entities),
                          "source_mentions": entities,
                          "preprocessing": {k: v for k, v in pre.items()
                                            if k in ("method", "sections", "sentence_count",
                                                     "normalized_preview")}},
        "relationships.json": {"count": len(canonical_relationships),
                               "relationships": canonical_relationships,
                               "note": "Every relationship is derived from a validated canonical resume record with source-path provenance."},
        "blocks.json": blocks,
        "block_audit.json": block_audit,
        "resume_blocks.json": section_keyed_output,
        "canonical_resume.json": canonical_resume,
        "semantic_blocks.json": semantic,
        "embeddings.json": {"model": emb["model"], "dim": emb["dim"],
                             "entity_texts": emb["entity_texts"],
                             "vectors_preview": emb["vectors_preview"]},
        "similarity.json": sim,
        "clusters.json": clusters,
        "graph.json": graph,
    }
    if debug:
        for row in pre_group_blocks:
            print("SEMANTIC_BLOCK " + json.dumps({
                "index": row["index"], "raw_text": row["raw_text"],
                "normalized_text": row["normalized_text"], "section": row["section"],
                "semantic_classification": row["semantic_classification"], "semantic_labels": row["semantic_labels"],
                "page_number": row["page_number"], "is_page_marker": row["is_page_marker"],
            }, ensure_ascii=False))
        employment_index = 0
        for group in semantic.get("groups", {}).values():
            if group.get("key") != "work_history":
                continue
            for record in group.get("value", []):
                for role in record.get("roles", []):
                    employment_index += 1
                    print(f"EMPLOYMENT #{employment_index}: company={record.get('company', '')}; "
                          f"role={role.get('designation', '')}; clients={role.get('clients', [])}; "
                          f"period={role.get('duration', '')}; "
                          f"responsibilities={len(role.get('role_and_responsibilities', []))}; "
                          f"source_blocks={len(role.get('source_blocks', []))}")
        coverage = semantic.get("coverage", {})
        print("BLOCK COVERAGE: "
              f"total={coverage.get('total_source_blocks', len(semantic_rows))} "
              f"classified={coverage.get('classified_blocks', 0)} "
              f"grouped={coverage.get('grouped_blocks', 0)} "
              f"page_numbers={coverage.get('ignored_page_numbers', 0)} "
              f"headers_footers={coverage.get('ignored_headers_footers', 0)} "
              f"bullet_markers={coverage.get('decorative_bullet_markers', 0)} "
              f"unknown={coverage.get('unclassified_blocks', 0)}")
        for unknown in semantic.get("unresolved_blocks", []):
            print(f"UNKNOWN BLOCK {unknown.get('source_block_id')}: {unknown.get('text')}")
    for fname, payload in artifacts.items():
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(payload, f, indent=2)
    # full vectors saved separately for reproducibility
    with open(os.path.join(out_dir, "embeddings.json"), "w") as f:
        json.dump({**artifacts["embeddings.json"], "vectors": emb["vectors"]}, f, indent=2)
    print(f"[{doc_id}] canonical_nodes={len(graph['nodes'])} canonical_rels={len(canonical_relationships)} "
          f"blocks={blocks['block_count']} edges={graph['stats']['edge_count']} "
          f"(explicit={graph['stats']['explicit_edges']}, semantic={graph['stats']['semantic_edges']})")
    return artifacts
