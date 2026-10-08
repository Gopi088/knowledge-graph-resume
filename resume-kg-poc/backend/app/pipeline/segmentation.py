"""High precision, structure-led resume context blocks.

Sentence embeddings and shared entity keywords never merge source contexts.
Sentences from one physical line stay together, wrapped lines stay with their
explicit list item, and education details can join a degree/institution entry
when their labels and order provide structural evidence.
"""
import re

_DEGREE = re.compile(r"\b(b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|bachelor(?:s)?|master(?:s)?|mba|ph\.?d|bca|mca)\b", re.I)
_INSTITUTION = re.compile(r"\b(university|institute|college|school|vidyalaya|iit|nit)\b", re.I)
_EDU_DETAIL = re.compile(r"\b(cgpa|gpa|percentage|year|graduat(?:ed|ion)|\d{4}\s*[-–]\s*\d{4})\b", re.I)
_JOB_HEADER = re.compile(
    r"\b(engineer|developer|scientist|analyst|manager|intern|designer|consultant|specialist|underwriter|executive|director|officer|architect|administrator|associate|accountant)\b"
    r".*(?:\bat\b|\b(?:19|20)\d{2}\b)", re.I,
)
_PROJECT_TITLE = re.compile(r"\b(project|app|dashboard|system|platform|website)\b", re.I)
_PROJECT_ACTION = re.compile(r"\b(built|build|developed|created|implemented|designed|using|powered\s+by|based\s+on)\b", re.I)


def _line_records(sentences: list[str], sentence_meta: list[dict]) -> list[dict]:
    by_line: dict[tuple[str, int], dict] = {}
    for index, sentence in enumerate(sentences):
        meta = sentence_meta[index] if index < len(sentence_meta) else {}
        section = meta.get("section", "Other")
        section_id = meta.get("section_id", f"s0:{section}")
        line_index = meta.get("line_index", index)
        key = (section_id, line_index)
        record = by_line.setdefault(key, {
            "section": section,
            "section_id": section_id,
            "line_index": line_index,
            "line_text": meta.get("line_text", sentence),
            "is_bullet": bool(meta.get("is_bullet", False)),
            "indent": meta.get("indent", 0),
            "source_path": meta.get("source_path"),
            "indices": [],
        })
        record["indices"].append(index)
    return sorted(by_line.values(), key=lambda record: record["line_index"])


def detect_blocks(
    sentences: list[str],
    sentence_meta: "list[dict] | None" = None,
    entities: "list[dict] | None" = None,
) -> dict:
    """Build blocks from explicit line/list/education structure only."""
    n = len(sentences)
    sentence_meta = sentence_meta or [{"index": i, "section": "Other", "line_index": i} for i in range(n)]
    lines = _line_records(sentences, sentence_meta)
    parent = list(range(n))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    decisions: list[dict] = []

    def join_lines(left: dict, right: dict, reason: str):
        if left["section_id"] != right["section_id"]:
            return
        for left_index in left["indices"]:
            for right_index in right["indices"]:
                union(left_index, right_index)
        decisions.append({
            "line_pair": [left["line_index"], right["line_index"]],
            "sentence_pair": [left["indices"][0], right["indices"][0]],
            "merged": True,
            "reason": reason,
        })

    # Sentences split from one physical line are one source bullet/context.
    for line in lines:
        for left, right in zip(line["indices"], line["indices"][1:]):
            union(left, right)
        if len(line["indices"]) > 1:
            decisions.append({
                "line_pair": [line["line_index"], line["line_index"]],
                "sentence_pair": [line["indices"][0], line["indices"][-1]],
                "merged": True,
                "reason": "sentences from the same physical source line",
            })

    # A list item may wrap onto following unbulleted lines. Stop at the next
    # list marker or section boundary, so neighboring bullets stay separate.
    active_item = None
    for line in lines:
        if active_item and line["section_id"] != active_item["section_id"]:
            active_item = None
        if line["is_bullet"]:
            active_item = line
        elif active_item:
            previous_is_open = not re.search(r"[.!?;]\s*$", active_item["line_text"])
            is_indented_continuation = line["indent"] > active_item["indent"]
            if previous_is_open or is_indented_continuation:
                join_lines(active_item, line, "wrapped or indented continuation of the same marked list item")
                active_item = line
            else:
                active_item = None

    # A separate project title can own the following description line when
    # that line has an implementation cue or is visibly indented. A new title
    # starts a new project context.
    active_project = None
    for line in (record for record in lines if record["section"].lower() == "projects"):
        if active_project and active_project["section_id"] != line["section_id"]:
            active_project = None
        is_title = bool(_PROJECT_TITLE.search(line["line_text"])) and not _PROJECT_ACTION.search(line["line_text"])
        if line["is_bullet"] and is_title:
            active_project = line
        elif is_title:
            active_project = line
        elif active_project and (_PROJECT_ACTION.search(line["line_text"]) or line["indent"] > active_project["indent"]):
            join_lines(active_project, line, "description structurally attached to the preceding project title")
            active_project = line
        else:
            active_project = None

    # Keep continuation lines together when the source explicitly carries a
    # wrapped comma list, as is common in Skills and Tools sections.
    for left, right in zip(lines, lines[1:]):
        if left["section_id"] != right["section_id"] or right["is_bullet"]:
            continue
        if left["section"].casefold() in {"skills", "technical skills", "tools", "technologies"}:
            if re.search(r",\s*$", left["line_text"]) or right["indent"] > left["indent"]:
                join_lines(left, right, "wrapped continuation of the same skill/tool list")

    # Summary/contact details are one section-level context when represented
    # as adjacent plain lines; explicit bullets remain separate list items.
    for left, right in zip(lines, lines[1:]):
        if (left["section_id"] == right["section_id"]
                and left["section"].casefold() in {"summary", "personal information", "header"}
                and right["line_index"] == left["line_index"] + 1
                and not left["is_bullet"] and not right["is_bullet"]):
            join_lines(left, right, "consecutive plain lines in the same summary/contact section")

    # Experience role headers and their responsibility lines form one job
    # entry. A new role/company/date header starts a new entry; every other
    # responsibility remains tied to the current entry, within Experience.
    active_job = None
    for line in (record for record in lines if record["section"].lower() == "experience"):
        if active_job and active_job["section_id"] != line["section_id"]:
            active_job = None
        if _JOB_HEADER.search(line["line_text"]):
            active_job = line
        elif active_job:
            join_lines(active_job, line, "responsibility line under the same experience role entry")

    # Education entries often print the institution, degree, and grade/date on
    # separate lines. Join only adjacent education lines with those explicit
    # structural labels; a new bullet or degree begins a separate entry.
    education_lines = [line for line in lines if line["section"].lower() == "education"]
    active_education: list[dict] = []
    active_has_degree = False
    for line in education_lines:
        text = line["line_text"]
        has_degree = bool(_DEGREE.search(text))
        has_institution = bool(_INSTITUTION.search(text))
        has_detail = bool(_EDU_DETAIL.search(text))
        eligible = has_degree or has_institution or has_detail
        if line["is_bullet"] or (has_degree and active_has_degree):
            active_education = []
            active_has_degree = False
        if not eligible:
            active_education = []
            active_has_degree = False
            continue
        if active_education:
            previous = active_education[-1]
            if (line["section_id"] != previous["section_id"]
                    or line["line_index"] != previous["line_index"] + 1):
                active_education = []
                active_has_degree = False
        if active_education:
            join_lines(active_education[-1], line, "adjacent education entry detail (degree, institution, or result)")
        active_education.append(line)
        active_has_degree = active_has_degree or has_degree

    # Record rejected adjacent lines to make the hard boundaries auditable.
    for left, right in zip(lines, lines[1:]):
        if left["section_id"] != right["section_id"]:
            reason = "different resume sections"
        elif find(left["indices"][0]) == find(right["indices"][0]):
            continue
        else:
            reason = "separate source lines without a list or structural entry link"
        decisions.append({
            "line_pair": [left["line_index"], right["line_index"]],
            "sentence_pair": [left["indices"][0], right["indices"][0]],
            "merged": False,
            "reason": reason,
        })

    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)

    blocks = []
    sentence_blocks = [""] * n
    for block_number, (_, indices) in enumerate(sorted(groups.items(), key=lambda pair: min(pair[1]))):
        block_id = f"b{block_number}"
        section_names = sorted({
            sentence_meta[index].get("section", "Other")
            for index in indices if index < len(sentence_meta)
        }) or ["Other"]
        section_ids = sorted({
            sentence_meta[index].get("section_id", f"s0:{sentence_meta[index].get('section', 'Other')}")
            for index in indices if index < len(sentence_meta)
        }) or [f"s0:{section_names[0]}"]
        source_lines = sorted({
            sentence_meta[index].get("line_index", index)
            for index in indices if index < len(sentence_meta)
        })
        source_line_texts = []
        source_paths = []
        for line in lines:
            if line["line_index"] in source_lines and line["section_id"] in section_ids:
                if line["line_text"] not in source_line_texts:
                    source_line_texts.append(line["line_text"])
                if line.get("source_path") and line["source_path"] not in source_paths:
                    source_paths.append(line["source_path"])
        root = find(indices[0])
        reasons = []
        for decision in decisions:
            if decision["merged"] and decision["sentence_pair"]:
                pair_roots = {find(i) for i in decision["sentence_pair"]}
                if root in pair_roots and decision["reason"] not in reasons:
                    reasons.append(decision["reason"])
        section = section_names[0] if len(section_names) == 1 else "|".join(section_names)
        joined_text = " ".join(source_line_texts)
        if section.casefold() == "education":
            entry_type = "education_entry"
        elif section.casefold() == "experience":
            entry_type = "experience_entry" if _JOB_HEADER.search(joined_text) else "experience_responsibility"
        elif section.casefold() == "projects":
            entry_type = "project_entry"
        elif section.casefold() == "summary":
            entry_type = "summary_context"
        elif section.casefold() in {"personal information", "header"}:
            entry_type = "personal_information_entry"
        elif section.casefold() in {"skills", "technical skills", "tools", "technologies"}:
            entry_type = "skill_list_entry"
        else:
            entry_type = "source_line_context"
        for index in indices:
            sentence_blocks[index] = block_id
        blocks.append({
            "id": block_id,
            "section_id": section_ids[0] if len(section_ids) == 1 else "|".join(section_ids),
            "section": section,
            "entry_type": entry_type,
            "sentence_indices": indices,
            "source_line_indices": source_lines,
            "source_lines": source_line_texts,
            "source_paths": source_paths,
            "sentences": [sentences[index] for index in indices],
            "assignment_method": "structural" if reasons else "source_line",
            "assignment_reason": "; ".join(reasons) if reasons else "Retained as a complete source-line context; no cross-line merge was justified.",
            "entities": [],
            "entity_ids": [],
            "relationship_ids": [],
            "related_block_ids": [],
        })

    line_blocks = {}
    sentence_layers = []
    for block in blocks:
        for index in block["sentence_indices"]:
            meta = sentence_meta[index] if index < len(sentence_meta) else {}
            line_index = meta.get("line_index", index)
            line_blocks[str(line_index)] = block["id"]
            sentence_layers.append({
                "sentence_index": index,
                "line_index": line_index,
                "section_id": block["section_id"],
                "block_id": block["id"],
            })

    section_layers = []
    for section_id in dict.fromkeys(block["section_id"] for block in blocks):
        section_blocks = [block for block in blocks if block["section_id"] == section_id]
        section_layers.append({
            "id": section_id,
            "name": section_blocks[0]["section"],
            "block_ids": [block["id"] for block in section_blocks],
            "sentence_indices": sorted({i for block in section_blocks for i in block["sentence_indices"]}),
        })

    return {
        "method": "layered source contexts: section instance > structural entry block > sentence/source line > entity occurrence; physical lines, explicit list continuations, experience entries, education details, and summary context; no embedding/keyword merging",
        "thresholds": {"semantic_merge_threshold": None},
        "layers": ["section", "context_block", "sentence", "entity", "verified_relationship"],
        "sections": section_layers,
        "blocks": blocks,
        "block_count": len(blocks),
        "sentence_blocks": sentence_blocks,
        "sentence_layers": sentence_layers,
        "line_blocks": line_blocks,
        "coverage": {
            "sentence_count": n,
            "assigned_sentence_count": sum(bool(block_id) for block_id in sentence_blocks),
            "source_line_count": len({line["line_index"] for line in lines}),
            "assigned_source_line_count": len(line_blocks),
            "complete": all(sentence_blocks) and len(line_blocks) == len({line["line_index"] for line in lines}),
        },
        "merge_decisions": decisions,
    }
