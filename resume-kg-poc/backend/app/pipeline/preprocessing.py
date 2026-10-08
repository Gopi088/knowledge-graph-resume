"""Step 2 — Text Normalization before parsing (structure-preserving).

Replaces the earlier destructive preprocessing (blind lowercasing,
punctuation stripping, stopword removal, flat token stream). Instead this
stage NORMALIZES the raw resume text while preserving everything the
later stages need for context:

  - original casing and punctuation are kept (needed for NER, SVO parsing
    and provenance),
  - section boundaries are detected (Summary, Experience, Projects,
    Education, Skills, Certifications, ...),
  - sentence boundaries are segmented per section (bullets/newlines first,
    then spaCy sentencizer within each line),
  - every sentence carries metadata: its section, position, and whether it
    came from a headline/bullet line.

Only conservative cleanup is applied: unicode NFKC, line-ending
unification, and trailing-whitespace stripping. Source line and blank-line
boundaries are preserved because they carry context for block grouping.

Output keeps the old `sentences: list[str]` contract (normalized sentence
strings) and adds `sentence_meta`, `sections` and `normalized_text`.
"""
import re
import json
import unicodedata
from difflib import get_close_matches

# Resume headings vary widely. Match normalized heading labels (not arbitrary
# occurrences in body text) and map common variants to stable section names.
SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "Summary": (
        "summary", "professional summary", "career summary", "profile",
        "professional profile", "profile summary", "career objective", "objective", "about me",
        "about", "personal statement", "executive summary",
    ),
    "Personal Information": (
        "personal information", "personal details", "personal detail", "contact",
        "contact information", "contact details", "personal profile",
    ),
    "Experience": (
        "experience", "work experience", "professional experience",
        "pofessional experience", "professional experiance", "work experiance",
        "employment history", "work history", "career history",
        "relevant experience", "industry experience", "project experience",
        "professional background", "internship experience", "internships", "internship",
        "career experience", "employment", "professional employment", "work background",
        "career profile", "experience summary", "career highlights", "employment experience",
    ),
    "Projects": (
        "project", "projects", "personal projects", "academic projects",
        "key projects", "research projects", "selected projects", "project experience",
        "project highlights", "key achievements and projects",
        "project portfolio", "selected work", "portfolio",
        "projects and achievements", "projects and highlights",
    ),
    "Education": (
        "education", "education qualification", "educational qualification", "educational background", "academic background",
        "academic qualifications", "academic qualification", "education qualification", "educational qualifications", "education and qualifications", "eeducation",
    ),
    "Domain Experience": (
        "domain experience", "domain expertise", "industry domain experience",
        "core banking and domain expertise",
    ),
    "Business Analysis and Product Management": (
        "business analysis and product management", "business analysis & product management",
    ),
    "Technical Program Management": ("technical program management",),
    "Skills": (
        "skills", "key skills", "core skills",
        "areas of expertise", "technical expertise",
        "skills and competencies", "skills and qualifications", "technical competencies",
        "technical proficiencies",
        "key competencies", "competencies", "areas of strength", "areas of competence",
        "technologies", "technical tools", "computer skills", "professional skills",
        "functional skills", "interpersonal skills", "soft skills", "key strengths",
        "tech stack", "tools and technologies",
    ),
    "Certifications": (
        "certification", "certifications", "certificate", "certificates",
        "licenses", "licences", "credentials", "licenses and certifications",
        "licences and certifications", "certifications and licenses",
        "certifications and licences", "certificates and licenses",
        "certificates and licences", "licenses and certificates",
        "licences and certificates", "professional certifications",
        "achievements and certifications", "achievement and certification",
        "certifications and awards", "certificates and awards", "certification and licenses",
        "certifications and licenses",
    ),
    "Technical Skills": ("technical skills", "technical skill set"),
    "Core Competencies": ("core competencies", "core competency", "core competences"),
    "Training": ("training", "professional training", "professional development", "courses", "coursework"),
    "Tools and Technology": ("tools and technology", "tools and technologies", "technology stack"),
    "Publications": ("publication", "publications", "research publications"),
    "Awards": ("award", "awards", "achievements", "honors", "honours"),
    "Volunteering": ("volunteering", "volunteer experience", "community involvement"),
    "Leadership": ("leadership", "leadership experience", "activities", "extracurricular activities"),
    "Languages": ("language", "languages"),
    "Interests": ("interests", "hobbies and interests", "hobbies"),
    "References": ("references", "professional references"),
}
_ALIAS_TO_SECTION = {
    re.sub(r"\s+", " ", alias.casefold().replace("&", "and")): section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

HEADER_CONTACT = "Header"
OTHER_SECTION = "Other"

_BULLET_CLASS = r"[\u2022\u25aa\u25cf\u25cb\u25e6\u00b7\u2013\u2014>\*\-]"
_LIST_PREFIX = re.compile(r"^\s*(?:(?P<bullet>[\u2022\u25aa\u25cf\u25cb\u25e6\u00b7\u2013\u2014>*\-])\s*|(?P<number>\d+[.)])\s*)")


def _is_header_line(line: str) -> "str | None":
    """Return canonical section name if the line looks like a section header."""
    s = line.strip()
    # PDF/text extraction often leaves bullets, numbering, or decorative
    # punctuation around a heading. Strip only those boundary characters.
    s = re.sub(r"^\s*(?:\d+[.)]|[•▪●*#])\s*", "", s)
    s = re.sub(r"^[|–—_\-:]+|[|–—_\-:]+$", "", s).strip()
    if not s or len(s) > 60:
        return None
    # A colon is a common heading delimiter (e.g. "Certificates:"). Do not
    # accept a sentence containing a section word as a heading.
    s = re.sub(r"\s*:\s*$", "", s).strip()
    normalized = re.sub(r"\s+", " ", s.casefold().replace("&", "and"))
    normalized = normalized.strip(" .:|–—_-\t")
    if normalized in _ALIAS_TO_SECTION:
        return _ALIAS_TO_SECTION[normalized]

    # Singular achievement labels commonly introduce a bullet inside an
    # employment entry ("Achievement: Increased revenue ..."); treating that
    # short label as the Awards section would cut off the active job context.
    if normalized in {"achievement", "accomplishment"}:
        return None

    # Distinguish common skill sub-headings from the parent skill inventory.
    if normalized in {"core competencies", "core competency", "core competences"}:
        return "Core Competencies"

    # Resume headings frequently contain OCR/typing noise. Correct only
    # heading-like lines and require a close match to a known heading.
    if len(normalized) <= 60 and not re.search(r"[.!?]", s):
        close = get_close_matches(normalized, _ALIAS_TO_SECTION.keys(), n=1, cutoff=0.88)
        if close:
            return _ALIAS_TO_SECTION[close[0]]

    # Preserve short custom headings when they are clearly formatted as
    # headings. This lets a resume use labels such as "CERTIFICATES & AWARDS"
    # without letting normal sentence text change the active section.
    custom_heading_terms = re.compile(
        r"\b(summary|experience|employment|education|skill|certification|certificate|"
        r"award|achievement|project|publication|training|objective|profile|expertise|"
        r"language|interest|reference|volunteer|leadership|contact|personal|technology)\b", re.I,
    )
    if (re.fullmatch(r"[A-Z][A-Z\s&/\\-]{2,58}", s)
            and " " in s.strip()
            and custom_heading_terms.search(s)
            and not re.search(r"[.!?/#]", s)):
        return s.title()
    return None


def _section_header_and_body(line: str) -> tuple[str | None, str]:
    """Recognize standalone headings and ``Heading: body`` source lines."""
    section = _is_header_line(line)
    if section:
        return section, ""
    # Some exports place the first item on the same line as its heading.
    # Only split at a colon when the left side is itself a known heading.
    if ":" in line:
        heading, body = line.split(":", 1)
        section = _is_header_line(heading)
        if section and body.strip():
            return section, body.strip()
    return None, ""


def _json_label(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", value)).strip().title()


def _json_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value).strip()


def _render_resume_json(data) -> tuple[str, dict[int, str]]:
    """Render structured resume JSON into readable source rows with JSON paths."""
    if not isinstance(data, dict):
        return "", {}
    rendered: list[str] = []
    paths: dict[int, str] = {}

    def add(text: str, path: str):
        if text.strip():
            paths[len(rendered)] = path
            rendered.append(text.strip())

    def scalar_fields(obj: dict, path: str) -> str:
        parts = []
        for key, value in obj.items():
            if isinstance(value, (str, int, float, bool)) and _json_scalar(value):
                parts.append(f"{_json_label(str(key))}: {_json_scalar(value)}")
        return " | ".join(parts)

    def section_name(key: str) -> str:
        key_norm = re.sub(r"\s+", " ", key.replace("_", " ").replace("-", " ").casefold()).strip()
        aliases = {
            "personal information": "Personal Information",
            "personal details": "Personal Information",
            "contact information": "Personal Information",
            "profile snapshot": "Summary",
            "professional summary": "Summary",
            "work history": "Experience",
            "employment history": "Experience",
            "experience": "Experience",
            "projects": "Projects",
            "education": "Education",
            "certifications": "Certifications",
            "certificates": "Certifications",
            "skills": "Skills",
            "tools and technology": "Tools and Technology",
            "tools and technologies": "Tools and Technology",
            "technical skills": "Technical Skills",
            "technical skill set": "Technical Skills",
            "core competencies": "Core Competencies",
            "core competency": "Core Competencies",
        }
        return aliases.get(key_norm, _json_label(key))

    identity_extras = {
        key: data[key] for key in ("place", "name_at_end") if key in data
    }
    root = data.items()
    for key, value in root:
        key_text = str(key)
        key_norm = re.sub(r"\s+", " ", key_text.replace("_", " ").replace("-", " ").casefold()).strip()
        # These trailing identity values belong with contact/personal details,
        # rather than becoming standalone resume sections.
        if key_norm in {"place", "name at end"}:
            # Fold trailing identity fields back into the contact section.
            continue

        heading = section_name(key_text)
        add(heading.upper(), f"$.{key_text}")
        path = f"$.{key_text}"
        if isinstance(value, dict):
            nested = list(value.items())
            if all(not isinstance(item, (dict, list)) for _, item in nested):
                text = scalar_fields(value, path)
                add(text, path)
                if key_norm in {"personal information", "personal details", "contact information"}:
                    for extra_key, extra_value in identity_extras.items():
                        add(f"{_json_label(extra_key)}: {_json_scalar(extra_value)}", f"$.{extra_key}")
            else:
                for nested_key, nested_value in nested:
                    nested_path = f"{path}.{nested_key}"
                    label = _json_label(str(nested_key))
                    if isinstance(nested_value, list):
                        values = [str(item).strip() for item in nested_value if not isinstance(item, (dict, list)) and str(item).strip()]
                        if values:
                            add(f"{label}: " + "; ".join(values), nested_path)
                        else:
                            for index, item in enumerate(nested_value):
                                add(scalar_fields(item, nested_path + f"[{index}]") if isinstance(item, dict) else _json_scalar(item), nested_path + f"[{index}]")
                    elif isinstance(nested_value, dict):
                        text = scalar_fields(nested_value, nested_path)
                        if text:
                            add(f"{label}: {text}", nested_path)
                    else:
                        add(f"{label}: {_json_scalar(nested_value)}", nested_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                item_path = f"{path}[{index}]"
                if isinstance(item, dict):
                    if key_norm in {"work history", "experience", "employment history"}:
                        company = _json_scalar(item.get("company", item.get("employer", "")))
                        role = _json_scalar(item.get("designation", item.get("role", item.get("title", ""))))
                        duration = _json_scalar(item.get("duration", item.get("dates", "")))
                        header = " at ".join(part for part in (role, company) if part)
                        if duration:
                            header += f" ({duration})"
                        add(header or scalar_fields(item, item_path), item_path)
                        for child_key, child_value in item.items():
                            if child_key in {"company", "employer", "designation", "role", "title", "duration", "dates"}:
                                continue
                            if isinstance(child_value, list):
                                for child_index, child in enumerate(child_value):
                                    if isinstance(child, (str, int, float)):
                                        add(f"• {_json_scalar(child)}", f"{item_path}.{child_key}[{child_index}]")
                            elif child_value:
                                add(f"{_json_label(str(child_key))}: {_json_scalar(child_value)}", f"{item_path}.{child_key}")
                    else:
                        prefix = "• " if key_norm == "education" else ""
                        add(prefix + scalar_fields(item, item_path), item_path)
                elif isinstance(item, list):
                    add("; ".join(_json_scalar(child) for child in item if _json_scalar(child)), item_path)
                else:
                    add(f"• {_json_scalar(item)}", item_path)
        elif value is not None:
            add(_json_scalar(value), path)

    return "\n".join(rendered), paths


def normalize_text(raw_text: str) -> str:
    """Conservative cleanup only — case, punctuation and layout survive."""
    text = unicodedata.normalize("NFKC", raw_text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Retain hyphens and line boundaries: both may carry source meaning.
    # Keep source line positions; continuation joining belongs to semantic grouping.
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    # Preserve blank lines as boundary evidence.
    return text.strip()


def _split_into_lines(normalized: str) -> list[str]:
    return [ln for ln in normalized.split("\n")]


def detect_sections(normalized: str) -> tuple[list[dict], list[dict]]:
    """Split normalized text into sections.

    Returns (sections, line_meta) where each line_meta entry is
    {"line": str, "section": str, "is_header": bool}.
    Text before the first header belongs to the Header (contact) section.
    """
    line_meta: list[dict] = []
    current = HEADER_CONTACT
    section_number = 0
    current_section_id = f"s{section_number}:Header"
    source_lines = _split_into_lines(normalized)
    split_heading_continuations: set[int] = set()
    for line_index, line in enumerate(source_lines):
        if line_index in split_heading_continuations:
            prefix = _LIST_PREFIX.match(line)
            line_meta.append({
                "line_index": line_index, "source_line": line.strip(), "line": line.strip(),
                "indent": len(line) - len(line.lstrip()), "is_bullet": bool(prefix),
                "list_marker": (prefix.group("bullet") or prefix.group("number")) if prefix else "",
                "section_id": current_section_id, "section": current, "is_header": True,
            })
            continue
        header, inline_body = _section_header_and_body(line)
        if not header and line.strip() and line_index + 1 < len(source_lines):
            combined = f"{line.strip()} {source_lines[line_index + 1].strip()}"
            combined_key = re.sub(r"\s+", " ", combined.casefold().replace("&", "and")).strip()
            if combined_key in _ALIAS_TO_SECTION:
                header = _ALIAS_TO_SECTION[combined_key]
                split_heading_continuations.add(line_index + 1)
        prefix = _LIST_PREFIX.match(line)
        is_page_number = bool(re.fullmatch(r"\s*(?:page\s+)?\d{1,3}(?:\s*(?:of|/)\s*\d+)?\s*", line, re.I))
        common = {
            "line_index": line_index,
            "source_line": line.strip(),
            "indent": len(line) - len(line.lstrip()),
            "is_bullet": bool(prefix),
            "list_marker": (prefix.group("bullet") or prefix.group("number")) if prefix else "",
            "is_page_number": is_page_number,
        }
        if header:
            current = header
            section_number += 1
            current_section_id = f"s{section_number}:{current}"
        common["section_id"] = current_section_id
        if header:
            # register section lazily below; header line itself is not body
            line_meta.append({**common, "line": line.strip(), "section": current, "is_header": True})
            if inline_body:
                line_meta.append({**common, "line": inline_body, "section": current, "is_header": False})
        elif line.strip() == "":
            line_meta.append({**common, "line": "", "section": current, "is_header": False})
        else:
            line_meta.append({**common, "line": line.strip(), "section": current, "is_header": False})

    # Headingless resumes are common. Infer only from high-signal record shapes
    # (dated job headers, qualification rows, project/certificate labels), then
    # let subsequent source lines inherit that local context. Avoid embedding
    # similarity or entity types as section evidence.
    if section_number == 0 or any(lm.get("section") == HEADER_CONTACT for lm in line_meta):
        active = HEADER_CONTACT
        inferred_index = 0
        saw_content = False
        role_words = re.compile(
            r"\b(analyst|engineer|developer|consultant|manager|director|designer|architect|"
            r"scientist|officer|specialist|associate|executive|intern|underwriter|administrator|"
            r"technician|nurse|teacher|coordinator|accountant|researcher|president|lead)\b", re.I
        )
        date_range = re.compile(
            r"(?:\b(?:19|20)\d{2}\s*[-–—/]|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}.*[-–—])",
            re.I,
        )
        degree_words = re.compile(
            r"\b(bachelor|master|ph\.?d|mba|b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|"
            r"b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|b\.?com\.?|m\.?com\.?|"
            r"diploma|associate degree|high school|secondary school|ssc|hsc|ged)\b", re.I
        )
        for lm in line_meta:
            if lm.get("is_header"):
                active = lm["section"]
                continue
            if lm.get("section") != HEADER_CONTACT:
                active = lm["section"]
                continue
            if lm.get("is_page_number") or not lm.get("line", "").strip():
                continue
            value = lm["line"].strip()
            folded = value.casefold().strip(" :.-")
            inferred = None
            if re.match(r"^(?:key\s+)?projects?\s*[:\-]", value, re.I):
                inferred = "Projects"
            elif re.match(r"^(?:education|academic qualifications?)\s*[:\-]", value, re.I):
                inferred = "Education"
            elif re.match(r"^(?:certifications?|certificates?|licenses?|licences?)\s*[:\-]", value, re.I):
                inferred = "Certifications"
            elif re.match(r"^(?:technical\s+)?skills?\s*[:\-]", value, re.I):
                inferred = "Skills"
            elif degree_words.search(value) and (re.search(r"\b(from|university|college|institute|school)\b", value, re.I)
                    or date_range.search(value)):
                inferred = "Education"
            elif (degree_words.match(value) and len(value) < 140
                  and not re.search(r"[.!?]", value)):
                inferred = "Education"
            elif date_range.search(value) and role_words.search(value) and len(value) < 260:
                inferred = "Experience"
            elif (role_words.search(value) and len(value) < 180
                  and re.search(r"\s(?:at|[@|])\s|\s[–—]\s", value, re.I)
                  and not re.search(r"[.!?]", value)):
                inferred = "Experience"
            elif re.search(r"\b(certified|certification|certificate)\b", value, re.I):
                inferred = "Certifications"
            elif (active != "Experience" and value.count(",") >= 2 and
                  len(re.findall(r"\b(python|java|sql|aws|azure|docker|react|excel|tensorflow|machine learning|git|postgresql|kubernetes|pandas|tableau)\b", value, re.I)) >= 2
                  and not re.search(r"\b(certified|certification|certificate)\b", value, re.I)):
                inferred = "Skills"
            elif re.match(r"^(?:project\s*(?:name|title)?\s*[:\-])", value, re.I):
                inferred = "Projects"
            elif (active == "Header" and saw_content and
                  len(value) > 45 and
                  (re.search(r"\b(experience|professional|skilled|expertise|speciali[sz]e|career objective)\b", value, re.I)
                   or re.search(r"[.!?]$", value))):
                inferred = "Summary"

            if inferred:
                active = inferred
                inferred_index += 1
            elif not saw_content and active == HEADER_CONTACT:
                active = HEADER_CONTACT
            lm["section"] = active
            lm["section_id"] = f"s{inferred_index}:{active}"
            lm["is_inferred_section"] = inferred is not None
            saw_content = True

    sections: list[dict] = []
    for lm in line_meta:
        if lm["is_header"] and not any(s["name"] == lm["section"] for s in sections):
            sections.append({"name": lm["section"], "is_header": True})
    # ensure every body section is represented even without explicit header
    for lm in line_meta:
        if not lm["is_header"] and lm["line"]:
            if not any(s["name"] == lm["section"] for s in sections):
                sections.append({"name": lm["section"], "is_header": False})
    if not sections:
        sections = [{"name": OTHER_SECTION, "is_header": False}]
    return sections, line_meta


def _segment_line(nlp, line: str) -> list[str]:
    """Segment one body line into sentence strings (case/punct preserved)."""
    # strip leading bullet markers but remember nothing else about them
    s = re.sub(rf"^{_BULLET_CLASS}\s*", "", line).strip()
    s = re.sub(r"^\d+[.)]\s*", "", s).strip()
    if not s:
        return []
    if nlp is not None:
        try:
            return [sp.text.strip() for sp in nlp(s).sents if sp.text.strip()]
        except Exception:
            pass
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    return parts or [s]


def get_nlp():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp
    except ImportError:
        return None


def preprocess(raw_text: str) -> dict:
    """Normalize + segment. Keeps `sentences: list[str]` contract."""
    nlp = get_nlp()
    normalized = normalize_text(raw_text)
    source_format = "resume_text"
    json_paths: dict[int, str] = {}
    structured_data = None
    try:
        parsed_json = json.loads(normalized)
        structured_text, json_paths = _render_resume_json(parsed_json)
        if structured_text:
            normalized = normalize_text(structured_text)
            source_format = "structured_resume_json"
            structured_data = parsed_json
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    sections, line_meta = detect_sections(normalized)
    for line in line_meta:
        if line["line_index"] in json_paths:
            line["source_path"] = json_paths[line["line_index"]]

    sentences: list[str] = []
    sentence_meta: list[dict] = []
    for idx, lm in enumerate(line_meta):
        if lm["is_header"] or lm.get("is_page_number") or not lm["line"]:
            continue
        for sent in _segment_line(nlp, lm["line"]):
            sentences.append(sent)
            sentence_meta.append({
                "index": len(sentences) - 1,
                "section": lm["section"],
                "section_id": lm.get("section_id", f"s0:{lm['section']}"),
                "line_index": lm.get("line_index", idx),
                "line_text": lm.get("source_line", lm["line"]),
                "indent": lm.get("indent", 0),
                "is_bullet": lm.get("is_bullet", False),
                "list_marker": lm.get("list_marker", ""),
                "source_path": lm.get("source_path"),
            })

    # sentence/section counts for research visibility (no destructive previews)
    per_section: dict[str, int] = {}
    section_names: dict[str, str] = {}
    for m in sentence_meta:
        sid = m.get("section_id", f"s0:{m['section']}")
        per_section[sid] = per_section.get(sid, 0) + 1
        section_names[sid] = m["section"]

    return {
        "method": ("structure-preserving text normalization (unicode NFKC, "
                   "trailing-whitespace cleanup; source line boundaries retained) + "
                   "section detection + per-section sentence segmentation; "
                   "casing, punctuation and section context preserved"),
        "normalized_text": normalized,
        "source_format": source_format,
        # Preserve the caller's original hierarchy for the structured-output
        # artifact. Flattened rows remain useful to NLP, but are lossy as a
        # replacement for an already structured resume.
        "structured_data": structured_data,
        "normalized_preview": normalized[:500],
        "sentences": sentences,
        "sentence_count": len(sentences),
        "sentence_meta": sentence_meta,
        # Kept for a complete source-to-section/block audit, including heading
        # lines that intentionally do not become entity-bearing sentences.
        "line_meta": line_meta,
        "sections": [{"id": sid, "name": name, "sentence_count": per_section[sid]}
                     for sid, name in section_names.items()],
    }
