"""Stable, source-preserving resume representation used by graph generation."""
from __future__ import annotations

from copy import deepcopy
import re


ALIASES = {
    "contact": "personal_information",
    "contact_details": "personal_information", "contact_information": "personal_information",
    "summary": "profile_snapshot", "professional_summary": "profile_snapshot",
    "career_summary": "profile_snapshot", "experience": "work_history",
    "employment_history": "work_history", "professional_experience": "work_history",
    "professional_background": "work_history", "projects_highlights": "projects",
    "project_highlights": "projects", "key_projects": "projects",
    "domain_expertise": "domain_experience", "industry_domain_experience": "domain_experience",
    "core_banking_and_domain_expertise": "domain_experience",
    "technical_skill": "technical_skills", "tools_and_technology": "technical_skills",
    "tools_and_technologies": "technical_skills", "certificates": "certifications",
    "licenses": "certifications", "licences": "certifications",
    "honors": "awards", "honours": "awards", "publication": "publications",
}


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        flattened = []
        for item in value:
            if isinstance(item, list):
                flattened.extend(_as_list(item))
            elif item not in (None, ""):
                flattened.append(item)
        return flattened
    return [value] if value != "" else []


def _normalize_roles(job: dict) -> list[dict]:
    raw_roles = job.get("roles")
    if not isinstance(raw_roles, list):
        raw_roles = [job] if any(k in job for k in ("designation", "role", "title", "position")) else []
    roles = []
    for raw in raw_roles:
        if isinstance(raw, str):
            raw = {"designation": raw}
        if not isinstance(raw, dict):
            continue
        role = deepcopy(raw)
        role["designation"] = (role.get("designation") or role.get("job_title") or
                               role.get("title") or role.get("role") or role.get("position") or "")
        responsibilities = (role.get("role_and_responsibilities") or role.get("responsibilities") or
                             role.get("duties") or [])
        role["role_and_responsibilities"] = [str(item).strip() for item in _as_list(responsibilities)
                                               if str(item).strip()]
        if "achievements" in role:
            role["achievements"] = [str(item).strip() for item in _as_list(role["achievements"])
                                    if str(item).strip()]
        raw_period = role.get("employment_period")
        if not role.get("duration"):
            role["duration"] = (raw.get("dates") or raw.get("date_range") or raw.get("period") or
                                (raw_period.get("source_text") if isinstance(raw_period, dict) else raw_period) or "")
        clients = role.get("clients", role.get("client", []))
        role["clients"] = [str(item).strip() for item in _as_list(clients) if str(item).strip()]
        if not role["clients"] and job.get("client"):
            # A company-level client can be assigned without ambiguity only
            # when the employer has a single role.
            siblings = job.get("roles")
            if not isinstance(siblings, list) or len(siblings) <= 1:
                role["clients"] = [str(item).strip() for item in _as_list(job["client"]) if str(item).strip()]
        period = str(role.get("duration") or "").strip()
        if isinstance(raw_period, dict):
            structured_period = deepcopy(raw_period)
            structured_period.setdefault("source_text", period)
            role["employment_periods"] = [structured_period]
            if not period:
                period = str(structured_period.get("source_text") or "").strip()
                role["duration"] = period
        else:
            role["employment_periods"] = ([{"source_text": period, **_parse_period(period)}] if period else [])
        if role["employment_periods"]:
            role["employment_period"] = role["employment_periods"][0]
            if role["employment_period"].get("start_date"):
                role["start_date"] = role["employment_period"]["start_date"]
            if role["employment_period"].get("end_date"):
                role["end_date"] = role["employment_period"]["end_date"]
        for alias in ("company", "company_name", "employer", "organization", "responsibilities",
                      "duties", "job_title", "title", "role", "position",
                      "dates", "date_range", "period", "client"):
            role.pop(alias, None)
        roles.append(role)
    return roles


def _parse_period(value: str) -> dict:
    """Expose dates only when a clear date range is present in source text."""
    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    atom = rf"(?:{months}\.?\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
    match = re.fullmatch(rf"\s*({atom})\s*(?:[-–—]|to)\s*({atom}|present|current|till\s+date|to\s+date)\s*", value, re.I)
    if not match:
        return {}
    end = re.sub(r"^(?:till|to)\s+date$", "Present", match.group(2), flags=re.I)
    return {"start_date": match.group(1), "end_date": end}


def canonicalize_resume(resume: dict) -> dict:
    """Return predictable core collections while retaining unknown sections.

    The original section-keyed extraction remains intact in ``source_sections``
    so normalization never discards fields the canonical core does not model.
    """
    source = deepcopy(resume or {})
    source.pop("_audit", None)
    normalized = {}
    for key, value in source.items():
        canonical_key = ALIASES.get(_key(key), _key(key))
        if canonical_key in normalized and isinstance(normalized[canonical_key], list):
            normalized[canonical_key].extend(_as_list(value))
        else:
            normalized[canonical_key] = value

    personal = normalized.get("personal_information", {})
    if not isinstance(personal, dict):
        personal = {"additional_details": _as_list(personal)} if personal else {}
    # A few upstream structured extractors use list positions as object keys
    # (for example {"0": "Candidate Name"}). Promote a plausible first
    # identity string to name and keep other numeric-key values as details.
    for key in list(personal):
        if str(key).isdecimal():
            value = personal.pop(key)
            if (str(key) == "0" and not personal.get("name") and isinstance(value, str)
                    and value.strip() and not re.search(r"@|\d", value)):
                personal["name"] = value.strip()
            elif value not in (None, ""):
                personal.setdefault("additional_details", []).extend(_as_list(value))
    personal_details = normalized.get("personal_details", {})
    if not isinstance(personal_details, dict):
        personal_details = {"additional_details": _as_list(personal_details)} if personal_details else {}
    embedded_details = personal.pop("personal_details", {})
    if isinstance(embedded_details, dict):
        personal_details = {**embedded_details, **personal_details}
    for key in ("name", "title", "email", "phone", "location"):
        if key in personal_details:
            personal.setdefault(key, personal_details.pop(key))
        if key not in personal and key in normalized:
            personal[key] = normalized[key]
    for key in ("date_of_birth", "nationality", "languages"):
        if key in personal:
            personal_details.setdefault(key, personal.pop(key))
        if key in normalized:
            personal_details.setdefault(key, normalized[key])
    if "languages" in personal_details:
        personal_details["languages"] = _as_list(personal_details["languages"])
    if personal.get("email") and isinstance(personal["email"], str):
        contact = personal["email"]
        email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", contact, re.I)
        phone_match = re.search(r"(?<!\w)\+?\d[\d().\s-]{6,}\d(?!\w)", contact)
        if email_match:
            personal["email"] = email_match.group(0)
        if phone_match and not personal.get("phone"):
            personal["phone"] = phone_match.group(0).strip()

    jobs = []
    for raw_job in _as_list(normalized.get("work_history")):
        if not isinstance(raw_job, dict):
            raise ValueError(
                "Raw work-history text cannot be serialized as an employment record; "
                "it must be classified and grouped into company/role records first."
            )
        job = deepcopy(raw_job)
        company = (job.get("company") or job.get("employer") or job.get("company_name") or
                   job.get("organization") or "")
        if isinstance(company, dict):
            job.setdefault("company_details", deepcopy(company))
            company = company.get("name") or company.get("company_name") or company.get("organization") or ""
        job["company"] = company
        job["duration"] = (job.get("duration") or job.get("dates") or job.get("date_range") or
                           job.get("period") or job.get("employment_period") or "")
        job["roles"] = _normalize_roles(job)
        if len(job["roles"]) == 1 and not job["roles"][0].get("duration") and job.get("duration"):
            role = job["roles"][0]
            role["duration"] = str(job["duration"])
            role["employment_periods"] = [{"source_text": role["duration"], **_parse_period(role["duration"])}]
            role["employment_period"] = role["employment_periods"][0]
        for alias in ("employer", "company_name", "organization", "dates", "date_range", "period",
                      "employment_period", "designation", "job_title", "title", "role", "position",
                      "responsibilities", "duties", "achievements"):
            job.pop(alias, None)
        jobs.append(job)

    education = []
    for item in _as_list(normalized.get("education")):
        if isinstance(item, dict):
            record = deepcopy(item)
            if record.get("degree") and not record.get("qualification"):
                record["qualification"] = record["degree"]
            elif record.get("qualification") and not record.get("degree"):
                record["degree"] = record["qualification"]
            education.append(record)
        else:
            education.append({"qualification": str(item).strip()})

    projects = []
    for item in _as_list(normalized.get("projects")):
        projects.append(deepcopy(item) if isinstance(item, dict) else {"description": str(item).strip()})

    skills = []
    for item in _as_list(normalized.get("skills")):
        if isinstance(item, dict):
            skills.append(deepcopy(item))
        else:
            skills.append({"name": str(item).strip()})

    certs = []
    raw_certs = normalized.get("certifications", [])
    if isinstance(raw_certs, dict):
        for category, values in raw_certs.items():
            for item in _as_list(values):
                certs.append({"name": str(item).strip(), "category": str(category)})
    else:
        for item in _as_list(raw_certs):
            certs.append(deepcopy(item) if isinstance(item, dict) else {"name": str(item).strip()})

    awards = _as_list(normalized.get("awards"))
    achievements = _as_list(normalized.get("achievements"))
    publications = _as_list(normalized.get("publications"))
    known = {"personal_information", "personal_details", "profile_snapshot", "work_history", "education", "projects",
             "skills", "technical_skills", "certifications", "domain_experience", "_audit"}
    known.update({"awards", "achievements", "publications"})
    custom = {key: deepcopy(value) for key, value in source.items()
              if ALIASES.get(_key(key), _key(key)) not in known}
    for key in ("technical_skills", "tools_and_technology", "domain_experience"):
        if key in normalized and key not in known:
            custom[key] = deepcopy(normalized[key])
    technical = normalized.get("technical_skills", normalized.get("tools_and_technology", {}))

    result = {
        "schema_version": "canonical-resume/v1",
        "personal_information": personal,
        "personal_details": personal_details,
        "profile_snapshot": [str(item).strip() for item in _as_list(normalized.get("profile_snapshot"))
                             if str(item).strip()],
        "work_history": jobs,
        "education": education,
        "projects": projects,
        "skills": skills,
        "technical_skills": deepcopy(technical) if isinstance(technical, (dict, list)) else _as_list(technical),
        "certifications": certs,
        "awards": [deepcopy(item) if isinstance(item, dict) else {"name": str(item).strip()} for item in awards],
        "achievements": [deepcopy(item) if isinstance(item, dict) else {"description": str(item).strip()} for item in achievements],
        "publications": [deepcopy(item) if isinstance(item, dict) else {"title": str(item).strip()} for item in publications],
        "domain_experience": _as_list(normalized.get("domain_experience")),
        "source_sections": custom,
        "source_audit": deepcopy(resume.get("_audit", {})) if isinstance(resume, dict) else {},
    }
    validate_canonical_resume(result)
    return result


def validate_canonical_resume(resume: dict) -> None:
    """Reject malformed hierarchy and source blocks assigned to the wrong section."""
    required = {
        "schema_version", "personal_information", "personal_details", "profile_snapshot", "work_history",
        "education", "projects", "skills", "technical_skills", "certifications",
        "domain_experience", "awards", "achievements", "publications", "source_sections", "source_audit",
    }
    missing = required - set(resume)
    if missing:
        raise ValueError(f"Canonical resume is missing fields: {sorted(missing)}")
    if resume["schema_version"] != "canonical-resume/v1":
        raise ValueError("Unsupported canonical resume schema version")
    if not isinstance(resume["personal_information"], dict):
        raise ValueError("Canonical personal_information must be an object")
    if not isinstance(resume["personal_details"], dict):
        raise ValueError("Canonical personal_details must be an object")
    for key in ("profile_snapshot", "work_history", "education", "projects", "skills",
                "certifications", "domain_experience", "awards", "achievements", "publications"):
        if not isinstance(resume[key], list):
            raise ValueError(f"Canonical {key} must be a list")
    for job in resume["work_history"]:
        if not isinstance(job, dict) or not isinstance(job.get("roles"), list):
            raise ValueError("Each canonical work record must contain a roles list")
        if "details" in job:
            raise ValueError("Legacy details-only work records are not valid canonical employment parents")
        company = str(job.get("company") or job.get("employer") or job.get("organization") or "").strip()
        if not company:
            raise ValueError("Each canonical work record must identify a company/employer")
        for role in job["roles"]:
            if (not isinstance(role, dict)
                    or not isinstance(role.get("role_and_responsibilities"), list)):
                raise ValueError("Each canonical role must contain a responsibility list")
            if "details" in role:
                raise ValueError("Legacy details-only roles are not valid canonical employment children")
            for field in ("clients", "role_and_responsibilities", "achievements", "projects"):
                if field in role and not isinstance(role[field], list):
                    raise ValueError(f"Canonical role field {field} must be a list")
    for education in resume["education"]:
        if not isinstance(education, dict):
            raise ValueError("Canonical education must contain row-based objects")
        if any(key in education for key in ("date_of_birth", "languages", "nationality")):
            raise ValueError("Personal details cannot be stored in an education record")
    for key in resume["personal_information"]:
        if str(key).isdecimal():
            raise ValueError("Canonical personal information cannot use numeric keys for identity fields")

    # Use the source block map as a provenance guard against cross-section
    # leakage. Inferred profile text inside an Experience heading is allowed
    # only when the classifier explicitly labels that block as a summary.
    block_map = resume.get("source_audit", {}).get("semantic_block_map", {})
    page_marker_values = {str(block.get("raw_text") or block.get("text") or "").strip().casefold()
                          for block in block_map.values()
                          if str(block.get("block_type", "")).upper() == "PAGE_ARTIFACT"
                          and (block.get("raw_text") or block.get("text"))}
    def content_values(value, parent_key=""):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"source_blocks", "responsibility_sources", "achievement_sources", "source_audit"}:
                    continue
                yield from content_values(child, str(key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from content_values(child, parent_key)
        elif isinstance(value, (str, int, float)) and parent_key not in {
                "year", "start_date", "end_date", "score", "gpa", "cgpa", "percentage"}:
            yield str(value).strip().casefold()
    semantic_fields = {key: value for key, value in resume.items()
                       if key not in {"source_audit", "unresolved_blocks"}}
    if page_marker_values and page_marker_values.intersection(content_values(semantic_fields)):
        raise ValueError("A source page-marker value leaked into canonical semantic content")

    domain_source_ids, technical_source_ids, personal_detail_source_ids = set(), set(), set()
    for source_id, block in block_map.items():
        section = re.sub(r"\s+", " ", str(block.get("section", "")).casefold()).strip()
        destination = str(block.get("destination") or "")
        kind = str(block.get("block_type") or "").upper()
        source_id = str(source_id)
        if "domain" in section or "expertise" in section:
            domain_source_ids.add(source_id)
        if "technical skill" in section or "tools and technolog" in section:
            technical_source_ids.add(source_id)
        if section in {"personal details", "personal information", "languages"}:
            personal_detail_source_ids.add(source_id)
        if section in {"experience", "professional experience", "work experience", "work history",
                       "employment history", "career history"}:
            if destination.startswith("$.skills"):
                raise ValueError("Employment source content cannot be assigned to skills")
            if destination.startswith("$.profile_snapshot") and kind != "PROFILE_SUMMARY":
                raise ValueError("Employment source content cannot be assigned to profile summary")
        if kind == "PAGE_ARTIFACT" and destination:
            raise ValueError("Page-marker source block cannot be assigned to a canonical field")

    for job in resume["work_history"]:
        work_ids = set(job.get("source_blocks", []))
        for role in job["roles"]:
            work_ids.update(role.get("source_blocks", []))
        if work_ids.intersection(domain_source_ids | technical_source_ids | personal_detail_source_ids):
            raise ValueError("A non-employment section source block was grouped into work history")
    education_ids = {source_id for record in resume["education"] for source_id in record.get("source_blocks", [])}
    if education_ids.intersection(personal_detail_source_ids):
        raise ValueError("Personal detail source blocks cannot be grouped into education")

    # Content detected in explicit sections must have a corresponding canonical
    # collection so the serializer cannot silently drop an entire section.
    project_blocks = [block for block in block_map.values()
                      if str(block.get("section", "")).casefold() in {"project", "projects", "project highlights"}
                      and str(block.get("block_type", "")).upper() not in {"SECTION_HEADING", "PAGE_ARTIFACT"}]
    if project_blocks and not resume["projects"]:
        raise ValueError("Project source blocks were detected but no canonical projects were produced")
    if not isinstance(resume["source_sections"], dict) or not isinstance(resume["source_audit"], dict):
        raise ValueError("Canonical provenance fields must be objects")
