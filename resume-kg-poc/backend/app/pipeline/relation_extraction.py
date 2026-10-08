"""High-precision, source-grounded relationship extraction for resumes.

An edge is emitted only when its endpoints are supported by a source sentence
or a narrowly defined structural entry (currently education details). Embedding
similarity, entity type, and co-occurrence elsewhere in a section never create
an edge.
"""
import re

SKILLISH = {"PROGRAMMING_LANGUAGE", "FRAMEWORK", "DATABASE", "CLOUD", "TOOL", "SKILL", "TECHNOLOGY"}
_ACTION_CUE = re.compile(
    r"\b(use[sd]?|using|built|build|develop(?:ed|s)?|implement(?:ed|s)?|"
    r"design(?:ed|s)?|creat(?:ed|es)?|integrat(?:ed|es)?|deploy(?:ed|s)?|"
    r"powered\s+by|based\s+on|work(?:ed|s)?\s+with)\b", re.I,
)


def _candidate(entities: list[dict]) -> dict | None:
    return next((entity for entity in entities if entity.get("type") == "PERSON"), None)


def extract_relationships(
    sentences: list[str],
    entities: list[dict],
    sentence_meta: "list[dict] | None" = None,
    sentence_blocks: "list[str] | None" = None,
) -> list[dict]:
    sentence_meta = sentence_meta or [{"index": i, "section": "Other"} for i in range(len(sentences))]
    sentence_blocks = sentence_blocks or [f"b{i}" for i in range(len(sentences))]
    meta_by_index = {meta.get("index", i): meta for i, meta in enumerate(sentence_meta)}
    by_sentence: dict[int, list[dict]] = {}
    by_block: dict[str, list[dict]] = {}
    for entity in entities:
        for index in entity.get("sentence_indices", []):
            by_sentence.setdefault(index, []).append(entity)
        for block_id in entity.get("block_ids", []):
            by_block.setdefault(block_id, []).append(entity)

    candidate = _candidate(entities)
    relationships: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(source: dict, target: dict, label: str, confidence: float,
            evidence_indices: list[int], evidence_kind: str = "same_sentence"):
        if source["id"] == target["id"] or not evidence_indices:
            return
        evidence_indices = sorted(set(evidence_indices))
        evidence = [sentences[index] for index in evidence_indices]
        # Validate both named endpoints against this evidence. The candidate is
        # the one allowed implicit subject in resume bullet sentences.
        evidence_text = "\n".join(evidence).casefold()
        for endpoint in (source, target):
            if endpoint.get("type") == "PERSON":
                continue
            if endpoint["text"].casefold() not in evidence_text:
                return
        block_ids = {sentence_blocks[i] for i in evidence_indices if i < len(sentence_blocks)}
        if len(block_ids) != 1:
            return
        block_id = next(iter(block_ids))
        sections = {meta_by_index.get(i, {}).get("section", "Other") for i in evidence_indices}
        if len(sections) != 1:
            return
        section = next(iter(sections))
        key = (source["id"], target["id"], label, block_id)
        if key in seen:
            return
        seen.add(key)
        relationships.append({
            "source": source["text"], "target": target["text"],
            "source_id": source["id"], "target_id": target["id"],
            "relationship": label, "confidence": confidence,
            "source_sentence": evidence[0] if len(evidence) == 1 else "\n".join(evidence),
            "evidence_sentences": evidence,
            "evidence_sentence_indices": evidence_indices,
            "evidence_kind": evidence_kind,
            "method": "source-context rules",
            "section": section, "block_id": block_id,
        })

    # Relationships established by one source sentence.
    for index, sentence in enumerate(sentences):
        mentioned = [entity for entity in by_sentence.get(index, [])
                     if not candidate or entity["id"] != candidate["id"]]
        if not mentioned:
            continue
        section = meta_by_index.get(index, {}).get("section", "Other").casefold()
        block_id = sentence_blocks[index] if index < len(sentence_blocks) else f"b{index}"
        skills = [entity for entity in mentioned if entity.get("type") in SKILLISH]
        projects = [entity for entity in mentioned if entity.get("type") == "PROJECT"]
        degrees = [entity for entity in mentioned if entity.get("type") == "DEGREE"]
        universities = [entity for entity in mentioned if entity.get("type") == "UNIVERSITY"]
        certifications = [entity for entity in mentioned if entity.get("type") == "CERTIFICATION"]
        companies = [entity for entity in mentioned if entity.get("type") == "COMPANY"]
        roles = [entity for entity in mentioned if entity.get("type") == "JOB_ROLE"]

        # Skill lists are evidence that the candidate has those skills only in
        # a skills/technology section; incidental keywords elsewhere are not.
        if candidate and section in {"skills", "technical skills", "tech stack", "technologies"}:
            for skill in skills:
                add(candidate, skill, "HAS_SKILL", 0.86, [index])

        if candidate and section == "certifications":
            for certification in certifications:
                add(candidate, certification, "EARNED", 0.90, [index])

        # A project and its technologies must occur in the same descriptive
        # source line, with an explicit implementation/technology cue.
        if projects and skills and _ACTION_CUE.search(sentence):
            for project in projects:
                for skill in skills:
                    add(project, skill, "USES_TECHNOLOGY", 0.88, [index])
                if candidate:
                    add(candidate, project, "WORKED_ON", 0.86, [index])
        elif candidate and skills and _ACTION_CUE.search(sentence):
            for skill in skills:
                add(candidate, skill, "USED_OR_APPLIED", 0.78, [index])

        # Employment relationship is emitted only for a role/company pair
        # explicitly co-mentioned in an experience source line.
        if section == "experience" and companies and roles:
            for role in roles:
                for company in companies:
                    add(role, company, "AT_COMPANY", 0.90, [index])
                    if candidate:
                        add(candidate, role, "HAS_ROLE", 0.88, [index])
                        add(candidate, company, "WORKED_AT", 0.88, [index])

        if candidate and degrees:
            for degree in degrees:
                add(candidate, degree, "HOLDS_DEGREE", 0.90, [index])

    # Structural experience entry: a role header plus its responsibilities is
    # a linked source context even when the role and a technology occur on
    # different lines. Require an action cue on the technology's own line.
    for block_id, block_entities in by_block.items():
        entries = {entity["id"]: entity for entity in block_entities}
        section = next((meta_by_index.get(index, {}).get("section", "Other").casefold()
                        for entity in entries.values()
                        for index in entity.get("sentence_indices", [])), "")

        if section == "projects":
            projects = [entity for entity in entries.values() if entity.get("type") == "PROJECT"]
            skills = [entity for entity in entries.values() if entity.get("type") in SKILLISH]
            for project in projects:
                for skill in skills:
                    action_lines = [index for index in skill.get("sentence_indices", [])
                                    if _ACTION_CUE.search(sentences[index])]
                    if action_lines:
                        add(project, skill, "USES_TECHNOLOGY", 0.88,
                            project.get("sentence_indices", []) + action_lines,
                            "structured_project_entry")

        roles = [entity for entity in entries.values() if entity.get("type") == "JOB_ROLE"]
        skills = [entity for entity in entries.values() if entity.get("type") in SKILLISH]
        evidence_indices = sorted({
            index for entity in entries.values()
            for index in entity.get("sentence_indices", [])
        })
        if section != "experience" or not roles:
            continue
        for skill in skills:
            skill_lines = skill.get("sentence_indices", [])
            action_lines = [index for index in skill_lines if _ACTION_CUE.search(sentences[index])]
            if action_lines:
                for role in roles:
                    add(role, skill, "ROLE_USES_TECHNOLOGY", 0.86,
                        role.get("sentence_indices", []) + action_lines,
                        "structured_experience_entry")

    # Education entries can put degree, institution, and result on adjacent
    # physical lines. The block detector groups those only when the lines have
    # education labels and no new-entry boundary. Preserve every line as edge
    # evidence, and require both endpoint labels to appear in that evidence.
    if candidate:
        for block_id, block_entities in by_block.items():
            entries = {entity["id"]: entity for entity in block_entities}
            degrees = [entity for entity in entries.values() if entity.get("type") == "DEGREE"]
            universities = [entity for entity in entries.values() if entity.get("type") == "UNIVERSITY"]
            evidence_indices = sorted({
                index for entity in entries.values()
                for index in entity.get("sentence_indices", [])
            })
            section = (meta_by_index.get(evidence_indices[0], {}).get("section", "Other").casefold()
                       if evidence_indices else "")
            if section != "education":
                continue
            for degree in degrees:
                for university in universities:
                    add(degree, university, "FROM_UNIVERSITY", 0.90,
                        evidence_indices, "structured_education_entry")

    return relationships
