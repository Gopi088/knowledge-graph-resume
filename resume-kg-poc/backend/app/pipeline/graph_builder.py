"""Generate and validate the knowledge graph from canonical resume records."""
import re



def build_resume_graph(canonical: dict, candidate_name: str = "Candidate") -> dict:
    """Build graph nodes and edges only from canonical resume hierarchy.

    Occurrence nodes are scoped to their canonical record. Technology links
    are only emitted when the technology name is present in that exact role
    responsibility or project description.
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    by_path: dict[str, dict] = {}
    omissions = []
    def education_evidence(education):
        """Return resume content only; provenance IDs are not source text."""
        def values(value):
            if isinstance(value, dict):
                return " ".join(values(child) for child in value.values())
            if isinstance(value, (list, tuple)):
                return " ".join(values(child) for child in value)
            return str(value or "")
        return " ".join(values(value) for key, value in education.items() if key != "source_blocks")

    def add_education_source_nodes(index, education, entry_path):
        """Add academic record components; avoid turning raw numbers into graph entities."""
        degree_field = next((field for field in ("degree", "qualification", "program", "title")
                             if education.get(field)), None)
        degree_label = education.get(degree_field) if degree_field else None
        degree = add_node(degree_label, "DEGREE", f"{entry_path}.{degree_field}", degree_label, "Education") if degree_label else None
        institution_field = next((field for field in ("university_or_board", "university", "institution",
                                                       "college_school", "school") if education.get(field)), None)
        institution_label = education.get(institution_field) if institution_field else None
        if degree:
            person_edge(degree, "COMPLETED_DEGREE", entry_path, entry_path,
                        education_evidence(education), "Education")
        if institution_label and degree:
            institution_path = f"{entry_path}.{institution_field}"
            institution = add_node(institution_label, "INSTITUTION", institution_path, institution_label, "Education")
            edge(degree, institution, "AWARDED_BY", entry_path, entry_path,
                 education_evidence(education), "Education")
        for field in ("year", "graduation_year"):
            value = education.get(field)
            if value:
                path = f"{entry_path}.{field}"
                node = add_node(value, "DATE", path, value, "Education")
                if degree:
                    edge(degree, node, "COMPLETED_IN", entry_path, path, str(value), "Education")
        for field in ("gpa", "cgpa", "score", "percentage", "percentage_and_pointer"):
            value = education.get(field)
            if value:
                # GPA/percentages stay as structured record values. Linking
                # numeric fragments as standalone graph entities is noisy.
                continue

    candidate = canonical.get("personal_information", {})
    if not isinstance(candidate, dict):
        candidate = {}
    personal_details = canonical.get("personal_details", {})
    if not isinstance(personal_details, dict):
        personal_details = {}
    person_name = str(candidate.get("name") or (candidate_name if candidate_name != "Candidate" else "") or "").strip()

    def add_node(label, node_type, path, source_text=None, section="Other", node_id=None):
        label = str(label or "").strip()
        if not label:
            return None
        node = {
            "id": node_id or f"n{len(nodes)}", "label": label, "type": node_type,
            "confidence": 1.0, "method": "canonical resume structure",
            "source_sentences": [str(source_text or label)], "frequency": 1,
            "sections": [section], "block_ids": [], "source_path": path,
            "source_text": str(source_text or label),
        }
        lookup = path.split("#term:", 1)[0]
        provenance = canonical.get("source_audit", {}).get("semantic_block_map", {})
        responsibility_match = re.match(r"(.+)\.role_and_responsibilities\[(\d+)\]$", lookup)
        if responsibility_match:
            role = get_canonical_value(canonical, responsibility_match.group(1))
            index = int(responsibility_match.group(2))
            if isinstance(role, dict) and index < len(role.get("responsibility_sources", [])):
                node["source_blocks"] = role["responsibility_sources"][index]
        field_match = re.match(r"(.+)\.(?:clients\[\d+\]|employment_periods\[\d+\]\.source_text)$", lookup)
        if field_match and not node.get("source_blocks"):
            role = get_canonical_value(canonical, field_match.group(1))
            field = "clients" if ".clients[" in lookup else "employment_period"
            if isinstance(role, dict):
                node["source_blocks"] = role.get("field_sources", {}).get(field, [])
        while lookup:
            record = get_canonical_value(canonical, lookup)
            if not node.get("source_blocks") and isinstance(record, dict) and record.get("source_blocks"):
                node["source_blocks"] = record["source_blocks"]
                node["block_ids"] = record["source_blocks"]
                break
            next_lookup = re.sub(r"(?:\.[\w-]+|\[\d+\])$", "", lookup)
            if next_lookup == lookup:
                break
            lookup = next_lookup
        if node.get("source_blocks"):
            node["block_ids"] = node["source_blocks"]
            node["source_evidence"] = [provenance[key] for key in node["source_blocks"] if key in provenance]
        nodes.append(node)
        by_path[path] = node
        return node

    def edge(source, target, relation, scope_path, evidence_path, evidence_text, section):
        if not source or not target or not evidence_text:
            return
        edges.append({
            "source": source["id"], "target": target["id"],
            "source_label": source["label"], "target_label": target["label"],
            "relationship": relation, "confidence": 1.0,
            "source_sentence": evidence_text, "evidence_sentences": [evidence_text],
            "evidence_kind": "canonical_structural_context",
            "method": "validated canonical hierarchy", "kind": "explicit",
            "section": section, "block_id": None,
            "scope_path": scope_path, "evidence_path": evidence_path,
            "source_blocks": list(dict.fromkeys(source.get("source_blocks", []) + target.get("source_blocks", []))),
        })

    def person_edge(target, relation, scope_path, evidence_path, evidence_text, section):
        edge(person, target, relation, scope_path, evidence_path, evidence_text, section)

    person_path = "$.personal_information.name" if candidate.get("name") else "$.personal_information"
    person = add_node(person_name, "PERSON", person_path,
                      person_name, "Personal Information", node_id="person") if person_name else add_node(
                          "Resume subject", "PERSON", "$.personal_information", "Resume subject", "Personal Information", node_id="person")
    if not person_name:
        person["unnamed_subject"] = True

    # Identity fields are meaningful source nodes when present.
    for field, node_type, relation in (
        ("title", "JOB_TITLE", "HAS_PROFESSIONAL_TITLE"),
        ("location", "LOCATION", "LOCATED_IN"),
    ):
        value = candidate.get(field)
        if value:
            path = f"$.personal_information.{field}"
            node = add_node(value, node_type, path, value, "Personal Information")
            person_edge(node, relation, "$.personal_information", path, str(value), "Personal Information")
    for field, node_type, relation in (
        ("date_of_birth", "DATE", "HAS_DATE_OF_BIRTH"),
        ("nationality", "NATIONALITY", "HAS_NATIONALITY"),
    ):
        value = personal_details.get(field)
        if value:
            path = f"$.personal_details.{field}"
            node = add_node(value, node_type, path, value, "Personal Details")
            person_edge(node, relation, "$.personal_details", path, str(value), "Personal Details")
    for index, language in enumerate(personal_details.get("languages", []) if isinstance(personal_details.get("languages"), list)
                                     else [personal_details.get("languages")]):
        if language:
            path = f"$.personal_details.languages[{index}]"
            node = add_node(language, "LANGUAGE", path, language, "Personal Details")
            person_edge(node, "SPEAKS", "$.personal_details", path, str(language), "Personal Details")

    for index, summary in enumerate(canonical.get("profile_snapshot", [])):
        path = f"$.profile_snapshot[{index}]"
        node = add_node(summary, "SUMMARY", path, summary, "Summary")
        person_edge(node, "HAS_SUMMARY", "$.profile_snapshot", path, str(summary), "Summary")

    for job_index, job in enumerate(canonical.get("work_history", [])):
        if not isinstance(job, dict):
            continue
        job_path = f"$.work_history[{job_index}]"
        company_name = job.get("company") or job.get("employer")
        company = None
        company_path = f"{job_path}.company"
        has_named_role = any(isinstance(role, dict) and
                             (role.get("designation") or role.get("title") or role.get("role"))
                             for role in job.get("roles", []))
        if company_name:
            company = add_node(company_name, "COMPANY", company_path, company_name, "Experience")
            role_context = " ".join(str(role.get("designation") or role.get("title") or role.get("role") or "")
                                     for role in job.get("roles", []) if isinstance(role, dict))
            person_edge(company, "WORKED_AT", job_path, job_path,
                        " ".join(part for part in (company_name, role_context) if part), "Experience")
        for role_index, role in enumerate(job.get("roles", [])):
            if not isinstance(role, dict):
                continue
            role_path = f"{job_path}.roles[{role_index}]"
            role_record = role
            designation = role_record.get("designation") or role_record.get("title") or role_record.get("role") or "Role"
            if designation == "Role" and not (role_record.get("designation") or role_record.get("title") or role_record.get("role")):
                omissions.append({"source_path": role_path, "reason": "Role title absent; responsibilities retained in canonical JSON."})
                continue
            role_node = add_node(designation, "JOB_ROLE", f"{role_path}.designation",
                                 designation, "Experience")
            role_source = " ".join(str(item) for item in (
                company_name, designation, role_record.get("duration") or job.get("duration")
            ) if item)
            person_edge(role_node, "HAS_ROLE", job_path, job_path, role_source, "Experience")
            if company:
                edge(company, role_node, "HAS_ROLE", job_path, job_path, role_source, "Experience")
            duration = role_record.get("duration")
            if duration:
                period_path = f"{role_path}.employment_periods[0].source_text" if role_record.get("employment_periods") else f"{role_path}.duration"
                period = add_node(duration, "EMPLOYMENT_PERIOD", period_path,
                                  duration, "Experience")
                edge(role_node, period, "HAS_EMPLOYMENT_PERIOD", role_path,
                     period_path, str(duration), "Experience")
            for client_index, client_name in enumerate(role_record.get("clients", [])):
                client_path = f"{role_path}.clients[{client_index}]"
                client_node = add_node(client_name, "CLIENT", client_path, client_name, "Experience")
                edge(role_node, client_node, "WORKED_FOR_CLIENT", role_path, client_path,
                     str(client_name), "Experience")
            for responsibility_index, responsibility in enumerate(role_record.get("role_and_responsibilities", [])):
                if isinstance(responsibility, dict):
                    text = str(responsibility.get("text") or responsibility.get("description") or "").strip()
                else:
                    text = str(responsibility).strip()
                if not text:
                    continue
                responsibility_path = f"{role_path}.role_and_responsibilities[{responsibility_index}]"
                responsibility_node = add_node(text, "RESPONSIBILITY", responsibility_path,
                                               text, "Experience")
                edge(role_node, responsibility_node, "HAS_RESPONSIBILITY", role_path,
                     responsibility_path, text, "Experience")


    for index, education in enumerate(canonical.get("education", [])):
        if not isinstance(education, dict):
            continue
        entry_path = f"$.education[{index}]"
        add_education_source_nodes(index, education, entry_path)

    for index, project in enumerate(canonical.get("projects", [])):
        if not isinstance(project, dict):
            continue
        project_path = f"$.projects[{index}]"
        label = project.get("name") or project.get("title") or project.get("project_name") or project.get("description")
        if not label:
            continue
        project_node = add_node(label, "PROJECT", project_path, str(label), "Projects")
        person_edge(project_node, "COMPLETED_PROJECT", project_path, project_path,
                    " ".join(str(value) for value in project.values() if value), "Projects")
        for field in ("technologies", "technology", "tools", "tech_stack"):
            for tech_index, value in enumerate(project.get(field, []) if isinstance(project.get(field), list)
                                               else [project.get(field)] if project.get(field) else []):
                tech_path = f"{project_path}.{field}[{tech_index}]" if isinstance(project.get(field), list) else f"{project_path}.{field}"
                tech = add_node(value, "TECHNOLOGY", tech_path, value, "Projects")
                edge(project_node, tech, "USED_TECHNOLOGY", project_path, tech_path,
                     str(value), "Projects")

    for index, skill in enumerate(canonical.get("skills", [])):
        if isinstance(skill, dict):
            label = skill.get("name") or skill.get("skill") or skill.get("label")
            category = skill.get("category")
        else:
            label, category = skill, None
        if not label:
            continue
        path = f"$.skills[{index}]"
        node = add_node(label, "SKILL", path, str(label), "Skills")
        if category:
            node["category"] = category
        person_edge(node, "HAS_SKILL", "$.skills", path, str(label), "Skills")

    def walk_technical(value, path, category=""):
        if isinstance(value, dict):
            for key, child in value.items():
                walk_technical(child, f"{path}.{key}", str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk_technical(child, f"{path}[{index}]", category)
        elif value:
            node = add_node(value, "TECHNOLOGY", path, value, "Technical Skills")
            if node:
                node["category"] = category
                person_edge(node, "HAS_TECHNOLOGY_SKILL", "$.technical_skills", path,
                            str(value), "Technical Skills")
    walk_technical(canonical.get("technical_skills", {}), "$.technical_skills")

    for index, certification in enumerate(canonical.get("certifications", [])):
        label = certification.get("name") or certification.get("title") if isinstance(certification, dict) else certification
        if label:
            path = f"$.certifications[{index}]"
            node = add_node(label, "CERTIFICATION", path, str(label), "Certifications")
            person_edge(node, "HAS_CERTIFICATION", "$.certifications", path,
                        str(label), "Certifications")

    for index, award in enumerate(canonical.get("awards", [])):
        label = (award.get("name") or award.get("title") or award.get("award")
                 if isinstance(award, dict) else award)
        if label:
            path = f"$.awards[{index}]"
            node = add_node(label, "AWARD", path, str(label), "Awards")
            person_edge(node, "HAS_AWARD", "$.awards", path, str(label), "Awards")

    for index, achievement in enumerate(canonical.get("achievements", [])):
        label = (achievement.get("description") or achievement.get("name") or achievement.get("title")
                 if isinstance(achievement, dict) else achievement)
        if label:
            path = f"$.achievements[{index}]"
            node = add_node(label, "ACHIEVEMENT", path, str(label), "Achievements")
            person_edge(node, "HAS_ACHIEVEMENT", "$.achievements", path,
                        str(label), "Achievements")

    for index, publication in enumerate(canonical.get("publications", [])):
        label = (publication.get("title") or publication.get("name")
                 if isinstance(publication, dict) else publication)
        if label:
            path = f"$.publications[{index}]"
            node = add_node(label, "PUBLICATION", path, str(label), "Publications")
            person_edge(node, "AUTHORED_PUBLICATION", "$.publications", path,
                        str(label), "Publications")

    for index, domain in enumerate(canonical.get("domain_experience", [])):
        label = domain.get("name") or domain.get("domain") if isinstance(domain, dict) else domain
        if label:
            path = f"$.domain_experience[{index}]"
            node = add_node(label, "DOMAIN", path, str(label), "Domain Experience")
            person_edge(node, "HAS_DOMAIN_EXPERIENCE", "$.domain_experience", path,
                        str(label), "Domain Experience")

    graph = {"nodes": nodes, "edges": edges, "stats": {
        "node_count": len(nodes), "edge_count": len(edges),
        "explicit_edges": len(edges), "semantic_edges": 0,
        "note": "Graph is generated from validated canonical resume records; no similarity edges are emitted.",
    }, "clusters": {}, "block_count": 0, "source": "canonical_resume/v1", "omissions": omissions}
    validate_resume_graph(canonical, graph)
    return graph


def validate_resume_graph(canonical: dict, graph: dict) -> None:
    """Raise ValueError if a graph edge lacks canonical hierarchy evidence."""
    node_list = graph.get("nodes", [])
    nodes = {node["id"]: node for node in node_list}
    if len(nodes) != len(node_list):
        raise ValueError("Graph contains duplicate node IDs")
    semantic_keys = [(node.get("type"), str(node.get("label", "")).casefold(), node.get("source_path"))
                     for node in node_list]
    if len(set(semantic_keys)) != len(semantic_keys):
        raise ValueError("Graph contains duplicate entities at the same canonical source path")
    def source_text(value) -> str:
        if isinstance(value, dict):
            return " ".join(source_text(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return " ".join(source_text(child) for child in value)
        return str(value or "")

    def within(path: str, parent: str) -> bool:
        return path == parent or path.startswith(parent + ".") or path.startswith(parent + "[")
    allowed = {
        ("PERSON", "HAS_ROLE", "JOB_ROLE"), ("PERSON", "WORKED_AT", "COMPANY"),
        ("COMPANY", "HAS_ROLE", "JOB_ROLE"),
        ("JOB_ROLE", "HAS_EMPLOYMENT_PERIOD", "EMPLOYMENT_PERIOD"),
        ("JOB_ROLE", "WORKED_FOR_CLIENT", "CLIENT"),
        ("JOB_ROLE", "HAS_RESPONSIBILITY", "RESPONSIBILITY"),
        ("JOB_ROLE", "USED_TECHNOLOGY", "TECHNOLOGY"),
        ("JOB_ROLE", "USED_TECHNOLOGY", "SKILL"),
        ("PERSON", "COMPLETED_DEGREE", "DEGREE"), ("DEGREE", "AWARDED_BY", "INSTITUTION"),
        ("DEGREE", "COMPLETED_IN", "DATE"), ("DEGREE", "HAS_RESULT", "RESULT"),
        ("PERSON", "COMPLETED_PROJECT", "PROJECT"), ("PROJECT", "USED_TECHNOLOGY", "TECHNOLOGY"),
        ("PROJECT", "USED_TECHNOLOGY", "SKILL"),
        ("PERSON", "HAS_SKILL", "SKILL"), ("PERSON", "HAS_TECHNOLOGY_SKILL", "TECHNOLOGY"),
        ("PERSON", "HAS_CERTIFICATION", "CERTIFICATION"), ("PERSON", "HAS_DOMAIN_EXPERIENCE", "DOMAIN"),
        ("PERSON", "HAS_AWARD", "AWARD"), ("PERSON", "AUTHORED_PUBLICATION", "PUBLICATION"),
        ("PERSON", "HAS_ACHIEVEMENT", "ACHIEVEMENT"),
        ("PERSON", "SPEAKS", "LANGUAGE"), ("PERSON", "LOCATED_IN", "LOCATION"),
        ("PERSON", "HAS_DATE_OF_BIRTH", "DATE"), ("PERSON", "HAS_NATIONALITY", "NATIONALITY"),
        ("PERSON", "HAS_PROFESSIONAL_TITLE", "JOB_TITLE"), ("PERSON", "HAS_SUMMARY", "SUMMARY"),
    }
    allowed_node_types = {item for relation in allowed for item in (relation[0], relation[2])}
    for node in graph.get("nodes", []):
        if node.get("type") not in allowed_node_types:
            raise ValueError(f"Graph contains an invalid entity type: {node.get('type')}")
        value = get_canonical_value(canonical, node.get("source_path", ""))
        if not node.get("source_path") or value is None:
            raise ValueError(f"Graph node lacks canonical provenance: {node.get('id')}")
        source_value = source_text(value).casefold()
        if node["label"].casefold() not in source_value and not (node.get("unnamed_subject") and node["type"] == "PERSON" and not canonical.get("personal_information", {}).get("name")):
            raise ValueError(f"Graph node label is not supported by its source: {node.get('label')}")
    edges_seen = set()
    incident = set()
    person_roles = set()
    company_roles = set()
    person_companies = set()
    for edge in graph.get("edges", []):
        source, target = nodes.get(edge.get("source")), nodes.get(edge.get("target"))
        if not source or not target:
            raise ValueError("Graph edge endpoint is missing")
        if (source["type"], edge.get("relationship"), target["type"]) not in allowed:
            raise ValueError(f"Unsupported graph relationship: {source['type']} {edge.get('relationship')} {target['type']}")
        edge_key = (source["id"], edge.get("relationship"), target["id"])
        if edge_key in edges_seen:
            raise ValueError("Graph contains a duplicate relationship edge")
        edges_seen.add(edge_key)
        incident.update((source["id"], target["id"]))
        if edge.get("relationship") == "HAS_ROLE" and source["type"] == "PERSON":
            person_roles.add(target["id"])
        elif edge.get("relationship") == "HAS_ROLE" and source["type"] == "COMPANY":
            company_roles.add(target["id"])
        elif edge.get("relationship") == "WORKED_AT":
            person_companies.add(target["id"])
        scope = edge.get("scope_path", "")
        evidence_path = edge.get("evidence_path", "")
        if get_canonical_value(canonical, scope) is None or get_canonical_value(canonical, evidence_path) is None:
            raise ValueError(f"Graph edge has no canonical source context: {edge.get('relationship')}")
        if not edge.get("source_sentence"):
            raise ValueError(f"Graph edge has no source evidence: {edge.get('relationship')}")
        evidence_value = source_text(get_canonical_value(canonical, evidence_path)).casefold()
        target_label = target["label"].casefold()
        if target_label not in evidence_value:
            raise ValueError(f"Graph edge target is absent from its source evidence: {target['label']}")
        evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_value))
        sentence_tokens = set(re.findall(r"[a-z0-9]+", edge["source_sentence"].casefold()))
        if sentence_tokens and not sentence_tokens.issubset(evidence_tokens):
            raise ValueError(f"Graph edge evidence text is not present in its canonical source path")
        source_path = source["source_path"].split("#term:", 1)[0]
        target_path = target["source_path"].split("#term:", 1)[0]
        relationship = edge["relationship"]
        if relationship == "HAS_EMPLOYMENT_PERIOD":
            structurally_scoped = (source_path == scope + ".designation" and
                                   target_path.startswith(scope + ".employment_periods["))
        elif relationship == "WORKED_FOR_CLIENT":
            structurally_scoped = (source_path == scope + ".designation" and
                                   target_path.startswith(scope + ".clients["))
        elif relationship == "HAS_ROLE" and source["type"] == "COMPANY":
            structurally_scoped = (source_path == scope + ".company" and
                                   target_path.startswith(scope + ".roles["))
        elif relationship == "HAS_RESPONSIBILITY":
            structurally_scoped = source_path == scope + ".designation" and target_path.startswith(scope + ".role_and_responsibilities[")
        elif relationship == "HAS_ROLE":
            structurally_scoped = target_path.startswith(scope + ".roles[")
        elif relationship == "WORKED_AT":
            structurally_scoped = target_path == scope + ".company"
        elif relationship in {"AWARDED_BY", "COMPLETED_IN", "HAS_RESULT"}:
            structurally_scoped = (source_path.startswith(scope + ".") and
                                   target_path.startswith(scope + ".") and source_path != target_path)
        elif relationship == "USED_TECHNOLOGY" and source["type"] == "JOB_ROLE":
            structurally_scoped = (source_path.startswith(scope + ".") and
                                   (target_path.startswith(scope + ".role_and_responsibilities[") or
                                    target_path.startswith(scope + ".technologies")))
        elif relationship == "USED_TECHNOLOGY" and source["type"] == "PROJECT":
            structurally_scoped = source_path.startswith(scope) and target_path.startswith(scope + ".")
        elif source["type"] == "PERSON":
            structurally_scoped = within(target_path, scope)
        else:
            structurally_scoped = within(source_path, scope) and within(target_path, scope)
        if not structurally_scoped:
            raise ValueError(f"Graph edge endpoints do not share the declared hierarchy: {relationship}")

    orphan_nodes = [node["id"] for node in node_list
                    if node["id"] not in incident and node["type"] != "PERSON"]
    if orphan_nodes:
        raise ValueError(f"Graph contains orphan nodes: {orphan_nodes}")
    for node in node_list:
        if node["type"] == "JOB_ROLE":
            if node["id"] not in person_roles:
                raise ValueError(f"Employment role is disconnected from the person: {node['label']}")
            role_path = node["source_path"].rsplit(".designation", 1)[0]
            has_company = any(company_node["type"] == "COMPANY" and
                              company_node["source_path"] == role_path.split(".roles[", 1)[0] + ".company"
                              for company_node in node_list)
            if has_company and node["id"] not in company_roles:
                raise ValueError(f"Employment role is not linked to its canonical company: {node['label']}")
        elif node["type"] == "COMPANY" and node["id"] not in person_companies:
            raise ValueError(f"Company is not linked to the person: {node['label']}")


def get_canonical_value(canonical: dict, path: str):
    """Resolve the canonical JSON path notation used by graph provenance."""
    path = path.split("#term:", 1)[0]
    if path == "$":
        return canonical
    value = canonical
    for key, index in re.findall(r"(?:^\$|\.([A-Za-z_][\w-]*)|\[(\d+)\])", path):
        if not key and not index:
            continue
        try:
            value = value[int(index)] if index else value[key]
        except (KeyError, IndexError, TypeError):
            return None
    return value
