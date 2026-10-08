"""Pipeline orchestrator — runs the full paper flow for one resume
and writes the 6 required JSON artifacts:
raw_text.json, entities.json, relationships.json,
embeddings.json, clusters.json, graph.json
"""
import json
import os
import re

from .text_repository import TextRepository, TextDocument
from .preprocessing import preprocess
from .segmentation import detect_blocks
from .entity_extraction import extract_entities
from .relation_extraction import extract_relationships
from .embeddings import embed_entities
from .similarity import cosine_similarity_matrix
from .clustering import cluster
from .graph_builder import build_graph


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
        })
        row["is_section_heading"] = row["is_section_heading"] or bool(meta.get("is_header"))
        if not meta.get("is_header") and meta.get("line", "").strip() != text:
            row["inline_body"] = meta.get("line", "").strip()

    assignments = []
    for line_index in sorted(source_by_index):
        row = source_by_index[line_index]
        block_id = line_to_block.get(str(line_index))
        row["block_id"] = block_id
        if row["is_section_heading"] and block_id:
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


def _build_section_keyed_output(block_audit: dict) -> dict:
    """Render resume content cleanly by section; keep provenance in _audit."""
    aliases = {
        "header": "personal_information",
        "personal information": "personal_information",
        "summary": "profile_snapshot",
        "experience": "work_history",
        "projects": "projects",
        "education": "education",
        "skills": "skills",
        "certifications": "certifications",
        "achievements": "achievements",
        "achievements and certifications": "achievements_and_certifications",
        "tools and technology": "tools_and_technology",
    }
    # Common misspellings in resume headings should not become their own
    # output sections after preprocessing has already recognized the section.
    aliases.update({
        "pofessional experience": "work_history",
        "professional experience": "work_history",
        "eeducation": "education",
        "technical skills": "skills",
        "core competencies": "skills",
        "professional summary": "profile_snapshot",
    })
    output: dict = {}
    block_map = []
    top_level_extras = {}

    def clean(lines: list[str]) -> list[str]:
        return [re.sub(r"^[\s•▪●*-]+", "", str(line)).strip()
                for line in lines if str(line).strip()]

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
        for row in rows:
            text = re.sub(r"^[\s•▪●*-]+", "", str(row.get("text", ""))).strip()
            if not text:
                continue
            starts_bullet = bool(row.get("is_bullet"))
            # A new employer/role heading is a boundary even if PDF extraction
            # omitted its bullet marker.
            is_job_heading = (not starts_bullet and len(text) < 220 and bool(re.search(
                r"\b(?:19|20)\d{2}\b", text, re.I
            )))
            if starts_bullet or is_job_heading or not result:
                result.append(text)
            else:
                result[-1] += " " + text
        return result

    def labeled_fields(lines: list[str]) -> dict:
        fields = {}
        known_keys = {
            "name", "title", "location", "phone", "email", "place", "name at end",
            "year", "qualification", "university", "institution", "degree", "gpa",
            "cgpa", "percentage", "company", "duration", "designation", "role",
        }
        for line in lines:
            current_key = None
            for part in re.split(r"\s*\|\s*", line):
                match = re.match(r"^([^:]{2,40}):\s*(.+)$", part.strip())
                if match and match.group(1).casefold().strip() in known_keys:
                    field_key = "_".join(match.group(1).casefold().replace("&", "and").split())
                    fields[field_key] = match.group(2).strip()
                    current_key = field_key
                elif current_key:
                    fields[current_key] += " | " + part.strip()
        return fields

    for section in block_audit.get("sections", []):
        name = section["name"].strip()
        key = aliases.get(name.casefold(), "_".join(name.casefold().replace("&", "and").split()))
        source_rows = complete_source_lines(section)
        # Build the readable section content from source lines rather than
        # relying on entity-bearing blocks. This preserves plain text such as
        # coursework, long responsibility bullets, and skill details that may
        # not produce any entities or relationships.
        source_text = [str(row.get("text", "")).strip() for row in source_rows]
        if key == "profile_snapshot":
            # PDF extraction commonly wraps a summary mid-sentence.
            source_text = [" ".join(source_text)] if source_text else []
        elif key in {"work_history", "achievements_and_certifications", "certifications", "education"}:
            source_text = reflow_bullets(source_rows)
        elif key == "skills":
            # Keep skill rows individually inspectable while joining only
            # obvious visual wraps (open parentheses or a trailing separator).
            joined: list[str] = []
            for line in source_text:
                if joined and (joined[-1].count("(") > joined[-1].count(")")
                               or re.search(r"(?:[,|/]\s*)$", joined[-1])):
                    joined[-1] += " " + line
                else:
                    joined.append(line)
            source_text = joined
        if key in {"profile_snapshot", "skills", "work_history", "education",
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
                value = {}
                if lines:
                    match = re.match(r"^(.*?)\s+at\s+(.*?)\s*\(([^()]*)\)\s*$", lines[0], re.I)
                    if match:
                        value = {
                            "company": match.group(2).strip(),
                            "duration": match.group(3).strip(),
                            "designation": match.group(1).strip(),
                            "role_and_responsibilities": lines[1:],
                        }
                    else:
                        value = {"details": lines, "role_and_responsibilities": []}
                else:
                    value = {"details": [], "role_and_responsibilities": []}
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
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def run_pipeline(raw_text: str, doc_id: str, out_dir: str, candidate_name: str = "Candidate") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Text Repository
    repo = TextRepository()
    doc = repo.add(TextDocument(doc_id=doc_id, raw_text=raw_text))
    raw_json = {"doc_id": doc.doc_id, "source": doc.source,
                "char_count": doc.char_count, "created_at": doc.created_at,
                "raw_text": doc.raw_text}

    # 2. Data preprocessing: structure-preserving normalization and segmentation
    pre = preprocess(raw_text)
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
    section_keyed_output = _build_section_keyed_output(block_audit)
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
    graph = build_graph(entities, relationships, sim, clusters, blocks=blocks)

    artifacts = {
        "raw_text.json": raw_json,
        "entities.json": {"count": len(entities), "entities": entities,
                          "preprocessing": {k: v for k, v in pre.items()
                                            if k in ("method", "sections", "sentence_count",
                                                     "normalized_preview")}},
        "relationships.json": {"count": len(relationships), "relationships": relationships,
                               "note": "Every edge is supported by its source sentence or a structurally linked resume entry, with section/block provenance."},
        "blocks.json": blocks,
        "block_audit.json": block_audit,
        "resume_blocks.json": section_keyed_output,
        "embeddings.json": {"model": emb["model"], "dim": emb["dim"],
                             "entity_texts": emb["entity_texts"],
                             "vectors_preview": emb["vectors_preview"]},
        "similarity.json": sim,
        "clusters.json": clusters,
        "graph.json": graph,
    }
    for fname, payload in artifacts.items():
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(payload, f, indent=2)
    # full vectors saved separately for reproducibility
    with open(os.path.join(out_dir, "embeddings.json"), "w") as f:
        json.dump({**artifacts["embeddings.json"], "vectors": emb["vectors"]}, f, indent=2)
    print(f"[{doc_id}] entities={len(entities)} rels={len(relationships)} "
          f"blocks={blocks['block_count']} edges={graph['stats']['edge_count']} "
          f"(explicit={graph['stats']['explicit_edges']}, semantic={graph['stats']['semantic_edges']})")
    return artifacts
