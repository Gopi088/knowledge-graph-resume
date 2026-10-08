"""Create graph nodes and source-verified relationship edges.

Similarity and cluster artifacts remain available for analysis, but neither is
used to create graph relationships. Every emitted edge points to entity
occurrences and evidence produced by the source-context extractor.
"""


def build_graph(entities, relationships, sim_data=None, clusters=None, blocks=None,
                semantic_threshold=None) -> dict:
    nodes = [
        {"id": entity["id"], "label": entity["text"], "type": entity["type"],
         "confidence": entity["confidence"], "method": entity["method"],
         "source_sentences": entity["source_sentences"],
         "frequency": entity["frequency"],
         "sections": entity.get("sections", []),
         "block_ids": entity.get("block_ids", [])}
        for entity in entities
    ]
    entity_by_id = {entity["id"]: entity for entity in entities}
    edges = []
    for relationship in relationships:
        source_id = relationship.get("source_id")
        target_id = relationship.get("target_id")
        source = entity_by_id.get(source_id)
        target = entity_by_id.get(target_id)
        if not source or not target:
            continue

        evidence = relationship.get("evidence_sentences") or [relationship.get("source_sentence", "")]
        evidence_text = "\n".join(evidence).casefold()
        if not evidence_text:
            continue
        if source.get("type") != "PERSON" and source["text"].casefold() not in evidence_text:
            continue
        if target.get("type") != "PERSON" and target["text"].casefold() not in evidence_text:
            continue
        edge_block = relationship.get("block_id")
        if source.get("type") == "PERSON":
            block_supported = edge_block in target.get("block_ids", [])
        elif target.get("type") == "PERSON":
            block_supported = edge_block in source.get("block_ids", [])
        else:
            block_supported = edge_block in (set(source.get("block_ids", [])) &
                                             set(target.get("block_ids", [])))
        if not block_supported:
            continue

        edges.append({
            "source": source_id,
            "target": target_id,
            "source_label": source["text"],
            "target_label": target["text"],
            "relationship": relationship["relationship"],
            "confidence": relationship["confidence"],
            "source_sentence": relationship["source_sentence"],
            "evidence_sentences": evidence,
            "evidence_kind": relationship.get("evidence_kind", "same_sentence"),
            "method": relationship["method"],
            "kind": "explicit",
            "section": relationship.get("section", "Other"),
            "block_id": relationship["block_id"],
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "explicit_edges": len(edges),
            "semantic_edges": 0,
            "note": ("Graph edges are source-verified relationships. Similarity and "
                     "cluster assignments do not create graph edges."),
        },
        "clusters": clusters or {},
        "block_count": blocks.get("block_count", 0) if blocks else 0,
    }
