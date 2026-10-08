import { useCallback, useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";

// ---------- types mirroring backend artifacts ----------
interface Entity {
  id: string; text: string; type: string; confidence: number;
  method: string; source_sentences: string[]; frequency: number;
  sections?: string[]; block_ids?: string[];
}
interface Relationship {
  source: string; target: string; relationship: string;
  confidence: number; source_sentence: string; method: string;
  section?: string; block_id?: string;
}
interface GraphEdge extends Relationship { kind: "explicit" | "semantic"; }
interface GraphNode {
  id: string; label: string; type: string; confidence: number;
  method: string; source_sentences: string[]; frequency: number;
}
interface Block {
  id: string; section: string; section_id?: string; sentence_indices: number[];
  sentences: string[]; entities: string[]; source_lines?: string[];
  source_paths?: string[]; source_line_indices?: number[];
  entity_ids?: string[]; relationship_ids?: string[]; entry_type?: string;
}
interface BlockAudit {
  format?: string;
  source_format: string;
  summary: {
    source_line_count: number; assigned_source_line_count: number;
    unassigned_source_line_indices: number[]; section_count: number;
    block_count: number; sentence_count: number; entity_count: number;
    relationship_count: number; complete: boolean;
  };
  coverage?: Record<string, any>;
  sections: {
    id: string; name: string; heading_lines: string[];
    source_line_indices: number[]; block_ids: string[];
    blocks: {
      id: string; entry_type: string; source_line_indices: number[];
      source_lines: string[]; source_paths: string[];
      sentences: string[]; entity_ids: string[]; relationship_ids: string[];
    }[];
  }[];
  source_line_assignments: {
    line_index: number; text: string; section_id: string; section: string;
    source_path?: string; is_section_heading: boolean; block_id?: string;
    assignment: string; assigned: boolean; reason?: string;
  }[];
  derived_from_blocks?: boolean;
}
interface Artifacts {
  "raw_text.json"?: { raw_text: string; doc_id: string };
  "entities.json"?: { count: number; entities: Entity[]; preprocessing?: any };
  "relationships.json"?: { count: number; relationships: Relationship[] };
  "blocks.json"?: {
    method: string; blocks: Block[]; block_count: number; sentence_blocks: string[];
    coverage?: Record<string, any>;
  };
  "block_audit.json"?: BlockAudit;
  "resume_blocks.json"?: Record<string, any>;
  "canonical_resume.json"?: Record<string, any>;
  "embeddings.json"?: { model: string; dim: number; entity_texts: string[]; vectors_preview?: number[][] };
  "clusters.json"?: any;
  "similarity.json"?: { top_pairs: { a: string; b: string; similarity: number }[] };
  "graph.json"?: { nodes: GraphNode[]; edges: GraphEdge[]; stats: any };
  similarity?: { top_pairs: { a: string; b: string; similarity: number }[] };
}

const TYPE_COLORS: Record<string, string> = {
  PERSON: "#111827", COMPANY: "#7c3aed", JOB_ROLE: "#0e7490",
  PROGRAMMING_LANGUAGE: "#2563eb", FRAMEWORK: "#16a34a", DATABASE: "#b45309",
  CLOUD: "#ea580c", TOOL: "#4d7c0f", SKILL: "#0284c7", TECHNOLOGY: "#475569",
  PROJECT: "#be185d", DEGREE: "#6d28d9", UNIVERSITY: "#0f766e",
  CERTIFICATION: "#a16207", DOMAIN: "#9a3412",
};

function auditFromBlocks(blocksDoc: Artifacts["blocks.json"]): BlockAudit | undefined {
  if (!blocksDoc?.blocks?.length) return undefined;
  const grouped = new Map<string, { id: string; name: string; blocks: BlockAudit["sections"][number]["blocks"]; lineIndices: number[] }>();
  const assignments = new Map<number, BlockAudit["source_line_assignments"][number]>();
  let fallbackLineIndex = 0;

  for (const block of blocksDoc.blocks) {
    const sectionId = block.section_id ?? `section:${block.section}`;
    if (!grouped.has(sectionId)) grouped.set(sectionId, { id: sectionId, name: block.section, blocks: [], lineIndices: [] });
    const section = grouped.get(sectionId)!;
    const lines = block.source_lines?.length ? block.source_lines : block.sentences;
    const lineIndices = block.source_line_indices?.length ? block.source_line_indices : block.sentence_indices;
    const normalizedIndices: number[] = [];
    lines.forEach((text, index) => {
      const lineIndex = lineIndices[index] ?? fallbackLineIndex++;
      normalizedIndices.push(lineIndex);
      section.lineIndices.push(lineIndex);
      assignments.set(lineIndex, {
        line_index: lineIndex, text, section_id: sectionId, section: block.section,
        source_path: block.source_paths?.[index], is_section_heading: false,
        block_id: block.id, assignment: "context_block", assigned: true,
      });
    });
    section.blocks.push({
      id: block.id, entry_type: block.entry_type ?? "context_block",
      source_line_indices: normalizedIndices, source_lines: lines,
      source_paths: block.source_paths ?? [], sentences: block.sentences,
      entity_ids: block.entity_ids ?? [], relationship_ids: block.relationship_ids ?? [],
    });
    fallbackLineIndex = Math.max(fallbackLineIndex, ...normalizedIndices.map((index) => index + 1), 0);
  }

  const sourceLineAssignments = [...assignments.values()].sort((a, b) => a.line_index - b.line_index);
  const coverage = blocksDoc.coverage ?? {};
  const sourceLineCount = coverage.source_line_count ?? sourceLineAssignments.length;
  const assignedLineCount = coverage.assigned_source_line_count ?? sourceLineAssignments.length;
  return {
    format: "resume-block-audit/fallback",
    source_format: "derived_from_blocks_artifact",
    derived_from_blocks: true,
    summary: {
      source_line_count: sourceLineCount,
      assigned_source_line_count: assignedLineCount,
      unassigned_source_line_indices: coverage.unassigned_source_line_indices ?? [],
      section_count: grouped.size,
      block_count: blocksDoc.block_count ?? blocksDoc.blocks.length,
      sentence_count: coverage.sentence_count ?? blocksDoc.sentence_blocks?.length ?? 0,
      entity_count: coverage.entity_count ?? 0,
      relationship_count: coverage.relationship_count ?? 0,
      complete: coverage.complete ?? assignedLineCount >= sourceLineCount,
    },
    coverage,
    sections: [...grouped.values()].map((section) => ({
      id: section.id, name: section.name, heading_lines: [],
      source_line_indices: [...new Set(section.lineIndices)].sort((a, b) => a - b),
      block_ids: section.blocks.map((block) => block.id), blocks: section.blocks,
    })),
    source_line_assignments: sourceLineAssignments,
  };
}

function sectionKeyedOutputFromAudit(audit?: BlockAudit): Record<string, any> | undefined {
  if (!audit) return undefined;
  const aliases: Record<string, string> = {
    header: "personal_information", "personal information": "personal_information",
    summary: "profile_snapshot", experience: "work_history", projects: "projects",
    education: "education", skills: "skills", certifications: "certifications",
    "tools and technology": "tools_and_technology",
  };
  const output: Record<string, any> = {};
  const blockMap: Record<string, any>[] = [];
  const topLevelExtras: Record<string, string> = {};
  const clean = (lines: string[]) => lines.map((line) => line.replace(/^[\s•▪●*-]+/, "").trim()).filter(Boolean);
  const fieldsFromLines = (lines: string[]) => {
    const fields: Record<string, string> = {};
    const known = new Set(["name", "title", "location", "phone", "email", "place", "name at end", "year", "qualification", "university", "institution", "degree", "gpa", "cgpa", "percentage"]);
    lines.forEach((line) => {
      let currentKey: string | undefined;
      line.split(/\s*\|\s*/).forEach((part) => {
        const match = part.match(/^([^:]{2,40}):\s*(.+)$/);
        if (match && known.has(match[1].toLowerCase().trim())) {
          currentKey = match[1].toLowerCase().replace(/&/g, "and").trim().replace(/\s+/g, "_");
          fields[currentKey] = match[2].trim();
        } else if (currentKey) fields[currentKey] += ` | ${part.trim()}`;
      });
    });
    return fields;
  };
  audit.sections.forEach((section) => {
    const normalized = section.name.toLowerCase().replace(/&/g, "and").trim();
    const key = aliases[normalized] ?? normalized.replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
    section.blocks.forEach((block) => {
      const lines = clean(block.source_lines.length ? block.source_lines : block.sentences);
      let value: any = lines;
      let itemIndices: number[] = [];
      if (key === "certifications" || key === "tools_and_technology") {
        const groups: Record<string, string[]> = {};
        lines.forEach((line) => {
          const splitAt = line.indexOf(":");
          if (splitAt < 0) return;
          const group = line.slice(0, splitAt).toLowerCase().replace(/&/g, "and").trim().replace(/\s+/g, "_");
          groups[group] = line.slice(splitAt + 1).split(";").map((item) => item.trim()).filter(Boolean);
        });
        if (Object.keys(groups).length) value = groups;
      } else if (key === "work_history") {
        const header = lines[0]?.match(/^(.*?)\s+at\s+(.*?)\s*\(([^()]*)\)\s*$/i);
        if (header) {
          value = { company: header[2].trim(), duration: header[3].trim(), designation: header[1].trim(), role_and_responsibilities: lines.slice(1) };
        } else value = { details: lines, role_and_responsibilities: [] };
      } else if (key === "education" || key === "personal_information") {
        const fields = fieldsFromLines(lines);
        if (Object.keys(fields).length) value = fields;
      }
      if (key === "personal_information" && !Array.isArray(value)) {
        output[key] = { ...(output[key] ?? {}), ...value };
        if (value.place) topLevelExtras.place = value.place;
        if (value.name_at_end) topLevelExtras.name_at_end = value.name_at_end;
        delete output[key].place;
        delete output[key].name_at_end;
      } else if (["certifications", "tools_and_technology"].includes(key) && !Array.isArray(value)) {
        output[key] ??= {};
        Object.entries(value).forEach(([category, values]) => {
          if (Array.isArray(values) && Array.isArray(output[key][category])) output[key][category].push(...values);
          else output[key][category] = values;
        });
      } else if (["education", "work_history"].includes(key)) {
        output[key] ??= [];
        output[key].push(value);
        itemIndices = [output[key].length - 1];
      } else {
        output[key] ??= [];
        output[key].push(...lines);
        itemIndices = Array.from({ length: lines.length }, (_, index) => output[key].length - lines.length + index);
      }
      blockMap.push({ block_id: block.id, section_id: section.id, section_key: key, item_indices: itemIndices,
        source_line_indices: block.source_line_indices, source_paths: block.source_paths });
    });
  });
  Object.assign(output, topLevelExtras);
  output._audit = {
    format: "section-keyed-resume-blocks/v1",
    source_format: audit.source_format,
    summary: audit.summary,
    unassigned_source_line_indices: audit.summary.unassigned_source_line_indices,
    block_map: blockMap,
    source_line_assignments: audit.source_line_assignments,
  };
  return output;
}

function layoutNodes(nodes: GraphNode[], edges: GraphEdge[]): Node[] {
  // Keep the radial layout, but color by explicit relation groups instead of
  // entity type. Ignore the Candidate hub so it does not color the whole graph
  // as one group merely because many resume facts point back to the person.
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const adjacency = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  edges.filter((edge) => edge.kind === "explicit").forEach((edge) => {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (!source || !target || source.type === "PERSON" || target.type === "PERSON") return;
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  });

  const colorById = new Map<string, string>();
  const visited = new Set<string>();
  let groupIndex = 0;
  nodes.forEach((node) => {
    if (visited.has(node.id)) return;
    const group: string[] = [];
    const pending = [node.id];
    visited.add(node.id);
    while (pending.length) {
      const id = pending.pop()!;
      group.push(id);
      adjacency.get(id)?.forEach((neighbor) => {
        if (!visited.has(neighbor)) { visited.add(neighbor); pending.push(neighbor); }
      });
    }
    const hue = (groupIndex * 137.508) % 360;
    const color = `hsl(${hue} 68% 40%)`;
    group.forEach((id) => colorById.set(id, color));
    groupIndex += 1;
  });

  // Simple radial layout: PERSON in center, others in rings by type.
  const center = nodes.find((n) => n.type === "PERSON");
  const rest = nodes.filter((n) => n !== center);
  const out: Node[] = [];
  if (center) {
    out.push({
      id: center.id, position: { x: 400, y: 250 },
      data: { label: `${center.label} (${center.type})` },
      style: { background: TYPE_COLORS[center.type] ?? "#333", color: "#fff", borderRadius: 12, padding: 8, fontSize: 12, fontWeight: 700 },
    });
  }
  const R = 260;
  rest.forEach((n, i) => {
    const a = (2 * Math.PI * i) / Math.max(rest.length, 1);
    out.push({
      id: n.id, position: { x: 400 + R * Math.cos(a), y: 250 + R * Math.sin(a) },
      data: { label: `${n.label} (${n.type})` },
      style: { background: colorById.get(n.id) ?? TYPE_COLORS[n.type] ?? "#64748b", color: "#fff", borderRadius: 8, padding: 6, fontSize: 11 },
    });
  });
  return out;
}

function toEdges(edges: GraphEdge[]): Edge[] {
  return edges.map((e, i) => ({
    id: `e${i}`,
    source: e.source,
    target: e.target,
    label: e.relationship,
    animated: e.kind === "semantic",
    style: e.kind === "explicit"
      ? { stroke: "#2563eb", strokeWidth: 2 }
      : { stroke: "#f59e0b", strokeWidth: 2, strokeDasharray: "6 4" },
    labelStyle: { fontSize: 9 },
    markerEnd: { type: MarkerType.ArrowClosed },
    data: e,
  }));
}

type Tab = "text" | "entities" | "relationships" | "blocks" | "structured output" | "block audit" | "similarity" | "clusters" | "graph";

export default function App() {
  const [samples, setSamples] = useState<string[]>([]);
  const [art, setArt] = useState<Artifacts | null>(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<Tab>("graph");
  const [selected, setSelected] = useState<any>(null);
  const [paste, setPaste] = useState("");

  const fetchSamples = useCallback(async () => {
    try {
      const r = await fetch("/api/samples");
      const j = await r.json();
      setSamples(j.samples ?? []);
    } catch { /* backend may not be running */ }
  }, []);
  useEffect(() => { fetchSamples(); }, [fetchSamples]);

  const loadSample = async (rid: string) => {
    setLoading(true);
    try {
      const r = await fetch(`/api/samples/${rid}`);
      const j: Artifacts = await r.json();
      // reconstruct similarity pairs from embeddings step: use graph semantic edges
      const sem = (j["graph.json"]?.edges ?? []).filter((e) => e.kind === "semantic");
      (j as any).similarity = { top_pairs: sem.map((e) => ({ a: e.source, b: e.target, similarity: e.confidence })) };
      setArt(j); setSelected(null);
      // Sample resumes are loaded specifically to inspect their generated
      // artifacts; take the user directly to the complete source audit.
      setTab(j["resume_blocks.json"] || j["block_audit.json"] || j["blocks.json"] ? "structured output" : "graph");
    } finally { setLoading(false); }
  };

  const uploadPdf = async (f: File | undefined) => {
    if (!f) return;
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const j: Artifacts = await r.json();
      setArt(j); setSelected(null); setTab("structured output");
    } finally { setLoading(false); }
  };

  const processPaste = async () => {
    if (!paste.trim()) return;
    setLoading(true);
    try {
      const r = await fetch("/api/process-text", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: paste, doc_id: "pasted", candidate_name: "Candidate" }),
      });
      const j: Artifacts = await r.json();
      setArt(j); setSelected(null); setTab("structured output");
    } finally { setLoading(false); }
  };

  const flowNodes = useMemo(() => art?.["graph.json"] ? layoutNodes(art["graph.json"].nodes, art["graph.json"].edges) : [], [art]);
  const flowEdges = useMemo(() => art?.["graph.json"] ? toEdges(art["graph.json"].edges) : [], [art]);

  const entities = art?.["entities.json"]?.entities ?? [];
  const relationships = art?.["relationships.json"]?.relationships ?? [];
  const blocksDoc = art?.["blocks.json"];
  const blockAudit = art?.["block_audit.json"] ?? auditFromBlocks(blocksDoc);
  const resumeBlocks = art?.["resume_blocks.json"] ?? sectionKeyedOutputFromAudit(blockAudit);
  const graph = art?.["graph.json"];
  const clusters = art?.["clusters.json"];
  const emb = art?.["embeddings.json"];
  const simPairs = (art?.["similarity.json"]?.top_pairs ?? art?.similarity?.top_pairs ?? []).slice(0, 25);

  const downloadBlockAudit = () => {
    if (!blockAudit) return;
    const blob = new Blob([JSON.stringify(blockAudit, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "block_audit.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadResumeBlocks = () => {
    if (!resumeBlocks) return;
    const blob = new Blob([JSON.stringify(resumeBlocks, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "resume_blocks.json";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <header>
        <h1>Resume Knowledge Graph — PoC (IJISRT25MAY2182 methodology)</h1>
        <p>Resume PDF → Text Repository → Normalization (sections preserved) → Entities → Context Blocks → SVO/Relations → BERT/SBERT → Cosine → DBSCAN + K-Means → KG → React Flow</p>
      </header>
      <div className="layout">
        {/* left: inputs */}
        <div className="panel">
          <h2>1 · Input</h2>
          <h3>Sample resumes (first experiment)</h3>
          {samples.length === 0 && <p className="hint">Backend not reachable or no samples yet. Start backend: <code>uvicorn app.main:app --reload</code> from backend/.</p>}
          {samples.map((s) => (
            <div key={s}><button onClick={() => loadSample(s)} disabled={loading}>Load {s}</button></div>
          ))}
          <h3>Upload resume PDF</h3>
          <input type="file" accept=".pdf,.txt" onChange={(e) => uploadPdf(e.target.files?.[0])} />
          <h3>Or paste resume text</h3>
          <textarea rows={8} value={paste} onChange={(e) => setPaste(e.target.value)} placeholder="Paste resume text here…" />
          <button className="primary" onClick={processPaste} disabled={loading || !paste.trim()}>Run pipeline</button>
          {loading && <p className="hint">Running pipeline…</p>}
          {graph && (
            <div className="detail">
              <b>Stats:</b> {graph.stats.node_count} nodes · {graph.stats.edge_count} edges
              ({graph.stats.explicit_edges} explicit solid · {graph.stats.semantic_edges} semantic dashed)
              <br /><span className="hint">{graph.stats.note}</span>
            </div>
          )}
          {selected && (
            <div className="detail">
              <b>Selected:</b>
              <pre style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>{JSON.stringify(selected, null, 2)}</pre>
            </div>
          )}
        </div>

        {/* right: artifacts */}
        <div className="panel">
          <div className="tabs">
            {(["text", "entities", "relationships", "blocks", "structured output", "block audit", "similarity", "clusters", "graph"] as Tab[]).map((t) => (
              <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>
            ))}
          </div>

          {tab === "text" && (
            <><h2>Raw text (Text Repository)</h2>
            <pre className="raw">{art?.["raw_text.json"]?.raw_text ?? "Load a sample or upload a resume."}</pre></>
          )}

          {tab === "entities" && (
            <><h2>Entities ({entities.length}) — Entity Identification</h2>
            <table><thead><tr><th>Text</th><th>Type</th><th>Conf</th><th>Sections</th><th>Method</th><th>Source sentence</th></tr></thead>
            <tbody>{entities.map((e, i) => (
              <tr key={i} onClick={() => setSelected(e)} style={{ cursor: "pointer" }}>
                <td><span className="badge" style={{ background: TYPE_COLORS[e.type] ?? "#666", color: "#fff" }}>{e.text}</span></td>
                <td>{e.type}</td><td>{e.confidence}</td><td className="hint">{(e.sections ?? []).join(", ")}</td><td>{e.method}</td>
                <td className="hint">{(e.source_sentences ?? []).slice(0, 2).join(" / ").slice(0, 160)}</td>
              </tr>))}</tbody></table></>
          )}

          {tab === "relationships" && (
            <><h2>Validated relationships ({relationships.length}) — canonical resume structure</h2>
            <p className="hint">Edges come from grouped resume records and retain source-path evidence. Similarity does not create relationships.</p>
            <table><thead><tr><th>Source</th><th>Relationship</th><th>Target</th><th>Conf</th><th>Section</th><th>Source sentence</th></tr></thead>
            <tbody>{relationships.map((r, i) => (
              <tr key={i} onClick={() => setSelected(r)} style={{ cursor: "pointer" }}>
                <td>{r.source}</td><td><b>{r.relationship}</b></td><td>{r.target}</td>
                <td>{r.confidence}</td><td className="hint">{r.section ?? ""}</td><td className="hint">{(r.source_sentence ?? "").slice(0, 160)}</td>
              </tr>))}</tbody></table></>
          )}

          {tab === "blocks" && (
            <><h2>Context blocks ({blocksDoc?.block_count ?? 0}) — section-aware segmentation</h2>
            <p className="hint">{blocksDoc?.method ?? "No data yet — load a sample."}</p>
            <table><thead><tr><th>Block</th><th>Section</th><th>Sentences</th><th>Entities</th></tr></thead>
            <tbody>{(blocksDoc?.blocks ?? []).map((b) => (
              <tr key={b.id} onClick={() => setSelected(b)} style={{ cursor: "pointer" }}>
                <td><b>{b.id}</b></td><td>{b.section}</td>
                <td className="hint">{b.sentences.join(" / ").slice(0, 220)}</td>
                <td className="hint">{b.entities.join(", ").slice(0, 160)}</td>
              </tr>))}</tbody></table></>
          )}

          {tab === "block audit" && (
            <>
              <h2>Block audit — source coverage</h2>
              {!blockAudit && <p className="hint">No block data was returned for this resume. Rerun the pipeline or restart the backend and try again.</p>}
              {blockAudit && <>
                {blockAudit.derived_from_blocks && <p className="hint">Showing blocks reconstructed from blocks.json. Restart the backend to include exact source-line coverage and source paths.</p>}
                <div className="detail">
                  <b>{blockAudit.summary.complete ? "Complete coverage" : "Coverage needs review"}</b>
                  <div className="hint">
                    Source: {blockAudit.source_format} · {blockAudit.summary.section_count} sections · {blockAudit.summary.block_count} blocks
                    · {blockAudit.summary.assigned_source_line_count}/{blockAudit.summary.source_line_count} source rows assigned
                    · {blockAudit.summary.sentence_count} sentences · {blockAudit.summary.entity_count} entities
                    · {blockAudit.summary.relationship_count} relationships
                  </div>
                  <button onClick={downloadBlockAudit} style={{ marginTop: 8 }}>Download block_audit.json</button>
                </div>
                {blockAudit.summary.unassigned_source_line_indices.length > 0 && (
                  <div className="detail">
                    <b>Unassigned source lines</b>
                    <p className="hint">Line indices: {blockAudit.summary.unassigned_source_line_indices.join(", ")}</p>
                    {blockAudit.source_line_assignments.filter((row) => !row.assigned).map((row) => (
                      <p key={row.line_index}><b>Line {row.line_index}:</b> {row.text} <span className="hint">{row.reason}</span></p>
                    ))}
                  </div>
                )}
                {blockAudit.sections.map((section) => (
                  <details className="audit-section" key={section.id}>
                    <summary>
                      <b>{section.name}</b> <span className="hint">· {section.blocks.length} blocks · {section.source_line_indices.length} source rows</span>
                    </summary>
                    {section.heading_lines.map((heading, index) => <p className="hint" key={index}>Section heading: {heading}</p>)}
                    {section.blocks.map((block) => (
                      <details className="audit-block" key={block.id}>
                        <summary><b>{block.id}</b> · {block.entry_type} <span className="hint">· lines {block.source_line_indices.join(", ")}</span></summary>
                        {block.source_paths.length > 0 && <p className="hint">Source path: {block.source_paths.join(", ")}</p>}
                        <div className="audit-source">
                          {block.source_lines.map((line, index) => <div key={index}>{line}</div>)}
                        </div>
                        {block.entity_ids.length > 0 && <p className="hint">Entities: {block.entity_ids.join(", ")}</p>}
                        {block.relationship_ids.length > 0 && <p className="hint">Relationships: {block.relationship_ids.join(", ")}</p>}
                      </details>
                    ))}
                  </details>
                ))}
                <details className="audit-section">
                  <summary><b>All source row assignments</b> <span className="hint">· {blockAudit.source_line_assignments.length} rows</span></summary>
                  <table><thead><tr><th>Line</th><th>Section</th><th>Block</th><th>Source path</th><th>Text</th></tr></thead>
                    <tbody>{blockAudit.source_line_assignments.map((row) => (
                      <tr key={row.line_index}>
                        <td>{row.line_index}</td><td>{row.section}</td><td>{row.block_id ?? (row.is_section_heading ? "heading" : "unassigned")}</td>
                        <td className="hint">{row.source_path ?? "—"}</td><td>{row.text}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </details>
              </>}
            </>
          )}

          {tab === "structured output" && (
            <>
              <h2>Resume output — section-keyed JSON</h2>
              {!resumeBlocks && <p className="hint">Run the pipeline to generate section-keyed resume blocks.</p>}
              {resumeBlocks && <>
                <p className="hint">Resume content is grouped under readable section keys, like the reference output. Block IDs, source paths, and coverage are available in the audit details below.</p>
                <button onClick={downloadResumeBlocks}>Download resume_blocks.json</button>
                {resumeBlocks._audit?.summary && <div className="detail">
                  <b>{resumeBlocks._audit.summary.complete ? "All source rows assigned" : "Some source rows need review"}</b>
                  <span className="hint"> · {resumeBlocks._audit.summary.assigned_source_line_count}/{resumeBlocks._audit.summary.source_line_count} rows · {resumeBlocks._audit.summary.block_count} blocks</span>
                </div>}
                <pre className="raw structured-output">{JSON.stringify(Object.fromEntries(Object.entries(resumeBlocks).filter(([key]) => key !== "_audit")), null, 2)}</pre>
                {resumeBlocks._audit && <details className="audit-section">
                  <summary><b>Block and source audit details</b></summary>
                  <pre className="raw structured-output">{JSON.stringify(resumeBlocks._audit, null, 2)}</pre>
                </details>}
              </>}
            </>
          )}

          {tab === "similarity" && (
            <><h2>Semantic similarity — cosine over {emb?.model ?? "embeddings"} (dim {emb?.dim ?? "?"})</h2>
            <p className="hint">High similarity ≠ factual relation. It only yields SEMANTICALLY_SIMILAR dashed edges (see Graph tab).</p>
            <table><thead><tr><th>A</th><th>B</th><th>Cosine</th></tr></thead>
            <tbody>{simPairs.map((p, i) => (
              <tr key={i}><td>{p.a}</td><td>{p.b}</td><td>{p.similarity}</td></tr>))}</tbody></table>
              {!simPairs.length && <p className="hint">No data yet — load a sample.</p>}</>
          )}

          {tab === "clusters" && (
            <><h2>DBSCAN + K-Means (Sentence-BERT space)</h2>
            {!clusters && <p className="hint">No data yet — load a sample.</p>}
            {clusters && (
              <>
                <h3>DBSCAN — natural groups + outliers ({clusters.dbscan?.algorithm})</h3>
                <pre className="raw">{JSON.stringify({ clusters: clusters.dbscan?.clusters, noise_outliers: clusters.dbscan?.noise_outliers }, null, 2)}</pre>
                <h3>K-Means — k={clusters.kmeans?.k} (silhouette {clusters.kmeans?.silhouette_score ?? "n/a"})</h3>
                <pre className="raw">{JSON.stringify(clusters.kmeans?.clusters, null, 2)}</pre>
              </>
            )}</>
          )}

          {tab === "graph" && (
            <><h2>Knowledge Graph — nodes + edges</h2>
            <div className="legend">
              <span><span className="solid-line" /> explicit (NLP parsing)</span>
              <span><span className="dashed-line" /> semantic/inferred (cosine — not factual)</span>
            </div>
            {!graph && <p className="hint">Load a sample or upload a resume to render the graph.</p>}
            {graph && (
              <div className="graph-wrap">
                <ReactFlow
                  nodes={flowNodes} edges={flowEdges} fitView
                  onNodeClick={(_, n) => setSelected(graph.nodes.find((x) => x.id === n.id))}
                  onEdgeClick={(_, e) => setSelected((e.data as GraphEdge))}
                >
                  <Background /><Controls /><MiniMap />
                </ReactFlow>
              </div>
            )}
            <p className="flow-note">Click any node/edge to see its source sentence and confidence (left panel). Solid = extracted relation, dashed = semantic closeness only.</p></>
          )}
        </div>
      </div>
    </>
  );
}
