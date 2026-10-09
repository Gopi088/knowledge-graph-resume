import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

const provenanceKeys = new Set([
  "source_blocks", "source_block_ids", "source_lines", "source_paths", "source_line_indices",
  "field_sources", "responsibility_sources", "employment_periods", "employment_period",
  "schema_version", "source_sections", "unresolved_blocks",
]);

function labelForKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function hasDisplayValue(value: unknown): boolean {
  if (value == null || value === "") return false;
  if (Array.isArray(value)) return value.some(hasDisplayValue);
  if (typeof value === "object") return Object.entries(value as Record<string, unknown>)
    .some(([key, nested]) => !provenanceKeys.has(key) && hasDisplayValue(nested));
  return true;
}

function phoneFromResumeHeader(rawText: string | undefined): string | undefined {
  if (!rawText) return undefined;
  const sectionHeading = /^(?:professional summary|profile summary|summary|about me|professional experience|work experience|employment history|work history|career history|experience|projects?|education|technical skills|skills|core competencies|certifications?|achievements|personal details)\s*:?$/i;
  const headerLines: string[] = [];
  for (const line of rawText.split(/\r?\n/)) {
    const cleaned = line.trim();
    if (sectionHeading.test(cleaned)) break;
    if (cleaned) headerLines.push(cleaned);
    if (headerLines.length >= 18) break;
  }
  const phonePattern = /(?:^|[^\w])(\+?\d[\d().\s-]{5,}\d)(?!\w)/g;
  for (const line of headerLines) {
    for (const match of line.matchAll(phonePattern)) {
      const candidate = match[1].trim();
      const digitCount = candidate.replace(/\D/g, "").length;
      const looksLikeDate = /^(?:\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2})$/.test(candidate);
      if (!looksLikeDate && digitCount >= 7 && digitCount <= 15) return candidate;
    }
  }
  return undefined;
}

function contactsFromResumeHeader(rawText: string | undefined): Record<string, string> {
  if (!rawText) return {};
  const sectionHeading = /^(?:professional summary|profile summary|summary|about me|professional experience|work experience|employment history|work history|career history|experience|projects?|education|technical skills|skills|core competencies|certifications?|achievements|personal details)\s*:?$/i;
  const lines: string[] = [];
  for (const line of rawText.split(/\r?\n/)) {
    const cleaned = line.trim();
    if (sectionHeading.test(cleaned)) break;
    if (cleaned) lines.push(cleaned);
    if (lines.length >= 18) break;
  }
  const text = lines.join("\n");
  const contacts: Record<string, string> = {};
  const email = text.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i)?.[0];
  if (email) contacts.email = email;
  const phone = phoneFromResumeHeader(rawText);
  if (phone) contacts.phone = phone;
  for (const [key, pattern] of Object.entries({
    linkedin: /(?:(?:https?:\/\/)?(?:www\.)?)linkedin\.com\/[^\s|,;]+/i,
    github: /(?:(?:https?:\/\/)?(?:www\.)?)github\.com\/[^\s|,;]+/i,
  })) {
    const link = text.match(pattern)?.[0]?.replace(/[.,)]+$/, "");
    if (link) contacts[key] = link;
  }
  const contactLine = lines.find((line) => /@|\+?\d[\d().\s-]{5,}\d/.test(line));
  if (contactLine) {
    const location = contactLine
      .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/ig, " ")
      .replace(/(?:^|[^\w])\+?\d[\d().\s-]{5,}\d(?!\w)/g, " ")
      .replace(/(?:(?:https?:\/\/)?(?:www\.)?)?(?:linkedin\.com|github\.com)\/[^\s|,;]+/ig, " ")
      .replace(/\b(?:linkedin|github|email|e-mail|phone|mobile|location|address)\s*:?/ig, " ")
      .replace(/[|·•]+/g, " ").replace(/\s+/g, " ").trim().replace(/^[,;:-]+|[,;:-]+$/g, "");
    if (location && !/\d|@/.test(location)) contacts.location = location;
  }
  return contacts;
}

function ReadableValue({ value }: { value: unknown }) {
  if (value == null || typeof value === "boolean") return null;
  if (typeof value === "string" || typeof value === "number") return <>{String(value)}</>;
  if (Array.isArray(value)) {
    const visible = value.filter(hasDisplayValue);
    if (!visible.length) return null;
    return <ul className="canonical-list">{visible.map((item, index) => <li key={index}><ReadableValue value={item} /></li>)}</ul>;
  }
  if (typeof value === "object") {
    const fields = Object.entries(value as Record<string, unknown>)
      .filter(([key, nested]) => !provenanceKeys.has(key) && hasDisplayValue(nested));
    if (!fields.length) return null;
    return <dl className="canonical-fields">{fields.map(([key, nested]) => (
      <div key={key}><dt>{labelForKey(key)}</dt><dd><ReadableValue value={nested} /></dd></div>
    ))}</dl>;
  }
  return null;
}

function CanonicalSection({ title, value }: { title: string; value: unknown }) {
  if (!hasDisplayValue(value)) return null;
  return <section className="canonical-section">
    <h3>{title}</h3>
    <ReadableValue value={value} />
  </section>;
}

function CanonicalWorkHistory({ value }: { value: unknown }) {
  if (!Array.isArray(value) || !value.some(hasDisplayValue)) return null;
  return <section className="canonical-section">
    <h3>Work Experience</h3>
    {value.filter(hasDisplayValue).map((company, companyIndex) => {
      if (!company || typeof company !== "object" || Array.isArray(company)) {
        return <div className="canonical-record" key={companyIndex}><ReadableValue value={company} /></div>;
      }
      const record = company as Record<string, unknown>;
      const roles = Array.isArray(record.roles) ? record.roles : [];
      const companyName = record.company ?? record.name;
      return <article className="canonical-record" key={companyIndex}>
        {hasDisplayValue(companyName) && <h4>{String(companyName)}</h4>}
        {roles.length > 0 ? roles.map((role, roleIndex) => {
          if (!role || typeof role !== "object" || Array.isArray(role)) {
            return <div className="canonical-role" key={roleIndex}><ReadableValue value={role} /></div>;
          }
          const roleRecord = role as Record<string, unknown>;
          const designation = roleRecord.designation ?? roleRecord.title ?? roleRecord.role;
          const responsibilities = roleRecord.role_and_responsibilities ?? roleRecord.responsibilities;
          const roleFields = Object.fromEntries(Object.entries(roleRecord).filter(([key]) =>
            !["designation", "title", "role", "role_and_responsibilities", "responsibilities"].includes(key)));
          return <div className="canonical-role" key={roleIndex}>
            {hasDisplayValue(designation) && <h4>{String(designation)}</h4>}
            <ReadableValue value={roleFields} />
            {hasDisplayValue(responsibilities) && <div className="canonical-responsibilities">
              <b>Responsibilities</b><ReadableValue value={responsibilities} />
            </div>}
          </div>;
        }) : <ReadableValue value={Object.fromEntries(Object.entries(record).filter(([key]) => key !== "company" && key !== "name"))} />}
      </article>;
    })}
  </section>;
}

async function readArtifacts(response: Response): Promise<Artifacts> {
  let data: any;
  try { data = await response.json(); }
  catch { throw new Error(`The server returned an unreadable response (HTTP ${response.status}).`); }
  if (!response.ok) {
    const detail = data?.detail ?? data?.message ?? data?.error;
    throw new Error(typeof detail === "string" ? detail : `Request failed (HTTP ${response.status}).`);
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("The server returned an invalid resume response.");
  }
  return data as Artifacts;
}

type Tab = "text" | "entities" | "relationships" | "blocks" | "structured output" | "block audit" | "similarity" | "clusters" | "graph";

export default function App() {
  const [samples, setSamples] = useState<string[]>([]);
  const [art, setArt] = useState<Artifacts | null>(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<Tab>("graph");
  const [selected, setSelected] = useState<any>(null);
  const [paste, setPaste] = useState("");
  const [requestError, setRequestError] = useState<string | null>(null);
  const requestSequence = useRef(0);

  const beginRequest = () => {
    requestSequence.current += 1;
    setArt(null);
    setSelected(null);
    setRequestError(null);
    setLoading(true);
    return requestSequence.current;
  };

  const fetchSamples = useCallback(async () => {
    try {
      const r = await fetch("/api/samples");
      const j = await r.json();
      setSamples(j.samples ?? []);
    } catch { /* backend may not be running */ }
  }, []);
  useEffect(() => { fetchSamples(); }, [fetchSamples]);

  const loadSample = async (rid: string) => {
    const requestId = beginRequest();
    try {
      const r = await fetch(`/api/samples/${rid}`);
      const j = await readArtifacts(r);
      // reconstruct similarity pairs from embeddings step: use graph semantic edges
      const sem = (j["graph.json"]?.edges ?? []).filter((e) => e.kind === "semantic");
      (j as any).similarity = { top_pairs: sem.map((e) => ({ a: e.source, b: e.target, similarity: e.confidence })) };
      if (requestId !== requestSequence.current) return;
      setArt(j);
      // Sample resumes are loaded specifically to inspect their generated
      // artifacts; take the user directly to the complete source audit.
      setTab("structured output");
    } catch (error) {
      if (requestId === requestSequence.current) setRequestError(error instanceof Error ? error.message : "Unable to load this sample.");
    } finally {
      if (requestId === requestSequence.current) setLoading(false);
    }
  };

  const uploadPdf = async (f: File | undefined) => {
    if (!f) return;
    const requestId = beginRequest();
    try {
      const fd = new FormData();
      fd.append("file", f, f.name);
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const j = await readArtifacts(r);
      if (requestId !== requestSequence.current) return;
      setArt(j);
      setTab("structured output");
    } catch (error) {
      if (requestId === requestSequence.current) setRequestError(error instanceof Error ? error.message : "Unable to process this resume.");
    } finally {
      if (requestId === requestSequence.current) setLoading(false);
    }
  };

  const processPaste = async () => {
    if (!paste.trim()) return;
    const requestId = beginRequest();
    try {
      const r = await fetch("/api/process-text", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: paste, doc_id: "pasted", candidate_name: "Candidate" }),
      });
      const j = await readArtifacts(r);
      if (requestId !== requestSequence.current) return;
      setArt(j);
      setTab("structured output");
    } catch (error) {
      if (requestId === requestSequence.current) setRequestError(error instanceof Error ? error.message : "Unable to process the pasted resume text.");
    } finally {
      if (requestId === requestSequence.current) setLoading(false);
    }
  };

  const flowNodes = useMemo(() => art?.["graph.json"] ? layoutNodes(art["graph.json"].nodes, art["graph.json"].edges) : [], [art]);
  const flowEdges = useMemo(() => art?.["graph.json"] ? toEdges(art["graph.json"].edges) : [], [art]);

  const entities = art?.["entities.json"]?.entities ?? [];
  const relationships = art?.["relationships.json"]?.relationships ?? [];
  const blocksDoc = art?.["blocks.json"];
  const blockAudit = art?.["block_audit.json"] ?? auditFromBlocks(blocksDoc);
  const canonicalResume = art?.["canonical_resume.json"];
  const canonicalPersonal = canonicalResume?.personal_information;
  const sourceHeaderContacts = contactsFromResumeHeader(art?.["raw_text.json"]?.raw_text);
  const recoveredHeaderContacts = Object.fromEntries(Object.entries(sourceHeaderContacts)
    .filter(([key, value]) => !hasDisplayValue(canonicalPersonal?.[key]) && value));
  const contactsWereRecovered = Boolean(canonicalResume && Object.keys(recoveredHeaderContacts).length);
  const personalInformationForDisplay = contactsWereRecovered
    ? { ...canonicalPersonal, ...recoveredHeaderContacts }
    : canonicalPersonal;
  const canonicalResumeForDisplay = contactsWereRecovered
    ? { ...canonicalResume, personal_information: personalInformationForDisplay }
    : canonicalResume;
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

  const downloadCanonicalResume = () => {
    if (!canonicalResumeForDisplay) return;
    const blob = new Blob([JSON.stringify(canonicalResumeForDisplay, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "canonical_resume.json";
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
          <input type="file" accept=".pdf,.txt" disabled={loading} onChange={(e) => {
            const file = e.currentTarget.files?.[0];
            e.currentTarget.value = "";
            void uploadPdf(file);
          }} />
          <h3>Or paste resume text</h3>
          <textarea rows={8} value={paste} onChange={(e) => setPaste(e.target.value)} placeholder="Paste resume text here…" />
          <button className="primary" onClick={processPaste} disabled={loading || !paste.trim()}>Run pipeline</button>
          {loading && <p className="hint">Running pipeline…</p>}
          {requestError && <div className="detail error" role="alert"><b>Could not load resume</b><div>{requestError}</div></div>}
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
              <h2>Parsed resume</h2>
              {!art && !loading && <p className="hint">Load a sample, upload a resume, or paste resume text to view its parsed data.</p>}
              {art && !canonicalResume && <div className="detail error" role="status">
                <b>The latest response did not include <code>canonical_resume.json</code>.</b>
                <div>No previous or reconstructed resume is being shown.</div>
                <div className="hint">Artifacts received: {Object.keys(art).filter((key) => key.endsWith(".json")).join(", ") || "none"}</div>
                <div className="hint">If the selected sample has a canonical output file, restart the backend serving port 8000 and load the sample again.</div>
              </div>}
              {canonicalResume && <>
                <p className="hint">Showing the canonical resume returned by the latest backend request.</p>
                {!Object.entries(canonicalResume).some(([key, value]) => !provenanceKeys.has(key) && hasDisplayValue(value)) &&
                  <p className="hint">The latest response contains no parsed resume fields.</p>}
                <button onClick={downloadCanonicalResume}>Download canonical_resume.json</button>
                <CanonicalSection title="Personal Information" value={personalInformationForDisplay} />
                {contactsWereRecovered && <p className="hint">Missing contact fields were recovered from this resume’s header text.</p>}
                <CanonicalSection title="Profile Snapshot" value={canonicalResume.profile_snapshot} />
                <CanonicalWorkHistory value={canonicalResume.work_history} />
                <CanonicalSection title="Projects" value={canonicalResume.projects} />
                <CanonicalSection title="Skills" value={canonicalResume.skills} />
                <CanonicalSection title="Technical Skills" value={canonicalResume.technical_skills} />
                <CanonicalSection title="Domain Experience" value={canonicalResume.domain_experience} />
                <CanonicalSection title="Education" value={canonicalResume.education} />
                <CanonicalSection title="Certifications" value={canonicalResume.certifications} />
                <CanonicalSection title="Awards" value={canonicalResume.awards} />
                <CanonicalSection title="Achievements" value={canonicalResume.achievements} />
                <CanonicalSection title="Publications" value={canonicalResume.publications} />
                <CanonicalSection title="Personal Details" value={canonicalResume.personal_details} />
                <details className="audit-section">
                  <summary><b>Canonical JSON</b></summary>
                  <pre className="raw structured-output">{JSON.stringify(canonicalResumeForDisplay, null, 2)}</pre>
                </details>
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
