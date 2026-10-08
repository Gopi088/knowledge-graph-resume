"""Step 3 — Entity Identification (paper: custom logic to identify important entities).

Adapted from generic PERSON/ORG/CONCEPT to resume ontology:
PERSON, COMPANY, JOB_ROLE, SKILL, TECHNOLOGY, PROGRAMMING_LANGUAGE,
FRAMEWORK, DATABASE, CLOUD, TOOL, PROJECT, DEGREE, UNIVERSITY,
CERTIFICATION, DOMAIN.

Strategy (kept simple / research-friendly):
  1. spaCy NER -> PERSON / ORG(=COMPANY) / GPE etc.
  2. Rule-based gazetteers for resume vocabulary (case-insensitive phrase match).
  3. Degree/university/certification regex patterns.
Every entity keeps its source sentence for provenance.
"""
import re

# ---- Resume gazetteers (phrase -> entity type) ----
PROGRAMMING_LANGUAGES = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "ruby", "php", "kotlin", "swift", "scala", "rust", "sql",
]
FRAMEWORKS = [
    "django", "flask", "fastapi", "spring", "spring boot", "react", "angular",
    "vue", "vue.js", "node.js", "express", "next.js", ".net", "tensorflow",
    "pytorch", "keras", "scikit-learn", "pandas", "numpy",
]
DATABASES = ["postgresql", "postgres", "mysql", "mongodb", "redis", "sqlite",
             "elasticsearch", "dynamodb", "cassandra", "oracle db", "sql server"]
CLOUDS = ["aws", "amazon web services", "azure", "gcp", "google cloud",
          "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "github actions"]
TOOLS = ["git", "github", "gitlab", "jira", "postman", "kafka", "rabbitmq",
         "airflow", "spark", "hadoop", "tableau", "power bi", "figma", "linux"]
SKILLS_EXTRA = ["rest api", "rest apis", "graphql", "microservices", "machine learning",
                "deep learning", "nlp", "data analysis", "etl", "unit testing",
                "system design", "agile", "scrum", "ai", "artificial intelligence",
                "ai/ml", "rag", "cnn", "transfer learning"]
FRAMEWORKS.append("langchain")
TOOLS.append("faiss")
TECHNOLOGIES = ["html", "css", "nosql", "oauth", "jwt", "websockets"]

CERTIFICATIONS = ["aws certified", "aws solutions architect", "azure fundamentals",
                  "google cloud professional", "cka", "pmp", "csm", "oracle certified"]
DEGREE_PATTERNS = [
    r"\b(b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|bachelor(?:s)?|master(?:s)?|mba|ph\.?d|bca|mca|higher\s+secondary|secondary\s+education|class\s+(?:10|12)|(?:10|12)(?:th|\/th))\b[^.\n]{0,80}",
]
UNIVERSITY_PATTERNS = [
    r"\b((?:[A-Z][A-Za-z&.'-]*\s+){0,5}(?:University|Institute|College|School|Vidyalaya))\b|\b((?:IIT|NIT)\s+[A-Z][A-Za-z&.'-]+)\b",
]
JOB_ROLES = ["software engineer", "backend developer", "frontend developer",
             "full stack developer", "data scientist", "data engineer", "ml engineer",
             "devops engineer", "cloud engineer", "qa engineer", "product manager",
             "ui/ux designer", "system analyst", "intern", "senior software engineer"]
DOMAINS = ["fintech", "healthcare", "e-commerce", "ecommerce", "edtech",
           "saas", "banking", "retail", "logistics"]
_ORG_NOISE = re.compile(
    r"\b(software|data|backend|frontend|full stack|machine learning|deep learning|"
    r"transfer learning|artificial intelligence|ai/ml|nlp|rag|engineer|developer|"
    r"scientist|analyst|manager|designer|python|java|javascript|tensorflow|pytorch|"
    r"certified|architect)\b", re.I,
)
_ORG_SUFFIX = re.compile(r"\b(inc\.?|llc|ltd\.?|limited|corp\.?|corporation|technologies|technology|analytics|systems|labs)\b", re.I)

GAZETTEER: list[tuple[str, str]] = []
for p in PROGRAMMING_LANGUAGES:
    GAZETTEER.append((p, "PROGRAMMING_LANGUAGE"))
for p in FRAMEWORKS:
    GAZETTEER.append((p, "FRAMEWORK"))
for p in DATABASES:
    GAZETTEER.append((p, "DATABASE"))
for p in CLOUDS:
    GAZETTEER.append((p, "CLOUD"))
for p in TOOLS:
    GAZETTEER.append((p, "TOOL"))
for p in SKILLS_EXTRA + TECHNOLOGIES:
    label = "SKILL" if p in SKILLS_EXTRA else "TECHNOLOGY"
    GAZETTEER.append((p, label))
for p in CERTIFICATIONS:
    GAZETTEER.append((p, "CERTIFICATION"))
for p in JOB_ROLES:
    GAZETTEER.append((p, "JOB_ROLE"))
for p in DOMAINS:
    GAZETTEER.append((p, "DOMAIN"))

# longest phrases first so "spring boot" wins over "spring"
GAZETTEER.sort(key=lambda x: -len(x[0]))
_GAZETTEER_TERMS = {phrase.casefold() for phrase, _ in GAZETTEER}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_entities(
    sentences: list[str],
    candidate_name: str = "Candidate",
    sentence_blocks: list[str] | None = None,
) -> list[dict]:
    from .preprocessing import get_nlp
    nlp = get_nlp()

    sentence_blocks = sentence_blocks or [f"b{i}" for i in range(len(sentences))]
    entities: dict[tuple[str, str, str], dict] = {}  # text/type scoped to one source block

    def add(text: str, etype: str, sentence: str, sentence_index: int,
            method: str, confidence: float):
        text = _norm(text)
        if len(text) < 2:
            return
        block_id = sentence_blocks[sentence_index] if sentence_index < len(sentence_blocks) else f"b{sentence_index}"
        key = (text.lower(), etype, block_id)
        if key not in entities:
            entities[key] = {
                "text": text, "type": etype, "confidence": confidence,
                "method": method, "source_sentences": [sentence],
                "frequency": 1, "block_ids": [block_id],
                "sentence_indices": [sentence_index],
            }
        else:
            entities[key]["frequency"] += 1
            if sentence not in entities[key]["source_sentences"]:
                entities[key]["source_sentences"].append(sentence)
            if sentence_index not in entities[key]["sentence_indices"]:
                entities[key]["sentence_indices"].append(sentence_index)

    # Candidate node (resume subject). Prefer the known resume name passed in;
    # only fall back to spaCy PERSON when no explicit name was given.
    # (spaCy often over-captures e.g. "Priya Nair Data" from "Priya Nair\nData Scientist".)
    person_found = None
    if candidate_name and candidate_name != "Candidate":
        person_found = _norm(candidate_name)
    elif nlp is not None and sentences:
        try:
            doc0 = nlp(" ".join(sentences[:3])[:2000])
            for ent in doc0.ents:
                if ent.label_ == "PERSON":
                    # guard: keep at most 3 title-cased tokens
                    toks = _norm(ent.text).split()
                    person_found = _norm(" ".join(toks[:3]))
                    break
        except Exception:
            pass
    candidate = person_found or candidate_name
    if sentences:
        add(candidate, "PERSON", sentences[0], 0, "spaCy NER / resume subject", 0.95)

    for sentence_index, sent in enumerate(sentences):
        slow = sent.lower()
        # 1) gazetteer phrase match
        for phrase, etype in GAZETTEER:
            # "ai" must match as a whole word only ("email"/"said" contain "ai").
            if phrase == "ai":
                m = re.search(r"\bai\b", sent, re.IGNORECASE)
                if not m or re.match(r"\s*/\s*ml\b", sent[m.end():], re.IGNORECASE):
                    continue
                surface = "AI"
            else:
                matches = list(re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", sent, re.IGNORECASE))
                if not matches:
                    continue
                # recover original casing from sentence
                m = matches[0]
            if phrase != "ai":
                surface = m.group(0) if m else phrase
                # tidy: title-case short techs, keep known acronyms upper
                if surface.lower() in ("aws", "gcp", "sql", "nlp", "etl", "ci/cd"):
                    surface = surface.upper()
            add(surface, etype, sent, sentence_index, "gazetteer + NLP parsing", 0.90)
        # 2) spaCy NER for COMPANY/UNIVERSITY/PROJECT-ish
        if nlp is not None:
            try:
                doc = nlp(sent)
                for ent in doc.ents:
                    if ent.label_ == "ORG" and not _ORG_NOISE.search(ent.text):
                        # disambiguate: university vs company
                        if re.search(r"university|institute|college|iit|nit", ent.text, re.I):
                            # Use the bounded institution regex below rather
                            # than a potentially oversized NER span.
                            continue
                        elif (_ORG_SUFFIX.search(ent.text)
                              and ent.text.casefold() not in _GAZETTEER_TERMS):
                            add(ent.text, "COMPANY", sent, sentence_index, "spaCy NER", 0.85)
                    elif ent.label_ in ("PRODUCT", "WORK_OF_ART"):
                        add(ent.text, "PROJECT", sent, sentence_index, "spaCy NER", 0.70)
            except Exception:
                pass
        # 3) degree patterns
        for pat in DEGREE_PATTERNS:
            for m in re.finditer(pat, sent, re.IGNORECASE):
                add(m.group(0), "DEGREE", sent, sentence_index, "regex + NLP parsing", 0.88)
        # 4) university patterns. Require a full institution-name pattern so
        #    fragments such as "Tech in Data Science from IIT" are not emitted.
        for pat in UNIVERSITY_PATTERNS:
            for m in re.finditer(pat, sent):
                institution = next((group for group in m.groups() if group), "")
                if institution:
                    add(institution, "UNIVERSITY", sent, sentence_index, "regex + NLP parsing", 0.88)

        # Explicit employer phrasing is more reliable than broad ORG spans.
        for m in re.finditer(
            r"\bat\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,5})",
            sent,
        ):
            company = m.group(1).strip().rstrip(".,")
            if (company and not _ORG_NOISE.search(company)
                    and company.casefold() not in _GAZETTEER_TERMS):
                add(company, "COMPANY", sent, sentence_index, "employer context pattern", 0.90)
        # 5) "Project: <Name>" / "<Name> project" heuristic -> PROJECT
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9\- ]{2,40}?)\s*(?:project|app|platform|system|dashboard)\b", sent):
            name = _norm(m.group(1))
            if name.lower() not in [p for p, _ in GAZETTEER]:
                add(name, "PROJECT", sent, sentence_index, "heuristic + NLP parsing", 0.65)

    ents = sorted(entities.values(), key=lambda e: (e["type"], -e["frequency"], e["text"]))
    for i, e in enumerate(ents):
        e["id"] = f"e{i}"
    return ents
