"""FastAPI backend — thin wrapper over the research pipeline.

Endpoints:
  POST /api/upload        — upload a resume PDF, run full pipeline, return all artifacts
  POST /api/process-text  — same, from raw text
  GET  /api/samples       — list pre-generated sample outputs
  GET  /api/samples/{rid} — fetch one sample's graph.json etc.
"""
import json
import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline.pipeline import extract_pdf_text, run_pipeline

app = FastAPI(title="Resume Knowledge Graph PoC (IJISRT25MAY2182 methodology)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


class TextRequest(BaseModel):
    text: str
    doc_id: str = "upload"
    candidate_name: str = "Candidate"


@app.post("/api/process-text")
def process_text(req: TextRequest):
    out_dir = tempfile.mkdtemp(prefix="rkg_")
    artifacts = run_pipeline(req.text, req.doc_id, out_dir, candidate_name=req.candidate_name)
    # drop full vectors from API response to keep payload small
    artifacts["embeddings.json"] = {k: v for k, v in artifacts["embeddings.json"].items() if k != "vectors"}
    return artifacts


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = ".pdf" if (file.filename or "").lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            text = extract_pdf_text(tmp_path)
        else:
            with open(tmp_path) as f:
                text = f.read()
    finally:
        os.unlink(tmp_path)
    out_dir = tempfile.mkdtemp(prefix="rkg_")
    doc_id = (file.filename or "upload").rsplit(".", 1)[0]
    artifacts = run_pipeline(text, doc_id, out_dir)
    artifacts["embeddings.json"] = {k: v for k, v in artifacts["embeddings.json"].items() if k != "vectors"}
    return artifacts


@app.get("/api/samples")
def list_samples():
    outs = os.path.join(DATA_DIR, "outputs")
    if not os.path.isdir(outs):
        return {"samples": []}
    return {"samples": sorted(d for d in os.listdir(outs)
                              if os.path.isfile(os.path.join(outs, d, "graph.json")))}


@app.get("/api/samples/{rid}")
def get_sample(rid: str):
    base = os.path.join(DATA_DIR, "outputs", rid)
    out = {}
    for fname in ["raw_text.json", "entities.json", "relationships.json",
                   "blocks.json", "block_audit.json", "resume_blocks.json", "embeddings.json", "similarity.json",
                   "clusters.json", "graph.json"]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if fname == "embeddings.json":
                data = {k: v for k, v in data.items() if k != "vectors"}
            out[fname] = data
    return out
