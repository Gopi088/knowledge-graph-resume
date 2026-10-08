"""CLI: run the full pipeline over the 2 sample resumes, writing the 6 JSONs each."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from create_sample_pdfs import SAMPLES, create_pdfs  # noqa: E402
from app.pipeline.pipeline import extract_pdf_text, run_pipeline  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..")
PDF_DIR = os.path.join(BASE, "data", "sample_resumes")
OUT_BASE = os.path.join(BASE, "data", "outputs")

if __name__ == "__main__":
    create_pdfs(PDF_DIR)
    for rid, resume in SAMPLES.items():
        pdf = os.path.join(PDF_DIR, f"{rid}.pdf")
        text = extract_pdf_text(pdf)
        run_pipeline(text, rid, os.path.join(OUT_BASE, rid),
                     candidate_name=resume["name"])
    print("Done. Outputs in data/outputs/<resume_id>/*.json")
