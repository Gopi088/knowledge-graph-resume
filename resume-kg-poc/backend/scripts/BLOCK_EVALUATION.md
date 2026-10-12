# Hybrid resume block evaluation

`batch_pipeline_test.py` evaluates the existing parser without changing the
parser or its production workflow. It runs each supported resume through the
current extraction pipeline and saves every returned JSON artifact under the
evaluation output directory.

## Run against a resume folder

From `resume-kg-poc/backend`:

```bash
python scripts/batch_pipeline_test.py \
  --input-dir "/path/to/RESUMES_FLAT" \
  --output-dir "./hybrid_evaluation"
```

Without `--ground-truth-dir`, extraction still runs and all JSON artifacts are
saved, but accuracy is explicitly reported as **not measured**. Existing parser
outputs are never reused as reference data.

## Add manually verified references

Provide a folder containing one manually checked JSON file for each resume.
Mirror the input directory layout and use the resume basename with a `.json`
extension. For example:

```text
resumes/Engineering/Candidate A.pdf
ground_truth/Engineering/Candidate A.json
```

The reference should use the parser's canonical section names where possible:
`work_history` (or `work_experience`), `projects`, `skills` or
`technical_skills`, and `education`. The evaluator also accepts common aliases.
For entity scoring, include an `entities` array in the reference JSON; otherwise
entity metrics are marked unavailable. The system-side entity list is taken
from the pipeline's `entities.json` artifact.

```bash
python scripts/batch_pipeline_test.py \
  --input-dir "/path/to/RESUMES_FLAT" \
  --ground-truth-dir "/path/to/manually_verified_json" \
  --output-dir "./hybrid_evaluation" \
  --similarity-threshold 0.72
```

Use `--max-files 5` for a small operational run before processing the full
folder. The default processes every supported file, including PDF, DOC/DOCX,
text, RTF, and common image formats (image extraction requires Tesseract).

## What the scores mean

Experience, Projects, Skills, and Education are scored independently. The
evaluator flattens their meaningful string/number values while preserving
section, field path, and available record anchors (such as employer, project,
or degree/institution). Exact normalized value-and-record matches count as true
positives; unmatched extracted items count as false positives; unmatched
reference items count as false negatives. Precision, recall, and F1 are derived
from those counts. Section mismatches can be flagged as
`wrongly_associated_section`; content assigned to another employer, project,
or education record can be flagged as `wrongly_associated_block`. The report
also compares JSON paths and non-section values, and reports entity metrics
when references include entities.

Cosine similarity uses a deterministic local TF-IDF implementation. It flags
low-similarity unmatched values for review and pairs high-similarity unmatched
values as likely `corrupted_or_incorrect_value` cases. Similarity does **not**
change TP/FP/FN or the reported scores. Scores are only as reliable as the
manually verified reference set and are not a substitute for inspecting flagged
examples.

## Output files

- `extracted/<relative path>/<resume stem and file type>/`: all JSON artifacts
  returned by the existing pipeline, `processing.json` with status/error and
  embedding method, plus `evaluation.json` when reference JSON is valid.
- `resume_results.csv` and `resume_results.json`: one row per processed resume,
  including processing and ground-truth status and per-section F1.
- `hybrid_evaluation_report.md`: consolidated metrics, error categories,
  ground-truth availability, and per-resume outcomes.

Missing reference files and invalid JSON are recorded per resume; they do not
abort the batch. If no usable reference JSON is found, the report says accuracy
is not measured instead of inventing expected results.

If the requested output directory already contains files, the run uses a new
timestamped sibling directory so prior reports and extracted outputs are not
overwritten.
