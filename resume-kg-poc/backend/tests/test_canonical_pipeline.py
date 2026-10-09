import json
from pathlib import Path
import shutil
import tempfile
import zipfile
import unittest

from PIL import Image, ImageDraw, ImageFont
from app.pipeline.canonical_resume import canonicalize_resume
from app.pipeline.graph_builder import build_resume_graph, validate_resume_graph
from app.pipeline.pipeline import (
    _build_section_keyed_output, extract_docx_text, extract_image_text, extract_pdf_text, run_pipeline,
)
from app.pipeline.preprocessing import preprocess
from app.main import TextRequest, process_text


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "data" / "outputs"


def run_sample(name):
    raw = json.loads((OUTPUTS / name / "raw_text.json").read_text())["raw_text"]
    out_dir = tempfile.TemporaryDirectory(prefix=f"resume-parser-{name}-")
    run_pipeline(raw, name, out_dir.name)
    artifacts = {
        key: json.loads((Path(out_dir.name) / f"{key}.json").read_text())
        for key in ("resume_blocks", "canonical_resume", "graph", "block_audit")
    }
    return out_dir, artifacts


class CanonicalPipelineTests(unittest.TestCase):
    def test_personal_information_sections_merge_dict_and_list_in_either_order(self):
        for sections in (
            [
                {"id": "header", "name": "Header"},
                {"id": "contact", "name": "Contact Details"},
            ],
            [
                {"id": "contact", "name": "Contact Details"},
                {"id": "header", "name": "Header"},
            ],
        ):
            with self.subTest(sections=sections):
                audit = {"sections": sections, "source_line_assignments": [], "summary": {}}
                semantic = {"groups": {
                    "header": {"key": "personal_information", "value": {"name": "Jordan Lee", "email": "jordan@example.com"}},
                    "contact": {"key": "personal_information", "value": ["Phone: +1 415 555 0123"]},
                }, "blocks": []}
                result = _build_section_keyed_output(audit, semantic=semantic)
                self.assertEqual(result["personal_information"]["name"], "Jordan Lee")
                self.assertEqual(result["personal_information"]["email"], "jordan@example.com")
                self.assertEqual(result["personal_information"]["phone"], "+1 415 555 0123")

    def test_existing_samples_keep_source_coverage_and_core_records(self):
        for name in ("resume_01", "resume_02"):
            with self.subTest(name=name):
                temp_dir, artifacts = run_sample(name)
                self.addCleanup(temp_dir.cleanup)
                audit = artifacts["block_audit"]["summary"]
                self.assertTrue(audit["complete"])
                self.assertEqual(audit["unassigned_source_line_indices"], [])
                canonical = artifacts["canonical_resume"]
                self.assertEqual(len(canonical["work_history"]), 1)
                role = canonical["work_history"][0]["roles"][0]
                self.assertTrue(role["designation"])
                self.assertTrue(role["role_and_responsibilities"])
                self.assertEqual(len(canonical["projects"]), 2)
                self.assertEqual(len(canonical["education"]), 1)
                self.assertTrue(canonical["skills"])
                validate_resume_graph(canonical, artifacts["graph"])
                self.assertEqual(audit["relationship_count"], len(artifacts["graph"]["edges"]))
                self.assertIn("source_context_relationship_count", audit)

    def test_aashish_resume_keeps_employer_hierarchies_and_sections(self):
        temp_dir, artifacts = run_sample("aashish_golhani")
        self.addCleanup(temp_dir.cleanup)
        canonical = artifacts["canonical_resume"]
        self.assertEqual(len(canonical["work_history"]), 5)
        self.assertTrue(all(job["company"] and job["roles"] for job in canonical["work_history"]))
        self.assertTrue(all(role["role_and_responsibilities"]
                            for job in canonical["work_history"] for role in job["roles"]))
        self.assertEqual(len(canonical["education"]), 3)
        self.assertTrue(canonical["certifications"])
        self.assertTrue(canonical["source_sections"].get("business_analysis_and_product_management"))
        self.assertTrue(artifacts["block_audit"]["summary"]["complete"])
        validate_resume_graph(canonical, artifacts["graph"])

    def test_multiple_roles_remain_under_one_company(self):
        canonical = canonicalize_resume({
            "personal_information": {"name": "Jordan Lee"},
            "work_history": [{
                "company": "Northwind Systems",
                "roles": [
                    {"title": "Analyst", "dates": "2020-2022",
                     "responsibilities": ["Prepared forecasts."]},
                    {"title": "Senior Analyst", "dates": "2022-present",
                     "responsibilities": ["Led planning."]},
                ],
            }],
        })
        graph = build_resume_graph(canonical)
        company = next(node for node in graph["nodes"] if node["type"] == "COMPANY")
        role_edges = [edge for edge in graph["edges"] if edge["relationship"] == "HAS_ROLE"
                      and edge["source"] == company["id"]]
        self.assertEqual(len(role_edges), 2)
        self.assertTrue(all(graph_node["type"] == "JOB_ROLE" for edge in role_edges
                            for graph_node in graph["nodes"] if graph_node["id"] == edge["target"]))
        validate_resume_graph(canonical, graph)

        text = (
            "Taylor Morgan\nExperience\n"
            "Northwind Systems – Analyst (2020 - 2022)\nPrepared forecasts.\n"
            "Northwind Systems – Senior Analyst (2022 - Present)\nLed planning."
        )
        temp_dir = tempfile.TemporaryDirectory(prefix="resume-multi-role-")
        self.addCleanup(temp_dir.cleanup)
        run_pipeline(text, "multi-role", temp_dir.name)
        parsed = json.loads((Path(temp_dir.name) / "canonical_resume.json").read_text())
        self.assertEqual(len(parsed["work_history"]), 1)
        self.assertEqual([role["designation"] for role in parsed["work_history"][0]["roles"]],
                         ["Analyst", "Senior Analyst"])
        self.assertEqual(parsed["work_history"][0]["roles"][1]["duration"], "2022 - Present")
        self.assertEqual(parsed["work_history"][0]["roles"][1]["role_and_responsibilities"], ["Led planning."])

    def test_clients_dates_and_responsibilities_stay_with_their_role(self):
        source = (
            "Casey Rivera\nExperience\n"
            "Northwind Systems – Analyst (2020 - 2022)\n"
            "Client: HDFC Bank\nPrepared monthly forecasts.\n"
            "Contoso Technologies – Senior Analyst (2022 - Present)\n"
            "Client: Acme Finance\nLed risk reporting."
        )
        temp_dir = tempfile.TemporaryDirectory(prefix="resume-role-scope-")
        self.addCleanup(temp_dir.cleanup)
        run_pipeline(source, "role-scope", temp_dir.name)
        canonical = json.loads((Path(temp_dir.name) / "canonical_resume.json").read_text())
        jobs = canonical["work_history"]
        self.assertEqual([job["company"] for job in jobs], ["Northwind Systems", "Contoso Technologies"])
        first, second = (job["roles"][0] for job in jobs)
        self.assertEqual(first["clients"], ["HDFC Bank"])
        self.assertEqual(first["employment_period"]["start_date"], "2020")
        self.assertEqual(first["employment_period"]["end_date"], "2022")
        self.assertEqual(first["role_and_responsibilities"], ["Prepared monthly forecasts."])
        self.assertEqual(second["clients"], ["Acme Finance"])
        self.assertEqual(second["role_and_responsibilities"], ["Led risk reporting."])
        graph = json.loads((Path(temp_dir.name) / "graph.json").read_text())
        self.assertEqual(sum(edge["relationship"] == "WORKED_FOR_CLIENT" for edge in graph["edges"]), 2)
        self.assertEqual(sum(edge["relationship"] == "HAS_EMPLOYMENT_PERIOD" for edge in graph["edges"]), 2)
        validate_resume_graph(canonical, graph)

    def test_standalone_employer_headers_and_numeric_month_dates_are_grouped(self):
        source = (
            "Morgan Smith\nmorgan@example.com | +1 415 555 0123\n"
            "LinkedIn: https://linkedin.com/in/morgansmith | GitHub: https://github.com/morgansmith\n"
            "Professional Summary\nBusiness analyst focused on reliable delivery.\n"
            "Experience\nWipro\nSenior Business Analyst\nClient: ICICI Bank\n"
            "03/2020 - 08/2022\nPrepared business requirements.\n"
            "Société Générale\nBusiness Analyst\n2023 - Present\nManaged UAT.\n"
            "Skills\nSQL, Jira\n2"
        )
        temp_dir = tempfile.TemporaryDirectory(prefix="resume-standalone-employers-")
        self.addCleanup(temp_dir.cleanup)
        run_pipeline(source, "standalone-employers", temp_dir.name)
        canonical = json.loads((Path(temp_dir.name) / "canonical_resume.json").read_text())
        self.assertEqual(canonical["personal_information"]["email"], "morgan@example.com")
        self.assertEqual(canonical["personal_information"]["phone"], "+1 415 555 0123")
        self.assertIn("linkedin", canonical["personal_information"])
        self.assertIn("github", canonical["personal_information"])
        self.assertEqual(canonical["profile_snapshot"], ["Business analyst focused on reliable delivery."])
        self.assertEqual([job["company"] for job in canonical["work_history"]],
                         ["Wipro", "Société Générale"])
        first, second = (job["roles"][0] for job in canonical["work_history"])
        self.assertEqual(first["clients"], ["ICICI Bank"])
        self.assertEqual((first["start_date"], first["end_date"]), ("03/2020", "08/2022"))
        self.assertEqual(first["role_and_responsibilities"], ["Prepared business requirements."])
        self.assertEqual((second["start_date"], second["end_date"]), ("2023", "Present"))
        self.assertEqual(second["role_and_responsibilities"], ["Managed UAT."])
        self.assertNotIn("2", [item["name"] for item in canonical["skills"]])
        graph = json.loads((Path(temp_dir.name) / "graph.json").read_text())
        clients = [node for node in graph["nodes"] if node["type"] == "CLIENT"]
        self.assertEqual([node["label"] for node in clients], ["ICICI Bank"])
        self.assertFalse(any(node["label"] == "ICICI Bank" and node["type"] == "COMPANY"
                             for node in graph["nodes"]))
        validate_resume_graph(canonical, graph)

    def test_docx_extraction_preserves_paragraphs_and_table_rows(self):
        document_xml = '''<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body><w:p><w:r><w:t>Jordan Lee</w:t></w:r></w:p>
          <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Degree</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:r><w:t>University</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
          </w:body></w:document>'''
        with tempfile.TemporaryDirectory(prefix="resume-docx-") as directory:
            docx_path = Path(directory) / "resume.docx"
            with zipfile.ZipFile(docx_path, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            self.assertEqual(extract_docx_text(str(docx_path)), "Jordan Lee\nDegree | University")

    def test_custom_academic_and_achievement_sections_are_preserved(self):
        canonical = canonicalize_resume({
            "personal_information": {"name": "Alex Morgan"},
            "awards": ["Dean's List"],
            "achievements": ["Published a research result."],
            "publications": [{"title": "A Study of Systems", "journal": "Example Journal"}],
            "additional_section": ["Candidate supplied content"],
        })
        graph = build_resume_graph(canonical)
        relations = {edge["relationship"] for edge in graph["edges"]}
        self.assertIn("HAS_AWARD", relations)
        self.assertIn("HAS_ACHIEVEMENT", relations)
        self.assertIn("AUTHORED_PUBLICATION", relations)
        self.assertEqual(canonical["source_sections"]["additional_section"],
                         ["Candidate supplied content"])
        validate_resume_graph(canonical, graph)

    def test_canonical_validation_rejects_legacy_details_only_employment(self):
        with self.assertRaisesRegex(ValueError, "details-only work records"):
            canonicalize_resume({"work_history": [{"details": ["Client: Example Bank"],
                                                     "role_and_responsibilities": []}]})
        with self.assertRaisesRegex(ValueError, "Raw work-history text cannot be serialized"):
            canonicalize_resume({"work_history": ["Client: Example Bank"]})

    def test_numeric_personal_keys_are_repaired_as_name_instead_of_returned(self):
        canonical = canonicalize_resume({"personal_information": {
            "0": "Jordan Lee", "email": "jordan@example.com | +1 555 123 4567"}})
        self.assertEqual(canonical["personal_information"]["name"], "Jordan Lee")
        self.assertEqual(canonical["personal_information"]["email"], "jordan@example.com")
        self.assertEqual(canonical["personal_information"]["phone"], "+1 555 123 4567")
        self.assertNotIn("0", canonical["personal_information"])

    def test_personal_details_section_splits_contact_from_personal_attributes(self):
        canonical = canonicalize_resume({"personal_details": {
            "name": "Jordan Lee", "email": "jordan@example.com", "phone": "555-0100",
            "location": "Bangalore", "date_of_birth": "1 Jan 1990",
            "languages": ["English"], "nationality": "Indian",
        }})
        self.assertEqual(canonical["personal_information"], {
            "name": "Jordan Lee", "email": "jordan@example.com",
            "phone": "555-0100", "location": "Bangalore",
        })
        self.assertEqual(canonical["personal_details"]["date_of_birth"], "1 Jan 1990")
        self.assertEqual(canonical["personal_details"]["languages"], ["English"])
        self.assertEqual(canonical["personal_details"]["nationality"], "Indian")

    def test_unheaded_resume_infers_context_without_crossing_records(self):
        raw = (
            "Jordan Lee\nProduct Analyst | jordan@example.com | +1 555 123 4567\n"
            "Product analyst focused on customer research and delivery.\n"
            "Product Analyst at Northwind Systems (2021 - Present)\n"
            "Coordinated customer research and roadmap delivery.\n"
            "B.S. Information Systems from State University (2017 - 2021)\n"
            "Certified Scrum Master\nPython, SQL, Jira"
        )
        preprocessed = preprocess(raw)
        sections = {line["line"]: line["section"] for line in preprocessed["line_meta"] if line["line"]}
        self.assertEqual(sections["Product analyst focused on customer research and delivery."], "Summary")
        self.assertEqual(sections["Product Analyst at Northwind Systems (2021 - Present)"], "Experience")
        self.assertEqual(sections["B.S. Information Systems from State University (2017 - 2021)"], "Education")
        self.assertEqual(sections["Certified Scrum Master"], "Certifications")
        self.assertEqual(sections["Python, SQL, Jira"], "Skills")
        temp_dir = tempfile.TemporaryDirectory(prefix="resume-parser-unheaded-")
        self.addCleanup(temp_dir.cleanup)
        run_pipeline(raw, "unheaded", temp_dir.name)
        canonical = json.loads((Path(temp_dir.name) / "canonical_resume.json").read_text())
        job = canonical["work_history"][0]
        self.assertEqual(job["company"], "Northwind Systems")
        self.assertEqual(job["roles"][0]["role_and_responsibilities"],
                         ["Coordinated customer research and roadmap delivery."])
        self.assertEqual(canonical["education"][0]["qualification"],
                         "B.S. Information Systems")
        self.assertEqual(canonical["skills"], [{"name": "Python"}, {"name": "SQL"}, {"name": "Jira"}])

    def test_graph_has_only_supported_relation_types_and_same_record_scope(self):
        canonical = canonicalize_resume({
            "personal_information": {"name": "Sam Example"},
            "work_history": [
                {"company": "Company A", "designation": "Engineer",
                 "duration": "2020-2021", "responsibilities": ["Built a Python service."]},
                {"company": "Company B", "designation": "Analyst",
                 "duration": "2022-2023", "responsibilities": ["Prepared reports."]},
            ],
        })
        graph = build_resume_graph(canonical)
        responsibility_edges = [edge for edge in graph["edges"]
                                if edge["relationship"] == "HAS_RESPONSIBILITY"]
        self.assertEqual(len(responsibility_edges), 2)
        self.assertNotIn("Python", [
            edge["target_label"] for edge in graph["edges"]
            if edge["source_label"] == "Analyst"
        ])
        tampered = json.loads(json.dumps(graph))
        responsibility_edge = next(edge for edge in tampered["edges"]
                                   if edge["relationship"] == "HAS_RESPONSIBILITY")
        responsibility_edge["source_sentence"] = "Invented responsibility unrelated to this resume."
        with self.assertRaises(ValueError):
            validate_resume_graph(canonical, tampered)
        invalid = json.loads(json.dumps(graph))
        invalid["edges"].append({
            "source": "person", "target": "missing", "source_label": "Sam Example",
            "target_label": "Unrelated", "relationship": "RANDOM_LINK",
            "scope_path": "$.work_history[0]", "evidence_path": "$.work_history[0]",
            "source_sentence": "no evidence",
        })
        with self.assertRaises(ValueError):
            validate_resume_graph(canonical, invalid)
        duplicate_edge = json.loads(json.dumps(graph))
        duplicate_edge["edges"].append(json.loads(json.dumps(duplicate_edge["edges"][0])))
        with self.assertRaisesRegex(ValueError, "duplicate relationship"):
            validate_resume_graph(canonical, duplicate_edge)
        duplicate_node = json.loads(json.dumps(graph))
        copied = json.loads(json.dumps(duplicate_node["nodes"][-1]))
        copied["id"] = "duplicate-node"
        duplicate_node["nodes"].append(copied)
        with self.assertRaisesRegex(ValueError, "duplicate entities"):
            validate_resume_graph(canonical, duplicate_node)

    def test_pdf_text_extraction_reads_repository_samples(self):
        text = extract_pdf_text(str(ROOT / "data" / "sample_resumes" / "resume_01.pdf"))
        self.assertIn("Aarav Sharma", text)
        self.assertIn("TechNova Solutions", text)

    def test_text_api_returns_the_canonical_graph_artifacts(self):
        result = process_text(TextRequest(
            text="Morgan Reed\nData Analyst | morgan@example.com\nExperience\n"
                 "Data Analyst at Example Analytics (2023 - Present)\nPrepared reports.",
            doc_id="api-resume",
        ))
        canonical = result["canonical_resume.json"]
        graph = result["graph.json"]
        self.assertEqual(canonical["schema_version"], "canonical-resume/v1")
        self.assertEqual(graph["source"], "canonical_resume/v1")
        self.assertEqual(result["relationships.json"]["count"], len(graph["edges"]))
        validate_resume_graph(canonical, graph)

    @unittest.skipUnless(shutil.which("tesseract"), "Tesseract OCR is not installed")
    def test_image_and_scanned_pdf_text_extraction(self):
        image = Image.new("RGB", (1400, 240), "white")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
        ImageDraw.Draw(image).text((40, 40), "Resume Candidate Jordan Lee", fill="black", font=font)
        with tempfile.TemporaryDirectory(prefix="resume-ocr-") as directory:
            image_path = Path(directory) / "resume.png"
            image.save(image_path)
            image_text = extract_image_text(str(image_path))
            self.assertIn("Jordan Lee", image_text)

            import pymupdf
            pdf_path = Path(directory) / "scanned.pdf"
            document = pymupdf.open()
            page = document.new_page(width=700, height=120)
            page.insert_image(page.rect, filename=str(image_path))
            document.save(pdf_path)
            document.close()
            self.assertIn("Jordan Lee", extract_pdf_text(str(pdf_path)))


if __name__ == "__main__":
    unittest.main()
