"""Generates 2 sample resume PDFs (reportlab) for the first experiment."""
import os

SAMPLES = {
    "resume_01": {
        "name": "Aarav Sharma",
        "lines": [
            "Aarav Sharma",
            "Backend Developer | aarav.sharma@example.com | +91-98765-43210",
            "",
            "Summary",
            "Backend Developer with 3 years of experience building scalable APIs and microservices.",
            "",
            "Experience",
            "Software Engineer at TechNova Solutions, Bengaluru (2022 - Present)",
            "Developed REST APIs using Python and Django and deployed them on AWS.",
            "Built microservices with FastAPI and PostgreSQL, containerized with Docker.",
            "Worked with Kubernetes for orchestration and Jenkins for CI/CD pipelines.",
            "Integrated Redis for caching and Kafka for event-driven messaging.",
            "",
            "Projects",
            "ShopKart API project: built an e-commerce backend using Django, PostgreSQL and Docker.",
            "Real-time analytics dashboard using Python, Pandas and PostgreSQL.",
            "",
            "Education",
            "B.Tech in Computer Science from Mumbai University (2018 - 2022).",
            "",
            "Certifications",
            "AWS Certified Solutions Architect.",
            "",
            "Skills",
            "Python, Django, FastAPI, Flask, REST API, PostgreSQL, Redis, Docker, Kubernetes, AWS, Jenkins, CI/CD, Kafka, Git, Agile.",
        ],
    },
    "resume_02": {
        "name": "Priya Nair",
        "lines": [
            "Priya Nair",
            "Data Scientist | priya.nair@example.com | +91-99887-76655",
            "",
            "Summary",
            "Data Scientist with 2 years of experience in machine learning and data analysis.",
            "",
            "Experience",
            "Data Scientist at FinEdge Analytics, Hyderabad (2023 - Present)",
            "Developed machine learning models using Python, Scikit-learn and TensorFlow.",
            "Built ETL pipelines with Airflow and SQL, deployed dashboards with Tableau.",
            "Used AWS and Docker to deploy ML models in the fintech domain.",
            "Performed data analysis with Pandas, NumPy and Matplotlib.",
            "",
            "Projects",
            "Credit risk scoring project using Python, Scikit-learn and PostgreSQL.",
            "NLP sentiment analysis app built with PyTorch and FastAPI.",
            "",
            "Education",
            "M.Tech in Data Science from IIT Madras (2021 - 2023).",
            "",
            "Certifications",
            "Google Cloud Professional Data Engineer.",
            "",
            "Skills",
            "Python, SQL, Scikit-learn, TensorFlow, PyTorch, Pandas, NumPy, Airflow, Tableau, PostgreSQL, Docker, AWS, Machine Learning, NLP, Git.",
        ],
    },
}


def create_pdfs(out_dir: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for rid, resume in SAMPLES.items():
        path = os.path.join(out_dir, f"{rid}.pdf")
        c = canvas.Canvas(path, pagesize=A4)
        text = c.beginText(50, 800)
        text.setFont("Helvetica", 11)
        for line in resume["lines"]:
            if line == "":
                text.textLine("")
                continue
            if line in (resume["name"], "Summary", "Experience", "Projects", "Education", "Certifications", "Skills"):
                text.setFont("Helvetica-Bold", 12 if line == resume["name"] else 11)
                text.textLine(line)
                text.setFont("Helvetica", 11)
            else:
                # simple word-wrap at ~95 chars
                while len(line) > 95:
                    cut = line[:95].rfind(" ")
                    cut = cut if cut > 0 else 95
                    text.textLine(line[:cut])
                    line = line[cut:].strip()
                text.textLine(line)
        c.drawText(text)
        c.save()
        paths[rid] = path
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    create_pdfs(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "sample_resumes"))
