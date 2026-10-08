"""Semantic regressions for grouping and the locally available Levi source text."""
import json
import re
import tempfile
from pathlib import Path

import pymupdf
import pytest

from app.pipeline.pipeline import extract_pdf_text, run_pipeline
from app.pipeline.graph_builder import validate_resume_graph


def parse(text):
    with tempfile.TemporaryDirectory() as directory:
        return run_pipeline(text, 'semantic-regression', directory)


def test_generic_company_role_client_date_bullets_and_shared_client():
    result = parse('''Taylor Morgan
Experience
Orion Consulting – Senior Business Analyst
Client – Shared Bank
July 2025 – June 2026
• Leading project delivery.
• Liaise with Risk, Compliance and Operations teams to understand ORM
  objectives and pain points.
Vega Services – Functional Analyst
Client: Shared Bank
03/2020 - 08/2022 (2y 5mon)
• Prepared requirements.''')
    resume = result['canonical_resume.json']
    assert [job['company'] for job in resume['work_history']] == ['Orion Consulting', 'Vega Services']
    first, second = [job['roles'][0] for job in resume['work_history']]
    assert first['clients'] == second['clients'] == ['Shared Bank']
    assert first['role_and_responsibilities'] == ['Leading project delivery.', 'Liaise with Risk, Compliance and Operations teams to understand ORM objectives and pain points.']
    assert second['role_and_responsibilities'] == ['Prepared requirements.']
    assert second['reported_duration'] == '2y 5mon'
    assert first['source_blocks'] and first['responsibility_sources']
    assert all(node['label'] != 'Shared Bank' for node in result['graph.json']['nodes'] if node['type'] == 'COMPANY')


def test_employment_blocks_are_grouped_and_client_is_not_employer():
    result = parse('''Experience
Example Bank – Risk Analyst
Client – Retail Bank
Jan 2022 – Dec 2023
• Led UAT.''')
    jobs = result['canonical_resume.json']['work_history']
    assert len(jobs) == 1
    assert jobs[0]['company'] == 'Example Bank'
    assert jobs[0]['roles'][0]['clients'] == ['Retail Bank']
    assert jobs[0]['roles'][0]['role_and_responsibilities'] == ['Led UAT.']
    assert jobs[0]['company'] != jobs[0]['roles'][0]['clients'][0]


def test_labeled_achievement_attaches_to_current_role():
    result = parse('''Experience
Example Capital – Investment Analyst
2021 – 2024
Achievement: Reduced reconciliation breaks by 30%.
• Managed daily control reviews.''')
    role = result['canonical_resume.json']['work_history'][0]['roles'][0]
    assert role['achievements'] == ['Reduced reconciliation breaks by 30%.']
    assert role['role_and_responsibilities'] == ['Managed daily control reviews.']
    block = next(row for row in result['semantic_blocks.json']['pre_group_blocks']
                 if row['semantic_classification'] == 'achievement')
    assert block['semantic_labels'] == ['achievement']


def test_page_number_does_not_break_employment_context():
    result = parse('''Experience
Wipro Ltd – Senior Analyst
2020 – 2024
• Led delivery.
3
• Managed production support.''')
    canonical = result['canonical_resume.json']
    assert len(canonical['work_history']) == 1
    assert canonical['work_history'][0]['roles'][0]['role_and_responsibilities'] == [
        'Led delivery.', 'Managed production support.']
    assert not any(node.get('label') == '3' for node in result['graph.json']['nodes'])
    marker = next(row for row in result['semantic_blocks.json']['pre_group_blocks']
                  if row['is_page_marker'])
    assert marker['raw_text'] == marker['normalized_text'] == '3'
    assert marker['semantic_classification'] == 'page_marker'
    assert marker['page_number'] == 3
    responsibility = next(row for row in result['semantic_blocks.json']['pre_group_blocks']
                          if row['semantic_classification'] == 'responsibility')
    assert {'index', 'raw_text', 'normalized_text', 'section', 'page_number'} <= set(responsibility)


def test_profile_does_not_contain_work_responsibilities():
    result = parse('''Professional Summary
Senior analyst with banking experience.
Professional Experience
Example Bank – Senior Analyst
• Gather feedback from operational users.
''')
    profile = result['canonical_resume.json']['profile_snapshot']
    assert profile == ['Senior analyst with banking experience.']
    assert not any('Gather feedback from operational users' in item for item in profile)


def test_skills_do_not_contain_employment_blocks():
    result = parse('''Core Competencies
Risk analysis, UAT
Experience
Example Bank – Senior Analyst
• Requirement gathering from stakeholders.
''')
    skills = result['canonical_resume.json']['skills']
    assert all('Requirement gathering from stakeholders' not in str(item) for item in skills)


def test_domain_and_technical_skills_are_distinct_from_work_history():
    result = parse('''Professional Experience
Example Bank – Analyst
• Managed delivery.
Domain Experience
Banking: Example Bank, Other Bank
Technical Skills
Tools: Jira, Postman
Databases: MySQL
Operating Systems: Windows, Linux''')
    canonical = result['canonical_resume.json']
    assert canonical['work_history'][0]['company'] == 'Example Bank'
    assert canonical['domain_experience'][0]['name'] == 'Banking'
    assert canonical['technical_skills']['tools'] == ['Jira', 'Postman']
    assert canonical['technical_skills']['databases'] == ['MySQL']
    assert canonical['technical_skills']['operating_systems'] == ['Windows', 'Linux']


def test_columnar_education_and_detached_final_row_stay_with_their_records():
    result = parse('''Levi Arbole
Education Qualification
Degree
College/School
University/Board
Year of Passing
BE Computer
Percentage &
Pointer
PCE, New Panvel
Third year of
Engineering
Mumbai University 2017
60.00/ 6.75
PCE, New Panvel
Second year of
Engineering
Mumbai University 2016
60.20 / 6.79
PCE, New Panvel
Diploma in I.T
Dr. NYTP, Bhivpuri
Road
Mumbai University 2015
52.82 / 5.75
Mumbai University 2014
74.44
SSC
Kendriya Vidyalaya
Ambernath
CBSE
Personal Details
Date of Birth: 22nd June 1995
Languages: English, Hindi,
Nationality: Indian
2011
67.7''')
    canonical = result['canonical_resume.json']
    assert [row['degree'] for row in canonical['education']] == [
        'BE Computer', 'Diploma in I.T', 'SSC']
    bachelor, diploma, ssc = canonical['education']
    assert [row['year'] for row in bachelor['academic_results']] == ['2017', '2016']
    assert [row['score'] for row in bachelor['academic_results']] == ['60.00/ 6.75', '60.20 / 6.79']
    assert diploma['year'] == '2015' and diploma['score'] == '52.82 / 5.75'
    assert diploma['academic_results'] == [{
        'university_or_board': 'Mumbai University', 'year': '2014', 'score': '74.44'}]
    assert ssc['year'] == '2011' and ssc['score'] == '67.7'
    assert canonical['personal_details']['languages'] == ['English', 'Hindi']
    assert canonical['personal_details']['nationality'] == 'Indian'
    assert result['semantic_blocks.json']['coverage']['complete']
    validate_resume_graph(canonical, result['graph.json'])


def test_promotions_share_company_but_not_responsibilities():
    result = parse('''Alex Rivera
Experience
Société Générale
Business Analyst
2023 - Present
• Managed UAT.
Senior Analyst
2019 - 2023
• Prepared reports.''')
    jobs = result['canonical_resume.json']['work_history']
    assert len(jobs) == 1 and jobs[0]['company'] == 'Société Générale'
    assert [role['designation'] for role in jobs[0]['roles']] == ['Business Analyst', 'Senior Analyst']
    assert [role['role_and_responsibilities'] for role in jobs[0]['roles']] == [['Managed UAT.'], ['Prepared reports.']]


def test_same_employer_groups_multiple_roles_with_now_dates_and_inline_title():
    result = parse('''Professional Experience
Business Analyst with 7 years of Capital Markets experience specializing in reference data and regulatory reporting.
Skills
Requirements gathering, SQL, UAT
Experience:
Global Markets Services
2023 - NOW
Business Analyst
Led migration delivery across operations and technology teams.
2019 - 2023 Senior Analyst
Managed reference data and executive stakeholder reporting.''')
    canonical = result['canonical_resume.json']
    assert canonical['profile_snapshot'] == [
        'Business Analyst with 7 years of Capital Markets experience specializing in reference data and regulatory reporting.']
    assert canonical['skills'] == [{'name': 'Requirements gathering'}, {'name': 'SQL'}, {'name': 'UAT'}]
    jobs = canonical['work_history']
    assert len(jobs) == 1
    assert jobs[0]['company'] == 'Global Markets Services'
    roles = jobs[0]['roles']
    assert [role['designation'] for role in roles] == ['Business Analyst', 'Senior Analyst']
    assert [role['duration'] for role in roles] == ['2023 - NOW', '2019 - 2023']
    assert roles[0]['role_and_responsibilities'] == [
        'Led migration delivery across operations and technology teams.']
    assert roles[1]['role_and_responsibilities'] == [
        'Managed reference data and executive stakeholder reporting.']
    assert all('details' not in job for job in jobs)
    assert all(row.get('destination', '').startswith('$.work_history[')
               for row in result['semantic_blocks.json']['blocks']
               if row['block_type'] in {'COMPANY', 'ROLE', 'DATE_RANGE', 'RESPONSIBILITY'})


def test_page_break_and_nested_bullets_preserve_responsibility_owner():
    result = parse('''Alex Rivera
Work History
Example Consulting – Business Analyst
2020 - 2024
• Document and analyze risk-related processes such as:
    ○ Risk & Control Self Assessments (RCSA)
    ○ Key Risk Indicators (KRI)
    ○ Loss Event Management
    ○ Issue and Action Tracking
Page 3
• Collaborating with clients on production issues and
  audit gap closures.
4
• Ensure tools capture correct risk data.''')
    role = result['canonical_resume.json']['work_history'][0]['roles'][0]
    assert len(role['role_and_responsibilities']) == 3
    assert all(term in role['role_and_responsibilities'][0] for term in ['RCSA', 'KRI', 'Loss Event', 'Issue and Action'])
    assert role['role_and_responsibilities'][1] == 'Collaborating with clients on production issues and audit gap closures.'
    assert len(role['responsibility_sources'][0]) == 5
    assert result['semantic_blocks.json']['coverage']['ignored_artifacts'] == 2


def test_tables_personal_details_projects_domains_and_technical_skills():
    result = parse('''Alex Rivera
Professional Experience
Example Services – Business Analyst
2020 - Present
• Managed delivery.
Project Highlights
1. Integration of ORM with ServiceNow GRC Module.
2. Trading Application (PINS).
3. Implementing report automation.
4. CKYC Compliance Project for a banking client.
5. Implementation of UIPath for an insurance portal.
Domain Experience
Banking: Example Bank, Other Bank
Insurance: Example Insurer
Technical Skills
Tools: JIRA, Service Now, Postman
Databases: MySQL, Oracle SQL
Operating Systems: Windows, Linux
Education Qualification
Degree | College/School | University/Board | Year of Passing | Percentage & Pointer
BE Computer | PCE, New Panvel | Mumbai University | 2017 | 60.00 / 6.75
Diploma in I.T | Dr. NYTP | Mumbai University | 2014 | 74.44
SSC | Kendriya Vidyalaya | CBSE | 2011 | 67.7
Personal Details
Date of Birth: 22nd June 1995
Languages: English, Hindi
Nationality: Indian''')
    resume = result['canonical_resume.json']
    assert len(resume['projects']) == 5
    assert len(resume['education']) == 3
    assert resume['education'][-1]['university_or_board'] == 'CBSE'
    assert resume['education'][-1]['year'] == '2011'
    assert resume['education'][-1]['score'] == '67.7'
    assert resume['personal_details']['nationality'] == 'Indian'
    assert resume['personal_details']['languages'] == ['English', 'Hindi']
    assert resume['domain_experience'][0]['organizations'] == ['Example Bank', 'Other Bank']
    assert resume['technical_skills']['databases'] == ['MySQL', 'Oracle SQL']
    assert resume['work_history'][0]['roles'][0]['role_and_responsibilities'] == ['Managed delivery.']
    assert result['semantic_blocks.json']['coverage']['complete']


def test_education_column_rows_reconstructed_from_source_text():
    result = parse('''Education Qualification
Degree
College/School
University/Board
Year of Passing
Percentage & Pointer
BE Computer
PCE, New Panvel
Mumbai University
2017
60.00 / 6.75
SSC
Kendriya Vidyalaya Ambernath
CBSE
2011
67.7''')
    education = result['canonical_resume.json']['education']
    assert len(education) == 2
    assert education[0]['qualification'] == 'BE Computer'
    assert education[0]['institution'] == 'PCE, New Panvel'
    assert education[0]['year'] == '2017'
    assert education[1]['institution'] == 'Kendriya Vidyalaya Ambernath'
    assert education[1]['university_or_board'] == 'CBSE'


def test_personal_contact_and_personal_details_are_semantic_fields():
    result = parse('''Levi Arbole
Email: arbole60@gmail.com | 9823506339
Location: Ambernath, Thane
Personal Details
Date of Birth: 22nd June 1995
Languages: English, Hindi
Nationality: Indian''')
    personal = result['canonical_resume.json']['personal_information']
    assert personal['name'] == 'Levi Arbole'
    assert personal['email'] == 'arbole60@gmail.com'
    assert personal['phone'] == '9823506339'
    assert personal['location'] == 'Ambernath, Thane'
    details = result['canonical_resume.json']['personal_details']
    assert not {'date_of_birth', 'languages', 'nationality'} & personal.keys()
    assert details['date_of_birth'] == '22nd June 1995'
    assert details['languages'] == ['English', 'Hindi']
    assert details['nationality'] == 'Indian'


def test_levi_full_regression_source_has_expected_hierarchy():
    source_path = Path('/tmp/levi_repro/raw_text.json')
    if not source_path.exists():
        pytest.skip('Local Levi extracted source is unavailable')
    raw = json.loads(source_path.read_text())['raw_text']
    result = parse(raw)
    canonical = result['canonical_resume.json']
    assert len(canonical['work_history']) == 5
    expected_jobs = [
        ('Clover Infotech Pvt Ltd', 'Senior Business Analyst', 'HDFC Bank'),
        ('Wipro Ltd', 'Senior Business Analyst', 'ICICI'),
        ('Tata Consultancy Services (TCS)', 'Functional Analyst', 'SBI'),
        ('Nelito Systems (DTS Group)', 'Functional Analyst', 'SBI'),
        ('QualityKiosk Technologies Pvt. Ltd.', 'Transitions Executive', None),
    ]
    for job, (company, title, client) in zip(canonical['work_history'], expected_jobs):
        assert job['company'] == company
        assert 'details' not in job
        assert len(job['roles']) == 1
        role = job['roles'][0]
        assert role['designation'] == title
        assert role['role_and_responsibilities']
        assert (role['clients'][0] if role['clients'] else None) == client
    semantic = result['semantic_blocks.json']
    for index, (_, _, client) in enumerate(expected_jobs):
        role_path = f'$.work_history[{index}].roles[0]'
        rows = [row for row in semantic['blocks'] if (row.get('destination') or '').startswith(role_path)]
        assert rows
        if client:
            client_row = next(row for row in rows if row['block_type'] == 'CLIENT')
            assert client_row['destination'] == role_path
        assert any(row['block_type'] == 'DATE_RANGE' and row['destination'] == role_path for row in rows)
        assert all(row['destination'] != f'$.work_history[{index}]' for row in rows
                   if row['block_type'] in {'CLIENT', 'DATE_RANGE'})
    assert all('details' not in role for job in canonical['work_history'] for role in job['roles'])
    assert len(canonical['projects']) == 5
    assert len(canonical['education']) == 3
    assert [record['degree'] for record in canonical['education']] == [
        'BE Computer', 'Diploma in I.T', 'SSC']
    assert canonical['education'][0]['year'] == '2017'
    assert [result['year'] for result in canonical['education'][0]['academic_results']] == ['2016', '2015']
    assert canonical['education'][1]['year'] == '2014'
    assert canonical['education'][1]['score'] == '74.44'
    assert canonical['education'][2]['year'] == '2011'
    assert canonical['education'][2]['score'] == '67.7'
    assert len(canonical['domain_experience']) == 5
    assert len(canonical['profile_snapshot']) == 1
    assert canonical['personal_information']['name'] == 'Levi Arbole'
    assert canonical['personal_information']['email'] == 'arbole60@gmail.com'
    assert canonical['personal_information']['phone'] == '9823506339'
    assert canonical['personal_information']['location'] == 'Ambernath, Thane'
    assert canonical['personal_details']['date_of_birth'] == '22nd June 1995'
    assert canonical['personal_details']['languages'] == ['English', 'Hindi']
    assert canonical['personal_details']['nationality'] == 'Indian'
    assert canonical['technical_skills']['tools']
    assert not any('Gather feedback from operational users' in item
                   for item in canonical['profile_snapshot'])
    assert not any('Requirement gathering from stakeholders' in str(item)
                   for item in canonical['skills'])
    canonical_text = []
    def collect_strings(value):
        if isinstance(value, dict):
            for item in value.values():
                collect_strings(item)
        elif isinstance(value, list):
            for item in value:
                collect_strings(item)
        elif isinstance(value, str):
            canonical_text.append(value.strip())
    collect_strings({key: canonical[key] for key in
                     ('profile_snapshot', 'skills', 'work_history', 'projects', 'education')})
    assert not set(canonical_text).intersection({'2', '3', '4', '5', '6'})
    assert not any(str(node.get('label', '')) in {'2', '3', '4', '5', '6'}
                   for node in result['graph.json']['nodes'])
    validate_resume_graph(canonical, result['graph.json'])
    assert all(job['roles'] and any(role['role_and_responsibilities'] for role in job['roles'])
               for job in canonical['work_history'])


def test_unknown_content_is_preserved_with_a_review_destination():
    result = parse('''Alex Rivera
Experience
An ambiguous sentence without an employer or title.''')
    canonical = result['canonical_resume.json']
    assert canonical['unresolved_blocks'][0]['text'] == 'An ambiguous sentence without an employer or title.'
    assert not result['semantic_blocks.json']['coverage']['complete']


def test_summary_and_skills_stop_at_strong_employment_context_without_heading():
    result = parse('''Alex Rivera
Executive Summary
Experienced professional focused on delivery.
Core Competencies
Requirements gathering, SQL, JIRA
Example Services – Business Analyst
Client: Example Bank
Jan 2023 - Present
• Managed UAT.''')
    canonical = result['canonical_resume.json']
    assert canonical['profile_snapshot'] == ['Experienced professional focused on delivery.']
    assert canonical['skills'] == [{'name': 'Requirements gathering'}, {'name': 'SQL'}, {'name': 'JIRA'}]
    assert canonical['work_history'][0]['roles'][0]['role_and_responsibilities'] == ['Managed UAT.']


def test_two_column_pdf_reorders_interleaved_drawing_commands_and_retains_geometry(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page(width=620, height=800)
    # Deliberately draw rows in an interleaved order. Reading top-to-bottom
    # across both columns would mix skill lists with employment responsibilities.
    left = ['Alex Rivera', 'Skills', 'Python, SQL, Jira', 'Education', 'BSc Computer Science', 'State University', '2020']
    right = ['Experience', 'Example Services', 'Business Analyst', '2021 - Present', 'Managed delivery.', 'Projects', '1. Inventory platform.']
    for i in range(max(len(left), len(right))):
        if i < right.__len__():
            page.insert_text((310, 70 + i * 25), right[i], fontsize=11)
        if i < left.__len__():
            page.insert_text((40, 70 + i * 25), left[i], fontsize=11)
    path = tmp_path / 'columns.pdf'
    pdf.save(path)
    pdf.close()
    text = extract_pdf_text(str(path))
    assert text.index('2020') < text.index('Experience')
    assert all(block['bbox'] and block['page'] == 1 for block in text.blocks)
    result = parse(text)
    canonical = result['canonical_resume.json']
    assert canonical['skills'] == [{'name': 'Python'}, {'name': 'SQL'}, {'name': 'Jira'}]
    assert canonical['work_history'][0]['company'] == 'Example Services'
    assert canonical['work_history'][0]['roles'][0]['role_and_responsibilities'] == ['Managed delivery.']


@pytest.mark.parametrize('sample', ['resume_01', 'resume_02'])
def test_existing_pdf_runs_through_layout_and_semantic_pipeline(sample, tmp_path):
    source = Path(__file__).resolve().parents[2] / 'data' / 'sample_resumes' / f'{sample}.pdf'
    extracted = extract_pdf_text(str(source))
    assert extracted.blocks and all('reading_order' in block for block in extracted.blocks)
    result = run_pipeline(extracted, sample, str(tmp_path / sample))
    canonical = result['canonical_resume.json']
    assert canonical['work_history'] and canonical['work_history'][0]['roles']
    assert result['semantic_blocks.json']['coverage']['complete']
    validate_resume_graph(canonical, result['graph.json'])


def test_ruled_pdf_education_table_is_grouped_by_row(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page(width=620, height=300)
    page.insert_text((35, 30), 'Education', fontsize=14)
    xs, ys = [30, 155, 265, 405, 500, 600], [50, 105, 155, 205]
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), width=1)
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), width=1)
    rows = [
        ['Degree', 'College/School', 'University/Board', 'Year of Passing', 'Percentage & Pointer'],
        ['BE Computer', 'PCE, New Panvel', 'Mumbai University', '2017', '60.00 / 6.75'],
        ['SSC', 'Kendriya Vidyalaya', 'CBSE', '2011', '67.7'],
    ]
    for row_index, cells in enumerate(rows):
        for cell_index, value in enumerate(cells):
            page.insert_text((xs[cell_index] + 3, ys[row_index] + 20), value, fontsize=7)
    path = tmp_path / 'education-table.pdf'
    pdf.save(path)
    pdf.close()
    extracted = extract_pdf_text(str(path))
    assert any(block.get('extraction') == 'pdf_table' for block in extracted.blocks)
    result = run_pipeline(extracted, 'table-resume', str(tmp_path / 'table-output'))
    education = result['canonical_resume.json']['education']
    assert len(education) == 2
    assert education[0]['degree'] == 'BE Computer'
    assert education[0]['institution'] == 'PCE, New Panvel'
    assert education[0]['university_or_board'] == 'Mumbai University'
    assert education[0]['score'] == '60.00 / 6.75'
    assert education[1]['year'] == '2011' and education[1]['university_or_board'] == 'CBSE'


def test_graph_rejects_cross_role_responsibility_reassignment():
    result = parse('''Alex Rivera
Experience
One Services – Analyst
2020 - 2021
Prepared reports.
Two Services – Analyst
2022 - 2023
Managed UAT.''')
    graph = json.loads(json.dumps(result['graph.json']))
    edges = [edge for edge in graph['edges'] if edge['relationship'] == 'HAS_RESPONSIBILITY']
    edges[0]['source'] = edges[1]['source']
    with pytest.raises(ValueError, match='hierarchy'):
        validate_resume_graph(result['canonical_resume.json'], graph)


def test_incidental_technology_mentions_do_not_assert_usage():
    result = parse('''Alex Rivera
Experience
Example Services – Analyst
2020 - Present
Evaluated alternatives to Python; the project did not use Python.''')
    assert not any(edge['relationship'] == 'USED_TECHNOLOGY' for edge in result['graph.json']['edges'])
