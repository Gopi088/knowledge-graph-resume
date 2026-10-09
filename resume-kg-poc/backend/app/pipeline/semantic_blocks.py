"""Classify source blocks, resolve owners, then produce hierarchical section records.

No embeddings or graph layout participate. Every grouping decision retains its
source IDs and reason. Ambiguous blocks remain in the unresolved collection.
"""
import re
from dataclasses import dataclass, field

MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
DATE_ATOM = rf'(?:{MONTH}\.?\s+\d{{4}}|\d{{1,2}}/\d{{4}}|(?:19|20)\d{{2}})'
PERIOD = re.compile(rf'(?<!\d)({DATE_ATOM})\s*(?:[-–—]|\bto\b)\s*({DATE_ATOM}|present|current|now|till\s+date|to\s+date)(?!\d)', re.I)
SINGLE_DATE = re.compile(rf'^(?:{DATE_ATOM}|present|current|till\s+date)$', re.I)
ROLE = re.compile(r'\b(analyst|engineer|developer|consultant|manager|director|designer|architect|scientist|officer|specialist|associate|executive|lead|intern|owner|president|chief|researcher|accountant|administrator|coordinator|technician|teacher|nurse|professor|fellow)\b', re.I)
ROLE = re.compile(r'\b(analyst|engineer|developer|consultant|manager|director|designer|architect|scientist|officer|specialist|associate|executive|lead|intern|owner|president|chief|researcher|accountant|administrator|coordinator|technician|teacher|nurse|professor|fellow|transitions? executive|underwriter|functional analyst)\b', re.I)
ACTION = re.compile(r'^(?:responsible|led|leading|liaise|gather|define|document|manag|collaborat|work(?:ed|ing)?\b|support|ensure|prepar|implement|develop|creat|built|design|deliver|conduct|perform|review|provid|coordinat|interact|handl|driv|translat|involved|deployed|integrat|improv|mentor|achiev)', re.I)
ORG_SUFFIX = re.compile(r'\b(?:ltd|limited|inc|llc|corp|corporation|systems|services|technologies|infotech|bank|university|laboratories)\b', re.I)
MARKER = re.compile(r'^\s*(?:(?P<bullet>[•▪●○◦*–—-])\s*|(?P<number>\d+[.)])(?=\s+))')
EDUCATION = re.compile(r'^(?:b\.?\s?(?:e|tech|sc|com|s|a|ca)\b|m\.?\s?(?:e|tech|sc|s|a|ba|ca)\b|bachelor|master|ph\.?d|doctor|diploma|ssc\b|hsc\b|high school|secondary|class\s+\d|(?:first|second|third|final)\s+year)', re.I)
EDUCATION_YEAR = re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)')


def clean(text):
    return MARKER.sub('', text).strip()


def extract_header_identity(rows):
    """Extract identity/contact fields from the resume header with line provenance.

    Contact details are commonly laid out as icons or separate visual columns,
    so extracted text can be unlabeled and may put phone, city, and email on
    the same line. This parser is deliberately limited to the header section.
    """
    lines = [(row, clean(str(row.get('text', '')))) for row in rows]
    fields, field_rows = {}, {}
    email_re = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
    phone_re = re.compile(r'(?<![\w])\+?\d[\d().\s-]{5,}\d(?![\w])')
    # Resume links are often printed as bare domains (linkedin.com/in/...) or
    # preceded by a label on the same contact row. Keep the domain in the
    # extracted value so it remains clickable in downstream consumers.
    link_re = re.compile(
        r'(?:(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/[^\s|,;]+|'
        r'(?:https?://|www\.)[^\s|,;]+)', re.I)
    label_re = re.compile(
        r'\b(name|title|email|e-mail|phone|mobile|contact(?: number)?|location|address|based in|city|'
        r'linkedin|github|website|portfolio)\s*[:：–—-]\s*',
        re.I,
    )

    def put(key, value, row):
        value = str(value or '').strip(' \t,;|·•')
        if value and key not in fields:
            fields[key] = value
            field_rows[key] = row

    # First collect explicit fields and independent contact tokens.
    for row, text in lines:
        for match in label_re.finditer(text):
            label = re.sub(r'[^a-z]+', '_', match.group(1).casefold()).strip('_')
            end = label_re.search(text, match.end())
            raw = text[match.end():end.start() if end else len(text)].strip(' \t,;|·•')
            key = {'e_mail': 'email', 'mobile': 'phone', 'contact': 'phone',
                   'contact_number': 'phone', 'address': 'location',
                   'based_in': 'location', 'city': 'location',
                   'website': 'website', 'portfolio': 'portfolio'}.get(label, label)
            if key == 'email':
                found = email_re.search(raw)
                raw = found.group(0) if found else raw
            elif key == 'phone':
                found = phone_re.search(raw)
                raw = found.group(0) if found and 7 <= len(re.sub(r'\D', '', found.group(0))) <= 15 else raw
            elif key in {'linkedin', 'github', 'website', 'portfolio'}:
                found = link_re.search(raw)
                raw = found.group(0).rstrip('.,)') if found else raw
            elif key == 'location':
                raw = email_re.sub('', phone_re.sub('', link_re.sub('', raw))).strip(' \t,;|·•')
            put(key, raw, row)
        email = email_re.search(text)
        if email:
            put('email', email.group(0), row)
        for key, pattern in (
            ('linkedin', re.compile(r'(?:(?:https?://)?(?:www\.)?)linkedin\.com/[^\s|,;]+', re.I)),
            ('github', re.compile(r'(?:(?:https?://)?(?:www\.)?)github\.com/[^\s|,;]+', re.I)),
        ):
            match = pattern.search(text)
            if match:
                put(key, match.group(0).rstrip('.,)'), row)
        if not any(key in fields and field_rows[key] is row for key in ('linkedin', 'github')):
            website = re.search(r'(?:(?:https?://)|www\.)[^\s|,;]+', text, re.I)
            if website:
                put('website', website.group(0).rstrip('.,)'), row)
        for match in phone_re.finditer(text):
            digits = re.sub(r'\D', '', match.group(0))
            if 7 <= len(digits) <= 15:
                put('phone', match.group(0), row)
                break

    # Detect short professional titles separately so they are not mistaken
    # for a name or city when contact details share their line.
    for row, text in lines:
        if (ROLE.search(text) and len(text.split()) <= 8 and not ACTION.search(text)
                and not email_re.search(text) and not phone_re.search(text)):
            put('title', text, row)
            break

    # The first short, human-name-like header line is the candidate name.
    for row, text in lines:
        candidate = email_re.sub(' ', text)
        candidate = phone_re.sub(' ', candidate)
        candidate = link_re.sub(' ', candidate)
        candidate = label_re.sub(' ', candidate)
        candidate = re.sub(r'[|·•]+', ' ', candidate).strip(' ,;:-')
        if (candidate and len(candidate.split()) <= 6 and not re.search(r'\d|@|[.!?]', candidate)
                and not ROLE.search(candidate) and not ACTION.search(candidate)
                and not re.search(r'\b(email|phone|mobile|location|address|linkedin|github)\b', candidate, re.I)):
            # Do not absorb the title when name and title are printed together.
            if fields.get('title') and candidate.casefold().endswith(fields['title'].casefold()):
                candidate = candidate[:-len(fields['title'])].strip(' ,;:-')
            put('name', candidate, row)
            if fields.get('name'):
                break

    # Prefer an explicit location label. Otherwise collect the unclaimed
    # short text on a contact line or in its own header line (icon columns are
    # often emitted this way by PDF text extraction).
    if not fields.get('location') and ('email' in fields or 'phone' in fields):
        candidates = []
        for row, text in lines:
            residual = email_re.sub(' ', text)
            residual = phone_re.sub(' ', residual)
            residual = link_re.sub(' ', residual)
            residual = label_re.sub(' ', residual)
            for known in (fields.get('name'), fields.get('title')):
                if known:
                    residual = re.sub(re.escape(known), ' ', residual, flags=re.I)
            residual = re.sub(
                r'\b(?:email|e-mail|phone|mobile|contact|location|address|based in|city|'
                r'linkedin|github|website|portfolio)\b', ' ', residual, flags=re.I)
            residual = re.sub(r'[|·•]+', ' ', residual)
            candidate = re.sub(r'\s+', ' ', residual).strip(' ,;:-')
            if (candidate and len(candidate.split()) <= 6 and not re.search(r'\d|@|[.!?]', candidate)
                    and not ROLE.search(candidate) and not ACTION.search(candidate)
                    and candidate.casefold() not in {str(fields.get('name', '')).casefold(), str(fields.get('title', '')).casefold()}):
                candidates.append((candidate, row))
        if candidates:
            put('location', candidates[-1][0], candidates[-1][1])

    return fields, field_rows


def name_like(text):
    words = text.split()
    return (0 < len(words) <= 14 and len(text) < 140 and not ACTION.search(text)
            and not re.search(r'[!?;:]|https?://|@', text)
            and (sum(w[:1].isupper() for w in words) >= max(1, len(words) * .6) or bool(ORG_SUFFIX.search(text))))


def role_title(text):
    body = PERIOD.sub('', text).strip(' ()|,–—-')
    # A role keyword inside a responsibility sentence (for example,
    # "executive stakeholder communication") is not a job title. Titles are
    # short noun phrases; source prose has sentence punctuation or many words.
    return (bool(ROLE.search(body)) and len(body) < 110 and len(body.split()) <= 8
            and not ACTION.search(body) and not re.search(r'[.!?;,@]', body))


def title_like(text):
    body = PERIOD.sub('', text).strip(' ()|,–—-')
    if re.match(r'^(?:clients?|role|designation|job title|title|dates?|duration|period)\b', body, re.I):
        return False
    return role_title(body) or (name_like(body) and len(body.split()) <= 8
                                and not re.search(r'[.!?@]', body))


def employment_header(text):
    label = re.match(r'^(?:company|employer|organization)\s*[:–—-]\s*(.+)$', text, re.I)
    if label:
        return {'company': label[1]} if not re.match(r'^(?:clients?|role|designation|job title|title|dates?|duration|period)\b', label[1], re.I) else None
    at = re.match(r'^(.+?)\s+at\s+(.+)$', text, re.I)
    if at and title_like(at[1]):
        date = PERIOD.search(at[2])
        return {'company': (at[2][:date.start()] if date else at[2]).strip(' (),|–—-'),
                'designation': at[1].strip(), 'period': date.group(0) if date else None}
    split = re.match(r'^(.+?)\s+[–—|\-]\s+(.+)$', text)
    if (split and not re.match(r'^(?:clients?|role|designation|job title|title|dates?|duration|period)\b', split[1], re.I)
            and name_like(split[1]) and title_like(split[2])):
        date = PERIOD.search(split[2])
        return {'company': split[1].strip(), 'designation': (split[2][:date.start()] if date else split[2]).strip(' ()|,–—-'),
                'period': date.group(0) if date else None}
    return None


def _classify(row, following):
    text = clean(row['text'])
    section = row['section'].casefold()
    if not text or row.get('artifact') or row.get('is_page_number') or re.fullmatch(r'(?:page\s+\d+(?:\s*(?:of|/)\s*\d+)?)|[\s•▪●○◦*|_–—-]+', text, re.I):
        return 'PAGE_ARTIFACT', {}
    if row['is_header']:
        return 'SECTION_HEADING', {}
    if section == 'experience' and re.match(r'^clients?\s*[:–—-]\s*(.+)$', text, re.I):
        value = re.match(r'^clients?\s*[:–—-]\s*(.+)$', text, re.I)[1]
        return 'CLIENT', {'clients': [v.strip() for v in re.split(r'[;|]', value) if v.strip()]}
    if section in {'header', 'personal information'} and ':' in text:
        label = text.split(':', 1)[0].strip().casefold()
        if label in {'email', 'e-mail', 'phone', 'mobile', 'contact number', 'location', 'address', 'date of birth', 'dob', 'nationality', 'languages'}:
            return 'PERSONAL_DETAIL', {'field': label, 'value': text.split(':', 1)[1].strip()}
    if section == 'experience':
        achievement = re.match(r'^(?:key\s+)?(?:achievement|accomplishment)s?\s*[:–—-]\s*(.+)$', text, re.I)
        if achievement:
            return 'ACHIEVEMENT', {'text': achievement[1].strip()}
        header = employment_header(text)
        if header:
            return ('COMPANY_ROLE' if header.get('designation') else 'COMPANY'), header
        title = re.match(r'^(?:role|designation|job title|title)\s*[:–—-]\s*(.+)$', text, re.I)
        if title or role_title(text):
            value = title[1] if title else text
            date = PERIOD.search(value)
            designation = value[:date.start()] if date else value
            if date and not designation.strip():
                designation = value[date.end():]
            fields = {'designation': designation.strip(' ()|,–—-'),
                      'period': date.group(0) if date else None}
            if date:
                fields.update(start_date=date[1], end_date=date[2])
            return 'ROLE', fields
        date = PERIOD.search(text)
        if date and not ACTION.search(text):
            rest = text[:date.start()].strip(' (),|–—-')
            trailing = text[date.end():].strip(' (),|–—-')
            # Some layouts place a new role title after its date range on the
            # same line: "2019 - 2023 Senior Analyst". Keep the date and role
            # attached as one semantic block instead of treating the title as
            # a company name or responsibility.
            if trailing and title_like(trailing):
                return 'DATE_RANGE', {'period': date.group(0), 'start_date': date[1],
                                      'end_date': date[2], 'designation': trailing}
            if rest and name_like(rest) and not re.match(r'^(?:dates?|duration|period)\b', rest, re.I):
                return 'COMPANY', {'company': rest, 'period': date.group(0)}
            return 'DATE_RANGE', {'period': date.group(0), 'start_date': date[1], 'end_date': date[2],
                                  'reported_duration': text[date.end():].strip(' ()|,')}
        if SINGLE_DATE.fullmatch(text.strip()):
            value = re.sub(r'^(?:till\s+date|current)$', 'Present', text.strip(), flags=re.I)
            return 'DATE_RANGE', {'period': value, 'date_value': value}
        if (not row['is_bullet'] and name_like(text) and following and
                (title_like(clean(following['text'])) or PERIOD.search(following['text']) or
                 re.match(r'^(?:Role|Client|Designation)\s*[:–—-]', following['text'], re.I))):
            return 'COMPANY', {'company': text}
        if re.match(r'^(?:(?:role\s*(?:&|and)\s*)?responsibilities|key achievements|achievements|accomplishments)\s*:?$', text, re.I):
            return 'SUBSECTION_HEADING', {}
        return 'RESPONSIBILITY', {}
    return ({'summary': 'PROFILE_SUMMARY', 'skills': 'SKILL', 'core competencies': 'SKILL',
             'business analysis and product management': 'SKILL',
             'technical skills': 'TECHNOLOGY',
             'projects': 'PROJECT', 'education': 'EDUCATION', 'certifications': 'CERTIFICATION',
             'domain experience': 'DOMAIN', 'header': 'CONTACT', 'personal information': 'PERSONAL_DETAIL',
             'languages': 'PERSONAL_DETAIL', 'tools and technology': 'TECHNOLOGY'}.get(section, 'UNKNOWN'), {})


def prepare_blocks(pre, layout=None):
    """Classify before entity detection; preserve layout and repair strong section boundaries."""
    rows = []
    layout = layout or []
    median_font = sorted(block['font_size'] for block in layout if block.get('font_size'))
    median_font = median_font[len(median_font) // 2] if median_font else 0
    layout_section = None
    for meta in pre['line_meta']:
        index = meta['line_index']
        geometry = layout[index] if index < len(layout) else {}
        if meta.get('is_header'):
            layout_section = None
        if layout_section and not meta.get('is_header'):
            meta['section'], meta['section_id'] = layout_section
        text = clean(meta.get('line', ''))
        next_text = clean(pre['line_meta'][index + 1].get('line', '')) if index + 1 < len(pre['line_meta']) else ''
        is_name_header = meta.get('section') == 'Header' and index <= 2 and len(text.split()) <= 5
        is_employer_header = name_like(text) and (title_like(next_text) or PERIOD.search(next_text))
        styled_heading = bool(geometry and (geometry.get('bold') or
                              (median_font and geometry.get('font_size') and geometry['font_size'] >= median_font * 1.25)))
        if (not meta.get('is_header') and not is_name_header and not is_employer_header
                and styled_heading and 2 <= len(text) <= 60
                and not employment_header(text) and not title_like(text) and not PERIOD.search(text)
                and not ACTION.search(text) and not re.search(r'[.!?@]', text)
                and not text.endswith(':')):
            section_name = text.strip(' :.-').title()
            layout_section = (section_name, f'layout:{index}:{re.sub(r"[^a-z0-9]+", "_", section_name.casefold()).strip("_")}')
            meta.update(section=layout_section[0], section_id=layout_section[1], is_header=True,
                        is_layout_heading=True)
    active_override = None
    active_origin = None
    metas = pre['line_meta']
    # Some PDF extractors emit the last education table cells after the
    # Personal Details heading. Reattach only a contiguous bare year + score
    # pair when the same source already contains an Education section.
    education_section = next((m for m in reversed(metas)
                              if m.get('section', '').casefold() in {'education', 'education qualification',
                                  'educational qualification', 'academic qualifications'}), None)
    if education_section:
        for index in range(len(metas) - 1):
            current, following = metas[index], metas[index + 1]
            year = clean(current.get('line', ''))
            score = clean(following.get('line', ''))
            if (current.get('section', '').casefold() in {'personal information', 'languages'}
                    and re.fullmatch(r'(?:19|20)\d{2}', year)
                    and re.fullmatch(r'\d{1,3}(?:\.\d+)?%?', score)
                    and following.get('section_id') == current.get('section_id')):
                current['section'], current['section_id'] = education_section['section'], education_section['section_id']
                following['section'], following['section_id'] = education_section['section'], education_section['section_id']
    original_sections = {meta['line_index']: meta['section'] for meta in metas}
    # Do not let a plausible-looking name/title near the top of a structured
    # resume start a synthetic employment section. In PDFs, a candidate name,
    # headline, or contact row can look like an employer/title pair after line
    # wrapping. Once an explicit work-experience heading exists, only content
    # beneath that heading can enter Experience; heuristic recovery is reserved
    # for genuinely headingless resumes.
    experience_heading_names = {
        'experience', 'professional experience', 'work experience', 'work history',
        'employment history', 'career history', 'relevant experience', 'employment',
    }
    has_explicit_experience_heading = any(
        meta.get('is_header') and meta.get('section', '').casefold().strip(' :.-')
        in experience_heading_names for meta in metas
    )
    for index, meta in enumerate(metas):
        if not meta.get('line', '').strip():
            continue
        geometry = layout[meta['line_index']] if meta['line_index'] < len(layout) else {}
        # Geometry only attaches to the matching source row, never to a guessed row.
        if geometry and clean(geometry['text']) != clean(meta.get('source_line', meta['line'])):
            geometry = {}
        text = meta['line']
        original_section = original_sections[meta['line_index']]
        if meta.get('is_header') or (active_override and original_section not in
                                      {active_origin, 'Experience'}):
            active_override = None
            active_origin = None
        next_meta = next((m for m in metas[index + 1:] if m.get('line', '').strip() and not m.get('is_page_number')), None)
        if (not has_explicit_experience_heading and not meta.get('is_header')
                and original_section in {'Header', 'Summary', 'Skills', 'Core Competencies', 'Other'}):
            value = clean(text)
            start = employment_header(value)
            standalone = (name_like(value) and ',' not in value and not title_like(value) and next_meta and
                          not next_meta.get('is_header') and title_like(clean(next_meta['line'])) and
                          (ORG_SUFFIX.search(value) or any(PERIOD.search(m.get('line', ''))
                           for m in metas[index + 1:index + 5])))
            if start or standalone:
                active_override = f'semantic:{meta["line_index"]}:Experience'
                active_origin = original_section
        if active_override and not meta.get('is_header') and original_section in {active_origin, 'Experience'}:
            meta['section'], meta['section_id'] = 'Experience', active_override
        marker = MARKER.match(text)
        row = {'id': f'd{meta["line_index"]}' + (':heading' if meta.get('is_header') else ''),
               'text': text, 'page': geometry.get('page'), 'bbox': geometry.get('bbox'),
               'block_index': geometry.get('block_index', meta['line_index']), 'reading_order': len(rows),
               'font_size': geometry.get('font_size'), 'bold': geometry.get('bold'),
               'indent': meta.get('indent', 0), 'is_bullet': bool(marker) or meta.get('is_bullet', False),
               'marker': (marker.group('bullet') or marker.group('number')) if marker else '',
               'is_header': meta.get('is_header', False), 'is_page_number': meta.get('is_page_number', False),
               'artifact': geometry.get('artifact'), 'section': meta['section'], 'section_id': meta['section_id'],
               'line_index': meta['line_index'], 'source_path': meta.get('source_path'), 'subsection': None,
               'entities': [], 'parent_candidate': None, 'destination': None, 'confidence': 0.0}
        if geometry.get('bbox') and not geometry.get('artifact'):
            # A numeric table cell in the page body is data, even if a text-only
            # page-marker heuristic would have removed it.
            row['is_page_number'] = False
            meta['is_page_number'] = False
        rows.append(row)
    for index, row in enumerate(rows):
        following = next((r for r in rows[index + 1:] if not r['is_page_number'] and not r.get('artifact')), None)
        if following and following['section_id'] != row['section_id']:
            following = None
        row['block_type'], row['entities'] = _classify(row, following)
        source_geometry = layout[row['line_index']] if row['line_index'] < len(layout) else {}
        row['raw_text'] = next((meta.get('source_line', meta.get('line', ''))
                                for meta in metas if meta['line_index'] == row['line_index']), row['text'])
        row['normalized_text'] = row['text']
        row['page_number'] = source_geometry.get('page')
        if row['block_type'] == 'PAGE_ARTIFACT':
            marker = re.fullmatch(r'(?:page\s+)?(\d{1,3})(?:\s*(?:of|/)\s*\d+)?', clean(row['text']), re.I)
            if marker:
                row['page_number'] = int(marker.group(1))
        semantic_names = {
            'SECTION_HEADING': 'section_heading', 'PROFILE_SUMMARY': 'profile_summary',
            'COMPANY': 'company', 'COMPANY_ROLE': 'company', 'ROLE': 'job_title',
            'CLIENT': 'client', 'DATE_RANGE': 'date_range', 'RESPONSIBILITY': 'responsibility',
            'ACHIEVEMENT': 'achievement', 'PROJECT': 'project_description', 'SKILL': 'skill',
            'DOMAIN': 'domain_experience', 'TECHNOLOGY': 'technical_skill',
            'EDUCATION': 'education_row', 'TABLE_HEADING': 'education_header',
            'PERSONAL_DETAIL': 'personal_detail', 'PAGE_ARTIFACT': 'page_marker',
            'UNKNOWN': 'unknown', 'SUBSECTION_HEADING': 'section_heading',
            'CONTACT': 'personal_detail',
        }
        row['semantic_classification'] = semantic_names.get(row['block_type'], 'unknown')
        row['semantic_labels'] = (["company", "job_title"] if row['block_type'] == 'COMPANY_ROLE'
                                   else [row['semantic_classification']])
        row['confidence'] = .95 if row['block_type'] not in {'UNKNOWN', 'RESPONSIBILITY'} else .7 if row['block_type'] == 'RESPONSIBILITY' else 0
        row['reason'] = 'explicit heading/field or local section and neighboring record structure'
        if row.get('source_path'):
            row['destination'] = row['source_path']
            row['parent_candidate'] = re.sub(r'(?:\.[\w-]+|\[\d+\])$', '', row['source_path'])
            row['reason'] = 'preserved source structure from supplied canonical data'
    by_line = {r['line_index']: r for r in rows if not r['is_header']}
    for meta in pre['sentence_meta']:
        if meta['line_index'] in by_line:
            row = by_line[meta['line_index']]
            meta.update(section=row['section'], section_id=row['section_id'])
    artifact_lines = {r['line_index'] for r in rows if r['block_type'] == 'PAGE_ARTIFACT'}
    for meta in metas:
        if meta['line_index'] in artifact_lines:
            meta['is_page_number'] = True
    sentences, sentence_meta = [], []
    for sentence, meta in zip(pre['sentences'], pre['sentence_meta']):
        if meta['line_index'] not in artifact_lines:
            meta['index'] = len(sentences)
            sentences.append(sentence)
            sentence_meta.append(meta)
    pre['sentences'], pre['sentence_meta'] = sentences, sentence_meta
    pre['sentence_count'] = len(sentences)
    pre['semantic_blocks'] = rows
    return rows


def _own(row, owner, path, reason):
    source_blocks = (owner.source_blocks if isinstance(owner, EmploymentRecord)
                     else owner.setdefault('source_blocks', []))
    if row['id'] not in source_blocks:
        source_blocks.append(row['id'])
    row.update(parent_candidate=path, destination=path, reason=reason)


@dataclass
class EmploymentRecord:
    """Intermediate employer parent; roles and source blocks are attached before serialization."""
    company: str
    roles: list[dict] = field(default_factory=list)
    source_blocks: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'company': self.company, **self.attributes,
                'roles': self.roles, 'source_blocks': self.source_blocks}


def group_employment(rows, offset=0):
    jobs, unresolved = [], []
    job = role = None
    role_path = job_path = None
    last = None
    pending = {}
    for row in rows:
        kind, fields, text = row['block_type'], row['entities'], clean(row['text'])
        if kind in {'PAGE_ARTIFACT', 'SECTION_HEADING'}:
            continue
        if kind in {'COMPANY', 'COMPANY_ROLE'}:
            company = fields['company']
            existing = next((record for record in reversed(jobs)
                             if record.company.casefold() == company.casefold()), None)
            if existing is None or (jobs and jobs[-1] is not existing):
                jobs.append(EmploymentRecord(company=company))
                existing = jobs[-1]
            job = existing
            job_path = f'$.work_history[{offset + len(jobs) - 1}]'
            role, role_path, last, pending = None, None, None, {}
            _own(row, job, job_path, 'employer header opens a new employment context')
            if fields.get('period'):
                job.attributes['duration'] = fields['period']
            if kind == 'COMPANY':
                continue
        if kind in {'ROLE', 'COMPANY_ROLE'}:
            if job is None:
                unresolved.append(row)
                continue
            role = {'designation': fields['designation'], 'role_and_responsibilities': [],
                    'responsibility_sources': [], 'source_blocks': []}
            job.roles.append(role)
            role_path = f'{job_path}.roles[{len(job.roles) - 1}]'
            _own(row, role, role_path, 'title opens a role under the active employer')
            for key, (value, source) in pending.items():
                role[key] = value
                _own(source, role, role_path, 'field between employer and title belongs to this role')
            pending = {}
            if fields.get('period'):
                role['duration'] = fields['period']
            last = None
            continue
        if job is None:
            unresolved.append(row)
            continue
        _own(row, job, job_path, 'source block inside the active employer context')
        if kind in {'CLIENT', 'DATE_RANGE'}:
            values = {'clients': fields['clients']} if kind == 'CLIENT' else {'duration': fields['period']}
            if fields.get('reported_duration'):
                values['reported_duration'] = fields['reported_duration']
            if kind == 'DATE_RANGE' and fields.get('designation'):
                role = {'designation': fields['designation'], 'role_and_responsibilities': [],
                        'responsibility_sources': [], 'source_blocks': []}
                job.roles.append(role)
                role_path = f'{job_path}.roles[{len(job.roles) - 1}]'
                role.update(values)
                role['start_date'] = fields.get('start_date', '')
                role['end_date'] = fields.get('end_date', '')
                role.setdefault('field_sources', {}).setdefault('employment_period', []).append(row['id'])
                _own(row, role, role_path, 'date range and role title share a source block')
                last = None
                continue
            if role is not None:
                role.update(values)
                if kind == 'DATE_RANGE':
                    if fields.get('start_date'):
                        role['start_date'] = fields['start_date']
                    if fields.get('end_date'):
                        role['end_date'] = fields['end_date']
                    if fields.get('date_value'):
                        if fields['date_value'].casefold() in {'present', 'current', 'now', 'till date', 'to date'}:
                            role['end_date'] = 'Present'
                        elif role.get('duration'):
                            role['end_date'] = fields['date_value']
                        else:
                            role['start_date'] = fields['date_value']
                    role.setdefault('field_sources', {}).setdefault('employment_period', []).append(row['id'])
                elif kind == 'CLIENT':
                    role.setdefault('field_sources', {}).setdefault('clients', []).append(row['id'])
                _own(row, role, role_path, 'explicit client/date field under the active role')
            else:
                if kind == 'DATE_RANGE':
                    if fields.get('start_date'):
                        values['start_date'] = fields['start_date']
                    if fields.get('end_date'):
                        values['end_date'] = fields['end_date']
                    if fields.get('date_value'):
                        values['end_date' if fields['date_value'].casefold() in {'present', 'current', 'now', 'till date', 'to date'}
                               else 'start_date'] = 'Present' if fields['date_value'].casefold() in {
                                   'present', 'current', 'now', 'till date', 'to date'} else fields['date_value']
                for key, value in values.items():
                    pending[key] = (value, row)
            continue
        if kind == 'ACHIEVEMENT':
            if role is None:
                role = {'designation': '', 'role_and_responsibilities': [],
                        'responsibility_sources': [], 'source_blocks': []}
                job.roles.append(role)
                role_path = f'{job_path}.roles[{len(job.roles) - 1}]'
            achievements = role.setdefault('achievements', [])
            achievements.append(fields.get('text') or text)
            role.setdefault('achievement_sources', []).append(row['id'])
            _own(row, role, f'{role_path}.achievements[{len(achievements) - 1}]',
                 'explicit achievement label within the active role')
            last = None
            continue
        if kind == 'SUBSECTION_HEADING':
            row['destination'] = role_path or job_path
            continue
        if role is None:
            # Preserve employment content without inventing a title.
            role = {'designation': '', 'role_and_responsibilities': [], 'responsibility_sources': [], 'source_blocks': []}
            job.roles.append(role)
            role_path = f'{job_path}.roles[{len(job.roles) - 1}]'
            for key, (value, source) in pending.items():
                role[key] = value
                _own(source, role, role_path, 'explicit field in employment with unspecified title')
            pending = {}
        bullets = role['role_and_responsibilities']
        nested = bool(last and row['is_bullet'] and
                      (row['indent'] > last['indent'] or row['marker'] in {'○', '◦'}) and
                      (':' in bullets[-1] or last.get('nested')))
        previous_text = last['text'].rstrip() if last else ''
        continuation = bool(last and (
            (not row['is_bullet'] and text[:1].islower()) or
            previous_text.casefold().endswith((' and', ' or', ' of', ' the', ' to', ',', ':')) or
            (not row['is_bullet'] and last['is_bullet']
             and not re.search(r'[.!?;]$', previous_text))
        ))
        if bullets and (nested or continuation):
            bullets[-1] += ('; ' if nested and last.get('nested') else ' ') + text
            role['responsibility_sources'][-1].append(row['id'])
        else:
            bullets.append(text)
            role['responsibility_sources'].append([row['id']])
        _own(row, role, f'{role_path}.role_and_responsibilities[{len(bullets) - 1}]',
             'nested/wrapped continuation of responsibility' if nested or continuation else 'responsibility in active role')
        row['nested'] = nested
        last = row
    for key, (value, row) in pending.items():
        job.attributes[key] = value
        _own(row, job, job_path, 'field belongs to employer; no role title supplied')
    return [record.to_dict() for record in jobs], unresolved


TABLE_KEYS = {'degree': 'degree', 'qualification': 'degree', 'college': 'institution', 'college/school': 'institution',
              'college school': 'institution', 'institution': 'institution', 'university': 'university_or_board',
              'university/board': 'university_or_board', 'university board': 'university_or_board',
              'year': 'year', 'year of passing': 'year', 'percentage': 'score', 'percentage & pointer': 'score',
              'percentage &': 'score', 'percentage and pointer': 'score', 'pointer': 'score', 'score': 'score', 'cgpa': 'score'}


def _table_cell(text):
    # PDF tables can be extracted as column blocks instead of rows. A newline
    # or pipe delimited line is converted to the same five-column row schema.
    return re.sub(r'\s+', ' ', text).strip()


def _group_columnar_education(rows, offset):
    # Distinguish a visual column dump from normal row-wise records. If one
    # degree is followed by institution/year/score and then another degree,
    # the source is row-wise and must not be zipped by field type.
    seen_first_degree = False
    seen_preceding_field = False
    repeated_degree_after_fields = False
    for row in rows:
        text = clean(row['text'])
        folded = text.casefold().rstrip(':')
        if folded in TABLE_KEYS:
            continue
        if EDUCATION.match(text):
            if seen_first_degree and seen_preceding_field:
                repeated_degree_after_fields = True
            seen_first_degree = True
        elif seen_first_degree and (PERIOD.search(text) or re.fullmatch(r'(?:19|20)\d{2}|\d{1,3}(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?%?', text)
                                    or re.search(r'\b(?:university|college|school|institute|cbse|icse)\b', text, re.I)):
            seen_preceding_field = True
    if repeated_degree_after_fields:
        return None
    columns = {}
    explicit_degree_count = 0
    for row in rows:
        cells = [_table_cell(cell).casefold().rstrip(':') for cell in clean(row['text']).split('|')]
        mapped = [TABLE_KEYS.get(cell) for cell in cells]
        if len(cells) > 1 and required_table_columns(mapped):
            continue
        label = cells[0]
        key = TABLE_KEYS.get(label)
        if key:
            if label in TABLE_KEYS:
                # Explicit column headers are metadata, not row values.
                row.update(block_type='TABLE_HEADING', destination='$.education', parent_candidate='$.education')
            else:
                columns.setdefault(key, []).append(row)
        elif len(cells) == 1 and re.fullmatch(r'(?:19|20)\d{2}', label):
            columns.setdefault('year', []).append(row)
        elif len(cells) == 1 and re.fullmatch(r'\d{1,3}(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?%?', label):
            columns.setdefault('score', []).append(row)
        elif len(cells) == 1 and EDUCATION.match(clean(row['text'])):
            columns.setdefault('degree', []).append(row)
            explicit_degree_count += 1
        elif len(cells) == 1 and re.search(r'\b(?:university|college|school|institute|cbse|icse)\b', label, re.I):
            target = ('university_or_board' if re.search(r'\b(?:university|board|cbse|icse)\b', label, re.I)
                      else 'institution')
            columns.setdefault(target, []).append(row)
        elif len(cells) == 1 and label == 'road' and columns.get('institution'):
            columns['institution'][-1]['text'] += ' Road'
    required = {'degree', 'institution', 'university_or_board', 'year', 'score'}
    if not required.issubset(columns) or len(columns['degree']) < 2:
        return None
    # Column dumps can interleave rows: a qualification and its school may be
    # followed by subsequent-year records, each with university/year/score,
    # before the next qualification begins.  Keep these as separate records
    # and attach the shared school to the contiguous education run.
    if explicit_degree_count >= 2 and len(columns['institution']) >= explicit_degree_count:
        records = []
        for index, degree_row in enumerate(columns['degree']):
            record = {'degree': _table_cell(degree_row['text'])}
            degree_year = EDUCATION_YEAR.search(record['degree'])
            if degree_year:
                record['year'] = degree_year.group(1)
            if index < len(columns['institution']):
                institution = columns['institution'][index]
                record['institution'] = _table_cell(institution['text'])
            # Assign board/university and score rows by reading order between
            # this degree and the next degree. This retains chronological rows
            # (including a degree's own year) without zipping unrelated fields.
            start = degree_row['line_index']
            end = columns['degree'][index + 1]['line_index'] if index + 1 < len(columns['degree']) else float('inf')
            between = [row for field in ('university_or_board', 'year', 'score')
                       for row in columns.get(field, []) if start < row['line_index'] < end]
            for field in ('university_or_board', 'year', 'score'):
                candidates = [item for item in columns.get(field, [])
                              if start < item['line_index'] < end]
                if candidates:
                    value = _table_cell(candidates[-1]['text'])
                    if field == 'year':
                        match = EDUCATION_YEAR.search(value)
                        if match:
                            value = match.group(1)
                    record[field] = value
            records.append(record)
            _own(degree_row, record, f'$.education[{offset + index}]', 'education qualification starts this record')
            if index < len(columns['institution']):
                _own(columns['institution'][index], record, f'$.education[{offset + index}]',
                     'institution in the same education record sequence')
            for field in ('university_or_board', 'year', 'score'):
                for source in columns.get(field, []):
                    if start < source['line_index'] < end and record.get(field) == _table_cell(source['text']):
                        _own(source, record, f'$.education[{offset + index}]',
                             'year, board, or score in the qualification record')
        if len(records) > explicit_degree_count and records[-1].get('degree', '').casefold() == 'ssc':
            # If a final school qualification is present, do not manufacture
            # an extra table row from stray numeric page/footer text.
            pass
        return records
    count = min(len(columns[key]) for key in required)
    records = []
    for index in range(count):
        record = {key: _table_cell(columns[key][index]['text']) for key in required}
        records.append(record)
        for key in required:
            source = columns[key][index]
            _own(source, record, f'$.education[{offset + index}]', 'aligned education table row cells')
    return records


def required_table_columns(mapped):
    return sum(bool(value) for value in mapped) >= 2


def group_education(rows, offset=0):
    columnar = _group_columnar_education(rows, offset)
    if columnar:
        return columnar
    records, columns = [], None
    current = None
    for row in rows:
        text = clean(row['text'])
        cells = [c.strip() for c in text.split('|')]
        mapped = [TABLE_KEYS.get(c.casefold().rstrip(':')) for c in cells]
        if len(cells) > 1 and sum(bool(c) for c in mapped) >= 2:
            columns = mapped
            row.update(block_type='TABLE_HEADING', destination='$.education', parent_candidate='$.education')
            continue
        if columns and len(cells) == len(columns):
            current = {key: value for key, value in zip(columns, cells) if key and value}
            extras = [value for key, value in zip(columns, cells) if not key and value]
            if extras:
                current['additional_details'] = extras
            records.append(current)
        elif len(cells) == 1 and cells[0].casefold().rstrip(':') in TABLE_KEYS:
            row.update(block_type='TABLE_HEADING', destination='$.education', parent_candidate='$.education')
            continue
        else:
            degree = bool(EDUCATION.search(text))
            academic_stage = bool(re.match(r'^(?:first|second|third|final)\s+year\b', text, re.I))
            starts_credential = degree and not academic_stage
            active_result = (current.get('academic_results', [])[current.get('_active_academic_result', -1)]
                             if current and current.get('_active_academic_result') is not None else None)
            if (active_result and active_result.get('stage', '').casefold().rstrip().endswith((' of', ' in', ' and'))
                    and not academic_stage and not re.search(r'\b(?:university|board|cbse|icse)\b', text, re.I)):
                active_result['stage'] += ' ' + text
                _own(row, current, f'$.education[{offset + len(records) - 1}]',
                     'wrapped continuation of an academic year/stage')
                continue
            if current is None or (starts_credential and current.get('degree')):
                current = {}
                records.append(current)
            date = PERIOD.search(text) or re.search(r'\b(?:19|20)\d{2}\b', text)
            if academic_stage:
                current.setdefault('academic_results', []).append({'stage': text})
                current['_active_academic_result'] = len(current['academic_results']) - 1
            elif starts_credential:
                body = text[:date.start()].strip(' (|,') if date else text
                qualification = re.split(r'\s+(?:from|at)\s+', body, maxsplit=1, flags=re.I)
                current['degree'] = qualification[0]
                if len(qualification) > 1:
                    current['institution'] = qualification[1].strip(' ()')
                if date:
                    current['year'] = date.group(0)
            elif current.get('degree') and re.search(r'\b(?:of|in|and|&)$', current['degree'], re.I):
                current['degree'] = f"{current['degree']} {text}".strip()
            elif re.fullmatch(r'(?:19|20)\d{2}', text):
                active = current.get('_active_academic_result')
                if active is not None and not current['academic_results'][active].get('year'):
                    current['academic_results'][active]['year'] = text
                elif current.get('year') and current.get('score'):
                    current.setdefault('academic_results', []).append({'year': text})
                    current['_active_academic_result'] = len(current['academic_results']) - 1
                else:
                    current['year'] = text
            elif re.fullmatch(r'\d{1,3}(?:\.\d+)?\s*(?:/\s*\d+(?:\.\d+)?)?%?', text) or re.match(r'^(?:CGPA|GPA|Percentage)\s*:', text, re.I):
                active = current.get('_active_academic_result')
                if active is not None and not current['academic_results'][active].get('score'):
                    current['academic_results'][active]['score'] = text
                elif current.get('score'):
                    current.setdefault('academic_results', []).append({'score': text})
                    current['_active_academic_result'] = len(current['academic_results']) - 1
                else:
                    current['score'] = text
            elif re.search(r'\b(?:university|board|cbse|icse)\b', text, re.I):
                year = EDUCATION_YEAR.search(text)
                board_name = text[:year.start()].strip() if year else text
                active = current.get('_active_academic_result')
                if active is not None:
                    current['academic_results'][active].setdefault('university_or_board', board_name)
                    if year:
                        current['academic_results'][active]['year'] = year.group(1)
                elif year:
                    if current.get('year') and current.get('score'):
                        current.setdefault('academic_results', []).append(
                            {'university_or_board': board_name, 'year': year.group(1)})
                        current['_active_academic_result'] = len(current['academic_results']) - 1
                    else:
                        current['university_or_board'] = board_name
                        current['year'] = year.group(1)
                elif not current.get('university_or_board'):
                    current['university_or_board'] = board_name
                elif current.get('score'):
                    current.setdefault('academic_results', []).append({'university_or_board': board_name})
                    current['_active_academic_result'] = len(current['academic_results']) - 1
                else:
                    current['university_or_board'] = (current['university_or_board'] + ' ' + board_name).strip()
            elif ('institution' in current and not current.get('university_or_board') and
                  re.search(r'\b(?:road|ambernath)$', text, re.I)):
                current['institution'] = f"{current['institution']} {text}".strip()
            elif 'institution' in current and not current.get('university_or_board'):
                if text.casefold() not in current['institution'].casefold():
                    current['institution'] = f"{current['institution']} {text}".strip()
            elif 'institution' not in current:
                current['institution'] = text
            elif text.casefold() not in current['institution'].casefold():
                current.setdefault('additional_details', []).append(text)
        _own(row, current, f'$.education[{offset + len(records) - 1}]', 'education row/qualification with its following fields')
    for record in records:
        record.pop('_active_academic_result', None)
    return records


def group_projects(rows, offset=0):
    records, current = [], None
    for index, row in enumerate(rows):
        text = clean(row['text'])
        field = re.match(r'^(technologies|tools|tech stack|client|description)\s*:\s*(.+)', text, re.I)
        numbered = bool(re.match(r'^\s*\d+[.)]', row['text']))
        named = re.match(r'^project(?: name| title)?\s*[:–—-]\s*(.+)', text, re.I)
        short_title = (not ACTION.search(text) and not re.search(r'[.!?]$', text) and len(text) < 100
                       and index + 1 < len(rows) and ACTION.search(clean(rows[index + 1]['text'])))
        independent = bool(current and not current.get('name') and current.get('description', '').endswith(('.', '!', '?')) and text[:1].isupper() and not field)
        if current is None or numbered or named or short_title or independent or (row['is_bullet'] and not field):
            current = {'name': named[1] if named else text} if named or short_title else {'description': text}
            records.append(current)
        elif field:
            key = field[1].casefold()
            if key in {'technologies', 'tools', 'tech stack'}:
                current.setdefault('technologies', []).extend(v.strip() for v in re.split('[,;]', field[2]) if v.strip())
            else:
                current[key] = field[2]
        elif 'description' in current:
            current['description'] += ' ' + text
        else:
            current['description'] = text
        if named or short_title or numbered:
            row['semantic_classification'] = 'project'
        else:
            row['semantic_classification'] = 'project_description'
        row['semantic_labels'] = [row['semantic_classification']]
        _own(row, current, f'$.projects[{offset + len(records) - 1}]', 'project heading/list item owns its description and labeled fields')
    return records


def group_sections(rows):
    groups, unresolved, offsets = {}, [], {}
    # Headings can be split into multiple source sections by OCR/PDF line
    # wraps. Group by resolved semantic section type, while each employer
    # heading still starts a distinct employment record within that context.
    semantic_sections = {'experience': 'work_history', 'professional experience': 'work_history',
                         'work experience': 'work_history', 'work history': 'work_history',
                         'employment history': 'work_history', 'career history': 'work_history',
                         'relevant experience': 'work_history', 'employment': 'work_history',
                         'education': 'education', 'projects': 'projects',
                         'domain experience': 'domain_experience', 'technical skills': 'technical_skills',
                         'skills': 'skills', 'summary': 'profile_snapshot', 'personal information': 'personal_information',
                         'header': 'personal_information', 'certifications': 'certifications',
                         'core competencies': 'skills', 'languages': 'personal_information',
                         'personal details': 'personal_information'}
    section_names = list(dict.fromkeys(row['section'] for row in rows))
    for section in section_names:
        body = [r for r in rows if r['section'].casefold() == section.casefold()
                and r['block_type'] not in {'SECTION_HEADING', 'PAGE_ARTIFACT'}]
        if not body:
            continue
        key = semantic_sections.get(section.casefold())
        sid = next((r['section_id'] for r in body), f'semantic:{section}')
        offset = offsets.get(key, 0)
        if key == 'work_history':
            first_employer = next((index for index, row in enumerate(body)
                                   if row['block_type'] in {'COMPANY', 'COMPANY_ROLE'}), len(body))
            preamble = body[:first_employer]
            overview_cues = re.compile(
                r'\b\d+\+?\s+years?\b.*\bexperience\b|\bexperienced\s+in\b|'
                r'\bspeciali[sz](?:e|es|ed|ing)\s+in\b|\bprofessional\s+with\b.*\bexperience\b|'
                r'\bproven\s+(?:expertise|ability|record)\b|\badept\s+at\b', re.I)
            if preamble and overview_cues.search(' '.join(clean(row['text']) for row in preamble)):
                summary_text = ' '.join(clean(row['text']) for row in preamble if clean(row['text']))
                summary = {'source_blocks': []}
                for row in preamble:
                    row['block_type'] = 'PROFILE_SUMMARY'
                    _own(row, summary, '$.profile_snapshot[0]',
                         'pre-employment overview prose describes the candidate rather than a job')
                groups[f'inferred:profile_snapshot:{sid}'] = {
                    'key': 'profile_snapshot', 'value': [summary_text], 'inferred': True,
                    'source_blocks': summary['source_blocks'],
                }
                body = body[first_employer:]
            value, unknown = group_employment(body, offset)
            unresolved.extend(unknown)
        elif key == 'education':
            value = group_education(body, offset)
        elif key == 'projects':
            value = group_projects(body, offset)
        elif key == 'domain_experience':
            value, current = [], None
            for row in body:
                text = clean(row['text'])
                label = re.match(r'^([^:]+):\s*(.*)$', text)
                if label or current is None:
                    current = {'name': label[1].strip() if label else text, 'organizations': []}
                    value.append(current)
                names = label[2] if label else text if row is not body[0] else ''
                current['organizations'].extend(v.strip() for v in re.split('[,;]', names) if v.strip())
                _own(row, current, f'$.domain_experience[{offset + len(value) - 1}]', 'domain section and explicit category')
        elif key in {'skills', 'personal_information', 'certifications'}:
            if key == 'personal_information':
                value = {}
                if section.casefold() == 'header':
                    header_fields, field_rows = extract_header_identity(body)
                    value.update(header_fields)
                    for row in body:
                        owned_fields = [field for field, source_row in field_rows.items() if source_row is row]
                        if owned_fields:
                            for field in owned_fields:
                                _own(row, value, f'$.personal_information.{field}',
                                     'identity/contact field extracted from resume header source text')
                        else:
                            _own(row, value, '$.personal_information',
                                 'unclassified header source retained in personal information provenance')
                    groups[sid] = {'key': key, 'value': value}
                    offsets[key] = offset + 1
                    continue
                for row in body:
                    text = clean(row['text'])
                    if row['block_type'] == 'CONTACT':
                        value.setdefault('name', text)
                        _own(row, value, '$.personal_information', 'contact identity from the resume header')
                        continue
                    elif row['block_type'] == 'PERSONAL_DETAIL':
                        label_match = re.match(
                            r'^(email|e-mail|phone|mobile|contact number|location|address|date of birth|dob|nationality|languages?)\s*:\s*(.*)$',
                            text, re.I)
                        raw_label = row['entities'].get('field') or (label_match.group(1) if label_match else '')
                        label = re.sub(r'[^a-z0-9]+', '_', raw_label.casefold()).strip('_')
                        field = {'e_mail': 'email', 'mobile': 'phone', 'contact_number': 'phone',
                                 'address': 'location', 'dob': 'date_of_birth'}.get(label, label)
                        raw = row['entities'].get('value') or (label_match.group(2) if label_match else text)
                        if section.casefold() == 'languages' and not field:
                            field = 'languages'
                        if field in {'date_of_birth', 'nationality', 'languages'}:
                            details = value.setdefault('personal_details', {'source_blocks': []})
                            if field == 'languages':
                                details.setdefault('languages', []).extend(
                                    item.strip() for item in re.split(r'[,;]', raw) if item.strip())
                            else:
                                details[field] = raw
                            _own(row, details, f'$.personal_details.{field}', 'personal detail field and source label')
                            continue
                        if field == 'email':
                            match = re.search(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', raw, re.I)
                            phone = re.search(r'(?<!\w)\+?\d[\d().\s-]{6,}\d(?!\w)', raw)
                            if match:
                                value[field] = match.group(0)
                            if phone and not value.get('phone'):
                                value['phone'] = phone.group(0).strip()
                        elif field == 'languages':
                            details = value.setdefault('personal_details', {'source_blocks': []})
                            details[field] = [item.strip() for item in re.split(r'[,;]', raw) if item.strip()]
                        elif field:
                            value[field] = raw
                        _own(row, value, f'$.personal_information.{field}' if field else '$.personal_information',
                             'contact field label in source')
                    elif section.casefold() == 'languages':
                        details = value.setdefault('personal_details', {'source_blocks': []})
                        details.setdefault('languages', []).extend(
                            item.strip() for item in re.split(r'[,;]', text) if item.strip())
                        _own(row, details, '$.personal_details.languages', 'language values in the Languages section')
                    else:
                        _own(row, value, '$.personal_information', 'personal information source block')
            else:
                value = []
                for row in body:
                    text = clean(row['text'])
                    continuation = bool(value and not row['is_bullet'] and text[:1].islower()
                                        and not re.search(r'[.!?;]$', value[-1]))
                    if continuation:
                        value[-1] += ' ' + text
                        row['destination'] = f'$.{key}[{offset + len(value) - 1}]'
                        row['parent_candidate'] = row['destination']
                        row['reason'] = 'wrapped continuation of the preceding skill bullet'
                    else:
                        value.append(text)
                        row['destination'] = f'$.{key}[{offset + len(value) - 1}]'
                        row['parent_candidate'] = row['destination']
                if key == 'skills' and len(value) == 1 and not body[0]['is_bullet'] and ',' in value[0]:
                    value = [item.strip(' .;,') for item in value[0].split(',') if item.strip(' .;,')]
                    body[0]['destination'] = f'$.skills[{offset}]'
                    body[0]['parent_candidate'] = body[0]['destination']
        elif key == 'technical_skills':
            value, category = {}, 'general'
            for row in body:
                text = clean(row['text'])
                label = re.match(r'^([^:]+):\s*(.*)$', text)
                if label:
                    category = re.sub(r'\W+', '_', label[1].casefold()).strip('_')
                    text = label[2]
                value.setdefault(category, []).extend(v.strip() for v in re.split('[,;]', text) if v.strip())
                row.update(destination=f'$.technical_skills.{category}', parent_candidate=f'$.technical_skills.{category}')
        elif key == 'profile_snapshot':
            value = [{'text': ' '.join(clean(row['text']) for row in body if clean(row['text'])),
                      'source_blocks': []}]
            for row in body:
                _own(row, value[0], '$.profile_snapshot[0]', 'source text within the summary section')
        else:
            for row in body:
                if row['block_type'] == 'UNKNOWN':
                    unresolved.append(row)
                else:
                    row['parent_candidate'] = sid
            continue
        groups[sid] = {'key': key, 'value': value}
        offsets[key] = offset + len(value)
    return {'groups': groups, 'blocks': rows, 'unresolved_blocks': [
        {'type': 'unknown', 'text': r['text'], 'source_block_id': r['id'], 'section': r['section']}
        for r in unresolved]}


def validate_semantic_employment(semantic):
    """Fail early if extracted employment rows are not nested under employer/role parents."""
    records = [record for group in semantic.get('groups', {}).values()
               if group.get('key') == 'work_history' for record in group.get('value', [])]
    for record in records:
        if not record.get('company') or not isinstance(record.get('roles'), list):
            raise ValueError('Semantic employment record must have a company and roles list')
        for role in record['roles']:
            if not isinstance(role.get('role_and_responsibilities'), list):
                raise ValueError('Semantic employment role must have a responsibility list')
            if 'details' in role:
                raise ValueError('Legacy details-only employment records are not valid semantic groups')
    record_count = len(records)
    unresolved_ids = {row.get('source_block_id') for row in semantic.get('unresolved_blocks', [])}
    for row in semantic.get('blocks', []):
        if (row.get('section', '').casefold() in {'experience', 'professional experience',
                'work experience', 'work history', 'employment history', 'career history'}
                and row.get('block_type') not in {'SECTION_HEADING', 'PAGE_ARTIFACT'}
                and not row.get('destination') and row.get('id') not in unresolved_ids):
            raise ValueError(f"Employment source block {row['id']} has no parent or unresolved review record")
        if row.get('block_type') not in {'COMPANY', 'COMPANY_ROLE'}:
            continue
        destination = row.get('destination') or ''
        match = re.match(r'^\$\.work_history\[(\d+)\]', destination)
        if not match or int(match.group(1)) >= record_count:
            raise ValueError(f"Employer block {row['id']} has no grouped employment parent")


def coverage_report(semantic):
    rows = semantic['blocks']
    artifacts = [r for r in rows if r['block_type'] == 'PAGE_ARTIFACT'
                 and (r.get('is_page_number') or r.get('artifact') == 'page_number')]
    margins = [r for r in rows if r.get('artifact') == 'repeated_margin']
    markers = [r for r in rows if r['block_type'] == 'PAGE_ARTIFACT' and r not in artifacts]
    meaningful = [r for r in rows if r['block_type'] not in {'PAGE_ARTIFACT', 'SECTION_HEADING', 'TABLE_HEADING'}]
    return {'total_source_blocks': len(rows), 'semantic_blocks': len(meaningful),
            'classified_blocks': sum(r['block_type'] != 'UNKNOWN' for r in meaningful),
            'grouped_blocks': sum(bool(r['destination']) for r in meaningful),
            'ignored_artifacts': len(artifacts), 'ignored_page_numbers': len(artifacts),
            'ignored_headers_footers': len(margins), 'decorative_bullet_markers': len(markers),
            'unclassified_blocks': len(semantic['unresolved_blocks']),
            'unresolved_source_ids': [r['source_block_id'] for r in semantic['unresolved_blocks']],
            'complete': not semantic['unresolved_blocks'] and all(r['destination'] for r in meaningful)}
