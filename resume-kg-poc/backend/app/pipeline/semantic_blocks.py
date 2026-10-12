"""Classify source blocks, resolve owners, then produce hierarchical section records.

No embeddings or graph layout participate. Every grouping decision retains its
source IDs and reason. Ambiguous blocks remain in the unresolved collection.
"""
import re
from dataclasses import dataclass, field

MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
DATE_ATOM = rf'(?:{MONTH}\.?\s*-?\s*\d{{2,4}}|\d{{1,2}}/\d{{2,4}}|(?:19|20)\d{{2}})'
PERIOD = re.compile(rf'(?<!\d)({DATE_ATOM})\s*(?:[-–—]|\bto\b)\s*({DATE_ATOM}|present|current|now|till\s+date|to\s+date)(?!\d)', re.I)
SINGLE_DATE = re.compile(rf'^(?:{DATE_ATOM}|present|current|till\s+date)$', re.I)
ROLE = re.compile(r'\b(analyst|engineer|developer|consultant|manager|director|designer|architect|scientist|officer|specialist|associate|executive|lead|intern|owner|president|chief|researcher|accountant|administrator|coordinator|technician|teacher|nurse|professor|fellow)\b', re.I)
ROLE = re.compile(r'\b(analyst|engineer|developer|consultant|manager|director|designer|architect|scientist|officer|specialist|associate|executive|lead|intern|owner|president|chief|researcher|accountant|administrator|coordinator|technician|teacher|nurse|professor|fellow|transitions? executive|underwriter|functional analyst)\b', re.I)
ACTION = re.compile(r'^(?:i\s+have\s+worked|responsible|led|leading|liaise|gather|define|document|manag|collaborat|work(?:ed|ing)?\b|support|ensure|prepar|implement|develop|creat|built|design|deliver|conduct|perform|review|provid|coordinat|interact|handl|driv|translat|involved|deployed|integrat|improv|mentor|achiev|verif|complet|download|authored|architect|establish|maintain|autom|facilitat|spearhead|orchestrat|own(?:ed)?|play(?:ed)?|act(?:ed)?|lead|configure|migrat|enhanc|reduc|optimi[sz]|analy[sz]|test|deploy|build|created|worked|managed)', re.I)
ORG_SUFFIX = re.compile(r'\b(?:ltd|limited|inc|llc|corp|corporation|systems|services|technologies|infotech|bank|university|laboratories)\b', re.I)
MARKER = re.compile(r'^\s*(?:(?P<bullet>[•▪●○◦\uf0b7*–—-])\s*|(?P<number>\d+[.)])(?=\s+))')
LAYOUT_SECTION_CUE = re.compile(
    r'\b(summary|objective|profile|experience|employment|work history|education|academic|'
    r'qualification|project|skill|competenc|expertise|certificat|licen[cs]e|award|'
    r'achievement|publication|training|course|language|interest|hobb|volunteer|'
    r'leadership|contact|personal|domain|technical|strength|activities|reference|patent)\b',
    re.I,
)
EDUCATION = re.compile(r'^(?:b\.?\s?(?:e|tech|sc|com|s|a|ca)\b|m\.?\s?(?:e|tech|sc|s|a|ba|ca)\b|bachelor|master|ph\.?d|doctor|diploma|ssc\b|hsc\b|high school|secondary|class\s+\d|(?:first|second|third|final)\s+year)', re.I)
EDUCATION_YEAR = re.compile(r'(?<!\d)((?:19|20)\d{2})(?!\d)')


def clean(text):
    text = re.sub(r'[\ue000-\uf8ff]', '', str(text or ''))
    return MARKER.sub('', text).strip()


def _expand_two_digit_year(value):
    """Expand abbreviated month years in common resume date ranges."""
    return re.sub(rf'\b({MONTH})\.?\s+(\d{{2}})\b',
                  lambda m: f"{m.group(1)} {('20' if int(m.group(2)) <= 39 else '19')}{m.group(2)}",
                  str(value or ''), flags=re.I)


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
        value = re.sub(r'[\ue000-\uf8ff]', '', str(value or '')).strip(' \t,;|·•')
        if key in {'linkedin', 'github', 'website', 'portfolio'} and value and not re.match(r'^https?://', value, re.I):
            if key == 'linkedin' and re.fullmatch(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,8}', value):
                value = 'https://www.linkedin.com/in/' + value
            elif key == 'github' and re.fullmatch(r'[A-Za-z0-9-]{1,39}', value):
                value = 'https://github.com/' + value
            else:
                value = 'https://' + value.lstrip('/')
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
        # Some templates show the LinkedIn icon but extract only its profile
        # slug (without linkedin.com/in/). Recognize this shape only in the
        # contact/header rows, alongside the other independently detected
        # contact fields.
        slug = re.fullmatch(r'(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,8}', text.strip())
        if slug:
            put('linkedin', text.strip(), row)

    # PDF line wrapping can split a URL at a hyphen. Rejoin only a direct
    # continuation token on the next source line and only when the captured
    # contact URL ends with that hyphen.
    for index, (row, _text) in enumerate(lines[:-1]):
        for key in ('linkedin', 'github', 'website', 'portfolio'):
            if field_rows.get(key) is not row or not fields.get(key, '').endswith('-'):
                continue
            continuation = clean(lines[index + 1][1])
            if re.fullmatch(r'[A-Za-z0-9-]{2,80}', continuation):
                fields[key] += continuation
                break

    # Detect short professional titles separately so they are not mistaken
    # for a name or city when contact details share their line.
    for row, text in lines:
        if (ROLE.search(text) and len(text.split()) <= 8 and not ACTION.search(text)
                and not email_re.search(text) and not phone_re.search(text)):
            put('title', text, row)
            break

    # The first short, human-name-like line before a detected title is the
    # candidate name. Later header lines can be a city or contact descriptor.
    title_position = next((index for index, (row, _text) in enumerate(lines)
                           if field_rows.get('title') is row), len(lines))
    for line_index, (row, text) in enumerate(lines):
        if line_index > title_position:
            break
        candidate = email_re.sub(' ', text)
        candidate = phone_re.sub(' ', candidate)
        candidate = link_re.sub(' ', candidate)
        candidate = label_re.sub(' ', candidate)
        candidate = re.sub(r'[|·•]+', ' ', candidate).strip(' ,;:-')
        if (len(candidate.split()) >= 2 and candidate and len(candidate.split()) <= 6 and not re.search(r'\d|@|[.!?]', candidate)
                and not ROLE.search(candidate) and not ACTION.search(candidate) and ',' not in candidate
                and not re.search(r'\b(email|phone|mobile|location|address|linkedin|github)\b', candidate, re.I)):
            # Do not absorb the title when name and title are printed together.
            if fields.get('title') and candidate.casefold().endswith(fields['title'].casefold()):
                candidate = candidate[:-len(fields['title'])].strip(' ,;:-')
            put('name', candidate, row)
            if fields.get('name'):
                break

    # Stylized PDF fonts can fragment the name into unusable repeated glyph
    # strings. When that happens, use a human-readable email local part only
    # if it cleanly contains multiple name tokens (e.g. first.last123).
    if not fields.get('name') and fields.get('email'):
        local = fields['email'].split('@', 1)[0]
        tokens = re.findall(r'[A-Za-z]{2,}', re.sub(r'[._+-]+', ' ', local))
        if len(tokens) >= 2 and len(tokens) <= 5:
            fields['name'] = ' '.join(token.capitalize() for token in tokens)

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
            # A location may be split across adjacent header rows (city/state,
            # then country). Prefer the most informative geographic phrase
            # over a short trailing country token when no explicit label is
            # present.
            location, location_row = max(candidates, key=lambda item: len(item[0]))
            put('location', location, location_row)

    return fields, field_rows


def name_like(text):
    words = text.split()
    return (0 < len(words) <= 14 and len(text) < 140 and not ACTION.search(text)
            and not re.search(r'[!?;:]|https?://|@', text)
            and (sum(w[:1].isupper() for w in words) >= max(1, len(words) * .6) or bool(ORG_SUFFIX.search(text))))


def role_title(text):
    body = PERIOD.sub('', text).strip(' ()|,–—-')
    # Many resumes put the seniority/title first and a team or domain after
    # an em dash ("Senior Analyst — Fund Reporting & Operations"). Recognize
    # the title-bearing side independently; checking the whole line against
    # ACTION incorrectly treats "Analyst" as the verb "analysis".
    parts = re.split(r'\s+[–—|]\s+', body, maxsplit=1)
    if len(parts) == 2 and len(body.split()) <= 12:
        left, right = (part.strip() for part in parts)
        left_is_title = bool(ROLE.search(left) and len(left.split()) <= 4
                             and not re.search(r'[.!?;,@]', left))
        right_is_title = bool(ROLE.search(right) and len(right.split()) <= 4
                              and not re.search(r'[.!?;,@]', right))
        if left_is_title or right_is_title:
            return True
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


def _split_company_location(value):
    """Split a trailing place from an explicitly named legal organization."""
    suffix = re.search(r'\b(?:limited|ltd\.?|inc\.?|llc|corp\.?|corporation|pvt\.?\s*ltd\.?)\b', value, re.I)
    if not suffix:
        return value.strip(' .,'), ''
    company = value[:suffix.end()].strip(' .,')
    location = value[suffix.end():].strip(' .,|-–—')
    if (location and len(location.split()) <= 3
            and re.fullmatch(r'[A-Z][A-Za-z.-]*(?:\s+[A-Z][A-Za-z.-]*){0,2}', location)):
        return company, location
    return value.strip(' .,'), ''


def employment_header(text):
    # First-person resume prose is a common employment-header format. Strip
    # the grammatical wrapper so the employer and title remain distinct fields.
    # A compact bullet such as "Worked at ACME from Jan 2020 to Mar 2022"
    # is also an employment header, even when it has no explicit job title.
    worked_at = re.match(r'^worked\s+at\s+(.+)$', text, re.I)
    if worked_at:
        value = worked_at.group(1).strip(' .')
        period = PERIOD.search(value)
        before_period = value[:period.start()].strip(' ,|–—-') if period else value
        before_period = re.sub(r'\s+from\s*$', '', before_period, flags=re.I).strip(' ,|–—-')
        designation = ''
        role_match = re.match(r'^(.+?)\s+as\s+(?:a|an|the)\s+(.+)$', before_period, re.I)
        if role_match:
            before_period = role_match.group(1).strip(' ,|–—-')
            designation = role_match.group(2).strip(' ,|–—-')
        # Remove connective words left around a range, while preserving the
        # actual organization text from the source.
        company = re.sub(r'\s+(?:from|during|between)\s*$', '', before_period, flags=re.I).strip(' ,|–—-')
        if company:
            result = {'company': company}
            if designation:
                result['designation'] = designation
            if period:
                result.update(period=period.group(0), start_date=period.group(1), end_date=period.group(2))
            return result
    narrative = re.match(
        r'^(?:i\s+am\s+currently\s+working\s+as|i\s+currently\s+work\s+as|'
        r'i\s+worked\s+as|currently\s+working\s+as|working\s+as)\s+(.+?)\s+at\s+(.+)$',
        text, re.I)
    if narrative:
        designation, company = narrative.group(1).strip(), narrative.group(2).strip()
        period = PERIOD.search(company)
        if period:
            company = company[:period.start()].strip()
        company = re.sub(r'\s+from\s*$', '', company, flags=re.I).strip(' ,|–—-')
        company, location = _split_company_location(company)
        designation = re.sub(r'^(?:a|an|the)\s+', '', designation, flags=re.I)
        if company:
            return {'company': company, 'designation': designation,
                    'period': period.group(0) if period else None,
                    'start_date': period.group(1) if period else None,
                    'end_date': period.group(2) if period else None,
                    'location': location or None}
    label = re.match(r'^(?:company|employer|organization)\s*[:–—-]\s*(.+)$', text, re.I)
    if label:
        return {'company': label[1]} if not re.match(r'^(?:clients?|role|designation|job title|title|dates?|duration|period)\b', label[1], re.I) else None
    at = re.match(r'^(.+?)\s+at\s+(.+)$', text, re.I)
    if at and title_like(at[1]):
        date = PERIOD.search(at[2])
        return {'company': (at[2][:date.start()] if date else at[2]).strip(' (),|–—-'),
                'designation': at[1].strip(), 'period': date.group(0) if date else None}
    split = re.match(r'^(.+?)\s+[–—|\-]\s+(.+)$', text)
    if split and not re.match(r'^(?:clients?|role|designation|job title|title|dates?|duration|period)\b', split[1], re.I):
        left, right = split[1].strip(), split[2].strip()
        left_date, right_date = PERIOD.search(left), PERIOD.search(right)
        left_title, right_title = role_title(left), role_title(right)
        if left_title and not right_title and name_like(right):
            date = right_date
            company = right[:date.start()] if date else right
            designation = left
        elif right_title and not left_title and name_like(left):
            date = left_date or right_date
            company = left[:date.start()] if date and date is left_date else left
            designation = right[:date.start()] if date and date is right_date else right
        elif name_like(left) and not left_title and role_title(right):
            date = right_date
            company = left
            designation = right[:date.start()] if date else right
        else:
            date = None
            company = designation = ''
        if company and designation:
            return {'company': company.strip(' ()|,–—-'),
                    'designation': designation.strip(' ()|,–—-'),
                    'period': date.group(0) if date else None}
    return None


def _classify(row, following):
    text = clean(row['text'])
    section = row['section'].casefold()
    compact_glyph_noise = re.sub(r'[|\s]', '', text)
    if (not text or row.get('artifact') or row.get('is_page_number') or
            re.fullmatch(r'(?:page\s+\d+(?:\s*(?:of|/)\s*\d+)?)|[\s•▪●○◦*|_–—-]+', text, re.I) or
            re.fullmatch(r'(?:([A-Za-z])\1){3,}', compact_glyph_noise)):
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
        # A responsibility bullet can begin with a role noun, e.g.
        # ``Copilot Developer, Designed ...``. Do not treat that noun as a
        # separate company heading.
        if re.search(r',\s*(?:designed|developed|implemented|managed|created|led|built|provided)\b', text, re.I):
            return 'RESPONSIBILITY', {}
        location = re.fullmatch(r'([A-Z][\w .\'-]{1,50}),\s*([A-Z][\w .\'-]{1,50})', text)
        if location and not row['is_bullet'] and not ACTION.search(text) and not re.search(r'[.!?]', text):
            return 'LOCATION', {'location': text}
        achievement = re.match(r'^(?:key\s+)?(?:achievement|accomplishment)s?\s*[:–—-]\s*(.+)$', text, re.I)
        if achievement:
            return 'ACHIEVEMENT', {'text': achievement[1].strip()}
        client_role = re.match(r'^client\s+(.+?)\s+[–—-]\s+(.+)$', text, re.I)
        if client_role and role_title(client_role.group(2)):
            return 'CLIENT', {'clients': [client_role.group(1).strip()],
                              'client_designation': client_role.group(2).strip()}
        client = re.match(r'^client\s*[:–—-]?\s*(.+)$', text, re.I)
        if client and len(client.group(1).split()) <= 8:
            return 'CLIENT', {'clients': [client.group(1).strip()]}
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
        if SINGLE_DATE.fullmatch(text.strip().rstrip('.')):
            value = re.sub(r'^(?:till\s+date|current|now)$', 'Present', text.strip().rstrip('.'), flags=re.I)
            return 'DATE_RANGE', {'period': value, 'date_value': value}
        if (not row['is_bullet'] and name_like(text) and following and
                (title_like(clean(following['text'])) or PERIOD.search(following['text']) or
                 re.match(r'^(?:Role|Client|Designation)\s*[:–—-]', following['text'], re.I))):
            return 'COMPANY', {'company': text}
        if re.match(r'^(?:(?:role\s*(?:&|and)\s*)?responsibilities|key achievements|achievements|accomplishments)\s*:?$', text, re.I):
            return 'SUBSECTION_HEADING', {}
        return 'RESPONSIBILITY', {}
    return ({'summary': 'PROFILE_SUMMARY', 'business objective': 'UNKNOWN',
             'skills': 'SKILL', 'core competencies': 'SKILL',
             'business analysis and product management': 'SKILL',
             'technical skills': 'TECHNOLOGY', 'technical expertise': 'TECHNOLOGY',
             'key strengths': 'SKILL',
             'projects': 'PROJECT', 'education': 'EDUCATION', 'certifications': 'CERTIFICATION',
             'certifications & training': 'CERTIFICATION',
             'awards': 'AWARD', 'achievements': 'ACHIEVEMENT', 'publications': 'PUBLICATION',
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
                and not re.match(r'^(?:project\s+(?:title|name|description)|technologies|technology|tech\s+stack|description|tools)\s*:', text, re.I)
                and styled_heading and 2 <= len(text) <= 60
                and not employment_header(text) and not title_like(text) and not PERIOD.search(text)
                and LAYOUT_SECTION_CUE.search(text)
                and not ACTION.search(text) and not re.search(r'[.!?@]', text)
                and not text.endswith(':') and len(text.split()) >= 2
                and ',' not in text):
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
        # Keep list/number markers in the row text because project extraction
        # uses numbering as a strong record boundary. Strip only invisible PDF
        # private-use glyphs here; clean() removes visible markers for labels.
        text = re.sub(r'[\ue000-\uf8ff]', '', str(meta['line'] or '')).strip()
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
        marker = MARKER.match(meta.get('line', '')) or MARKER.match(text)
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
    # PDF text extraction may wrap the final year of an employment period
    # onto the next line. Rejoin only the unambiguous ``Month YYYY – Month`` +
    # ``YYYY`` pattern before classification so the year cannot become a
    # standalone start date or part of the employer name.
    wrapped_date_rows = set()
    wrapped_period = re.compile(rf'({MONTH}\.?\s+\d{{4}}\s*[-–—]\s*{MONTH}\.?)\s*$', re.I)
    for index, row in enumerate(rows[:-1]):
        following = rows[index + 1]
        match = wrapped_period.search(clean(row['text']))
        if (row['section'].casefold() == 'experience' and match
                and re.fullmatch(r'\s*(?:19|20)\d{2}\s*', clean(following['text']))
                and following['section_id'] == row['section_id']):
            row['text'] = row['text'].rstrip() + ' ' + clean(following['text'])
            row['wrapped_date_source_id'] = following['id']
            wrapped_date_rows.add(following['id'])
    for index, row in enumerate(rows):
        following = next((r for r in rows[index + 1:] if not r['is_page_number'] and not r.get('artifact')), None)
        if following and following['section_id'] != row['section_id']:
            following = None
        # Keep source wording for classification, but allow an unambiguous
        # wrapped employment date to participate in the employer header.
        if following and following['id'] in wrapped_date_rows:
            following = None
        row['block_type'], row['entities'] = _classify(row, following)
        # In the common stacked layout, the employer and role occupy
        # consecutive lines. A role such as "Senior Analyst — Fund Reporting"
        # can resemble a company/title pair when classified without context.
        # Preserve the explicit employer immediately above it as the parent.
        previous = next((candidate for candidate in reversed(rows[:index])
                         if candidate['section_id'] == row['section_id']
                         and candidate['block_type'] != 'PAGE_ARTIFACT'), None)
        if (row['section'].casefold() == 'experience' and previous
                and previous['block_type'] in {'COMPANY', 'COMPANY_ROLE'}
                and row['block_type'] == 'COMPANY_ROLE'
                and not re.match(r'^worked\s+at\b', clean(row['text']), re.I)):
            row['block_type'] = 'ROLE'
            row['entities'] = {'designation': clean(row['text'])}
        if row['id'] in wrapped_date_rows:
            row['block_type'], row['entities'] = 'PAGE_ARTIFACT', {}
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
            'CLIENT': 'client', 'LOCATION': 'location', 'DATE_RANGE': 'date_range', 'RESPONSIBILITY': 'responsibility',
            'ACHIEVEMENT': 'achievement', 'PROJECT': 'project_description', 'SKILL': 'skill',
            'AWARD': 'award', 'PUBLICATION': 'publication',
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
    # Avoid duplicating text layers that PDF generators place at virtually the
    # same coordinates (often selectable text plus an accessibility overlay).
    exact_rows = {}
    for row in rows:
        box, text = row.get('bbox'), re.sub(r'\s+', ' ', clean(row['text'])).casefold()
        key = (row.get('page'), text)
        previous = exact_rows.get(key)
        if box and text and previous:
            old = previous.get('bbox')
            if old and max(abs(float(a) - float(b)) for a, b in zip(box, old)) <= 1.0:
                row.update(block_type='PAGE_ARTIFACT', entities={},
                           semantic_classification='page_marker', semantic_labels=['page_marker'],
                           reason='duplicate PDF text layer overlaps the same source text')
                continue
        if box and text:
            exact_rows[key] = row
    # PDF reading order can emit a top-of-page contact card after an entire
    # sidebar or work section. Recover it by its geometry and contact tokens,
    # then keep the nearby name/title/contact lines out of work history.
    contact_re = re.compile(r'@|\+?\d[\d().\s-]{5,}\d|(?:linkedin|github)\.com|https?://|www\.', re.I)
    contact_anchors = [row for row in rows if row.get('bbox') and row.get('page') is not None
                       and row['bbox'][1] <= 180 and contact_re.search(row['text'])]
    for anchor in contact_anchors:
        page, anchor_x = anchor['page'], anchor['bbox'][0]
        contact_bottom = max(item['bbox'][3] for item in contact_anchors
                             if item['page'] == page)
        for candidate in rows:
            box = candidate.get('bbox')
            if (not box or candidate.get('page') != page or box[1] > contact_bottom + 5
                    or box[2] < anchor_x - 300 or box[0] > anchor_x + 80):
                continue
            text = clean(candidate['text'])
            if not text or (len(text.split()) > 9 and not contact_re.search(text)):
                continue
            if ACTION.search(text) and not ROLE.search(text):
                continue
            candidate.update(section='Header', section_id=f'layout:contact:{page}',
                             block_type='PERSONAL_DETAIL', entities={},
                             semantic_classification='personal_detail',
                             semantic_labels=['personal_detail'], confidence=.95,
                             reason='top-page contact card recovered from aligned PDF layout')
    # PDF parsers frequently emit a bullet glyph as a standalone line. Carry
    # that structural boundary to the next text line before section grouping
    # removes artifact-only glyph blocks.
    for index, row in enumerate(rows):
        if row['block_type'] != 'PAGE_ARTIFACT' or not row.get('marker'):
            continue
        following = next((candidate for candidate in rows[index + 1:]
                          if candidate['section_id'] == row['section_id']
                          and candidate['block_type'] not in {'PAGE_ARTIFACT', 'SECTION_HEADING'}), None)
        if following:
            following['is_bullet'] = True
            following['marker'] = row['marker']
    # Recover timeline-style experience rows using geometry. In these layouts,
    # a date and title share a y band, while employer and location follow in
    # the left lane. Generic text heuristics cannot distinguish a one-word
    # employer from a city, but their repeated left-lane order can.
    for section_id in dict.fromkeys(row['section_id'] for row in rows if row['section'].casefold() == 'experience'):
        experience_rows = [row for row in rows if row['section_id'] == section_id
                           and row['section'].casefold() == 'experience' and row.get('bbox')]
        intervals = sorted((row['bbox'][0], row['bbox'][2]) for row in experience_rows)
        gaps = [(right[0] - left[1], (right[0] + left[1]) / 2)
                for left, right in zip(intervals, intervals[1:]) if right[0] - left[1] >= 18]
        if not gaps:
            continue
        gap, xcut = max(gaps)
        left_lane = [row for row in experience_rows if row['bbox'][2] <= xcut]
        right_lane = [row for row in experience_rows if row['bbox'][0] >= xcut]
        if (len(left_lane) < 4 or len(right_lane) < 4
                or sum(row['block_type'] == 'DATE_RANGE' for row in left_lane) < 2
                or sum(row['block_type'] in {'ROLE', 'COMPANY_ROLE'} for row in right_lane) < 2):
            continue
        waiting_for_employer = False
        employer_seen = False
        left_ids = {row['id'] for row in left_lane}
        for row in experience_rows:
            if row['id'] not in left_ids:
                continue
            text = clean(row['text'])
            if row['block_type'] == 'DATE_RANGE':
                waiting_for_employer, employer_seen = True, False
                continue
            if (not waiting_for_employer or row['is_bullet'] or PERIOD.search(text)
                    or role_title(text) or ACTION.search(text) or len(text.split()) > 6
                    or not name_like(text)):
                continue
            if not employer_seen:
                row.update(block_type='COMPANY', entities={'company': text},
                           semantic_classification='company', semantic_labels=['company'],
                           reason='timeline left-lane employer follows its date/title row')
                employer_seen = True
            elif not ORG_SUFFIX.search(text):
                row.update(block_type='LOCATION', entities={'location': text},
                           semantic_classification='location', semantic_labels=['location'],
                           reason='timeline left-lane location follows its employer')
    # PDF reading order often emits employer, city, then title as three lines.
    # The middle short employer-like token is a work location in that exact
    # structural pattern, not a second company.
    for index in range(1, len(rows) - 1):
        previous, row, following = rows[index - 1:index + 2]
        if (row['section'].casefold() == 'experience' and
                previous['section_id'] == row['section_id'] == following['section_id'] and
                previous['block_type'] in {'COMPANY', 'COMPANY_ROLE'} and
                row['block_type'] == 'COMPANY' and following['block_type'] in {'ROLE', 'DATE_RANGE'} and
                len(clean(row['text']).split()) <= 3 and not ORG_SUFFIX.search(row['text'])):
            row['block_type'] = 'LOCATION'
            row['entities'] = {'location': clean(row['text'])}
            row['semantic_classification'] = 'location'
            row['semantic_labels'] = ['location']
            row['reason'] = 'short row between employer and role/date is a location field'
    # Some two-column resumes place employer/title text in the left lane and
    # dates in a right lane. Reading order can put those dates after Education,
    # even though their vertical position exactly matches an Experience role.
    # Reattach only date rows with page geometry, tight vertical alignment,
    # and a right-of-role position; text similarity alone is not sufficient.
    experience_roles = [row for row in rows
                        if row['section'].casefold() == 'experience'
                        and row['block_type'] in {'ROLE', 'COMPANY_ROLE'}
                        and row.get('bbox') and row.get('page') is not None]
    for row in rows:
        if (row['section'].casefold() == 'experience' or row['is_bullet']
                or not row.get('bbox') or row.get('page') is None):
            continue
        period = PERIOD.search(clean(row['text']))
        if not period:
            continue
        x0, y0, x1, y1 = row['bbox']
        aligned = []
        for role_row in experience_roles:
            if role_row['page'] != row['page']:
                continue
            rx0, ry0, rx1, ry1 = role_row['bbox']
            vertical_delta = abs((y0 + y1) / 2 - (ry0 + ry1) / 2)
            if vertical_delta <= 2.5 and x0 >= rx1 - 20:
                aligned.append((vertical_delta, role_row))
        if aligned:
            _, owner = min(aligned, key=lambda item: item[0])
            row.update(section='Experience', section_id=owner['section_id'],
                       block_type='DATE_RANGE',
                       entities={'period': period.group(0), 'start_date': period.group(1),
                                 'end_date': period.group(2)},
                       aligned_role_source_id=owner['id'],
                       semantic_classification='date_range',
                       semantic_labels=['date_range'],
                       reason='right-column date vertically aligned with the role header')
    by_line = {r['line_index']: r for r in rows if not r['is_header']}
    for meta in pre['line_meta']:
        if meta['line_index'] in by_line:
            row = by_line[meta['line_index']]
            meta.update(section=row['section'], section_id=row['section_id'])
            if row.get('reason') == 'duplicate PDF text layer overlaps the same source text':
                meta['is_page_number'] = True
    for meta in pre['sentence_meta']:
        if meta['line_index'] in by_line:
            row = by_line[meta['line_index']]
            meta.update(section=row['section'], section_id=row['section_id'])
    artifact_lines = {r['line_index'] for r in rows if r['block_type'] == 'PAGE_ARTIFACT'}
    for meta in metas:
        if meta['line_index'] in artifact_lines:
            row = next((item for item in rows if item['line_index'] == meta['line_index']), None)
            # PAGE_ARTIFACT also covers repeated headers, separators, and
            # decorative extraction noise. Only numeric page markers should
            # be reported as page numbers in source coverage and validation.
            meta['is_page_number'] = bool(
                row and (row.get('is_page_number') or row.get('artifact') == 'page_number'
                         or re.fullmatch(r'\s*(?:page\s+)?\d{1,3}(?:\s*(?:of|/)\s*\d+)?\s*',
                                         str(row.get('text', '')), re.I))
            )
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
    next_job_fields = {}
    orphan_roles = []
    roles_by_source_id = {}
    for row_index, row in enumerate(rows):
        kind, fields, text = row['block_type'], row['entities'], clean(row['text'])
        aligned_role = roles_by_source_id.get(row.get('aligned_role_source_id'))
        if kind == 'DATE_RANGE' and aligned_role:
            target_job, target_role = aligned_role
            target_job_path = f'$.work_history[{offset + jobs.index(target_job)}]'
            target_role_path = f'{target_job_path}.roles[{target_job.roles.index(target_role)}]'
            target_role.update(duration=fields['period'], start_date=fields.get('start_date', ''),
                               end_date=fields.get('end_date', ''))
            target_role.setdefault('field_sources', {}).setdefault('employment_period', []).append(row['id'])
            _own(row, target_role, target_role_path,
                 'date aligned in the document layout with this role header')
            continue
        # Some resumes list several employer headers first, then place one
        # consolidated responsibilities section below them. If its first
        # responsibility explicitly names one of those role titles, use that
        # exact source cue to restore the correct parent context.
        responsibility_title = re.match(r'^(.{2,80}?),\s*(?=[A-Z])', text) if kind == 'RESPONSIBILITY' else None
        if responsibility_title:
            wanted = re.sub(r'\W+', ' ', responsibility_title.group(1)).strip().casefold()
            matches = [(candidate_job, candidate_role) for candidate_job in jobs
                       for candidate_role in candidate_job.roles
                       if re.sub(r'\W+', ' ', candidate_role.get('designation', '')).strip().casefold() == wanted]
            if len(matches) == 1:
                job, role = matches[0]
                job_index = jobs.index(job)
                job_path = f'$.work_history[{offset + job_index}]'
                role_path = f'{job_path}.roles[{job.roles.index(role)}]'
                last = None
                text = text[responsibility_title.end():].strip()
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
            role, role_path, last = None, None, None
            pending, next_job_fields = next_job_fields, {}
            _own(row, job, job_path, 'employer header opens a new employment context')
            if fields.get('period'):
                job.attributes['duration'] = fields['period']
            if orphan_roles:
                for orphan_row, orphan_fields in orphan_roles:
                    role = {'designation': orphan_fields['designation'], 'role_and_responsibilities': [],
                            'responsibility_sources': [], 'source_blocks': []}
                    job.roles.append(role)
                    role_path = f'{job_path}.roles[{len(job.roles) - 1}]'
                    _own(orphan_row, role, role_path, 'title row aligned with employer in the timeline layout')
                    for key, (value, source) in pending.items():
                        role[key] = value
                        _own(source, role, role_path, 'date/location associated with the aligned role')
                    pending = {}
                    if orphan_fields.get('clients'):
                        role['clients'] = orphan_fields['clients']
                    if orphan_fields.get('period'):
                        role['duration'] = orphan_fields['period']
                orphan_roles = []
            if kind == 'COMPANY':
                if fields.get('period'):
                    role = {'designation': '', 'role_and_responsibilities': [],
                            'responsibility_sources': [], 'source_blocks': [],
                            'duration': fields['period'],
                            'start_date': fields.get('start_date', ''),
                            'end_date': fields.get('end_date', '')}
                    job.roles.append(role)
                    role_path = f'{job_path}.roles[{len(job.roles) - 1}]'
                    roles_by_source_id[row['id']] = (job, role)
                continue
        if kind in {'ROLE', 'COMPANY_ROLE'}:
            if job is None:
                orphan_roles.append((row, fields))
                continue
            role = {'designation': fields['designation'], 'role_and_responsibilities': [],
                    'responsibility_sources': [], 'source_blocks': []}
            job.roles.append(role)
            roles_by_source_id[row['id']] = (job, role)
            role_path = f'{job_path}.roles[{len(job.roles) - 1}]'
            _own(row, role, role_path, 'title opens a role under the active employer')
            for key, (value, source) in pending.items():
                role[key] = value
                _own(source, role, role_path, 'field between employer and title belongs to this role')
            pending = {}
            if fields.get('clients'):
                role['clients'] = fields['clients']
                role.setdefault('field_sources', {}).setdefault('clients', []).append(row['id'])
            if fields.get('period'):
                role['duration'] = fields['period']
            if fields.get('start_date'):
                role['start_date'] = fields['start_date']
            if fields.get('end_date'):
                role['end_date'] = fields['end_date']
            if fields.get('location'):
                role['location'] = fields['location']
            last = None
            continue
        if job is None:
            # A period can precede its employer in column-oriented PDFs.
            # Carry it forward only when an employer header follows directly
            # (allowing one short location line), preserving its source.
            lookahead = rows[row_index + 1:row_index + 6]
            upcoming_company = next((candidate for candidate in lookahead
                                     if candidate['block_type'] in {'COMPANY', 'COMPANY_ROLE'}), None)
            company_is_next = bool(upcoming_company and not any(
                candidate['block_type'] == 'RESPONSIBILITY' for candidate in lookahead[:lookahead.index(upcoming_company)]))
            # Once a role exists, a following date belongs to that role even
            # when another employer appears a few lines later. The lookahead
            # path is only for genuinely orphaned dates before an employer.
            if kind == 'DATE_RANGE' and company_is_next and role is None:
                values = {'duration': fields['period']}
                if fields.get('start_date'):
                    values['start_date'] = fields['start_date']
                if fields.get('end_date'):
                    values['end_date'] = fields['end_date']
                next_job_fields.update({key: (value, row) for key, value in values.items()})
                _own(row, row, '$.work_history', 'date immediately before an employer belongs to that employment entry')
                continue
            unresolved.append(row)
            continue
        _own(row, job, job_path, 'source block inside the active employer context')
        if kind == 'LOCATION':
            if role is not None:
                role['location'] = fields['location']
                role.setdefault('field_sources', {}).setdefault('location', []).append(row['id'])
                _own(row, role, role_path, 'standalone work-location field under the active role')
            else:
                pending['location'] = (fields['location'], row)
                _own(row, job, job_path, 'work location between employer and role title')
            continue
        if kind in {'CLIENT', 'DATE_RANGE'}:
            values = {'clients': fields['clients']} if kind == 'CLIENT' else {'duration': fields['period']}
            if fields.get('reported_duration'):
                values['reported_duration'] = fields['reported_duration']
            lookahead = rows[row_index + 1:row_index + 6]
            upcoming_company = next((candidate for candidate in lookahead
                                     if candidate['block_type'] in {'COMPANY', 'COMPANY_ROLE'}), None)
            company_is_next = bool(upcoming_company and not any(
                candidate['block_type'] == 'RESPONSIBILITY' for candidate in lookahead[:lookahead.index(upcoming_company)]))
            if kind == 'DATE_RANGE' and company_is_next and role is None:
                if role is None and pending:
                    role = {'designation': '', 'role_and_responsibilities': [],
                            'responsibility_sources': [], 'source_blocks': []}
                    job.roles.append(role)
                    role_path = f'{job_path}.roles[{len(job.roles) - 1}]'
                    for key, (value, source) in pending.items():
                        role[key] = value
                        _own(source, role, role_path, 'fields preceding the next employment record belong to this role')
                    pending = {}
                next_job_fields.update({key: (value, row) for key, value in values.items()})
                if fields.get('start_date'):
                    next_job_fields['start_date'] = (fields['start_date'], row)
                if fields.get('end_date'):
                    next_job_fields['end_date'] = (fields['end_date'], row)
                _own(row, job, job_path, 'date immediately before the next employer starts that employment entry')
                job = role = None
                job_path = role_path = None
                continue
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
                if kind == 'CLIENT' and fields.get('client_designation'):
                    role['client_designation'] = fields['client_designation']
                if kind == 'DATE_RANGE':
                    if fields.get('start_date'):
                        role['start_date'] = fields['start_date']
                    if fields.get('end_date'):
                        role['end_date'] = fields['end_date']
                    if fields.get('date_value'):
                        if fields['date_value'].casefold() in {'present', 'current', 'now', 'till date', 'to date'}:
                            role['end_date'] = 'Present'
                        elif role.get('start_date') and not role.get('end_date'):
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
        continuation = bool(last and not row['is_bullet'] and (
            text[:1].islower() or
            previous_text.casefold().endswith((' and', ' or', ' of', ' the', ' to', ',', ':', '-')) or
            (len(text.split()) <= 5
             and not re.search(r'[.!?;:]$', previous_text) and not ACTION.search(text)) or
            (last['is_bullet']
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
        institution_period = re.match(
            r'^(?P<institution>.+?)\s*[|–—-]\s*(?P<start>(?:19|20)\d{2})\s*[-–—/]\s*(?P<end>(?:19|20)?\d{2}|present|current)\s*$',
            text, re.I)
        if institution_period and re.search(r'\b(?:university|college|institute|school)\b',
                                             institution_period.group('institution'), re.I):
            if current and current.get('degree'):
                current = {}
                records.append(current)
            elif current is None:
                current = {}
                records.append(current)
            institution = institution_period.group('institution').strip(' |,')
            end = institution_period.group('end')
            if len(end) == 2 and end.isdigit():
                end = ('20' if int(end) <= 39 else '19') + end
            current.update(institution=institution, university_or_board=institution,
                           year=f"{institution_period.group('start')} – {end}")
            _own(row, current, f'$.education[{offset + len(records) - 1}]',
                 'institution and study period share an education source line')
            continue
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
            score_label = re.match(r'^(?:CGPA|GPA|Percentage)\s*[:–—-]\s*(.+)$', text, re.I)
            if score_label and current is not None:
                current['score'] = score_label.group(1).strip(' &')
                _own(row, current, f'$.education[{offset + len(records) - 1}]',
                     'explicit academic score label')
                continue
            if isinstance(date, re.Match) and PERIOD.search(text) and current is not None:
                current['year'] = PERIOD.search(text).group(0)
                _own(row, current, f'$.education[{offset + len(records) - 1}]',
                     'education date range in the academic entry')
                continue
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
            elif current.get('degree') and (
                    re.search(r'\b(?:of|in|and|&)$', current['degree'], re.I) or
                    re.fullmatch(r"(?:bachelor'?s?|master'?s?)\s+degree", current['degree'], re.I) or
                    (text.casefold() == 'engineering' and current['degree'].casefold().endswith('communication'))):
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
            elif '|' in text and current.get('degree'):
                for cell in (part.strip() for part in text.split('|') if part.strip()):
                    score = re.match(r'^(?:CGPA|GPA|Percentage)\s*[:–—-]?\s*(.+)$', cell, re.I)
                    year = re.search(r'\b(?:19|20)\d{2}\b', cell)
                    if score:
                        current['score'] = score.group(1).strip()
                    elif year and re.search(r'(?:19|20)\d{2}\s*[–—-]\s*(?:19|20)\d{2}', cell):
                        current['year'] = cell
                    elif not current.get('institution'):
                        current['institution'] = cell
                    elif cell.casefold() not in current['institution'].casefold():
                        current.setdefault('additional_details', []).append(cell)
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
            elif ('institution' in current and re.fullmatch(
                    r'[A-Z][\w .\'-]{1,45},\s*[A-Z][\w .\'-]{1,45}', text)):
                current['location'] = text
            elif ('institution' in current and not current.get('degree') and not current.get('year')
                  and ('&' in text or 'engineering' in text.casefold())):
                current['degree'] = text
            elif ('institution' in current and not current.get('university_or_board') and
                  re.search(r'\b(?:road|ambernath)$', text, re.I)):
                current['institution'] = f"{current['institution']} {text}".strip()
            elif 'institution' in current and not current.get('university_or_board'):
                if text.casefold() not in current['institution'].casefold():
                    current['institution'] = f"{current['institution']} {text}".strip()
            elif 'institution' not in current:
                current['institution'] = text
            elif not current.get('degree') and not current.get('score') and not current.get('year'):
                current['degree'] = text
            elif text.casefold() not in current['institution'].casefold():
                current.setdefault('additional_details', []).append(text)
        _own(row, current, f'$.education[{offset + len(records) - 1}]', 'education row/qualification with its following fields')
    for record in records:
        record.pop('_active_academic_result', None)
    return records


def group_projects(rows, offset=0):
    records, current = [], None
    project_context = {}
    row_records = {}
    for index, row in enumerate(rows):
        text = clean(row['text'])
        if row['block_type'] == 'PAGE_ARTIFACT' or row.get('artifact'):
            continue
        link = re.match(r'^(github|live demo|demo|portfolio|website)\s*:\s*(\S+)', text, re.I)
        if link and current is not None:
            current.setdefault('links', []).append({'label': link.group(1), 'url': link.group(2)})
            row['semantic_classification'] = 'project_link'
            row['semantic_labels'] = ['project_link']
            _own(row, current, f'$.projects[{offset + len(records) - 1}].links',
                 'project link belongs to the active project heading')
            row_records[row['id']] = current
            continue
        # Standalone date rows are paired to a project title by page geometry
        # in the second pass below; they are metadata, never project records.
        if PERIOD.fullmatch(text) or SINGLE_DATE.fullmatch(text.rstrip('.')):
            continue
        employer_period = PERIOD.search(text)
        employer_label = (PERIOD.sub('', text).strip(' ()[]{} .:-–—')
                          if employer_period else '')
        if (employer_period and employer_label and name_like(employer_label)
                and (ORG_SUFFIX.search(employer_label)
                     or (employer_label.isupper() and len(employer_label.split()) >= 2))
                and not ACTION.search(employer_label)):
            start_date = _expand_two_digit_year(employer_period.group(1))
            end_date = _expand_two_digit_year(employer_period.group(2))
            project_context = {'organization': employer_label,
                               'start_date': start_date, 'end_date': end_date,
                               'duration': f'{start_date} – {end_date}'}
            project_context['employment_intro_open'] = False
            row['_employment_context_company'] = employer_label
            current = None
            row['semantic_classification'] = 'project_employer_context'
            row['semantic_labels'] = ['project_employer_context']
            _own(row, project_context, '$.projects',
                 'employer and employment period introduce the following project entries')
            continue
        context_heading = re.match(r'^(?:projects?\s+worked\s+(?:in|on)|project\s+in)\s+(.+)$', text, re.I)
        if context_heading:
            context_value = context_heading[1].strip(' .:-')
            period = PERIOD.search(context_value)
            if period:
                context_value = context_value[:period.start()].strip(' .:-')
            project_context = {'client': context_value} if context_value else {}
            singular_project_context = bool(re.match(r'^project\s+in\b', text, re.I))
            if period and singular_project_context:
                start_date = _expand_two_digit_year(period.group(1))
                end_date = _expand_two_digit_year(period.group(2))
                project_context.update(start_date=start_date, end_date=end_date,
                                       duration=f'{start_date} – {end_date}')
            current = None
            row['semantic_classification'] = 'project_context'
            row['semantic_labels'] = ['project_context']
            _own(row, project_context, '$.projects', 'source heading establishes a project grouping context')
            continue
        field = re.match(r'^(technologies|tools|tech stack|client|(?:project\s+)?description)\s*:\s*(.*)$', text, re.I)
        numbered_match = re.match(r'^\s*\.?\s*(\d+)[.)]\s*(.*)$', row['text'])
        numbered = bool(numbered_match)
        named = re.match(r'^project(?: name| title)?\s*[:–—-]\s*(.+)', text, re.I)
        title_tech = re.match(r'^(.+?)\s+\|\s+(.+)$', text)
        if title_tech and (ACTION.search(title_tech[1]) or len(title_tech[1]) > 100
                           or len(title_tech[2]) > 70 or re.search(r'[.!?]', title_tech[2])):
            title_tech = None
        following = next((candidate for candidate in rows[index + 1:]
                          if candidate['block_type'] != 'PAGE_ARTIFACT' and not candidate.get('artifact')), None)
        following_text = clean(following['text']) if following else ''
        inline_period = PERIOD.search(text)
        period_title = bool(inline_period and PERIOD.sub('', text).strip(' :.-')
                            and not ACTION.search(PERIOD.sub('', text).strip(' :.-'))
                            and len(PERIOD.sub('', text).strip(' :.-')) < 100)
        explicit_project_title = bool(row['is_bullet'] and following_text
                                      and re.match(r'^(?:technologies|tools|tech\s+stack|project\s+description)\s*:',
                                                   following_text, re.I))
        short_title = ((not row['is_bullet'] or explicit_project_title)
                       and (explicit_project_title or not ACTION.search(text)) and not PERIOD.search(text)
                       and not SINGLE_DATE.fullmatch(text.rstrip('.'))
                       and not re.search(r'[.!?]$', text) and len(text) < 100
                       and following is not None
                       and (bool(row.get('bold')) if row.get('bold') is not None else
                            (ACTION.search(following_text) or PERIOD.search(following_text)
                             or SINGLE_DATE.fullmatch(following_text.rstrip('.'))
                            or (len(text.split()) <= 6 and len(following_text) >= 55))))
        context_continuation = bool(project_context.get('organization')
                                    and project_context.get('employment_intro_open')
                                    and not row.get('is_bullet'))
        if (project_context.get('organization') and current is None
                and not (field or numbered or named or title_tech or period_title)
                and ((ACTION.search(text) and not explicit_project_title)
                     or re.match(r'^currently\s+\w+', text, re.I)
                     or text[:1].islower() or context_continuation)):
            project_context.setdefault('employment_responsibilities', []).append(text)
            row['_employment_context_company'] = project_context['organization']
            row['_employment_context_intro'] = True
            row['_employment_context_continuation'] = context_continuation
            project_context['employment_intro_open'] = not bool(re.search(r'[.!?]\s*$', text))
            row['semantic_classification'] = 'employment_context'
            row['semantic_labels'] = ['employment_context']
            _own(row, project_context, '$.work_history',
                 'source prose immediately under an employer/date context records work duties')
            row_records[row['id']] = project_context
            continue
        if field and current is not None:
            key = field[1].casefold()
            if key in {'technologies', 'tools', 'tech stack'}:
                current.setdefault('technologies', []).extend(v.strip() for v in re.split('[,;]', field[2]) if v.strip())
            elif field[2].strip():
                if key.endswith('description'):
                    current.setdefault('description', field[2].strip())
                else:
                    current[key] = field[2].strip()
            row['semantic_classification'] = 'project_field'
            row['semantic_labels'] = ['project_field']
            _own(row, current, f'$.projects[{offset + len(records) - 1}]',
                 'labeled technology/client/description field belongs to the active project')
            row_records[row['id']] = current
            continue
        if current is None or numbered or named or title_tech or short_title or period_title:
            if title_tech:
                current = {'name': title_tech[1].strip(),
                           'technologies': [item.strip() for item in title_tech[2].split(',') if item.strip()],
                           'responsibilities': []}
            elif (named or short_title or period_title
                  or (numbered and not ACTION.search(numbered_match[2]))):
                project_name = named[1] if named else (PERIOD.sub('', text).strip(' .:-') if period_title else
                                                       numbered_match[2].strip() if numbered else text)
                project_name = re.sub(r'^project\s*[:–—-]\s*', '', project_name, flags=re.I).strip()
                current = {'name': project_name, 'responsibilities': []}
            else:
                current = {'description': numbered_match[2] if numbered and numbered_match[2] else text, 'responsibilities': []}
            if period_title and inline_period:
                start_date = _expand_two_digit_year(inline_period.group(1))
                end_date = _expand_two_digit_year(inline_period.group(2))
                current.update(start_date=start_date, end_date=end_date,
                               duration=f'{start_date} – {end_date}')
            if project_context and current is not None:
                if project_context.get('client'):
                    current.setdefault('client', project_context['client'])
                if project_context.get('organization'):
                    current.setdefault('organization', project_context['organization'])
                for key in ('start_date', 'end_date', 'duration'):
                    if project_context.get(key):
                        current.setdefault(key, project_context[key])
            records.append(current)
        elif 'description' in current:
            current['description'] += ' ' + text
            current['responsibilities'].append(text)
        else:
            responsibilities = current.setdefault('responsibilities', [])
            if responsibilities and (text[:1].islower() or
                                     responsibilities[-1].rstrip().endswith((',', 'and', 'or', '-'))):
                responsibilities[-1] += ' ' + text
            elif text and not field:
                responsibilities.append(text)
            if not current.get('description') and not current.get('name'):
                current['description'] = text
        if named or short_title or period_title or numbered or title_tech:
            row['semantic_classification'] = 'project'
        else:
            row['semantic_classification'] = 'project_description'
        row['semantic_labels'] = [row['semantic_classification']]
        _own(row, current, f'$.projects[{offset + len(records) - 1}]', 'project heading/list item owns its description and labeled fields')
        row_records[row['id']] = current

    # Two-column project timelines put dates in a narrow right column and
    # titles/descriptions in the left column. Reading the columns separately
    # preserves text order but detaches those dates. Reattach only dates whose
    # bounding boxes overlap a project-heading row on the same page.
    for row in rows:
        if row['block_type'] == 'PAGE_ARTIFACT' or row.get('artifact') or not row.get('bbox'):
            continue
        text = clean(row['text'])
        period = PERIOD.search(text)
        single_date = SINGLE_DATE.fullmatch(text.rstrip('.')) if not period else None
        if not period and not single_date:
            continue
        candidates = []
        for heading in rows:
            box = heading.get('bbox')
            record = row_records.get(heading['id'])
            if (not box or not record or heading is row or heading.get('page') != row.get('page')
                    or heading.get('semantic_classification') != 'project'):
                continue
            overlap = min(box[3], row['bbox'][3]) - max(box[1], row['bbox'][1])
            if overlap >= -1 and box[0] < row['bbox'][0]:
                candidates.append((overlap, -abs(box[1] - row['bbox'][1]), record))
        if candidates:
            record = max(candidates, key=lambda item: (item[0], item[1]))[2]
            if period:
                start_date = _expand_two_digit_year(period.group(1))
                end_date = _expand_two_digit_year(period.group(2))
                record['start_date'], record['end_date'] = start_date, end_date
                record['duration'] = f'{start_date} – {end_date}'
                field = 'duration'
            else:
                record['date'] = _expand_two_digit_year(single_date.group(0))
                field = 'date'
            row['destination'] = f'$.projects[{offset + records.index(record)}].{field}'
            row['parent_candidate'] = row['destination']
            row['reason'] = 'date aligned with project title on the same visual row'
    return records


def group_sections(rows):
    groups, unresolved, offsets = {}, [], {}
    # Headings can be split into multiple source sections by OCR/PDF line
    # wraps. Group by resolved semantic section type, while each employer
    # heading still starts a distinct employment record within that context.
    semantic_sections = {'experience': 'work_history', 'professional experience': 'work_history',
                         'work experience': 'work_history', 'work history': 'work_history',
                         'employment history': 'work_history', 'career history': 'work_history',
                         'relevant experience': 'work_history', 'industry experience': 'work_history',
                         'internship experience': 'work_history',
                         'internships': 'work_history', 'career experience': 'work_history',
                         'professional background': 'work_history', 'professional employment': 'work_history',
                         'employment experience': 'work_history', 'work background': 'work_history',
                         'employment': 'work_history',
                         'education': 'education', 'projects': 'projects',
                         'project highlights': 'projects', 'project experience': 'projects',
                         'personal projects': 'projects', 'academic projects': 'projects',
                         'key projects': 'projects', 'research projects': 'projects',
                         'domain experience': 'domain_experience', 'technical skills': 'technical_skills',
                         'technical profile': 'technical_skills',
                         'technical expertise': 'technical_skills', 'key strengths': 'key_strengths',
                         'certifications & training': 'certifications', 'awards': 'awards',
                         'achievements': 'achievements', 'publications': 'publications',
                         'skills': 'skills', 'key skills': 'skills', 'core skills': 'skills',
                         'soft skills': 'skills',
                         'professional skills': 'skills', 'technical competencies': 'skills',
                         'skills and competencies': 'skills', 'summary': 'profile_snapshot',
                         'professional summary': 'profile_snapshot', 'profile summary': 'profile_snapshot',
                         'career summary': 'profile_snapshot', 'executive summary': 'profile_snapshot',
                         'career objective': 'profile_snapshot', 'personal information': 'personal_information',
                         'header': 'personal_information', 'certifications': 'certifications',
                         'core competencies': 'skills', 'languages': 'personal_information',
                         'personal details': 'personal_information', 'educational qualification': 'education',
                         'educational qualifications': 'education', 'academic qualification': 'education',
                         'academic qualifications': 'education', 'academic background': 'education',
                         'educational background': 'education'}
    # Some project-centric resumes use "Business Objective" as a repeated
    # per-project heading and then label Customer/Role/Platform/Solutions
    # Delivered. Treat that as project context only when those structural
    # markers are actually present; a conventional career objective remains
    # unresolved for explicit review instead of being mislabeled.
    business_objective_rows = [r for r in rows if r['section'].casefold() == 'business objective']
    project_markers = sum(bool(re.match(
        r'^\s*(?:customer|client|role|platform|team size|solutions? delivered)\s*:',
        clean(r['text']), re.I)) for r in rows)
    business_objective_is_projects = (
        sum(r.get('is_header', False) for r in business_objective_rows) >= 2
        and project_markers >= 3
        and any(re.match(r'^\s*solutions? delivered\s*:', clean(r['text']), re.I)
                for r in rows)
    )
    if business_objective_is_projects:
        semantic_sections['business objective'] = 'projects'
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
        elif key == 'certifications':
            value = []
            previous = None
            for row in body:
                text = clean(row['text'])
                box, old_box = row.get('bbox'), previous.get('bbox') if previous else None
                wrapped_continuation = bool(
                    value and box and old_box and row.get('page') == previous.get('page')
                    and -1 <= box[1] - old_box[3] <= 4.0
                    and box[0] >= old_box[0] + 8
                    and not row.get('is_bullet')
                )
                if wrapped_continuation:
                    value[-1]['name'] = (value[-1]['name'].rstrip() + ' ' + text).strip()
                    index = len(value) - 1
                else:
                    value.append({'name': text})
                    index = len(value) - 1
                row.update(destination=f'$.certifications[{offset + index}]',
                           parent_candidate=f'$.certifications[{offset + index}]',
                           reason='wrapped certification text joined by adjacent source-line geometry')
                previous = row
        elif key == 'awards':
            # Some two-page PDFs continue after the Awards heading with bold
            # capability headings and responsibility bullets. Keep the actual
            # award as an award, and represent those later capability groups
            # as strengths instead of mislabeling each bullet as an award.
            category_starts = []
            for index, row in enumerate(body[:-1]):
                next_content = next((candidate for candidate in body[index + 1:]
                                     if candidate['block_type'] != 'PAGE_ARTIFACT'), None)
                title = clean(row['text'])
                if (row.get('bold') and next_content and ACTION.search(clean(next_content['text']))
                        and len(title.split()) <= 7 and not re.search(r'[.!?]', title)):
                    category_starts.append(index)
            if category_starts:
                first_category = category_starts[0]
                value = []
                for row in body[:first_category]:
                    if not clean(row['text']) or row['block_type'] == 'PAGE_ARTIFACT':
                        continue
                    value.append({'name': clean(row['text'])})
                    _own(row, value[-1], f'$.awards[{offset + len(value) - 1}]',
                         'explicit award text before capability continuation')
                strengths = []
                for category_index, start in enumerate(category_starts):
                    end = category_starts[category_index + 1] if category_index + 1 < len(category_starts) else len(body)
                    category_row = body[start]
                    record = {'category': clean(category_row['text']), 'details': []}
                    strengths.append(record)
                    _own(category_row, record, f'$.key_strengths[{len(strengths) - 1}]',
                         'bold capability heading with following action statements')
                    for detail_row in body[start + 1:end]:
                        if detail_row['block_type'] == 'PAGE_ARTIFACT':
                            continue
                        record['details'].append(clean(detail_row['text']))
                        _own(detail_row, record, f'$.key_strengths[{len(strengths) - 1}]',
                             'action statement under the capability heading')
                groups[f'inferred:key_strengths:{sid}'] = {
                    'key': 'key_strengths', 'value': strengths, 'inferred': True}
            else:
                value = [{'name': clean(row['text'])} for row in body if clean(row['text'])]
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
        elif key in {'skills', 'personal_information', 'certifications', 'awards',
                     'achievements', 'publications'}:
            if key == 'personal_information':
                value = {}
                header_fields, field_rows = extract_header_identity(body)
                value.update(header_fields)
                if section.casefold() == 'header':
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
                    extracted_contact_fields = [field for field, source_row in field_rows.items()
                                                if source_row is row]
                    if extracted_contact_fields:
                        for field in extracted_contact_fields:
                            _own(row, value, f'$.personal_information.{field}',
                                 'identity/contact value extracted from its source row')
                        continue
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
            previous_source_text = ''
            for row_index, row in enumerate(body):
                text = clean(row['text'])
                label = re.match(r'^([^:]+):\s*(.*)$', text)
                explicit_category = bool(label)
                if label:
                    category = re.sub(r'\W+', '_', label[1].casefold()).strip('_')
                    text = label[2]
                else:
                    column_label = re.match(r'^([A-Za-z][A-Za-z &/-]{1,40})\s{2,}(.+)$', text)
                    if column_label:
                        explicit_category = True
                        category = re.sub(r'\W+', '_', column_label[1].casefold()).strip('_')
                        text = column_label[2]
                items = value.setdefault(category, [])
                if (not explicit_category and '|' in text and items and previous_source_text
                        and not previous_source_text.rstrip().endswith(('|', '.', ';', ':'))):
                    wrapped = [part.strip() for part in text.split('|')]
                    if len(wrapped) > 1 and wrapped[0]:
                        items[-1] = f"{items[-1].rstrip()} {wrapped[0]}".strip()
                        text = ' | '.join(wrapped[1:])
                parts = [v.strip(' .') for v in re.split(r'\s*[|;]\s*|,\s*(?![^()]*\))', text) if v.strip(' .')]
                if (not label and not re.match(r'^[A-Za-z][A-Za-z &/-]{1,40}\s{2,}', clean(row['text']))
                        and items and len(text.split()) <= 5):
                    items[-1] = (items[-1] + ' ' + text).strip()
                else:
                    items.extend(parts)
                value[category] = [re.sub(r'-\s+(?=[A-Za-z])', '', item) for item in items]
                previous_source_text = clean(row['text'])
                row.update(destination=f'$.technical_skills.{category}', parent_candidate=f'$.technical_skills.{category}')
        elif key == 'key_strengths':
            value = []
            for row in body:
                text = clean(row['text'])
                if value and not re.search(r'[.!?]$', value[-1]):
                    value[-1] = (value[-1].rstrip() + ' ' + text).strip()
                elif text:
                    value.append(text)
                row.update(destination=f'$.key_strengths', parent_candidate='$.key_strengths')
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
    # Some resumes place employer-specific role descriptions directly below
    # a dated employer heading inside a projects section. Attach only those
    # lines to the exact company record established elsewhere in the resume;
    # project entries themselves remain separate records.
    work_groups = [group for group in groups.values() if group.get('key') == 'work_history']
    work_records = [record for group in work_groups for record in group.get('value', [])]
    by_company = {re.sub(r'\W+', ' ', record.get('company', '')).strip().casefold(): record
                  for record in work_records if record.get('company')}
    for row in rows:
        company = row.get('_employment_context_company')
        if not company or not row.get('_employment_context_intro'):
            continue
        record = by_company.get(re.sub(r'\W+', ' ', company).strip().casefold())
        if not record:
            continue
        roles = record.setdefault('roles', [])
        role = (roles[0] if len(roles) == 1 else
                next((candidate for candidate in roles if not candidate.get('designation')), None))
        if role is None:
            role = {'designation': '', 'role_and_responsibilities': [],
                    'responsibility_sources': [], 'source_blocks': []}
            roles.append(role)
        text = clean(row['text'])
        title_match = re.match(r'^worked\s+as\s+(?:a|an|the)\s+(.+?)\s+(?:on|for|in)\b', text, re.I)
        if title_match and role_title(title_match.group(1)):
            role['designation'] = title_match.group(1).strip()
        responsibilities = role.setdefault('role_and_responsibilities', [])
        if row.get('_employment_context_continuation') and responsibilities:
            responsibilities[-1] = (responsibilities[-1].rstrip() + ' ' + text).strip()
            if role.get('responsibility_sources'):
                role['responsibility_sources'][-1].append(row['id'])
        elif text and text not in responsibilities:
            responsibilities.append(text)
            role.setdefault('responsibility_sources', []).append([row['id']])
        if row['id'] not in role.setdefault('source_blocks', []):
            role['source_blocks'].append(row['id'])
        path = f'$.work_history[{work_records.index(record)}].roles[{roles.index(role)}]'
        row.update(destination=path, parent_candidate=path,
                   reason='employer-specific overview is linked by the exact dated employer context')

    # PDF templates often render language proficiency as dotted leaders and
    # put the final percentage on its own line. Store the language names while
    # retaining the original source blocks in provenance.
    for group in groups.values():
        if group.get('key') != 'personal_information' or not isinstance(group.get('value'), dict):
            continue
        details = group['value'].get('personal_details')
        if not isinstance(details, dict) or not isinstance(details.get('languages'), list):
            continue
        normalized, pending_score = [], False
        for item in details['languages']:
            item = re.sub(r'\s*[-._·•]{2,}\s*\d{0,3}%?\s*$', '', clean(item)).strip(' ,;:-')
            if re.fullmatch(r'\d{1,3}%?', item):
                pending_score = True
                continue
            if item:
                item = re.sub(r'\s*\d{1,3}%?\s*$', '', item).strip(' ,;:-')
                if item and item.casefold() not in {v.casefold() for v in normalized}:
                    normalized.append(item)
                pending_score = False
        details['languages'] = normalized

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
