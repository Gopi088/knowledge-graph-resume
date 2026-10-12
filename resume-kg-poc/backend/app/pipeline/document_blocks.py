"""Source blocks and geometric reading order; text-only callers keep the str API."""
from collections import Counter
import re


class DocumentText(str):
    def __new__(cls, blocks):
        obj = super().__new__(cls, '\n'.join(block['text'] for block in blocks))
        obj.blocks = blocks
        return obj


def _gap(items, axis):
    intervals = sorted((b['bbox'][axis], b['bbox'][axis + 2]) for b in items)
    end = intervals[0][1]
    gaps = []
    for lo, hi in intervals[1:]:
        if lo > end:
            gaps.append((lo - end, (lo + end) / 2))
        end = max(end, hi)
    return max(gaps, default=(0, 0))


def reading_order(items):
    """Read columns top-to-bottom when their contents align as shared rows.

    Timeline resumes commonly put dates/employers in a narrow left column and
    the matching title/details in a right column. A page-wide x split reads
    every left-side entry before every right-side entry and disconnects those
    records. Detect repeated vertical alignment across the two columns and
    read those pages in row order; retain recursive column reading for layouts
    whose columns are independent.
    """
    if len(items) < 2:
        return items
    xgap, xcut = _gap(items, 0)
    # Text lines in columns often extend far enough to overlap on x (a long
    # paragraph can span most of its column), so interval-gap detection alone
    # misses the gutter. The start positions still expose a stable gutter in
    # conventional sidebar resumes. Use this only for a pronounced gap; small
    # indents and date/title layouts continue through the ordinary heuristic.
    page_width = max(b['bbox'][2] for b in items)
    start_position_split = False
    if xgap < page_width * 0.22:
        starts = sorted({b['bbox'][0] for b in items})
        if len(starts) >= 2:
            start_gaps = [(right - left, (right + left) / 2)
                          for left, right in zip(starts, starts[1:])]
            start_gap, start_cut = max(start_gaps, default=(0, 0))
            if start_gap >= page_width * 0.22:
                xgap, xcut = start_gap, start_cut
                start_position_split = True
    ygap, ycut = _gap(items, 1)
    if xgap >= 18:
        if start_position_split:
            left = [b for b in items if b['bbox'][0] < xcut]
            right = [b for b in items if b['bbox'][0] >= xcut]
        else:
            left = [b for b in items if b['bbox'][2] <= xcut]
            right = [b for b in items if b['bbox'][0] >= xcut]
        if len(left) >= 2 and len(right) >= 2:
            # For a pronounced start-position gutter, these are independent
            # sidebar and body flows. Their y-coordinate overlaps do not mean
            # the lines belong to the same row; reading them together changes
            # which section owns subsequent content.
            if start_position_split:
                return reading_order(left) + reading_order(right)
            aligned_left, aligned_right = set(), set()
            for li, first in enumerate(left):
                y0, y1 = first['bbox'][1], first['bbox'][3]
                for ri, second in enumerate(right):
                    # Small vertical offsets are normal for paired labels and
                    # values; avoid treating broadly overlapping columns as
                    # rows by limiting the overlap tolerance to line height.
                    overlap = min(y1, second['bbox'][3]) - max(y0, second['bbox'][1])
                    if overlap >= -2:
                        aligned_left.add(li)
                        aligned_right.add(ri)
            # A dense two-column resume can have many accidental y overlaps:
            # the main column has long paragraphs while the sidebar has skills,
            # education and contact details. Treating two matches as evidence
            # of paired rows interleaves unrelated sections (and corrupts the
            # section state downstream). Row-wise ordering is appropriate only
            # when a meaningful share of *both* columns participates in aligned
            # pairs, as in date/title timelines and tabular records.
            left_alignment = len(aligned_left) / max(1, len(left))
            right_alignment = len(aligned_right) / max(1, len(right))
            if (len(aligned_left) >= 2 and len(aligned_right) >= 2
                    and left_alignment >= 0.20 and right_alignment >= 0.20):
                # Group slightly offset baselines into the same visual row,
                # then keep the left value before the right-side title/text.
                return sorted(items, key=lambda b: (round(b['bbox'][1] / 4), b['bbox'][0], b['bbox'][1]))
            return reading_order(left) + reading_order(right)
    if ygap > 3:
        top = [b for b in items if b['bbox'][3] <= ycut]
        bottom = [b for b in items if b['bbox'][1] >= ycut]
        if top and bottom:
            return reading_order(top) + reading_order(bottom)
    return sorted(items, key=lambda b: (round(b['bbox'][1], 1), b['bbox'][0]))


def extract_pdf_blocks(path, ocr):
    import pymupdf
    pages = []
    with pymupdf.open(path) as document:
        for page_index, page in enumerate(document):
            rows = []
            for block_index, block in enumerate(page.get_text('dict')['blocks']):
                for line in block.get('lines', []):
                    spans = line.get('spans', [])
                    text = ''.join(span.get('text', '') for span in spans).strip()
                    if text:
                        rows.append({'text': text, 'page': page_index + 1, 'bbox': list(line['bbox']),
                                     'page_height': page.rect.height, 'block_index': block_index,
                                     'font_size': max((s.get('size', 0) for s in spans), default=None),
                                     'bold': any(s.get('flags', 0) & 16 for s in spans), 'extraction': 'pdf_text'})
            if sum(len(row['text']) for row in rows) < 20:
                text = ocr(page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes('png'))
                rows = [{'text': line, 'page': page_index + 1, 'bbox': None, 'block_index': i,
                         'font_size': None, 'bold': None, 'extraction': 'ocr'}
                        for i, line in enumerate(text.splitlines()) if line.strip()]
            else:
                # Ruled tables expose cell/row geometry. Preserve each row as a
                # unit so column sorting cannot turn cells into unrelated records.
                if hasattr(page, 'find_tables'):
                    tables = page.find_tables().tables
                    for table in tables:
                        box = table.bbox
                        contained = lambda row: (box[0] - 1 <= row['bbox'][0] and row['bbox'][2] <= box[2] + 1
                                                 and box[1] - 1 <= row['bbox'][1] and row['bbox'][3] <= box[3] + 1)
                        rows = [row for row in rows if not contained(row)]
                        for index, cells in enumerate(table.extract()):
                            rows.append({'text': ' | '.join((cell or '').replace('\n', ' ') for cell in cells),
                                         'page': page_index + 1, 'bbox': list(table.rows[index].bbox),
                                         'page_height': page.rect.height, 'block_index': f'table:{index}',
                                         'font_size': None, 'bold': None, 'extraction': 'pdf_table', 'table_row': True})
                rows = reading_order(rows)
            pages.append(rows)
    margins = Counter()
    for rows in pages:
        for text in {r['text'] for r in rows if r.get('bbox') and
                     (r['bbox'][1] < 55 or r['bbox'][3] > r['page_height'] - 45)}:
            margins[text] += 1
    result = []
    seen_margins = set()
    for rows in pages:
        for row in rows:
            box = row.get('bbox')
            margin = bool(box and (box[1] < 55 or box[3] > row['page_height'] - 45))
            # Some PDF extractors concatenate a bottom page number onto the
            # final body line. Split only the trailing artifact pattern so it
            # cannot leak into the responsibility or project text.
            trailing_page = re.search(r'\s+(\d{1,3})\s*$', row['text']) if margin or not box else None
            if trailing_page and trailing_page.start() > 0:
                artifact_text = trailing_page.group(1)
                row['text'] = row['text'][:trailing_page.start()].rstrip()
            else:
                artifact_text = None
            page_marker = bool(re.fullmatch(r'(?:page\s*)?\d{1,3}(?:\s*(?:of|/)\s*\d+)?', row['text'], re.I))
            row['artifact'] = ('page_number' if page_marker and (margin or not box) else
                               'repeated_margin' if margin and margins[row['text']] > 1 and row['text'] in seen_margins else None)
            if artifact_text:
                result.append({'id': f'd{len(result)}', 'text': artifact_text, 'page': row.get('page'),
                               'bbox': None, 'page_height': row.get('page_height'),
                               'block_index': row.get('block_index'), 'font_size': None,
                               'bold': None, 'extraction': row.get('extraction'), 'artifact': 'page_number',
                               'reading_order': len(result)})
            if margin:
                seen_margins.add(row['text'])
            row.update(id=f'd{len(result)}', reading_order=len(result))
            result.append(row)
    return DocumentText(result)
