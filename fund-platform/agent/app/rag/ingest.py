"""确定性解析 docx/md/txt；保留章节、表格和来源信息。"""
from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80,
    separators=['\n\n', '\n', '。', '；', '，', ' ', ''])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_corpus(path):
    files = yaml.safe_load(Path(path).read_text(encoding='utf-8'))['files']
    if any(f['authority'] not in {'authoritative', 'background', 'historical'} for f in files):
        raise ValueError('未知 authority')
    if len({f['path'] for f in files}) != len(files):
        raise ValueError('语料清单重复路径')
    return sorted(files, key=lambda f: f['path'])


def doc_date(path, root):
    try:
        result = subprocess.run(['git', 'log', '-1', '--format=%ad', '--date=iso-strict', '--',
                                 str(path.relative_to(root))], cwd=root, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('T')[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()


def heading(text, style=''):
    if style.lower().startswith('heading'):
        match = re.search(r'\d+', style)
        return int(match[0]) if match else 1
    if len(text) > 85:
        return None
    if re.match(r'^[一二三四五六七八九十〇]+、', text):
        return 1
    if re.match(r'^维度[一二三四五六七八九十]', text):
        return 2
    if re.match(r'^\d+(?:\.\d+)*[.、\s]', text) and len(text) < 45 and not any(w in text for w in ('阈值', '满分', '线性插值', '。', '；')):
        return 2
    return None


def markdown_table(rows):
    def clean(cell):
        return cell.replace('|', '\\|').replace('\n', '<br>')
    rows = [[clean(str(c)) for c in row] for row in rows]
    if not rows:
        return ''
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |',
                      '| ' + ' | '.join(['---']*len(rows[0])) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows[1:]])


def parse_docx(path):
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(path)
    sections = {}
    blocks = []
    for node in document.element.body:
        if node.tag == qn('w:p'):
            paragraph = Paragraph(node, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style else ''
            level = heading(text, style)
            if level:
                sections = {k: v for k, v in sections.items() if k < level}
                sections[level] = text
            numpr = paragraph._p.pPr.numPr if paragraph._p.pPr is not None else None
            numbering = numpr.xml if numpr is not None else ''
            blocks.append(dict(text=text, section=' / '.join(sections.values()) or '正文',
                               kind='paragraph', style=style, numbering=numbering))
        elif node.tag == qn('w:tbl'):
            table = Table(node, document)
            blocks.append(dict(text=markdown_table([[c.text for c in r.cells] for r in table.rows]),
                               section=' / '.join(sections.values()) or '正文', kind='table'))
    return blocks


def parse(path):
    path = Path(path)
    if path.suffix.lower() == '.docx':
        return parse_docx(path)
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.md':
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#'*i, f'h{i}') for i in range(1, 7)], strip_headers=False)
        blocks = []
        for doc in splitter.split_text(text):
            section = ' / '.join(doc.metadata.values()) or '正文'
            pending, kind = [], None
            for line in doc.page_content.splitlines():
                next_kind = 'table' if line.lstrip().startswith('|') else 'markdown'
                if pending and kind != next_kind:
                    blocks.append(dict(text='\n'.join(pending), section=section, kind=kind))
                    pending = []
                pending.append(line)
                kind = next_kind
            if pending:
                blocks.append(dict(text='\n'.join(pending), section=section, kind=kind))
        return blocks
    if path.suffix.lower() == '.txt':
        # Preserve explicit Chinese/decimal headings before recursive character splitting.
        sections, blocks, lines = {}, [], []
        for line in text.splitlines():
            level = heading(line.strip())
            if level:
                if lines:
                    blocks.append(dict(text='\n'.join(lines), section=' / '.join(sections.values()) or '正文', kind='text'))
                sections = {k: v for k, v in sections.items() if k < level}
                sections[level] = line.strip()
                lines = []
            lines.append(line)
        if lines:
            blocks.append(dict(text='\n'.join(lines), section=' / '.join(sections.values()) or '正文', kind='text'))
        return blocks
    raise ValueError(f'不支持的语料格式: {path.suffix}')


def split_table(text):
    lines = text.splitlines()
    if len(text) <= 800 or len(lines) < 3:
        return [text]
    header, batches, rows = lines[:2], [], []
    for row in lines[2:]:
        if rows and len('\n'.join(header + rows + [row])) > 800:
            batches.append('\n'.join(header + rows))
            rows = []
        rows.append(row)
    if rows:
        batches.append('\n'.join(header + rows))
    return batches


def chunks_for(path, root, authority):
    path, root = Path(path), Path(root)
    source = path.relative_to(root).as_posix()
    date = doc_date(path, root)
    chunks, pending, section, styles, numbering = [], [], None, set(), []

    def emit(text, current, kind):
        for fragment in split_table(text) if kind == 'table' else SPLITTER.split_text(text):
            if not fragment.strip():
                continue
            identity = f'{source}\0{current}\0{len(chunks)}\0{fragment}'
            chunks.append(dict(chunk_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                source=source, source_name=path.name, section=current or '正文', doc_date=date,
                authority=authority, text=fragment, styles=sorted(styles), numbering=list(numbering)))

    def flush():
        if pending:
            emit('\n'.join(pending), section, 'paragraph')
            pending.clear()
            styles.clear()
            numbering.clear()

    for block in parse(path):
        if block['kind'] == 'table':
            flush()
            section = block['section']
            emit(block['text'], section, 'table')
        else:
            if block['section'] != section:
                flush()
                section = block['section']
            pending.append(block['text'])
            if block.get('style'):
                styles.add(block['style'])
            if block.get('numbering'):
                numbering.append(block['numbering'])
    flush()
    # Keep a short paragraph and its adjacent table in the same evidence unit.
    # Every table row stays intact; oversized tables were already split with headers.
    merged = []
    for chunk in chunks:
        if merged and merged[-1]['section'] == chunk['section'] and len(merged[-1]['text']+'\n'+chunk['text']) <= 800:
            previous = merged[-1]
            previous['text'] += '\n' + chunk['text']
            previous['styles'] = sorted(set(previous['styles'] + chunk['styles']))
            previous['numbering'] += chunk['numbering']
        else:
            merged.append(chunk)
    for ordinal, chunk in enumerate(merged):
        identity = f"{source}\0{chunk['section']}\0{ordinal}\0{chunk['text']}"
        chunk['chunk_id'] = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return merged
