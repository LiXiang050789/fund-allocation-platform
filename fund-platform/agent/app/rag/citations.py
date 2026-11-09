"""检索引用映射与文档 authority 边界。"""
import re
from .textnorm import normalize_text

DOC_REF = re.compile(r'\[来源[:：]\s*(.*?)\s*§\s*([^\]]+)\]')
# Only used to reject obsolete facts or explicitly discuss obsolete values.
OLD_VALUES = (19.8, 12.4, 94.26, 94.3, 89.2, 26.9, 1.41, 177, 34.4, 37.1)
MARKERS = re.compile(r'旧值|禁用|历史|已废弃|不得')


def retrieved_chunks(results):
    return [c for r in results if r.get('ok') and r.get('tool') == 'search_docs'
            for c in r.get('data', {}).get('chunks', [])]


def citation(chunk):
    section = chunk['section']
    if len(section) > 75:
        parts = section.split(' / ')
        section = parts[0]
        for part in parts[1:]:
            if len(section + ' / ' + part) > 75:
                break
            section += ' / ' + part
    return f"[来源: {chunk['source_name']} §{section}]"


def citation_matches(answer, chunks):
    matched, invalid = [], []
    for name, section in DOC_REF.findall(answer):
        candidates = [c for c in chunks if c['source_name'] == name.strip()
                      and section.strip() and c['section'].startswith(section.strip())]
        if candidates:
            matched.extend(candidates)
        else:
            invalid.append(f'{name} §{section}')
    # Detect malformed citations as well as well-formed but fabricated ones.
    if answer.count('[来源') != len(DOC_REF.findall(answer)):
        invalid.append('引用格式错误')
    return matched, invalid


def marked_numbers(body):
    """An obsolete number may be quoted as obsolete, never as a current fact."""
    allowed, violations = [], []
    for clause in re.split(r'[。；\n]', normalize_text(body)):
        for raw in re.findall(r'-?\d+(?:\.\d+)?', clause):
            value = float(raw)
            if any(abs(abs(value)-old) < 1e-9 for old in OLD_VALUES):
                if MARKERS.search(clause):
                    allowed.append(value)
                else:
                    violations.append(value)
    return allowed, violations


def document_text(chunk):
    text = chunk['text']
    if chunk['authority'] != 'authoritative':
        # Background is explanatory only. Hide factual digits before generation.
        text = re.sub(r'[-+]?\d+(?:\.\d+)?[%％]?', '（背景数字省略）', text)
    return text
