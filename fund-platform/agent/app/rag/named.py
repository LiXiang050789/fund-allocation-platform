"""从问题识别明确文档名称；泛称只用于检索保留位。"""
from pathlib import Path
import re
import unicodedata

ALIASES = {
    '强化学习.txt': ('强化学习设计文档', '强化学习文档', '强化学习'),
    '项目彻底理解手册-答辩备战版.md': ('项目彻底理解手册', '答辩备战版', '理解手册', '手册'),
    '项目书修订说明.md': ('项目书修订说明', '修订说明'),
    'PPO奖励口径-最终代码.md': ('PPO奖励口径-最终代码', 'PPO奖励口径', '最终PPO奖励', '最终代码奖励'),
}


def key(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()


def named_sources(query, chunks, *, explicit=False) -> list[str]:
    query = key(query)
    found = []
    catalog = [*chunks, *({'source_name': name, 'section': ''} for name in ALIASES)]
    for chunk in catalog:
        name = chunk['source_name']
        if name in found:
            continue
        aliases = {name, Path(name).stem, *ALIASES.get(name, ())}
        # A short TXT heading is also its title (e.g. 一、债券).
        if name.endswith('.txt'):
            title = re.sub(r'^[一二三四五六七八九十\d]+[、.．]\s*', '', chunk['section'].split(' / ')[0])
            if 2 <= len(title) <= 12 and title not in {'正文', '概述', '说明', '附录'}:
                aliases.add(title)
        if explicit:
            aliases.discard('手册')
        if any(key(alias) in query for alias in aliases if alias):
            found.append(name)
    return found


def corpus_sources():
    """Names only: verification does not load corpus bodies or embeddings."""
    from .ingest import load_corpus
    return [{'source_name': Path(entry['path']).name, 'section': ''}
            for entry in load_corpus(Path(__file__).with_name('corpus.yaml'))]
