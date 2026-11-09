"""共享数值文本归一化；保留区间连接号。"""
import unicodedata


def normalize_text(text):
    return unicodedata.normalize('NFKC', text).translate(str.maketrans({'−': '-', '－': '-', '﹣': '-'}))


def numeric_text(text):
    """Separate binary subtraction from signed numeric literals in formulas."""
    import re
    # Chinese clause punctuation is not a thousands separator.
    text = normalize_text(text.replace('，', '；'))
    text = re.sub(r'(?<=\d)[-–—](?=\d)', ' 至 ', text)
    text = re.sub(r'(?<=[)）\]])-(?=\d)', ' - ', text)
    text = re.sub(r'(?<=[A-Za-z_])-(?=\d+(?:\.\d+)?(?:[)）/×*]|$))', ' - ', text)
    return re.sub(r'(?<=[\w)）\]])-(?=\d+(?:\.\d+)?\s*[×*])', ' - ', text)
