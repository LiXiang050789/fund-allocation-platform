"""Pure date parsing for platform score queries. Default year is the data cutoff year."""
import re
from datetime import date

LATEST_DATA_YEAR = 2026
DATE_TOKEN = re.compile(r"(?<![\d/-])(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:\d{4}年)?\d{1,2}月\d{1,2}日)(?!\d)")
SEPARATOR = re.compile(r"^\s*(?:到|至|~|～|—)\s*$")
# 截至/截止是“截止到某日”的单日语义，不是起止区间分隔符。
CUTOFF = re.compile(r"截(?:止|至)(?:到|至)?")


def _strip_cutoff(question: str) -> str:
    return CUTOFF.sub("", question)


def range_requested(question: str) -> bool:
    text = _strip_cutoff(question)
    return bool(re.search(r"到|至|~|～|—", text) and
                (DATE_TOKEN.search(text) or re.search(r"\d+月|每天.*评分|评分.*每天", text)))


def _date(token: str, year: int) -> str:
    if '月' in token:
        match = re.fullmatch(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日", token)
        return date(int(match[1] or year), int(match[2]), int(match[3])).isoformat()
    return date(*map(int, re.split(r"[-/]", token))).isoformat()


def _year(question: str) -> int:
    match = re.search(r"(?<!\d)(\d{4})(?:年|[-/])", question)
    return int(match[1]) if match else LATEST_DATA_YEAR


def parse_date_range(question: str) -> tuple[str, str] | None:
    text = _strip_cutoff(question)
    tokens = list(DATE_TOKEN.finditer(text))
    if not range_requested(text):
        return None
    if len(tokens) != 2 or not SEPARATOR.fullmatch(text[tokens[0].end():tokens[1].start()]):
        raise ValueError("无法解析日期区间，请使用完整起止日期")
    try:
        start, end = (_date(token.group(), _year(text)) for token in tokens)
    except ValueError as exc:
        raise ValueError("日期无效，请使用有效年月日") from exc
    if start > end:
        raise ValueError("起止日期倒置")
    return start, end


def parse_single_date(question: str) -> str | None:
    text = _strip_cutoff(question)
    if range_requested(text):
        return None
    tokens = list(DATE_TOKEN.finditer(text))
    if len(tokens) != 1:
        return None
    try:
        return _date(tokens[0].group(), _year(text))
    except ValueError as exc:
        raise ValueError("日期无效，请使用有效年月日") from exc


def history_error(results: list[dict], question: str = "") -> str | None:
    """Only disclose known date validation messages, never arbitrary exception text."""
    allowed = ("日期必须为YYYY-MM-DD", "起止日期倒置", "区间内无交易日",
               "日期越界：可用范围", "无法解析日期区间", "日期无效")
    for result in results:
        if result.get('tool') != 'get_market_score_history' or result.get('ok'):
            continue
        error = result.get('error') or ''
        message = error.removeprefix('ValueError: ')
        if message == "日期必须为YYYY-MM-DD" and question:
            try:
                parse_date_range(question)
                parse_single_date(question)
            except ValueError as exc:
                message = str(exc)
        if message.startswith(allowed):
            return "日期区间无法查询：" + message + "。请提供有效区间重试。"
    return None
