"""Shared sentence ending normalization, independent of models and transport."""
import re


def sentence_end(text: str, fallback: str = "当前数据无法回答") -> str:
    text = text.rstrip()
    match = re.search(r"[。！？!?,，、；：.;:…\s]+$", text)
    ending = match.group() if match else ""
    body = text[:match.start()].rstrip() if match else text
    if not body:
        return sentence_end(fallback, "当前数据无法回答") if fallback.strip("。！？!?,，、；：.;:… \n\t") else "当前数据无法回答。"
    terminal = next((char for char in ending if char in "。！？"), "。")
    return body + terminal
