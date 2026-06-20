import re
from typing import List


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int = 200) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[。！？；\n])\s*', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    import jieba.analyse
    keywords = jieba.analyse.extract_tags(text, topK=top_k)
    return keywords


def highlight_keywords(text: str, keywords: List[str]) -> str:
    for kw in keywords:
        if kw:
            text = re.sub(
                re.escape(kw),
                f"**{kw}**",
                text,
                flags=re.IGNORECASE
            )
    return text


def format_reference(doc_text: str, score: float, source: str = "") -> str:
    header = f"> **相关度**: {score:.4f}"
    if source:
        header += f" | **来源**: {source}"
    content = f"> {doc_text[:150].replace(chr(10), ' ')}..."
    return f"{header}\n{content}\n"
