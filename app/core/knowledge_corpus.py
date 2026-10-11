from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import BASE_DIR, settings


CORPUS_DIR = Path(settings.rag_corpus_dir)
if not CORPUS_DIR.is_absolute():
    CORPUS_DIR = BASE_DIR / CORPUS_DIR

MANIFEST_PATH = CORPUS_DIR / "manifest.json"
CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"


@dataclass(frozen=True)
class KnowledgeSource:
    key: str
    title: str
    category: str
    description: str
    source_name: str
    source_path: str
    file_size_kb: int
    chunk_count: int = 0
    source_file: str | None = None

    def as_reference(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "href": f"/knowledge-base/source/{self.key}",
            "source_name": self.source_name,
            "source_path": self.source_path,
            "file_size_kb": self.file_size_kb,
            "chunk_count": self.chunk_count,
        }


def _category_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _question_terms(question: str) -> list[str]:
    return [
        term
        for term in {
            token.strip(".,:;!?()[]{}\"'").lower()
            for token in re.split(r"\s+", question)
        }
        if len(term) >= 4
    ]


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"sources": []}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"sources": []}


@lru_cache(maxsize=1)
def knowledge_sources() -> tuple[KnowledgeSource, ...]:
    sources: list[KnowledgeSource] = []
    for raw in _manifest().get("sources", []):
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("source_id") or "").strip()
        title = str(raw.get("title") or raw.get("source_filename") or key).strip()
        source_path = str(raw.get("source_path") or raw.get("source_filename") or "").strip()
        if not key or not title:
            continue
        sources.append(
            KnowledgeSource(
                key=key,
                title=title,
                category=str(raw.get("category") or "Source Materials"),
                description=str(
                    raw.get("description")
                    or "Prepared source text included in the deployable knowledge corpus."
                ),
                source_name=str(raw.get("source_filename") or Path(source_path).name or title),
                source_path=source_path,
                file_size_kb=max(1, int(raw.get("text_size_kb") or raw.get("file_size_kb") or 1)),
                chunk_count=int(raw.get("chunk_count") or 0),
                source_file=raw.get("source_file"),
            )
        )
    return tuple(sources)


@lru_cache(maxsize=1)
def _source_lookup() -> dict[str, KnowledgeSource]:
    return {source.key: source for source in knowledge_sources()}


def _metadata_score(source: KnowledgeSource, terms: list[str]) -> int:
    haystack = f"{source.title} {source.source_name} {source.source_path} {source.category}".lower()
    return sum(4 if term in source.title.lower() else 1 for term in terms if term in haystack)


def _featured_score(source: KnowledgeSource) -> tuple[int, int, int]:
    name = source.source_name.lower()
    category_weight = {
        "IPEC Directives": 50,
        "Regulatory Guides": 45,
        "IPEC Circulars": 35,
        "Policy Wordings": 25,
        "Insurance Documents": 20,
        "Templates & Returns": 10,
    }.get(source.category, 0)
    keyword_weight = sum(
        weight
        for keyword, weight in (
            ("insurance act", 40),
            ("ipec", 32),
            ("sandbox", 30),
            ("zicarp", 28),
            ("risk based capital", 26),
            ("guideline", 20),
            ("framework", 20),
            ("circular", 12),
        )
        if keyword in name
    )
    return (category_weight + keyword_weight, source.chunk_count, source.file_size_kb)


def source_references(query: str = "", category: str = "") -> list[dict[str, Any]]:
    sources = list(knowledge_sources())
    if category:
        category_text = _category_slug(category)
        sources = [source for source in sources if _category_slug(source.category) == category_text]
    if query:
        terms = _question_terms(query)
        sources = [
            source
            for source in sources
            if any(
                term in f"{source.title} {source.source_name} {source.source_path} {source.category}".lower()
                for term in terms
            )
        ]
        sources.sort(key=lambda source: _metadata_score(source, terms), reverse=True)
    else:
        sources.sort(key=_featured_score, reverse=True)
    return [source.as_reference() for source in sources]


def category_stats() -> list[dict[str, Any]]:
    sources = knowledge_sources()
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.category] = counts.get(source.category, 0) + 1
    return [
        {"icon": "policy", "label": "IPEC Circulars", "count": counts.get("IPEC Circulars", 0), "color": "rgba(212,175,55,0.12)"},
        {"icon": "article", "label": "Regulatory Guides", "count": counts.get("Regulatory Guides", 0), "color": "rgba(96,165,250,0.12)"},
        {"icon": "auto_stories", "label": "Policy Wordings", "count": counts.get("Policy Wordings", 0), "color": "rgba(74,222,128,0.12)"},
        {"icon": "folder_open", "label": "Prepared Sources", "count": len(sources), "color": "rgba(251,191,36,0.12)"},
    ]


def resolve_source(source_key: str) -> KnowledgeSource | None:
    return _source_lookup().get(source_key)


@lru_cache(maxsize=1)
def _chunks() -> tuple[dict[str, Any], ...]:
    if not CHUNKS_PATH.exists():
        return tuple()
    chunks: list[dict[str, Any]] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("text"):
                chunks.append(item)
    return tuple(chunks)


def _chunk_score(chunk: dict[str, Any], terms: list[str]) -> int:
    metadata = " ".join(
        str(chunk.get(key) or "")
        for key in ("title", "source_filename", "source_path", "category")
    ).lower()
    text = str(chunk.get("text") or "").lower()
    return sum((6 if term in metadata else 0) + (text.count(term) * 2) for term in terms)


def source_text_chunks(question: str, limit: int = 8) -> list[dict[str, str]]:
    terms = _question_terms(question)
    chunks = list(_chunks())
    if terms:
        ranked = sorted(chunks, key=lambda chunk: _chunk_score(chunk, terms), reverse=True)
        ranked = [chunk for chunk in ranked if _chunk_score(chunk, terms) > 0] or ranked
    else:
        ranked = chunks

    selected: list[dict[str, str]] = []
    per_source: dict[str, int] = {}
    for chunk in ranked:
        source = resolve_source(str(chunk.get("source_id") or ""))
        source_id = str(chunk.get("source_id") or "")
        if per_source.get(source_id, 0) >= 2:
            continue
        title = str(chunk.get("title") or (source.title if source else "Knowledge Source"))
        source_name = str(chunk.get("source_filename") or (source.source_name if source else title))
        source_path = str(chunk.get("source_path") or (source.source_path if source else source_name))
        page = chunk.get("page")
        page_part = f"\nPage: {page}" if page not in (None, "") else ""
        text = str(chunk.get("text") or "")
        selected.append(
            {
                "title": title,
                "source": source_name,
                "path": source_path,
                "text": f"{title}\nSource file: {source_name}\nPath: {source_path}{page_part}\n{text}",
            }
        )
        per_source[source_id] = per_source.get(source_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def source_excerpts(question: str, chunks: list[dict[str, str]]) -> list[dict[str, str]]:
    terms = _question_terms(question)
    excerpts: list[dict[str, str]] = []
    for chunk in chunks:
        sentences = [
            re.sub(r"\s+", " ", sentence).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"])
        ]
        ranked = sorted(
            [sentence for sentence in sentences if len(sentence) > 40],
            key=lambda sentence: sum(1 for term in terms if term in sentence.lower()),
            reverse=True,
        )
        best = next((sentence for sentence in ranked if any(term in sentence.lower() for term in terms)), "")
        if best:
            excerpts.append(
                {
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "excerpt": best[:700],
                }
            )
    return excerpts


def source_plaintext(source_key: str) -> str:
    source = resolve_source(source_key)
    if source is None:
        return ""
    parts = [
        f"{source.title}",
        f"Category: {source.category}",
        f"Original source: {source.source_path}",
        "",
    ]
    for chunk in _chunks():
        if chunk.get("source_id") == source_key:
            page = chunk.get("page")
            label = f"Page {page}" if page not in (None, "") else f"Chunk {chunk.get('chunk_index', '')}"
            parts.append(f"[{label}]")
            parts.append(str(chunk.get("text") or "").strip())
            parts.append("")
    return "\n".join(parts).strip()
