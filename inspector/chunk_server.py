"""
Chunk Inspector + Dashboard Backend - Read-only viewer for chunks_v6.json
FastAPI on port 8001, loads chunks into memory at startup.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "chunks_v6.json")
)

# ---------------------------------------------------------------------------
# Load data (guard against double-import by uvicorn)
# ---------------------------------------------------------------------------
_loaded = False

def load_data():
    global all_chunks, ALL_SEMANTIC_TYPES, ALL_PARTS, ALL_CONTENT_TYPES, _loaded
    if _loaded:
        return
    print(f"Loading chunks from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_chunks = raw["chunks"]

    # Normalize: extract metadata fields to top level for easier filtering
    for c in all_chunks:
        meta = c.get("metadata", {})
        c["part"] = meta.get("part", "")
        c["section"] = meta.get("section", "")
        c["warning_level"] = c.get("warning_level") or meta.get("warning_level")

    # Collect unique filter options
    ALL_SEMANTIC_TYPES = sorted(set(
        c.get("semantic_type", "") for c in all_chunks if c.get("semantic_type")
    ))
    ALL_PARTS = sorted(set(
        c.get("part", "") for c in all_chunks if c.get("part")
    ), key=lambda x: int(x.replace("PART ", "")) if x.startswith("PART ") else 99)
    ALL_CONTENT_TYPES = sorted(set(
        c.get("content_type", "") for c in all_chunks if c.get("content_type")
    ))

    print(f"Loaded {len(all_chunks)} chunks")
    print(f"  Semantic types: {ALL_SEMANTIC_TYPES}")
    print(f"  Parts: {ALL_PARTS}")
    print(f"  Content types: {ALL_CONTENT_TYPES}")
    _loaded = True

all_chunks: list[dict] = []
ALL_SEMANTIC_TYPES: list[str] = []
ALL_PARTS: list[str] = []
ALL_CONTENT_TYPES: list[str] = []

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_data()
    yield

app = FastAPI(title="Chunk Inspector Dashboard", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def compute_stats(chunks: list[dict]) -> dict:
    """Compute full statistics for a filtered chunk list."""
    by_semantic = defaultdict(int)
    by_part = defaultdict(int)
    fragments = 0
    oversized = 0
    total_tokens = 0
    min_tokens = 999999
    max_tokens = 0
    fault_count = 0
    token_bins = defaultdict(int)

    for c in chunks:
        stype = c.get("semantic_type", "unknown")
        part = c.get("part", "unknown")
        by_semantic[stype] += 1
        by_part[part] += 1
        tc = c.get("token_count", 0)
        total_tokens += tc
        if tc < min_tokens:
            min_tokens = tc
        if tc > max_tokens:
            max_tokens = tc
        if tc < 50:
            fragments += 1
        if tc > 768:
            oversized += 1
        ft = c.get("fault_triplet")
        if ft and isinstance(ft, list) and len(ft) > 0:
            fault_count += len(ft)
        # Token distribution bins (100-token intervals)
        bin_key = min(tc // 100, 6)  # 0-100, 100-200, ..., 600+
        token_bins[bin_key] += 1

    n = max(len(chunks), 1)

    # Build ordered token_distribution
    bin_labels = ["0-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600+"]
    token_distribution = []
    for i, label in enumerate(bin_labels):
        token_distribution.append({"range": label, "count": token_bins.get(i, 0)})

    return {
        "total_chunks": len(chunks),
        "avg_tokens": round(total_tokens / n, 1),
        "min_tokens": min_tokens if chunks else 0,
        "max_tokens": max_tokens if chunks else 0,
        "fragment_count": fragments,
        "oversized_count": oversized,
        "fault_triplet_count": fault_count,
        "by_semantic_type": dict(sorted(by_semantic.items())),
        "by_part": dict(sorted(by_part.items(), key=lambda x: (
            int(x[0].replace("PART ", "")) if x[0].startswith("PART ") else 99, x[0]
        ))),
        "token_distribution": token_distribution,
    }


def filter_chunks(
    semantic_type: Optional[str] = None,
    part: Optional[str] = None,
    content_type: Optional[str] = None,
    search: Optional[str] = None,
    has_fault: Optional[bool] = None,
) -> list[dict]:
    """Filter chunks by criteria."""
    result = all_chunks

    if semantic_type:
        result = [c for c in result if c.get("semantic_type") == semantic_type]
    if part:
        result = [c for c in result if c.get("part") == part]
    if content_type:
        result = [c for c in result if c.get("content_type") == content_type]
    if search:
        q = search.lower()
        result = [
            c for c in result
            if q in c.get("chunk_id", "").lower()
            or q in c.get("text", "").lower()
            or q in c.get("part", "").lower()
            or q in c.get("section", "").lower()
        ]
    if has_fault:
        result = [
            c for c in result
            if c.get("fault_triplet") and len(c.get("fault_triplet", [])) > 0
        ]

    return result


def chunk_to_response(c: dict) -> dict:
    """Convert internal chunk dict to API response format."""
    return {
        "chunk_id": c.get("chunk_id", ""),
        "semantic_type": c.get("semantic_type", ""),
        "content_type": c.get("content_type", ""),
        "part": c.get("part", ""),
        "section": c.get("section", ""),
        "token_count": c.get("token_count", 0),
        "text": c.get("text", ""),
        "component": c.get("component") or [],
        "fault_triplet": c.get("fault_triplet") or [],
        "fault_symptom": c.get("fault_symptom") or "",
        "repair_action": c.get("repair_action") or "",
        "repair_level": c.get("repair_level") or "",
        "risk_level": c.get("risk_level") or "",
        "important_kwd": c.get("important_kwd") or "",
        "question_kwd": c.get("question_kwd") or "",
        "domain_tags": c.get("domain_tags") or [],
        "parent_id": c.get("parent_id") or "",
        "child_ids": c.get("child_ids") or [],
        "warning_level": c.get("warning_level") or "",
        "mom_id": c.get("mom_id") or "",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/chunks")
async def get_chunks(
    semantic_type: Optional[str] = Query(None, description="过滤语义类型"),
    part: Optional[str] = Query(None, description="过滤 PART"),
    content_type: Optional[str] = Query(None, description="过滤内容类型"),
    search: Optional[str] = Query(None, description="模糊搜索 chunk_id / text"),
    has_fault: Optional[bool] = Query(None, description="只看有故障三元组的"),
    sort_by: Optional[str] = Query(None, description="排序字段: token_count, part, semantic_type"),
    sort_order: Optional[str] = Query("asc", description="排序: asc / desc"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=200, description="每页数量"),
):
    # Filter
    filtered = filter_chunks(semantic_type, part, content_type, search, has_fault)

    # Sort
    if sort_by in ("token_count", "part", "semantic_type"):
        reverse = sort_order == "desc"
        if sort_by == "token_count":
            filtered = sorted(filtered, key=lambda c: c.get("token_count", 0), reverse=reverse)
        elif sort_by == "part":
            def part_key(c):
                p = c.get("part", "")
                if p.startswith("PART "):
                    return int(p.replace("PART ", ""))
                return 99
            filtered = sorted(filtered, key=part_key, reverse=reverse)
        elif sort_by == "semantic_type":
            filtered = sorted(filtered, key=lambda c: c.get("semantic_type", ""), reverse=reverse)

    # Stats (computed on filtered set)
    stats = compute_stats(filtered)

    # Paginate
    total = len(filtered)
    total_pages = max(1, math.ceil(total / page_size))
    if page > total_pages:
        page = total_pages
    start = (page - 1) * page_size
    page_chunks = filtered[start:start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "stats": stats,
        "chunks": [chunk_to_response(c) for c in page_chunks],
        "filters_available": {
            "semantic_types": ALL_SEMANTIC_TYPES,
            "parts": ALL_PARTS,
            "content_types": ALL_CONTENT_TYPES,
        },
    }


@app.get("/api/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Get a single chunk by ID."""
    for c in all_chunks:
        if c.get("chunk_id") == chunk_id:
            return chunk_to_response(c)
    raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
