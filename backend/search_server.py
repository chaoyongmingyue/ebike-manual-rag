"""
Yadea DM6 Electric Bicycle Manual Knowledge Base Q&A Backend v6
FastAPI + Qdrant (hybrid RRF) + BGE-M3 + Ollama (qwen2.5:7b)
v6 additions: Query expansion, Routing, LLM Rerank, Metadata Boost, Parent aggregation
"""

from __future__ import annotations

import json
import logging
import os
import re
import hashlib

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Prefetch, SparseVector, Filter, FieldCondition, MatchValue

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("ebike-search")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.environ.get("MODEL_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "bge-m3")))
QDRANT_PATH: str = os.environ.get("QDRANT_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")))
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "ebike_manual")
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

MAX_CONTEXT_CHARS: int = 4000
TEXT_PREVIEW_LEN: int = 120

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
model: Any = None
qdrant: QdrantClient | None = None
points_count: int = 0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, qdrant, points_count

    logger.info("Loading BGE-M3 ...")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)
    logger.info("BGE-M3 loaded")

    logger.info("Connecting Qdrant at %s ...", QDRANT_PATH)
    if not os.path.exists(QDRANT_PATH):
        logger.error("QDRANT_PATH does not exist: %s", repr(QDRANT_PATH))
        raise FileNotFoundError(f"QDRANT_PATH not found: {QDRANT_PATH}")
    qdrant = QdrantClient(path=QDRANT_PATH)
    info = qdrant.get_collection(COLLECTION_NAME)
    points_count = info.points_count
    logger.info("Connected. %d points.", points_count)
    print(f"Connected. {points_count} points.")

    yield

    if qdrant is not None:
        qdrant.close()
        logger.info("Qdrant closed")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="雅迪 DM6 说明书问答 API v6", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000","http://127.0.0.1:8000",
        "http://localhost:3000","http://127.0.0.1:3000","null",
    ],
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20, description="检索文档数量")

class SourceItem(BaseModel):
    chunk_id: str
    content_type: str
    semantic_type: str = ""
    part: str
    section: str
    text_preview: str
    text_full: str = ""
    text_is_short: bool = True
    in_context: bool = False
    score: float
    component: list[str] = []
    domain_tags: list[str] = []
    fault_symptom: str = ""
    repair_level: str = ""
    risk_level: str = ""

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

class HealthResponse(BaseModel):
    status: str
    qdrant_points: int

# ---------------------------------------------------------------------------
# Phase 1: Query Expansion
# ---------------------------------------------------------------------------
async def expand_query(query: str) -> str:
    """Use qwen2.5:7b to extract keywords and expand the query."""
    prompt = f"从以下用户问题中提取关键检索词，用空格分隔。只输出关键词，不要解释。\n\n问题：{query}\n关键词："
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0), proxy=None, trust_env=False) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 80}}
            )
            resp.raise_for_status()
            data = resp.json()
            keywords = data.get("response", "").strip()
            if keywords:
                logger.info("Expanded keywords: %s", keywords)
                return f"{query} {keywords}"
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
    # Fallback: route-based keyword expansion
    extra = _fallback_keywords(query)
    if extra:
        return f"{query} {extra}"
    return query


def _fallback_keywords(query: str) -> str:
    """Rule-based keyword expansion when LLM is unavailable."""
    mapping = {
        "保修": "三包 保修期限 质保",
        "充电": "充电器 电池 充电孔",
        "故障": "故障 排除 维修 原因",
        "调速": "调速转把 故障 更换",
        "刹车": "刹车 制动 刹把",
        "电池": "蓄电池 锂电池 充电",
        "安装": "安装 装配 连接",
        "安全": "安全 注意事项 警告",
        "仪表": "仪表盘 指示灯 显示",
        "灯": "前照灯 尾灯 转向灯 刹车灯",
        "保养": "保养 维护 检查",
    }
    extras = []
    for k, v in mapping.items():
        if k in query:
            extras.append(v)
    return " ".join(extras)

# ---------------------------------------------------------------------------
# Phase 2: Encode
# ---------------------------------------------------------------------------
def _encode_query(query: str) -> tuple[list[float], SparseVector]:
    output = model.encode([query], return_dense=True, return_sparse=True, max_length=512)
    dense_vec = output["dense_vecs"][0].tolist()
    lexical_weights: dict[int, float] = output["lexical_weights"][0]
    sparse_vec = SparseVector(indices=list(lexical_weights.keys()), values=list(lexical_weights.values()))
    return dense_vec, sparse_vec

# ---------------------------------------------------------------------------
# Phase 2.5: Query Routing
# ---------------------------------------------------------------------------
def route_query(query: str) -> dict:
    FAULT_KW = ["故障","不工作","坏了","异常","怎么办","修","排除","原因","处理","解决"]
    SAFETY_KW = ["安全","危险","警告","注意","禁止","不要"]
    is_fault = any(kw in query for kw in FAULT_KW)
    is_safety = any(kw in query for kw in SAFETY_KW)
    if is_safety:
        return {"top_k": 10, "filter": Filter(must=[FieldCondition(key="semantic_type", match=MatchValue(value="风险警告"))]), "rerank": True}
    elif is_fault:
        return {"top_k": 10, "filter": None, "rerank": True}
    else:
        return {"top_k": 10, "filter": None, "rerank": True}

# ---------------------------------------------------------------------------
# Phase 3: Qdrant Search
# ---------------------------------------------------------------------------
PAYLOAD_FIELDS = [
    "chunk_id","content_type","part","section","text",
    "semantic_type","component","fault_symptom","repair_level",
    "risk_level","fault_triplet","domain_tags","parent_id","mom_id"
]

# ---------------------------------------------------------------------------
# Phase 4: LLM Rerank
# ---------------------------------------------------------------------------
async def rerank_llm(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Use LLM to rerank candidates from 10 to top_k."""
    if len(chunks) <= top_k:
        return chunks
    prompt = f"从以下候选段落中选出与用户问题最相关的 {top_k} 个。只输出编号，逗号分隔，按相关度从高到低排列。\n\n用户问题：{query}\n\n"
    for i, c in enumerate(chunks):
        snippet = c.get("text","")[:300].replace("\n"," ")
        prompt += f"[{i}] {snippet}\n\n"
    prompt += f"最相关的 {top_k} 个编号（从高到低）："
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0), proxy=None, trust_env=False) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "stream": False, "options": {"temperature": 0.1, "num_predict": 30}}
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            reranked = [chunks[i] for i in indices if 0 <= i < len(chunks)]
            return reranked[:top_k] if reranked else chunks[:top_k]
    except Exception as e:
        logger.warning("Rerank failed: %s", e)
    return chunks[:top_k]

# ---------------------------------------------------------------------------
# Phase 5: Metadata Boost
# ---------------------------------------------------------------------------
def metadata_boost(query: str, chunk: dict) -> float:
    """Calculate relevance boost based on metadata match."""
    boost = 0.0
    comps = (chunk.get("component") or []) if isinstance(chunk.get("component"), list) else []
    for comp in comps:
        if comp in query:
            boost += 0.15; break
    symptom = chunk.get("fault_symptom","") or ""
    if symptom:
        sw = set(symptom.replace("，"," ").replace("、"," ").split())
        qw = set(query.replace("？"," ").split())
        boost += 0.1 * min(len(sw & qw), 3)
    stype = chunk.get("semantic_type","")
    if any(k in query for k in ["怎么修","故障","不工作","坏了","异常"]) and stype == "故障诊断":
        boost += 0.1
    if any(k in query for k in ["怎么","如何","步骤","安装","充电"]) and stype == "操作步骤":
        boost += 0.1
    if any(k in query for k in ["安全","注意","危险","警告"]) and stype == "风险警告":
        boost += 0.15
    tags = (chunk.get("domain_tags") or []) if isinstance(chunk.get("domain_tags"), list) else []
    for tag in tags:
        if tag in query:
            boost += 0.05; break
    return min(boost, 0.3)

# ---------------------------------------------------------------------------
# Phase 6: Parent Chunk Aggregation
# ---------------------------------------------------------------------------
def expand_context(chunks: list[dict]) -> list[dict]:
    """Add parent/mom chunks to the result set."""
    ids = set()
    for c in chunks:
        if c.get("parent_id"):
            ids.add(c["parent_id"])
        if c.get("mom_id"):
            ids.add(c["mom_id"])
    if not ids:
        return chunks
    try:
        parents = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=list(ids),
            with_payload=["text","chunk_id","part","section","semantic_type","domain_tags","content_type"]
        )
        for p in parents:
            chunks.append({
                "chunk_id": p.payload.get("chunk_id", p.id),
                "text": p.payload.get("text",""),
                "part": p.payload.get("part",""),
                "section": p.payload.get("section",""),
                "semantic_type": p.payload.get("semantic_type",""),
                "domain_tags": p.payload.get("domain_tags",[]),
                "content_type": p.payload.get("content_type","text"),
                "score": 0.0,
                "is_parent": True,
            })
    except Exception as e:
        logger.warning("Parent expansion failed: %s", e)
    return chunks

# ---------------------------------------------------------------------------
# Context Assembly
# ---------------------------------------------------------------------------
def _build_context(hits: list[dict]) -> tuple[str, list[dict], set[str]]:
    """Assemble context string from retrieved chunks."""
    MIN_TEXT_LEN = 50
    context = ""
    included_sources: list[dict] = []
    included_ids: set[str] = set()

    for r in hits:
        text = r.get("text","").strip()
        if len(text) < MIN_TEXT_LEN:
            continue
        part = r.get("part","未知章节")
        section = r.get("section","") or part
        chunk_id = r.get("chunk_id","")

        chunk = (
            f"[文档名称]\n{part}\n\n"
            f"[章节]\n{section}\n\n"
            f"[内容]\n{text}"
        )
        if len(context) + len(chunk) + 2 <= MAX_CONTEXT_CHARS:
            if context:
                context += "\n\n"
            context += chunk
            included_sources.append(r)
            included_ids.add(chunk_id)

    return context, included_sources, included_ids

def strip_thinking(text: str) -> str:
    markers = [
        r'\*\*答案\*\*[：:]?\s*', r'直接回答[：:]?\s*', r'最终回答[：:]?\s*',
        r'\n\n(?=#{1,3}\s)', r'\n\n(?=根据说明书)', r'\n\n(?=雅迪)',
        r'【答案】[：:]?\s*',
    ]
    for marker in markers:
        m = re.search(marker, text)
        if m:
            after = text[m.end():].strip()
            if len(after) > 20:
                return after
    parts = re.split(r'\n\n(?=[^\n]{20,})', text)
    if len(parts) > 1:
        candidates = [p for p in parts[1:] if len(p) > 50]
        if candidates:
            return candidates[-1].strip()
    cleaned = re.sub(
        r'^(?:Thinking Process:|#+\s*(?:Analysis|Analyze|思考|推理|Draft)).*?(?:\n|$)',
        '', text, flags=re.MULTILINE,
    )
    cleaned = cleaned.strip()
    if cleaned and len(cleaned) > 20:
        return cleaned
    return text

async def _generate_answer(ctx: str, query: str) -> str:
    prompt = f"""用户问题：

{{{{question}}}}

{query}

检索结果：

{{{{context}}}}

{ctx}

每条检索结果格式：

[文档名称]
{{{{document_name}}}}

[章节]
{{{{section_name}}}}

[内容]
{{{{chunk}}}}

请严格遵循以下规则：

1. 只能依据检索结果回答。
2. 不允许使用外部知识、推测或编造。
3. 如果无法回答，直接回复"知识库中未找到相关信息。"

回答格式要求（必须严格遵守）：
- 以"【答案】"开头，独占一行。
- 每条要点独占一行，用"1. 2. 3. "编号。
- 要点之间用空行分隔。
- 答案之后独占一行写"引用来源："。
- 每条引用独占一行，格式为"[序号] 文档名称 · 章节名"。
- 引用与答案要点对应，不重复。

开始回答。"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0), proxy=None, trust_env=False) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 1024}},
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
            return strip_thinking(raw)
    except httpx.HTTPStatusError as e:
        logger.error("Ollama HTTP error: %s", e)
        raise HTTPException(status_code=502, detail=f"Ollama 返回错误: {e.response.status_code}")
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
        logger.error("Ollama connection/timeout: %s", e)
        raise HTTPException(status_code=503, detail=f"Ollama 服务不可用: {str(e)}")

# ---------------------------------------------------------------------------
# Source Building
# ---------------------------------------------------------------------------
def _build_sources(hits: list[dict], included_ids: set[str] | None = None) -> list[SourceItem]:
    if included_ids is None:
        included_ids = set()
    sources: list[SourceItem] = []
    for r in hits:
        text = r.get("text","")
        cid = r.get("chunk_id","")
        preview = text[:TEXT_PREVIEW_LEN]
        if len(text) > TEXT_PREVIEW_LEN:
            preview += "…"
        comp = r.get("component") or []
        if isinstance(comp, str):
            comp = [comp]
        tags = r.get("domain_tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        sources.append(SourceItem(
            chunk_id=cid,
            content_type=r.get("content_type",""),
            semantic_type=r.get("semantic_type",""),
            part=r.get("part",""),
            section=r.get("section",""),
            text_preview=preview,
            text_full=text,
            text_is_short=len(text) <= TEXT_PREVIEW_LEN,
            in_context=cid in included_ids,
            score=round(r.get("score", 0), 3),
            component=comp,
            domain_tags=tags,
            fault_symptom=r.get("fault_symptom","") or "",
            repair_level=r.get("repair_level","") or "",
            risk_level=r.get("risk_level","") or "",
        ))
    return sources

# ---------------------------------------------------------------------------
# Type icon mapping
# ---------------------------------------------------------------------------
TYPE_ICONS = {
    "故障诊断": "🔧", "风险警告": "⚠️", "操作步骤": "📋",
    "参数查询": "📊", "部件说明": "🔍", "电路拓扑": "⚡",
    "概述说明": "📝",
}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", qdrant_points=points_count)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if model is None or qdrant is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    # Phase 1: Query expansion
    logger.info("Phase 1: Expanding query...")
    expanded_query = await expand_query(req.query)
    logger.info("Expanded: %s", expanded_query[:120])

    # Phase 2: Encode
    logger.info("Phase 2: Encoding...")
    dense_vec, sparse_vec = _encode_query(expanded_query)

    # Phase 2.5: Route
    route = route_query(req.query)
    logger.info("Phase 2.5: Routing -> filter=%s, top_k=%d", bool(route.get("filter")), route["top_k"])

    # Phase 3: Qdrant search
    logger.info("Phase 3: Searching Qdrant...")
    prefetch_limit = max(route["top_k"] * 4, 40)
    search_kwargs = {
        "collection_name": COLLECTION_NAME,
        "prefetch": [
            Prefetch(query=dense_vec, using="dense", limit=prefetch_limit),
            Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
        ],
        "query": FusionQuery(fusion="rrf"),
        "limit": route["top_k"],
        "with_payload": PAYLOAD_FIELDS,
    }
    if route.get("filter"):
        search_kwargs["query_filter"] = route["filter"]
    results = qdrant.query_points(**search_kwargs)
    hits_raw = results.points

    # Convert to dicts
    hits = []
    for pt in hits_raw:
        p = pt.payload or {}
        d = {k: p.get(k) for k in PAYLOAD_FIELDS}
        d["score"] = pt.score
        d["_point_id"] = pt.id
        hits.append(d)
    logger.info("Retrieved %d results", len(hits))

    # Phase 4: LLM Rerank
    if route.get("rerank") and len(hits) > 5:
        logger.info("Phase 4: Reranking %d -> 5...", len(hits))
        hits = await rerank_llm(req.query, hits, top_k=5)

    # Phase 5: Metadata Boost
    logger.info("Phase 5: Metadata boost...")
    for h in hits:
        boost = metadata_boost(req.query, h)
        h["score"] = round(h.get("score", 0) + boost, 3)
    hits.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Phase 6: Parent expansion
    logger.info("Phase 6: Parent expansion...")
    hits = expand_context(hits)

    # Build context
    context, _, included_ids = _build_context(hits)

    # Generate answer
    logger.info("Generating answer...")
    answer = await _generate_answer(context, req.query)
    logger.info("Answer generated (%d chars)", len(answer))

    # Build sources
    sources = _build_sources(hits, included_ids)

    return ChatResponse(answer=answer, sources=sources)


@app.post("/api/debug/context")
async def debug_context(req: ChatRequest):
    if model is None or qdrant is None:
        raise HTTPException(status_code=503, detail="Not ready")

    expanded_query = await expand_query(req.query)
    dense_vec, sparse_vec = _encode_query(expanded_query)
    route = route_query(req.query)
    prefetch_limit = max(route["top_k"] * 4, 40)
    search_kwargs = {
        "collection_name": COLLECTION_NAME,
        "prefetch": [
            Prefetch(query=dense_vec, using="dense", limit=prefetch_limit),
            Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
        ],
        "query": FusionQuery(fusion="rrf"),
        "limit": route["top_k"],
        "with_payload": PAYLOAD_FIELDS,
    }
    if route.get("filter"):
        search_kwargs["query_filter"] = route["filter"]
    results = qdrant.query_points(**search_kwargs)
    hits_raw = results.points

    raw_chunks = []
    for pt in hits_raw:
        p = pt.payload or {}
        text = p.get("text","")
        raw_chunks.append({
            "chunk_id": p.get("chunk_id",""),
            "part": p.get("part",""),
            "section": p.get("section",""),
            "semantic_type": p.get("semantic_type",""),
            "text_len": len(text),
            "text_preview": text[:200],
            "score": round(pt.score, 6),
        })

    return {
        "query": req.query,
        "expanded_query": expanded_query,
        "route": route,
        "total_retrieved": len(hits_raw),
        "raw_chunks": raw_chunks,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if model is None or qdrant is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    # Phase 1
    expanded_query = await expand_query(req.query)
    # Phase 2
    dense_vec, sparse_vec = _encode_query(expanded_query)
    # Phase 2.5
    route = route_query(req.query)
    # Phase 3
    prefetch_limit = max(route["top_k"] * 4, 40)
    search_kwargs = {
        "collection_name": COLLECTION_NAME,
        "prefetch": [
            Prefetch(query=dense_vec, using="dense", limit=prefetch_limit),
            Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
        ],
        "query": FusionQuery(fusion="rrf"),
        "limit": route["top_k"],
        "with_payload": PAYLOAD_FIELDS,
    }
    if route.get("filter"):
        search_kwargs["query_filter"] = route["filter"]
    results = qdrant.query_points(**search_kwargs)

    hits = []
    for pt in results.points:
        p = pt.payload or {}
        d = {k: p.get(k) for k in PAYLOAD_FIELDS}
        d["score"] = pt.score
        hits.append(d)
    logger.info("Stream: retrieved %d results", len(hits))

    # Phase 4
    if route.get("rerank") and len(hits) > 5:
        hits = await rerank_llm(req.query, hits, top_k=5)
    # Phase 5
    for h in hits:
        boost = metadata_boost(req.query, h)
        h["score"] = round(h.get("score", 0) + boost, 3)
    hits.sort(key=lambda x: x.get("score", 0), reverse=True)
    # Phase 6
    hits = expand_context(hits)

    context, _, included_ids = _build_context(hits)
    sources = _build_sources(hits, included_ids)

    prompt = f"""用户问题：

{{{{question}}}}

{req.query}

检索结果：

{{{{context}}}}

{context}

每条检索结果格式：

[文档名称]
{{{{document_name}}}}

[章节]
{{{{section_name}}}}

[内容]
{{{{chunk}}}}

请严格遵循以下规则：

1. 只能依据检索结果回答。
2. 不允许使用外部知识、推测或编造。
3. 如果无法回答，直接回复"知识库中未找到相关信息。"

回答格式要求（必须严格遵守）：
- 以"【答案】"开头，独占一行。
- 每条要点独占一行，用"1. 2. 3. "编号。
- 要点之间用空行分隔。
- 答案之后独占一行写"引用来源："。
- 每条引用独占一行，格式为"[序号] 文档名称 · 章节名"。
- 引用与答案要点对应，不重复。

开始回答。"""

    async def event_stream() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'start', 'message': '正在生成回答...'})}\n\n"
        full_answer = ""
        async for token in _stream_ollama(prompt):
            if token.startswith("__ERROR__"):
                err_msg = token[len("__ERROR__"):] or "Ollama 服务异常"
                yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
                return
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources]})}\n\n"
        clean = strip_thinking(full_answer)
        yield f"data: {json.dumps({'type': 'done', 'answer': clean})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _stream_ollama(prompt: str) -> AsyncGenerator[str, None]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0), proxy=None, trust_env=False) as client:
            async with client.stream(
                "POST", f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "options": {"temperature": 0.3, "num_predict": 1024}},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if chunk.get("done", False):
                                break
                            token = chunk.get("response", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
    except httpx.HTTPStatusError as e:
        logger.error("Ollama HTTP error in stream: %s", e)
        yield "__ERROR__Ollama 返回错误"
    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
        logger.error("Ollama connection/timeout in stream: %s", e)
        yield "__ERROR__Ollama 服务不可用"
    except Exception as e:
        logger.error("Unexpected error in stream: %s", e)
        yield f"__ERROR__{str(e)}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_server:app", host="0.0.0.0", port=8000, reload=False)
