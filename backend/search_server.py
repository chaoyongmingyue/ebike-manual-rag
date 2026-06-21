"""
Yadea DM6 Electric Bicycle Manual Knowledge Base Q&A Backend v6
FastAPI + Qdrant (hybrid RRF) + BGE-M3 + Ollama (qwen2.5:7b)
v6 additions: Query expansion, Routing, LLM Rerank, Metadata Boost, Parent aggregation
v6.1: Structured logging, deep health checks, memory metrics, degradation, config file
"""

from __future__ import annotations

import json
import logging
import os
import re
import hashlib
import time
import uuid
import yaml
from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FusionQuery, Prefetch, SparseVector, Filter, FieldCondition, MatchValue,
)

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    """Load config.json; merge with hardcoded defaults. Config values take precedence."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    defaults: dict[str, Any] = {
        "qdrant_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")),
        "collection": "ebike_manual",
        "ollama": {
            "url": "http://localhost:11434/api/generate",
            "model": "qwen2.5:7b",
            "temperature": 0.3,
            "num_predict": 512,
            "timeout_seconds": 120,
        },
        "retrieval": {
            "top_k": 10,
            "rerank_top_k": 5,
            "context_max_chars": 4000,
        },
        "retry": {
            "ollama_max_retries": 2,
            "qdrant_max_retries": 3,
            "retry_delay_ms": 500,
        },
        "server": {
            "port": 8000,
        },
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Deep merge: only override keys present in loaded config
            for section in defaults:
                if section in loaded and isinstance(defaults[section], dict) and isinstance(loaded[section], dict):
                    defaults[section].update(loaded[section])
                elif section in loaded:
                    defaults[section] = loaded[section]
        except Exception as e:
            print(f"WARNING: Failed to load config.json: {e}")

    return defaults


CONFIG = _load_config()

# ---------------------------------------------------------------------------
# Resolved configuration values
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.environ.get(
    "MODEL_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "bge-m3")),
)

_qdrant_raw: str = os.environ.get("QDRANT_PATH", CONFIG["qdrant_path"])
QDRANT_PATH: str = _qdrant_raw if os.path.isabs(_qdrant_raw) else os.path.abspath(
    os.path.join(os.path.dirname(__file__), _qdrant_raw)
)

COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", CONFIG["collection"])

_ollama_cfg = CONFIG["ollama"]
_ollama_url_raw: str = os.environ.get("OLLAMA_URL", _ollama_cfg["url"])
OLLAMA_URL: str = _ollama_url_raw.removesuffix("/api/generate").rstrip("/")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", _ollama_cfg["model"])
OLLAMA_TEMPERATURE: float = _ollama_cfg.get("temperature", 0.3)
OLLAMA_NUM_PREDICT: int = _ollama_cfg.get("num_predict", 512)
OLLAMA_TIMEOUT: int = _ollama_cfg.get("timeout_seconds", 120)

_retrieval_cfg = CONFIG["retrieval"]
MAX_CONTEXT_CHARS: int = _retrieval_cfg.get("context_max_chars", 4000)
RERANK_TOP_K: int = _retrieval_cfg.get("rerank_top_k", 5)
DEFAULT_TOP_K: int = _retrieval_cfg.get("top_k", 10)

_retry_cfg = CONFIG["retry"]
RETRY_OLLAMA_MAX: int = _retry_cfg.get("ollama_max_retries", 2)
RETRY_QDRANT_MAX: int = _retry_cfg.get("qdrant_max_retries", 3)
RETRY_DELAY_MS: int = _retry_cfg.get("retry_delay_ms", 500)

SERVER_PORT: int = int(os.environ.get("PORT", CONFIG["server"].get("port", 8000)))

TEXT_PREVIEW_LEN: int = 120

# ---------------------------------------------------------------------------
# Prompt loading from YAML
# ---------------------------------------------------------------------------
PROMPT_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(prompt_id: str, version: str | None = None) -> tuple[str, str]:
    """Load a prompt template from YAML.

    Args:
        prompt_id: e.g. "answer", "query_expand", "rerank"
        version: e.g. "v1", "v2". If None, auto-selects the latest version.

    Returns:
        (prompt_template, version_string)
    """
    prompt_path = PROMPT_DIR / prompt_id
    if not prompt_path.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_path}")

    if version is None:
        versions = sorted(prompt_path.glob("v*.yaml"))
        if not versions:
            raise FileNotFoundError(f"No prompt version found in: {prompt_path}")
        version = versions[-1].stem  # e.g. "v1"

    yaml_file = prompt_path / f"{version}.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {yaml_file}")

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["prompt"], data["version"]


# Load all prompts at startup — declared here, loaded after logger init
QUERY_EXPAND_PROMPT: str = ""
QUERY_EXPAND_VERSION: str = ""
RERANK_PROMPT: str = ""
RERANK_VERSION: str = ""
ANSWER_PROMPT: str = ""
ANSWER_VERSION: str = ""

# ---------------------------------------------------------------------------
# In-memory observability stores (restart clears)
# ---------------------------------------------------------------------------
struct_logs: deque[dict] = deque(maxlen=200)          # structured log ring buffer

request_total: int = 0
request_success: int = 0
request_error: int = 0
request_timestamps: deque[float] = deque(maxlen=500)   # timestamps for last_hour

latencies: deque[int] = deque(maxlen=200)              # all request total_ms
all_recall_counts: deque[int] = deque(maxlen=500)      # recall counts per request
all_reranked_counts: deque[int] = deque(maxlen=500)    # reranked counts per request
error_types: dict[str, int] = {}                        # e.g. {"ollama_timeout": 2}

_start_time: float = time.monotonic()                  # for uptime calculation
_degraded_flag: bool = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ebike-search")

# Load prompts from YAML (after logger is available)
QUERY_EXPAND_PROMPT, QUERY_EXPAND_VERSION = load_prompt("query_expand")
RERANK_PROMPT, RERANK_VERSION = load_prompt("rerank")
ANSWER_PROMPT, ANSWER_VERSION = load_prompt("answer")
logger.info(
    "Prompts loaded: query_expand=%s rerank=%s answer=%s",
    QUERY_EXPAND_VERSION, RERANK_VERSION, ANSWER_VERSION,
)

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
    global model, qdrant, points_count, _degraded_flag

    logger.info("Loading BGE-M3 ...")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)
    logger.info("BGE-M3 loaded")

    # Connect Qdrant with retry
    logger.info("Connecting Qdrant at %s ...", QDRANT_PATH)
    if not os.path.exists(QDRANT_PATH):
        logger.error("QDRANT_PATH does not exist: %s", repr(QDRANT_PATH))
        raise FileNotFoundError(f"QDRANT_PATH not found: {QDRANT_PATH}")

    connected = False
    last_error: Exception | None = None
    for attempt in range(RETRY_QDRANT_MAX + 1):  # initial + retries
        try:
            qdrant = QdrantClient(path=QDRANT_PATH)
            info = qdrant.get_collection(COLLECTION_NAME)
            points_count = info.points_count
            logger.info("Connected. %d points.", points_count)
            print(f"Connected. {points_count} points.")
            connected = True
            break
        except Exception as e:
            last_error = e
            if attempt < RETRY_QDRANT_MAX:
                delay = 2 ** (attempt + 1)  # 2s, 4s, 8s
                logger.warning(
                    "Qdrant connection attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, RETRY_QDRANT_MAX + 1, e, delay,
                )
                time.sleep(delay)

    if not connected:
        _degraded_flag = True
        logger.error(
            "Qdrant connection failed after %d attempts: %s. Starting in degraded mode.",
            RETRY_QDRANT_MAX + 1, last_error,
        )
        print(f"WARNING: Qdrant unavailable. Starting in degraded mode.")

    yield

    if qdrant is not None:
        qdrant.close()
        logger.info("Qdrant closed")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="雅迪 DM6 说明书问答 API v6", version="2.0.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:3000", "http://127.0.0.1:3000", "null",
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
    trace_id: str | None = Field(default=None, alias="_trace_id")
    degraded: bool = Field(default=False, alias="_degraded")
    degraded_reason: str | None = Field(default=None, alias="_degraded_reason")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_route_type(query: str) -> str:
    """Determine query route type for structured logging."""
    FAULT_KW = ["故障", "不工作", "坏了", "异常", "怎么办", "修", "排除", "原因", "处理", "解决"]
    SAFETY_KW = ["安全", "危险", "警告", "注意", "禁止", "不要"]
    if any(kw in query for kw in SAFETY_KW):
        return "safety"
    elif any(kw in query for kw in FAULT_KW):
        return "fault"
    return "general"


def _make_trace_id() -> str:
    return uuid.uuid4().hex[:8]


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Phase 1: Query Expansion
# ---------------------------------------------------------------------------
async def expand_query(query: str) -> str:
    """Use qwen2.5:7b to extract keywords and expand the query."""
    prompt = QUERY_EXPAND_PROMPT.format(query=query)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0), proxy=None, trust_env=False,
        ) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 80},
                },
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
    sparse_vec = SparseVector(
        indices=list(lexical_weights.keys()), values=list(lexical_weights.values()),
    )
    return dense_vec, sparse_vec


# ---------------------------------------------------------------------------
# Phase 2.5: Query Routing
# ---------------------------------------------------------------------------
def route_query(query: str) -> dict:
    FAULT_KW = ["故障", "不工作", "坏了", "异常", "怎么办", "修", "排除", "原因", "处理", "解决"]
    SAFETY_KW = ["安全", "危险", "警告", "注意", "禁止", "不要"]
    is_fault = any(kw in query for kw in FAULT_KW)
    is_safety = any(kw in query for kw in SAFETY_KW)
    if is_safety:
        return {
            "top_k": DEFAULT_TOP_K,
            "filter": Filter(must=[FieldCondition(key="semantic_type", match=MatchValue(value="风险警告"))]),
            "rerank": True,
        }
    elif is_fault:
        return {"top_k": DEFAULT_TOP_K, "filter": None, "rerank": True}
    else:
        return {"top_k": DEFAULT_TOP_K, "filter": None, "rerank": True}


# ---------------------------------------------------------------------------
# Phase 3: Qdrant Search
# ---------------------------------------------------------------------------
PAYLOAD_FIELDS = [
    "chunk_id", "content_type", "part", "section", "text",
    "semantic_type", "component", "fault_symptom", "repair_level",
    "risk_level", "fault_triplet", "domain_tags", "parent_id", "mom_id",
]


# ---------------------------------------------------------------------------
# Phase 4: LLM Rerank
# ---------------------------------------------------------------------------
async def rerank_llm(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Use LLM to rerank candidates to top_k."""
    if len(chunks) <= top_k:
        return chunks
    candidates_text = ""
    for i, c in enumerate(chunks):
        snippet = c.get("text", "")[:300].replace("\n", " ")
        candidates_text += f"[{i}] {snippet}\n\n"
    prompt = RERANK_PROMPT.format(top_k=top_k, query=query, candidates=candidates_text)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0), proxy=None, trust_env=False,
        ) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL, "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 30},
                },
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
            boost += 0.15
            break
    symptom = chunk.get("fault_symptom", "") or ""
    if symptom:
        sw = set(symptom.replace("，", " ").replace("、", " ").split())
        qw = set(query.replace("？", " ").split())
        boost += 0.1 * min(len(sw & qw), 3)
    stype = chunk.get("semantic_type", "")
    if any(k in query for k in ["怎么修", "故障", "不工作", "坏了", "异常"]) and stype == "故障诊断":
        boost += 0.1
    if any(k in query for k in ["怎么", "如何", "步骤", "安装", "充电"]) and stype == "操作步骤":
        boost += 0.1
    if any(k in query for k in ["安全", "注意", "危险", "警告"]) and stype == "风险警告":
        boost += 0.15
    tags = (chunk.get("domain_tags") or []) if isinstance(chunk.get("domain_tags"), list) else []
    for tag in tags:
        if tag in query:
            boost += 0.05
            break
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
            with_payload=[
                "text", "chunk_id", "part", "section",
                "semantic_type", "domain_tags", "content_type",
            ],
        )
        for p in parents:
            chunks.append({
                "chunk_id": p.payload.get("chunk_id", p.id),
                "text": p.payload.get("text", ""),
                "part": p.payload.get("part", ""),
                "section": p.payload.get("section", ""),
                "semantic_type": p.payload.get("semantic_type", ""),
                "domain_tags": p.payload.get("domain_tags", []),
                "content_type": p.payload.get("content_type", "text"),
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
        text = r.get("text", "").strip()
        if len(text) < MIN_TEXT_LEN:
            continue
        part = r.get("part", "未知章节")
        section = r.get("section", "") or part
        chunk_id = r.get("chunk_id", "")

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
    """Call Ollama to generate an answer. Raises raw httpx exceptions for caller to handle."""
    prompt = ANSWER_PROMPT.format(query=query, context=ctx)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0), proxy=None, trust_env=False,
    ) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": OLLAMA_TEMPERATURE, "num_predict": OLLAMA_NUM_PREDICT},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "")
        return strip_thinking(raw)


# ---------------------------------------------------------------------------
# Source Building
# ---------------------------------------------------------------------------
def _build_sources(hits: list[dict], included_ids: set[str] | None = None) -> list[SourceItem]:
    if included_ids is None:
        included_ids = set()
    sources: list[SourceItem] = []
    for r in hits:
        text = r.get("text", "")
        cid = r.get("chunk_id", "")
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
            content_type=r.get("content_type", ""),
            semantic_type=r.get("semantic_type", ""),
            part=r.get("part", ""),
            section=r.get("section", ""),
            text_preview=preview,
            text_full=text,
            text_is_short=len(text) <= TEXT_PREVIEW_LEN,
            in_context=cid in included_ids,
            score=round(r.get("score", 0), 3),
            component=comp,
            domain_tags=tags,
            fault_symptom=r.get("fault_symptom", "") or "",
            repair_level=r.get("repair_level", "") or "",
            risk_level=r.get("risk_level", "") or "",
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


# ===========================================================================
# Routes
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /api/health — Deep health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    # Check Qdrant
    qdrant_status = "ok"
    qdrant_latency_ms = 0
    qdrant_points_val = points_count
    try:
        t0 = time.perf_counter()
        if qdrant is not None:
            count_result = qdrant.count(COLLECTION_NAME)
            qdrant_points_val = count_result.count
        qdrant_latency_ms = round((time.perf_counter() - t0) * 1000)
    except Exception as e:
        qdrant_status = "error"
        qdrant_points_val = 0
        logger.warning("Health check: Qdrant error: %s", e)

    # Check Ollama
    ollama_status = "ok"
    ollama_latency_ms = 0
    test_response = ""
    try:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0), proxy=None, trust_env=False,
        ) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL, "prompt": "你好", "stream": False,
                    "options": {"num_predict": 5},
                },
            )
            ollama_latency_ms = round((time.perf_counter() - t0) * 1000)
            if resp.status_code == 200:
                test_response = (resp.json().get("response", "") or "")[:10]
            else:
                ollama_status = "error"
    except Exception as e:
        ollama_status = "error"
        logger.warning("Health check: Ollama error: %s", e)

    # Overall status
    if qdrant_status == "ok" and ollama_status == "ok":
        overall = "healthy"
    elif qdrant_status == "error" and ollama_status == "error":
        overall = "down"
    else:
        overall = "degraded"

    return {
        "status": overall,
        "qdrant": {
            "status": qdrant_status,
            "points": qdrant_points_val,
            "collection": COLLECTION_NAME,
            "latency_ms": qdrant_latency_ms,
        },
        "ollama": {
            "status": ollama_status,
            "model": OLLAMA_MODEL,
            "latency_ms": ollama_latency_ms,
            "test_response": test_response,
        },
        "uptime_seconds": round(time.monotonic() - _start_time),
        "version": "v6",
    }


# ---------------------------------------------------------------------------
# GET /api/metrics — Memory metrics
# ---------------------------------------------------------------------------
@app.get("/api/metrics")
async def metrics():
    global request_total, request_success, request_error

    # Last hour requests
    now = time.time()
    last_hour = sum(1 for ts in request_timestamps if now - ts <= 3600)

    # Latency stats
    lat_list = list(latencies)
    sorted_lat = sorted(lat_list) if lat_list else [0]
    n = len(sorted_lat)
    avg_ms = round(sum(sorted_lat) / n) if n > 0 else 0
    p50_ms = sorted_lat[n // 2] if n > 0 else 0
    p95_ms = sorted_lat[int(n * 0.95)] if n > 1 else (sorted_lat[0] if n > 0 else 0)
    max_ms = sorted_lat[-1] if n > 0 else 0
    recent = list(lat_list)[-10:] if len(lat_list) > 10 else lat_list

    # Recall stats
    recalls = list(all_recall_counts)
    rerankeds = list(all_reranked_counts)
    avg_top_k = round(sum(recalls) / len(recalls)) if recalls else 0
    avg_reranked = round(sum(rerankeds) / len(rerankeds)) if rerankeds else 0

    return {
        "requests": {
            "total": request_total,
            "success": request_success,
            "error": request_error,
            "last_hour": last_hour,
        },
        "latency": {
            "avg_ms": avg_ms,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "max_ms": max_ms,
            "recent": recent,
        },
        "recall": {
            "avg_top_k": avg_top_k,
            "avg_reranked": avg_reranked,
        },
        "errors": error_types,
        "memory": {
            "log_buffer_size": len(struct_logs),
        },
    }


# ---------------------------------------------------------------------------
# GET /api/logs — Structured log query
# ---------------------------------------------------------------------------
@app.get("/api/logs")
async def get_logs(limit: int = Query(default=20, ge=1, le=200), status: str | None = Query(default=None)):
    logs = list(struct_logs)
    if status:
        logs = [l for l in logs if l.get("status") == status]
    return logs[-limit:]


# ---------------------------------------------------------------------------
# POST /api/chat — Main Q&A endpoint with observability
# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global request_total, request_success, request_error, error_types

    trace_id = _make_trace_id()
    route_type = _get_route_type(req.query)
    request_total += 1
    request_timestamps.append(time.time())

    degraded = False
    degraded_reason: str | None = None

    if model is None or qdrant is None:
        request_error += 1
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    # ---- Timing ----
    t_start = time.perf_counter()

    # Phase 1: Query expansion
    expanded_query = await expand_query(req.query)
    t1 = time.perf_counter()

    # Phase 2: Encode
    dense_vec, sparse_vec = _encode_query(expanded_query)
    t2 = time.perf_counter()

    # Phase 2.5: Route
    route = route_query(req.query)
    t25 = time.perf_counter()

    # Phase 3: Qdrant search
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
    try:
        results = qdrant.query_points(**search_kwargs)
    except Exception as e:
        logger.error("Qdrant search failed: %s", e)
        error_types["qdrant_error"] = error_types.get("qdrant_error", 0) + 1
        request_error += 1
        raise HTTPException(status_code=503, detail=f"Qdrant 检索失败: {str(e)}")
    hits_raw = results.points
    t3 = time.perf_counter()

    # Convert to dicts
    hits = []
    for pt in hits_raw:
        p = pt.payload or {}
        d = {k: p.get(k) for k in PAYLOAD_FIELDS}
        d["score"] = pt.score
        d["_point_id"] = pt.id
        hits.append(d)
    recall_count = len(hits)

    # Phase 4: LLM Rerank
    reranked_count = len(hits)
    if route.get("rerank") and len(hits) > RERANK_TOP_K:
        logger.info("Phase 4: Reranking %d -> %d...", len(hits), RERANK_TOP_K)
        hits = await rerank_llm(req.query, hits, top_k=RERANK_TOP_K)
        reranked_count = len(hits)
    t4 = time.perf_counter()

    # Phase 5: Metadata Boost
    for h in hits:
        boost = metadata_boost(req.query, h)
        h["score"] = round(h.get("score", 0) + boost, 3)
    hits.sort(key=lambda x: x.get("score", 0), reverse=True)
    t5 = time.perf_counter()

    # Phase 6: Parent expansion
    hits = expand_context(hits)
    t6 = time.perf_counter()

    # Build context
    context, _, included_ids = _build_context(hits)

    # Generate answer — with degradation
    answer = ""
    t7 = time.perf_counter()  # default if degradation
    try:
        answer = await _generate_answer(context, req.query)
        t7 = time.perf_counter()
        request_success += 1
    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
        t7 = time.perf_counter()
        logger.warning("Ollama unavailable (timeout/connect), fallback to retrieval-only: %s", e)
        error_types["ollama_timeout"] = error_types.get("ollama_timeout", 0) + 1
        request_success += 1  # still counts as success — we served a degraded response
        fallback_parts = []
        for c in hits[:RERANK_TOP_K]:
            part_label = c.get("part", "")
            section_label = c.get("section", "")
            text_snippet = (c.get("text", "") or "")[:300]
            fallback_parts.append(f"[{part_label} {section_label}]\n{text_snippet}")
        answer = (
            "⚠️ AI 生成服务暂时不可用。以下是检索到的最相关内容。\n\n"
            + "\n\n".join(fallback_parts)
        )
        degraded = True
        degraded_reason = "Ollama 服务不可用，返回纯检索结果"
    except httpx.HTTPStatusError as e:
        t7 = time.perf_counter()
        logger.error("Ollama HTTP error: %s", e)
        error_types["ollama_timeout"] = error_types.get("ollama_timeout", 0) + 1
        request_error += 1
        raise HTTPException(status_code=502, detail=f"Ollama 返回错误: {e.response.status_code}")
    except Exception as e:
        t7 = time.perf_counter()
        logger.error("Ollama unexpected error: %s", e)
        error_types["ollama_timeout"] = error_types.get("ollama_timeout", 0) + 1
        fallback_parts = []
        for c in hits[:RERANK_TOP_K]:
            part_label = c.get("part", "")
            section_label = c.get("section", "")
            text_snippet = (c.get("text", "") or "")[:300]
            fallback_parts.append(f"[{part_label} {section_label}]\n{text_snippet}")
        answer = (
            "⚠️ AI 生成服务暂时不可用。以下是检索到的最相关内容。\n\n"
            + "\n\n".join(fallback_parts)
        )
        degraded = True
        degraded_reason = f"Ollama 服务不可用: {str(e)[:200]}"
        request_success += 1  # degraded but served

    total_ms = round((t7 - t_start) * 1000)
    latencies.append(total_ms)
    all_recall_counts.append(recall_count)
    all_reranked_counts.append(reranked_count)

    # Build structured log
    timing = {
        "query_expand_ms": round((t1 - t_start) * 1000),
        "encode_ms": round((t2 - t1) * 1000),
        "qdrant_search_ms": round((t3 - t2) * 1000),
        "rerank_ms": round((t4 - t3) * 1000),
        "boost_ms": round((t5 - t4) * 1000),
        "expand_parent_ms": round((t6 - t5) * 1000),
        "ollama_generate_ms": round((t7 - t6) * 1000),
        "total_ms": total_ms,
    }

    log_entry = {
        "trace_id": trace_id,
        "timestamp": _now_iso(),
        "query": req.query,
        "route": route_type,
        "timing": timing,
        "result": {
            "recall_count": recall_count,
            "reranked_count": reranked_count,
            "sources": [h.get("chunk_id", "") for h in hits[:RERANK_TOP_K]],
            "answer_length": len(answer),
        },
        "status": "success",
        "error": degraded_reason,
    }
    struct_logs.append(log_entry)
    logger.info("trace=%s route=%s total=%dms ok=%d", trace_id, route_type, total_ms, len(answer))

    # Build sources
    sources = _build_sources(hits, included_ids)

    return ChatResponse(
        answer=answer,
        sources=sources,
        trace_id=trace_id,
        degraded=degraded,
        degraded_reason=degraded_reason,
    )


# ---------------------------------------------------------------------------
# POST /api/debug/context
# ---------------------------------------------------------------------------
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
        text = p.get("text", "")
        raw_chunks.append({
            "chunk_id": p.get("chunk_id", ""),
            "part": p.get("part", ""),
            "section": p.get("section", ""),
            "semantic_type": p.get("semantic_type", ""),
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


# ---------------------------------------------------------------------------
# POST /api/chat/stream
# ---------------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if model is None or qdrant is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    trace_id = _make_trace_id()

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
    if route.get("rerank") and len(hits) > RERANK_TOP_K:
        hits = await rerank_llm(req.query, hits, top_k=RERANK_TOP_K)
    # Phase 5
    for h in hits:
        boost = metadata_boost(req.query, h)
        h["score"] = round(h.get("score", 0) + boost, 3)
    hits.sort(key=lambda x: x.get("score", 0), reverse=True)
    # Phase 6
    hits = expand_context(hits)

    context, _, included_ids = _build_context(hits)
    sources = _build_sources(hits, included_ids)

    prompt = ANSWER_PROMPT.format(query=req.query, context=context)

    async def event_stream() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'start', 'message': '正在生成回答...', '_trace_id': trace_id})}\n\n"
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
        yield f"data: {json.dumps({'type': 'done', 'answer': clean, '_trace_id': trace_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _stream_ollama(prompt: str) -> AsyncGenerator[str, None]:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0), proxy=None, trust_env=False,
        ) as client:
            async with client.stream(
                "POST", f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL, "prompt": prompt, "stream": True,
                    "options": {"temperature": OLLAMA_TEMPERATURE, "num_predict": OLLAMA_NUM_PREDICT},
                },
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
    uvicorn.run("search_server:app", host="0.0.0.0", port=SERVER_PORT, reload=False)
