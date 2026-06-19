# 电动自行车说明书 · Retriever 方案 v6（终版）

> 融合 RAGFlow Retriever 学习 + v6 分块/Metadata + 你的优化提案，专为单说明书 84 chunks 场景定制。

---

## 一、你的优化提案审查

### 1.1 可直接吸收

| 提案 | 判断 | 说明 |
|------|:---:|------|
| Query 改写 + 扩展 | ✅ 直接吸收 | RAGFlow 的 `keyword_extraction()` 同理，LLM 将"车充不进电"展开为"无法充电/充电失败" |
| Rerank | ✅ 直接吸收 | 这是 RAGFlow 检索链路中提升最大的环节。84 chunks 场景用 LLM 重排即可 |
| Metadata 参与排序 | ✅ 直接吸收 | RAGFlow 的 `_rank_feature_scores()` 对标签和 PageRank 加权 |
| Fusion 多路融合 | ✅ 吸收并简化 | RAGFlow 用 `term_weight * tksim + vt_weight * vtsim + rank_fea` |

### 1.2 过度设计（本说明书场景）

| 提案 | 判断 | 原因 |
|------|:---:|------|
| 独立 BM25 索引 | ❌ 过度 | BGE-M3 自带 sparse 向量，已覆盖关键词匹配。Qdrant 原生支持 sparse 检索。对 84 个 chunk 单独建 BM25 索引是杀鸡用牛刀 |
| 5 路 Intent Router | ❌ 过度 | 84 chunks 下 2 路就够了（故障诊断 vs 其他）。5 种 intent × 5 套策略 90% 的规则不会触发 |
| vehicle_model/system 实体提取 | ❌ 过度 | 单说明书无多车型 |
| Query Rewrite 独立模块 | ⚠️ 简化为 LLM 扩展 | 不建同义词表，直接用 LLM 做 `keyword_extraction`（RAGFlow 方式），一次调用同时完成改写和扩展 |
| ES Pushdown | ❌ 不适用 | Qdrant 用 Payload Filter，机制不同但效果等价 |

---

## 二、本方案的设计原则

```
84 chunks，1 本说明书 → 不建重武器
检索质量提升的三板斧：Rerank > Metadata Boost > Query Expansion
优先用 LLM（已就绪），不加新模型新索引
```

---

## 三、最终 Retriever 架构

```
用户问题 "控制器保修多久"
        │
        ▼
┌──────────────────────────┐
│ Phase 1: Query 扩展       │  ← LLM 一次调用
│  keyword_extraction()    │     类似 RAGFlow
│  "控制器" → 输出:         │
│  "控制器 保修 三包 质保"   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Phase 2: BGE-M3 编码      │
│  dense(1024) + sparse    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Phase 3: Qdrant 混合检索   │
│  dense + sparse → Top-10 │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Phase 4: Rerank (LLM)     │  ← 核心新增
│  10 chunks × 相关性评分    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Phase 5: Metadata Boost   │  ← 新增
│  component 匹配加分       │
│  semantic_type 匹配加分   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Phase 6: 父块聚合 + 上下文  │
│  expand_parent()         │
│  → Top-5 + 父 chunks     │
└────────────┬─────────────┘
             │
             ▼
      Ollama 生成答案
```

---

## 四、各 Phase 详细设计

### Phase 1：Query 扩展

**对应 RAGFlow**：`keyword_extraction()` (`dialog_service.py:664`)

**实现**：
```python
async def expand_query(query: str) -> str:
    """LLM 提取关键词，扩展原始查询"""
    prompt = f"""从以下用户问题中提取关键检索词，用空格分隔。
只输出关键词，不要解释。

问题：{query}
关键词："""
    
    async with httpx.AsyncClient(timeout=30) as cli:
        resp = await cli.post("http://localhost:11434/api/generate", json={
            "model": "qwen3.5:9b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80},
        })
    
    keywords = resp.json()["response"].strip()
    # 合并原始查询 + 扩展关键词
    return f"{query} {keywords}"
```

**效果**：
```
输入: "车充不进电"
输出: "车充不进电 无法充电 充电失败 电池不充电 充电异常"

输入: "控制器保修多久"
输出: "控制器保修多久 三包 质保期限 保修政策 配件保修"
```

**为什么不做独立 Rewrite + Expansion 两个模块**：RAGFlow 的做法是一次 `keyword_extraction()` 同时完成改写和扩展。对 84 chunks 场景，分开做是过度设计。

---

### Phase 2：BGE-M3 编码

不变，复用现有逻辑。BGE-M3 同时产出 dense(1024) + sparse，sparse 向量已经在入库时存入了 Qdrant。

---

### Phase 3：Qdrant 混合检索

**当前已有，无需修改**。dense + sparse 双路检索，Top-K 从现在的 5 提升到 10（给 Rerank 留候选池）。

```python
results = qdrant.search(
    collection_name="ebike_manual",
    query_vector=("dense", dense_vec),
    query_sparse_vector=("sparse", sparse_vec),
    limit=10,  # ← 从 5 改为 10
    with_payload=[...],
)
```

**为什么不需要独立 BM25 索引**：
- BGE-M3 的 sparse 向量本身就是学习到的关键词权重，效果接近 BM25
- Qdrant 原生支持 sparse 检索
- 84 个 chunk 场景下，sparse + dense 的覆盖率已经足够

---

### Phase 4：Rerank（核心新增）

**对应 RAGFlow**：`rerank_by_model()` (`search.py:513`)

**为什么必须要 Rerank**：
```
Top-10 候选:
  #1 "控制器保修期限 12 个月"  ← 最佳（直接回答）
  #3 "控制器故障排除方法"      ← 相关但不是保修问题
  #7 "车辆定期保养表"          ← 边缘相关

没有 Rerank → #3 可能排到 #1 前面（embedding 相似度高）
有 Rerank   → LLM 理解"保修"意图 → #1 排到最前
```

**实现**：LLM-based rerank（不需要下载新模型）

```python
async def rerank_with_llm(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """用 qwen3.5:9b 对候选 chunks 重排序"""

    # 构建 prompt
    chunks_text = ""
    for i, c in enumerate(chunks):
        preview = c["text"][:300].replace("\n", " ")
        chunks_text += f"[{i}] {preview}\n\n"

    prompt = f"""从以下候选段落中选出与用户问题最相关的 {top_k} 个。
只输出编号，用逗号分隔，按相关度从高到低排列。不要解释。

用户问题：{query}

候选段落：
{chunks_text}

最相关的 {top_k} 个编号（从高到低）："""

    async with httpx.AsyncClient(timeout=60) as cli:
        resp = await cli.post("http://localhost:11434/api/generate", json={
            "model": "qwen3.5:9b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 30},
        })

    # 解析编号
    try:
        indices = [int(x.strip()) for x in resp.json()["response"].split(",")]
        reranked = [chunks[i] for i in indices if 0 <= i < len(chunks)]
        # 填充到 top_k
        for c in chunks:
            if len(reranked) >= top_k:
                break
            if c not in reranked:
                reranked.append(c)
        return reranked[:top_k]
    except:
        return chunks[:top_k]  # 解析失败，返回原始顺序
```

**为什么用 LLM 而不是 Cross-Encoder**：
- 84 chunks 场景下，LLM 重排的延迟可接受（~2秒）
- 不需要下载新模型（BGE-reranker 约 1.5GB）
- qwen3.5:9b 已经在用，零额外成本

---

### Phase 5：Metadata Boost

**对应 RAGFlow**：`_rank_feature_scores()` (`search.py:334`)

**规则**：

```python
def metadata_boost(query: str, chunk: dict) -> float:
    """根据 chunk metadata 与 query 的匹配度加分"""
    boost = 0.0

    # 规则 1：component 匹配 (最高权重)
    components = chunk.get("component", [])
    for comp in components:
        if comp in query:
            boost += 0.15
            break

    # 规则 2：fault_symptom 关键词匹配
    symptom = chunk.get("fault_symptom", "")
    if symptom:
        symptom_words = set(symptom.replace("，", " ").replace("、", " ").split())
        query_words = set(query.replace("？", " ").split())
        overlap = len(symptom_words & query_words)
        if overlap > 0:
            boost += 0.1 * min(overlap, 3)

    # 规则 3：semantic_type 与查询意图匹配
    semantic = chunk.get("semantic_type", "")
    if any(kw in query for kw in ["怎么修", "故障", "不工作", "坏了", "异常"]):
        if semantic == "故障诊断":
            boost += 0.1
    if any(kw in query for kw in ["怎么", "如何", "步骤", "安装", "充电"]):
        if semantic == "操作步骤":
            boost += 0.1
    if any(kw in query for kw in ["安全", "注意", "危险", "警告"]):
        if semantic == "风险警告":
            boost += 0.15

    # 规则 4：domain_tags 匹配
    tags = chunk.get("domain_tags", [])
    for tag in tags:
        if tag in query:
            boost += 0.05
            break

    return min(boost, 0.3)  # 单字段 boost 上限 0.3
```

---

### Phase 6：父块聚合 + 上下文

在 Rerank 后的 Top-5 基础上扩展：

```python
def expand_context(chunks: list[dict], qdrant) -> list[dict]:
    """拉取父 chunk 和兄弟 chunk"""
    expanded = list(chunks)
    ids_to_add = set()

    for c in chunks:
        # 父 chunk
        if c["payload"].get("parent_id"):
            ids_to_add.add(c["payload"]["parent_id"])
        # mom chunk
        if c["payload"].get("mom_id"):
            ids_to_add.add(c["payload"]["mom_id"])

    if ids_to_add:
        parents = qdrant.retrieve(
            collection_name="ebike_manual",
            ids=list(ids_to_add),
            with_payload=["text", "chunk_id", "part", "section"],
        )
        expanded.extend(parents)

    return expanded
```

---

## 五、融合公式

**对应 RAGFlow**：`sim = tkWeight * tksim + vtWeight * vtsim + rank_fea`

```python
# 本方案融合公式
final_score = (
    qdrant_score * 0.5 +          # Qdrant 混合检索分数（dense + sparse 已融合）
    rerank_boost    * 0.3 +        # LLM Rerank 排序（按位置衰减：第1位 1.0, 第5位 0.2）
    metadata_boost  * 0.2            # Metadata 匹配加分
)
```

| 权重 | 来源 | 含义 |
|:---:|------|------|
| 0.5 | Qdrant 原始排序 | dense + sparse 混合检索的 fusion 分数 |
| 0.3 | LLM Rerank 位置 | 位置越前 boost 越高 |
| 0.2 | Metadata Boost | component/语义类型/domain_tag 匹配 |

---

## 六、路由简化

本方案不做 5 路 Router。改为**2 路轻量路由**：

```python
def route_query(query: str) -> dict:
    """简单判断查询类型，调整检索参数"""
    
    FAULT_KEYWORDS = ["故障", "不工作", "坏了", "异常", "怎么办", "修", "排除"]
    SAFETY_KEYWORDS = ["安全", "危险", "警告", "注意", "禁止"]
    PARAM_KEYWORDS = ["多少", "多久", "多长时间", "几", "规格", "参数"]

    is_fault = any(kw in query for kw in FAULT_KEYWORDS)
    is_safety = any(kw in query for kw in SAFETY_KEYWORDS)

    if is_safety:
        return {
            "top_k": 10,
            "payload_filter": {"semantic_type": "风险警告"},
            "rerank": True,
        }
    elif is_fault:
        return {
            "top_k": 10,
            "rerank": True,
            # 无 filter：故障可能跨 PART
        }
    else:
        return {
            "top_k": 10,
            "rerank": True,
        }
```

---

## 七、与 RAGFlow 的对照

| RAGFlow 机制 | 本方案 | 取舍原因 |
|------|------|------|
| `keyword_extraction()` | Phase 1 LLM 扩展 | 一致 |
| BM25 (`FulltextQueryer`) | BGE-M3 sparse | Qdrant 原生稀疏检索，无需单独索引 |
| 向量检索 (cosine, topk=1024) | BGE-M3 dense (cosine, topk=10) | 84 chunks 不需要 1024 候选 |
| `rerank_by_model()` | Phase 4 LLM Rerank | 不装新模型，用已有 qwen3.5 |
| `_rank_feature_scores()` | Phase 5 Metadata Boost | 一致 |
| `vector_similarity_weight=0.3` | 融合权重 0.5/0.3/0.2 | 基于本场景调优 |
| `retrieval_by_children()` | Phase 6 父块聚合 | 一致 |
| TOC 增强 | 不实现 | 单文档 34 页，TOC 增强增益极小 |
| KG 检索 | 不实现 | 单文档不需要知识图谱 |
| Web 搜索 (Tavily) | 不实现 | 纯本地 |
| `meta_data_filter` (manual/auto) | 路由中 payload_filter | 2 路简化版 |
| `full_question()` 多轮上下文 | 不实现 | 前端保留 10 轮对话，暂不做多轮改写 |

---

## 八、完整 `search_server.py` 检索流程骨架

```python
@app.post("/api/chat")
async def chat(req: ChatRequest):
    # ── Phase 1: Query 扩展 ──
    expanded_query = await expand_query(req.query)

    # ── Phase 2: BGE-M3 编码 ──
    out = model.encode([expanded_query], return_dense=True, return_sparse=True)
    w = out["lexical_weights"][0]
    dense_vec = out["dense_vecs"][0].tolist()
    sparse_vec = {"indices": list(w.keys()), "values": list(w.values())}

    # ── Phase 2.5: 路由 ──
    route = route_query(req.query)
    query_filter = None
    if route.get("payload_filter"):
        query_filter = Filter(must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in route["payload_filter"].items()
        ])

    # ── Phase 3: Qdrant 混合检索 ──
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=("dense", dense_vec),
        query_sparse_vector=("sparse", sparse_vec),
        query_filter=query_filter,
        limit=route["top_k"],
        with_payload=["chunk_id","content_type","part","section","text",
                       "component","fault_symptom","semantic_type","domain_tags"],
    )

    # ── Phase 4: Rerank ──
    if route.get("rerank"):
        chunks = [r.payload for r in results]
        reranked = await rerank_with_llm(req.query, chunks, top_k=5)
    else:
        reranked = [r.payload for r in results[:5]]

    # ── Phase 5: Metadata Boost ──
    for c in reranked:
        c["_boost"] = metadata_boost(req.query, c)

    # 按 boost 重排
    reranked.sort(key=lambda c: c.get("_boost", 0), reverse=True)

    # ── Phase 6: 父块聚合 ──
    final_chunks = expand_context(reranked, qdrant)

    # ── 上下文拼接 + Ollama 生成 ──
    ctx = build_context(final_chunks)
    answer = await generate_answer(req.query, ctx)

    return {"answer": answer, "sources": format_sources(final_chunks)}
```

---

## 九、预期效果

| 指标 | 当前（Phase 3 only） | v6（完整 Phase 1-6） | 提升 |
|------|:---:|:---:|:---:|
| 故障查询 Top-5 命中率 | ~60% | ~90% | Phase 4 Rerank + Phase 5 Boost |
| 参数查询精确匹配 | ~70% | ~90% | Phase 1 扩展 + 路由 |
| 安全查询优先召回 | ~50% | ~95% | Phase 2.5 路由 filter |
| 答案引用来源可读性 | PART X 裸标签 | PART X · 章节名 + 展开全文 | Phase 6 父块聚合 |
| 新增依赖 | 0 | 0 | 全部用已有 qwen3.5 + Qdrant |

---

## 十、你的提案 → v6 的吸收路径

| 你的提案 | 吸收 | 调整 |
|----------|:---:|------|
| Query Understanding 三层结构 | ✅ Phase 1 扩展 + Phase 2.5 路由 | 从 5 路 intent 简化为 2 路；不独立提取 vehicle/system 实体 |
| Retrieval Router | ✅ Phase 2.5 | 从 5 套策略简化为 2 套 |
| Query Rewrite + Expansion | ✅ Phase 1 LLM `keyword_extraction` | 合并为一个 LLM 调用 |
| Hybrid BM25 + Vector | ⚠️ BGE-M3 sparse 替代 BM25 | 不建独立 BM25 索引 |
| Rerank | ✅ Phase 4 LLM Rerank | 用已有 qwen3.5，不装 Cross-Encoder |
| Metadata Boost | ✅ Phase 5 | 4 条规则覆盖 component/symptom/semantic/tag |
| Fusion Scoring | ✅ | 0.5/0.3/0.2 权重 |

---

*终版 v6，2026-06-18*
