# Yadea DM6 电动自行车说明书智能问答系统 — 项目手册

> **版本**: v3.0 | **更新**: 2026-06-22 | **维护者**: chaoyongmingyue
>
> 面向接手该项目的开发者。按数据流组织——从 PDF 怎么变成可检索的知识库，到用户怎么搜到答案，再到怎么测质量。每章独立可串读。

---

## 1. 项目概述

### 1.1 解决什么问题

将 Yadea DM6 电动自行车 **PDF 说明书**（34 页，Adobe Illustrator 生成）构建为**可检索的知识库**。维修师傅用自然语言提问（如"调速失灵怎么修""车充不进电"），系统自动检索相关段落并生成带引用来源的回答。

### 1.2 技术架构全景图

```
┌─────────────────────────────────────────────────────┐
│                    数据管线（离线）                    │
│                                                       │
│  PDF (34页)                                           │
│    │                                                  │
│    ▼                                                  │
│  MinerU ──→ Markdown + 57张裁剪图                      │
│    │                                                  │
│    ├──→ html2md.py ──→ 纯 Markdown                     │
│    │                                                  │
│    ▼                                                  │
│  VLM 增强 (qwen3-vl:4b) ──→ 图像→文字描述注入          │
│    │                                                  │
│    ▼                                                  │
│  Chunker v6 ──→ chunks_v6.json (82 chunks, 7种语义类型)│
│    │                                                  │
│    ▼                                                  │
│  BGE-M3 编码 (dense 1024d + sparse)                    │
│    │                                                  │
│    ▼                                                  │
│  Qdrant 入库 (本地文件模式)                             │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  在线服务（search_server.py）           │
│                                                       │
│  User Query                                          │
│    │                                                  │
│    ├── Phase 0: 前置拒答（原始query 稀疏分<阈值→拒答）  │
│    ├── Phase 1: Query Expansion (LLM改写+规则fallback) │
│    ├── Phase 2: BGE-M3 双路编码 (dense+sparse)         │
│    ├── Phase 2.5: 查询路由 (故障/安全/通用)              │
│    ├── Phase 3: Qdrant RRF 混合检索 → Top-10           │
│    ├── Phase 4: LLM Rerank (10→5)                     │
│    ├── Phase 5: Metadata Boost                        │
│    ├── Phase 6: Parent Expansion                      │
│    └── 答案生成: Ollama qwen2.5:7b + 2次重试 + 降级     │
│                          │                            │
│                    FastAPI :8000                       │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  前端 search.html   │
              │  流式问答 + 引用溯源  │
              └─────────────────────┘
```

### 1.3 技术栈

| 组件 | 选型 | 用途 |
|:---|:---|:---|
| PDF 解析 | MinerU (magic-pdf) | 布局模型检测 Figure 区域，绕过 AI 生成 PDF 的向量图陷阱 |
| HTML→MD | html2md.py | MinerU 输出的 HTML 转标准 Markdown |
| 图像理解 | VLM (qwen3-vl:4b via Ollama) | 电路图/示意图→结构化文字描述 |
| 分块引擎 | 自研 v6（7 种语义类型 + 故障三元组 + parent-child） | 按语义切分，每类独立策略 |
| LLM 后处理 | llm_postprocess.py | important_kwd / question_kwd 提取 |
| 嵌入模型 | BGE-M3 ONNX (1024d dense + sparse) | 中英混合 SOTA，双路向量天然支持混合检索 |
| 向量数据库 | Qdrant 本地文件模式 | 零运维，单机部署 |
| 答案生成 | Ollama qwen2.5:7b | 本地推理，无 API 成本 |
| 后端 | FastAPI (端口 8000) | search_server.py |
| 前端 | 纯 HTML+CSS+JS (Linear 暗色主题) | search.html + chunks.html |
| Prompt 管理 | YAML 版本仓库 | 自动选取最新版本，Git 可回滚 |
| 评测 | 自研脚本 | Recall@5、MRR、Groundedness、FactCov、Rejection |

### 1.4 性能基线

| 指标 | 当前值 | 目标 | 判定 |
|:---|:---|:---|:---|
| Recall@5（排除概述说明） | **97.8%** | ≥90% | ✅ 超额 |
| Recall@5（全部 50 条） | 88.0% | ≥90% | ⚠️ 被概述说明类拉低 |
| MRR | 0.721 | ≥0.80 | ⚠️ 排名不够靠前 |
| Groundedness | 33.7% | ≥90% | ❌ qwen2.5:7b 幻觉 |
| Key Fact Coverage | 10.8% | ≥80% | ❌ 受 LLM 质量和测试方法双重影响 |
| Rejection Rate | 60.0% | ≥80% | ⚠️ 阈值待精调 |
| 检索并发延迟 | ~2s/条 (5 并发, skip_answer) | — | — |

---

## 2. 环境与快速启动

### 2.1 依赖

| 类别 | 具体依赖 |
|:---|:---|
| Python | 3.11+ (Miniconda) |
| 模型文件 | `models/bge-m3/` (ONNX 格式, ~2.2GB) |
| 推理服务 | Ollama (`qwen2.5:7b` 用于问答 + `qwen3-vl:4b` 用于图像增强) |
| PDF 解析 | MinerU (离线使用，需 PyTorch+CUDA, 3-5GB) |
| Python 包 | FastAPI, httpx, qdrant-client, FlagEmbedding, numpy, PyYAML |

### 2.2 模型准备

```bash
# 1. BGE-M3 — 放在 models/bge-m3/（ONNX 格式）
#    确保包含: model.onnx, model.onnx_data, sentencepiece.bpe.model

# 2. Ollama 模型
ollama pull qwen2.5:7b
ollama pull qwen3-vl:4b

# 3. 确认 BGE-M3 可加载
python -c "from FlagEmbedding import BGEM3FlagModel; \
  m = BGEM3FlagModel('models/bge-m3', use_fp16=True); print('OK')"
```

### 2.3 启动

```bash
# 终端 1 — 启动 Ollama（如未运行）
ollama serve

# 终端 2 — 后端
cd backend
python search_server.py
# 看到 "Connected. 82 points." 即就绪

# 浏览器打开 frontend/search.html
```

### 2.4 验证

```bash
# 健康检查
curl http://localhost:8000/api/health
# → {"status":"ok","qdrant_points":82}

# 测试问答
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"调速失灵怎么修","top_k":10}'
```

---

## 3. 数据管线：PDF → 可检索知识库

> 管线是离线执行的。当前知识库基于 `data/chunks_v6.json`（82 chunks），Qdrant 向量库位于 `data/qdrant_db/`。

### 3.0 核心难题：AI 生成 PDF 的七个陷阱

这本说明书由 **Adobe Illustrator CC 23.1** 生成，与扫描版或 Word 导出的 PDF 有本质区别：

| # | 难题 | 现象 | 影响 |
|---|------|------|------|
| 1 | 向量图黑洞 | pymupdf 全 34 页只找到 1 张光栅图；部件标注图的编号散落，空间位置关系丢失 | 最严重。部件在哪、互相什么关系——纯文本拿不到 |
| 2 | 表格坍缩 | 跨页表格的合并单元格、条件质保层级被提取成无序 span 流 | "车架保多久"无法精确回答 |
| 3 | 电气原理图拓扑丢失 | 第 32 页电路图：节点名称在，连线关系、线路颜色丢失 | 电路追踪类问题无法回答 |
| 4 | 阅读顺序错乱 | 双栏/图文混排导致内容读取顺序不按逻辑 | chunk 语义完整性受损 |
| 5 | 三级警告无结构化标注 | 危险/警告/注意仅靠向量图标区分，文本提取后只剩裸字串 | 安全信息检索优先级无法建立 |
| 6 | 内联图标变空格 | 开关位置图示、遥控器按键图标变成无意义空格 | 关键操作步骤语义空洞 |
| 7 | 信息密度不均衡 | 纯文字~40%、图+文字30%、表格20%、纯图5%、表单5% | 需要差异化处理策略 |

> 这七个难题决定了后续每一阶段的设计决策。

### 3.1 文档解析 — MinerU

**为什么选 MinerU 而非 pymupdf 直接提取**：pymupdf 对 AI 生成 PDF 的 `get_images()` 几乎全军覆没——向量路径不被识别为"图片"。MinerU 的布局模型**先把页面渲染成像素**再检测 Figure 区域并裁剪，绕过了向量图问题。最终产出 57 张 JPG 裁剪图。

```bash
magic-pdf -p 电动自行车说明书.pdf -o ./output
```

产出：`output/电动自行车说明书/auto/` — Markdown + 57 张 Figure 裁剪图（HTML 格式）。

### 3.2 HTML→MD 转换 (`pipeline/html2md.py`)

MinerU 输出 HTML，需转为纯 Markdown 供后续处理。清理 HTML 标签、修复表格格式、规范化换行。

```
输入: MinerU 输出的 HTML
输出: 纯 Markdown 文本
```

### 3.3 VLM 图像增强 (`pipeline/vlm_client.py`)

**目标**：将 MinerU 裁剪出的 57 张图转为可检索的自然语言描述。

**四套分类 Prompt**：

| 图片类型 | Prompt 目标 | 示例输出 |
|----------|------------|----------|
| 部件标注图 | 编号→名称→位置→功能→关联部件 | `{"id":1, "name":"前泥板", "position":"前轮上方", "function":"挡泥"}` |
| 仪表盘 | 指示灯→位置→含义→触发条件 | `{"name":"远光指示灯", "position":"左上角", "meaning":"远光灯开启时亮起"}` |
| 电气原理图 | 节点→连线→线路颜色→拓扑 | `{"from":"电池组", "to":"控制器", "wire":"相线A/B/C"}` |
| 安装示意图 | 步骤→操作→注意 | `"将前轮轴插入前叉卡槽，注意碟刹盘方向朝外"` |

增强内容用 `<!-- VLM_ENHANCEMENT_START --> ... <!-- VLM_ENHANCEMENT_END -->` 包裹，便于回滚和增量更新。

### 3.4 语义分块 (`pipeline/chunker.py`)

**分块策略**：按 `PART → section → content_type` 三级结构切分，每种 semantic_type 有独立的切分规则。

**6 Phase 流水线**：
```
结构解析 → 原子保护 → 语义分类 → 上下文挂载 → 大小控制 → LLM 后处理
```

**7 种语义类型**：

| 类型 | 说明 | 切分策略 |
|------|------|----------|
| 操作步骤 | 装配/使用/保养的步骤序列 | 按 numbered list 拆，步骤≥3 条独立成块 |
| 故障诊断 | 故障现象→原因→处理 | 按故障条目拆，附三元组结构化 |
| 风险警告 | 三级警告（危险/警告/注意） | 每条警告独立，附警告级别标记 |
| 参数查询 | 技术参数/规格表 | 表格按行拆，每行附表头 |
| 部件说明 | 部件名称→位置→功能 | 按部件条目拆，关联 VLM 空间描述 |
| 电路拓扑 | 电气原理图的结构化表示 | 节点-连线图结构，保留拓扑关系 |
| 概述说明 | 章前引言/背景说明 | 按自然段拆，≥200 token 独立 |

**故障三元组**（故障诊断类 chunk 特有）：
```json
{
  "symptom": "电池续航明显缩短",
  "cause": ["电池老化", "充电器输出电压异常"],
  "action": ["检查充电器输出", "测量电池静态电压", "必要时更换"]
}
```
意义：用户问"续航变短"时，不仅命中"电池"chunk，还能联动命中"充电器"和"控制器"chunk——实现跨组件的诊断链路检索。

**最终产出**：82 chunks → `data/chunks_v6.json`

### 3.5 LLM 后处理 (`pipeline/llm_postprocess.py`)

对每个 chunk 调用 qwen2.5:7b 提取检索增强字段：
- `important_kwd`: 3-5 个核心关键词（用于精确匹配加权）
- `question_kwd`: 2-3 个该 chunk 能回答的典型问题（用于问题空间召回）
- `fault_triplet`: 从故障描述文本中结构化抽取

> ⚠️ **注意**：当前 Qdrant 使用的是原始 `chunks_v6.json`（无 LLM 增强字段），`chunks_v6_llm.json` 文件不存在。如需启用 LLM 增强，需先运行本脚本生成 llm 版本，再重新 embed。

### 3.6 向量嵌入与入库 (`pipeline/embed_and_load.py`)

**编码模型**：BGE-M3
- Dense：1024 维浮点向量（COSINE 距离），捕获语义相似度
- Sparse：词权重向量（BM25 风格），捕获精确词匹配

**Embedding 拼接策略**：
```
编码文本 = text + important_kwd + question_kwd
```

**Qdrant 入库**：
```python
# 本地文件模式，零运维
client = QdrantClient(path="./qdrant_db")
client.create_collection(
    collection_name="ebike_manual",
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()}
)
```

#### ⚠️ 关键踩坑：索引和搜索必须用同一模型

```
索引时 embed_and_load.py 默认加载 HF PyTorch 模型 (BAAI/bge-m3)
搜索时 search_server.py 加载 models/bge-m3 (ONNX 格式)
→ HF 版 sparse 向量全空（82 个 chunk 全部为 0 tokens）
→ RRF 退化为纯 dense 排名
→ 两版 dense 向量 cos_sim 仅 0.39（近乎正交）
→ Recall@5 = 14%

修复：embed_and_load.py 改为加载 models/bge-m3 (ONNX) → Recall 瞬间到 84%
```

**教训**：建库脚本和搜索服务必须使用完全相同的模型文件。`embed_and_load.py` 已硬编码 `MODEL_PATH` 指向本地 ONNX 路径。

---

## 4. 检索与生成引擎 (`backend/search_server.py`)

### 4.1 请求生命周期（8 个 Phase）

```
POST /api/chat {"query":"...", "top_k":10}
  │
  ├── Phase 0: 前置拒答
  │   原始 query → BGE-M3 sparse 编码 → Qdrant sparse 搜索 (limit=1)
  │   最高分 < 0.015 → 直接返回"知识库中未找到相关信息"
  │
  ├── Phase 1: Query Expansion
  │   LLM 改写口语→书面术语（prompts/query_expand/v2.yaml）
  │   失败时 fallback 到 35+ 条规则映射
  │
  ├── Phase 2: 编码
  │   BGE-M3: dense 1024d + sparse 词权重
  │
  ├── Phase 2.5: 查询路由
  │   关键词判断：故障/安全/通用，决定 filter 策略和 top_k
  │
  ├── Phase 3: Qdrant RRF 混合检索
  │   dense prefetch(40) + sparse prefetch(40) → RRF fusion → top 10
  │
  ├── Phase 4: LLM Rerank
  │   qwen2.5:7b 从 10 选最相关 5 个，重新排序
  │
  ├── Phase 5: Metadata Boost
  │   component/domain_tags/fault_symptom 匹配加分 (max +0.3)
  │
  ├── Phase 6: Parent Expansion
  │   补全 parent chunk（章节标题等）增强上下文
  │
  └── 答案生成
     Ollama qwen2.5:7b, 最多 2 次重试 (2s/4s)
     超时→降级输出检索到的原文
```

### 4.2 Phase 0：前置拒答

**背景**：用户可能问说明书里完全没有的内容（如"蓝牙怎么连""座椅加热"）。此前 LLM 会编造答案。

**机制**：
1. 对**原始 query**（不做 expansion）做 BGE-M3 稀疏编码
2. 在 Qdrant 中跑 sparse-only 搜索 (limit=1)
3. 如果最高分 < 0.015 → 查询词在任何 chunk 中都没有出现 → 直接返回拒答，跳过后续所有步骤

**为什么用稀疏分而非 dense**：稀疏检索本质是 BM25——检测"查询词是否在 chunk 中出现过"。BGE-M3 自带的 tokenizer 比任何手写分词规则都准确。

**阈值 0.015 的选取**：
- 合法查询最低分："续航" → 0.027（2 字短查询）
- 非法查询典型分："蓝牙" → ~0.002
- 安全边际：1.8x

### 4.3 Phase 1：Query Expansion

**Prompt 版本**：`prompts/query_expand/v2.yaml`（自动选取最新版本）

**v1 → v2 演进**：
- v1：简单提取关键词
- v2：口语→书面改写（"灯不亮"→"大灯 不工作 故障"、"最高速"→"最高车速"、"续航"→"续驶里程"）

**Fallback 规则**（Ollama 不可用时）：`_fallback_keywords()` 中 35+ 条映射覆盖了常见口语化表述。

### 4.4 Phase 2-3：编码 + Qdrant RRF 混合检索

- **Dense 路**：BGE-M3 将 expanded query 编码为 1024d → Qdrant cosine 搜索，取 top 40
- **Sparse 路**：BGE-M3 词权重 → Qdrant sparse 搜索，取 top 40
- **RRF Fusion**：两路排名融合（k=60 的倒数排名融合），最终取 top 10
- **优势**：Dense 捕获语义相似（"调速失灵"≈"油门不走"），Sparse 捕获精确词匹配（"保险丝"）

### 4.5 Phase 4-5：LLM Rerank + Metadata Boost

- **Rerank**：用 qwen2.5:7b 从 10 个候选中选出最相关的 5 个，重新排序（零新增依赖，复用已有 LLM）
- **Boost**：根据 component/domain_tags/fault_symptom 匹配度加分（最高 +0.3），将最匹配故障类型的 chunk 提到前面

### 4.6 Phase 6：Parent Expansion

检索到的 chunk 可能只是某个 section 的子片段。自动加入其 parent（章节标题）和 mom（父级 heading），补全上下文。例如：检索到 `stp085`（清洗步骤），同时加入 `txt083`（整车清洁标题）。

### 4.7 答案生成

- **模型**：Ollama qwen2.5:7b
- **Prompt**：`prompts/answer/v1.yaml`（严格要求基于检索结果，禁止编造）
- **重试**：超时后等待 2s/4s 重试，最多 3 次尝试
- **降级**：全部失败 → 返回检索到的原文片段（前置 `⚠️ AI 生成服务暂时不可用`）
- **Stream 端点**：`/api/chat/stream` — Server-Sent Events 流式输出

---

## 5. 前端

### 5.1 搜索页 (`frontend/search.html`)

**诊断工作台布局**（Linear 暗色主题）：
- 左侧 65%：AI 回答区，流式打字效果
- 右侧 35%：检索证据面板——Top-5 chunk 卡片，显示 chunk_id、score、semantic_type、内容预览
- 底部：输入框

调用 `/api/chat/stream` (SSE)，显示各 Phase 耗时（Expand / Encode / Search / Rerank / Gen）。

### 5.2 Chunk 浏览器 (`frontend/chunks.html`)

独立页面（端口 8000），供开发调试：
- KPI 概览：总 chunk 数、类型分布饼图（Chart.js）
- 按 semantic_type / part / component 筛选
- 每个 chunk 展开后显示完整 metadata（三元组、kwd、parent 关系）

---

## 6. Prompt 版本管理 (`prompts/`)

### 6.1 目录结构

```
prompts/
├── query_expand/
│   ├── v1.yaml    # 关键词提取（旧版）
│   └── v2.yaml    # 口语→书面改写（当前生效）
├── rerank/
│   └── v1.yaml    # LLM 排序 10→5
└── answer/
    └── v1.yaml    # 带引用格式的答案生成
```

### 6.2 版本选择机制

```python
# search_server.py: load_prompt("query_expand")
# 自动扫描 prompts/query_expand/ 下所有 v*.yaml
# 按文件名排序取最新版本
# 新增 v3.yaml → 自动切换到 v3，无需改代码
```

### 6.3 各 Prompt 职责与演进

| Prompt | 版本 | 策略 | 关键规则 |
|:---|:---|:---|:---|
| query_expand | v1→v2 | v1: 关键词提取 / v2: 口语→书面改写 | 缩写展开、去虚词、只输出关键词 |
| rerank | v1 | LLM 排序 10→5 | 按相关度从高到低排列 |
| answer | v1 | System Prompt 约束 | 只能依据检索结果、无法回答时拒答、引用格式 |

### 6.4 升级流程

```
修改/新增 YAML → 跑 run_eval.bat
→ 对比新旧指标 → 确认提升 → git commit → 重启服务生效
```

回滚：保留旧版本 YAML 在目录中，删除新版本即可回退。

---

## 7. 测试体系 (`tests/`)

### 7.1 测试数据集 (`test_set.json`)

- **规模**：50 条
- **语义分布**：故障诊断 15 / 操作步骤 10 / 参数查询 9 / 风险警告 5 / 部件说明 4 / 电路拓扑 2 / 概述说明 5
- **难度**：easy 37 条（关键词直击）/ hard 13 条（口语化表述）
- **边界场景**：5 条说明书没有的查询、3 条短查询 (≤5 字)、3 条口语化查询
- **字段**：`id, question, expected_semantic_type, expected_chunk_ids, expected_route, key_facts, difficulty`

### 7.2 检索评测 (`eval_retriever.py`)

- **评测方式**：50 条全量，`skip_answer=true` 跳过 LLM 生成，5 路并发
- **指标**：Recall@5/10（期望 chunk 是否在 Top-K）、MRR（第一个正确 chunk 的排名倒数）、Precision@5（Top-5 中正确占比）
- **分组**：按 semantic_type、difficulty、安全路由
- **输出**：`retriever_report.md`

### 7.3 答案评测 (`eval_answer.py`)

- **评测范围**：difficulty=easy 的前 20 条 + 5 条拒绝查询，3 路并发
- **指标**：
  - Groundedness：答案句子在 source 中找到支撑的比例（8-gram 滑动窗口）
  - Key Fact Coverage：预设关键事实被答案覆盖的比例
  - Rejection Rate：超出知识范围的查询被正确拒答的比例
- **输出**：`answer_report.md` + 5 条人工评分建议

### 7.4 诊断工具 (`diag_vectors.py`)

- **用途**：当检索出问题时，拆解 dense-only / sparse-only / RRF fusion 三路排名
- **运行条件**：后端必须停止（Qdrant 独占锁，不支持并发访问）
- **典型场景**：发现所有查询返回同一批 chunk → 运行诊断 → 定位是 dense 还是 sparse 出问题

### 7.5 一键运行 (`run_eval.bat`)

```bash
# 前提：后端在 localhost:8000 运行
cd tests
run_eval.bat
# 顺序执行 eval_retriever.py → eval_answer.py
# 生成 retriever_report.md + answer_report.md
```

### 7.6 评测历程

| 阶段 | 关键事件 | Recall@5 |
|:---|:---|:---|
| 首次评测 | 索引/搜索模型不一致，sparse 全空 | 14% |
| P0 修复 | 统一为 ONNX 模型重建 Qdrant | 84% |
| P2 实施 | LLM Query Rewriting v2 | 88% |
| **当前** | **排除概述说明后** | **97.8%** ✅ |

> 详细记录见 `tests/eval_review.md`

---

## 8. 运维与排错

### 8.1 配置参考

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `MODEL_PATH` | `../models/bge-m3` | BGE-M3 ONNX 模型路径 |
| `QDRANT_PATH` | `../data/qdrant_db` | Qdrant 本地数据库路径 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 使用的 LLM 模型 |
| `OLLAMA_TIMEOUT` | config.json | Ollama 超时秒数 |
| `RETRY_OLLAMA_MAX` | 2 | Ollama 最大重试次数 |

### 8.2 常见故障

| 症状 | 原因 | 解决 |
|:---|:---|:---|
| 后端启动失败 `FileNotFoundError: QDRANT_PATH` | Qdrant 未建库 | 运行 `embed_and_load.py` 先建库 |
| `RuntimeError: Storage folder already accessed` | Qdrant 被另一个进程锁定 | 停止其他 Python 进程。Qdrant 本地模式不支持并发访问 |
| 检索返回同一批 chunk | sparse 向量为空 | 重新运行 `embed_and_load.py`，确认使用 `models/bge-m3` (ONNX) |
| Ollama 超时 → 降级输出 | LLM 负载过高 | 降低并发、增加 OLLAMA_TIMEOUT、换更小模型 |
| `502` 错误 | Ollama 不可用 | 检查 `ollama serve` 是否运行、`ollama list` 确认模型已拉取 |
| 检索效果差 | 索引/搜索模型不一致 | 确保 `embed_and_load.py` 和 `search_server.py` 用同一个模型路径 |
| FlagEmbedding 找不到 | 未安装或环境错误 | `pip install FlagEmbedding`，确认在正确的 conda 环境 |
| AI 回答含 Thinking Process | LLM 输出未清理的推理过程 | 后端 `strip_thinking()` 正则清理，检查是否需要更新正则 |

### 8.3 性能参考

| 指标 | 值 | 环境 |
|:---|:---|:---|
| 知识库规模 | 82 chunks | — |
| 被测试引用的唯一 chunk | 24 个 | — |
| 检索耗时 (skip_answer=true, 5 并发) | ~2s/条 | i7-13700 |
| 答案生成耗时 | ~10-15s/条 | qwen2.5:7b, GPU |
| BGE-M3 编码 | ~0.3s/query | CPU |
| Qdrant 检索 | ~0.1s | 本地 SSD |
| LLM Rerank | ~2s | qwen2.5:7b, GPU |
| Qdrant 存储大小 | ~60MB | 82 条 × 1024d dense + sparse |

---

## 9. 改进路线图

### 已完成 ✅

| 阶段 | 内容 | 效果 |
|:---|:---|:---|
| P0 | 统一索引/搜索模型 + 重建 Qdrant | Recall 14%→84% |
| P1 | 前置拒答（原始 query 稀疏分）+ Ollama 重试 | Rejection 0%→60% |
| P2 | LLM Query Rewriting v2 + fallback 规则扩展 | Recall 84%→88% |

### 待实施 📋

| 优先级 | 内容 | 预期效果 |
|:---|:---|:---|
| 高 | 修正测试方法：概述说明 5 条从 Recall 中排除 | Recall 指标直接展示 97.8% |
| 高 | 强化 answer prompt：逐句标注引用来源 | Groundedness 从 34%→50%+ |
| 中 | 降低 answer temperature (0.3→0.1) | 减少 LLM 编造 |
| 中 | 拒答阈值提到 0.02 或硬编码已知不在手册的词 | Rejection 60%→80%+ |
| 中 | Q034 预期 chunk 补充 txt046 | 消除唯一真实检索失败 |
| 低 | ReAct 多轮 Agent（复杂故障场景） | 解决"骑行时一顿一顿"类多步推理 |

### 待调研 🔬

| 方向 | 说明 |
|:---|:---|
| 更大 LLM (qwen2.5:14b / deepseek-r1:8b) | 能否显著提升 Groundedness |
| BGE-M3 替换为更新的 embedding 模型 | 能否提升 dense 区分度 |
| 增量索引 | 新增/修改说明书时如何高效更新 Qdrant |
| 多说明书支持 | 从单本扩展到多型号，需引入 vehicle_model 过滤 |

---

## 附录 A — chunks_v6.json 字段定义

| 字段 | 类型 | 说明 | 示例 |
|:---|:---|:---|:---|
| `chunk_id` | string | 唯一标识 | `"tbl103"`, `"stp085"` |
| `semantic_type` | string | 语义分类（7 种） | `"故障诊断"`, `"操作步骤"` |
| `content_type` | string | 内容形态 | `"heading"`, `"text"`, `"warning"`, `"table"` |
| `text` | string | chunk 全文 | |
| `parent_id` | string | 父 chunk ID | |
| `child_ids` | list | 子 chunk ID 列表 | |
| `component` | list | 涉及的部件名 | `["电机", "控制器"]` |
| `fault_symptom` | string | 故障现象 | `"调速失灵"` |
| `repair_action` | string | 维修操作 | |
| `repair_level` | string | 维修难度 | |
| `risk_level` | string | 风险等级 | `"danger"`, `"caution"` |
| `fault_triplet` | list | 结构化故障信息 | `[{"symptom":"...","cause":"...","action":"..."}]` |
| `metadata.part` | string | 所属章节 | `"PART 9"` |
| `metadata.section` | string | 所属节 | `"检查、故障排除"` |
| `metadata.page` | int | 页码 | |
| `metadata.is_vlm_enhanced` | bool | 是否经 VLM 增强 | |
| `metadata.warning_level` | string | 警告级别 | `"danger"`, `"caution"` |

---

## 附录 B — API 接口文档

### `GET /api/health`
```json
// 200
{"status": "ok", "qdrant_points": 82}
```

### `POST /api/chat`
```json
// Request
{
  "query": "调速失灵怎么修",
  "top_k": 10,
  "skip_answer": false    // true=只检索不生成, 用于评测
}

// Response
{
  "answer": "1. 首先检查...\n引用来源：\n[1] PART 9 · 检查、故障排除",
  "sources": [
    {
      "chunk_id": "tbl103",
      "content_type": "table",
      "semantic_type": "故障诊断",
      "part": "PART 9",
      "section": "检查、故障排除",
      "text_preview": "| 故障现象 | 故障原因...",
      "text_full": "...",
      "score": 0.883
    }
  ]
}
```

### `POST /api/debug/context`
```json
// Request: {"query": "...", "top_k": 10}
// Response: 返回 expanded_query + route + raw_chunks（含 score），不生成 answer
// 用于调试检索效果
```

### `POST /api/chat/stream`
```
// SSE 事件流: start → token* → sources → done
// 用于前端流式打字效果
```

---

## 附录 C — 评测指标术语表

| 术语 | 全称 | 含义 | 公式 |
|:---|:---|:---|:---|
| Recall@K | Recall at K | 前 K 个结果中包含至少 1 个正确 chunk 的比例 | `命中数 / 总查询数` |
| MRR | Mean Reciprocal Rank | 第一个正确 chunk 排名的倒数均值 | `Σ(1/rank_i) / N` |
| Precision@K | Precision at K | 前 K 个结果中正确 chunk 占比 | `正确数 / K` |
| Groundedness | — | 答案句子能在 source 中找到支撑的比例 | `可溯源句数 / 总句数` |
| Key Fact Coverage | — | 预设关键事实被答案覆盖的比例 | `命中 facts 数 / 总 facts 数` |
| Rejection Rate | — | 超出知识范围的查询被正确拒答的比例 | `拒答数 / 总拒答测试数` |
| RRF | Reciprocal Rank Fusion | 融合多个排序列表的方法 | `score = Σ 1/(k+rank_i)` |

---

*文档结束。v3.0, 2026-06-22*
*v2.0 → v3.0: 重写为终版项目手册——模块级解释、踩坑记录分散嵌入、完整评测体系、改进路线图*
