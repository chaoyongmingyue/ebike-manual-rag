# 电动自行车说明书 · Metadata 方案 v6（终版）

> 融合 RAGFlow Metadata 学习 + v6 分块方案 + 你的优化提案，专为单说明书场景定制。

---

## 一、你的优化提案审查

### 1.1 哪些可直接吸收

| 提案 | 判断 | 说明 |
|------|:---:|------|
| 语义优先原则 | ✅ 直接吸收 | `fault_symptom > chapter title > page structure` 是正确方向 |
| 检索增强字段 | ✅ 直接吸收 | `important_keywords` + `generated_questions` 来自 RAGFlow，本方案也需要 |
| 故障三元组 | ✅ 直接吸收 | 已在 v6 分块方案中设计 |
| Chunk 类型体系 | ✅ 吸收并简化 | 你的 8 种 → 本说明书实际需要 7 种 |
| Query 驱动设计原则 | ✅ 直接吸收 | Metadata 服务于检索，不服务于描述 |

### 1.2 哪些过度设计（本说明书场景）

| 提案 | 判断 | 原因 |
|------|:---:|------|
| `vehicle_model` / `model_year` | ❌ 过度 | **只有一本说明书，一个车型 TDR2699Z**。多车型字段在此场景零过滤价值 |
| `applicable_models` 列表 | ❌ 过度 | 同上，单一车型 |
| `system`（电池系统/电机系统） | ❌ 过度 | 说明书不是按系统组织，是按 PART 组织 |
| `fault_code`（如 B001） | ❌ 过度 | 说明书使用描述性故障名（"调速失灵"），不使用故障码 |
| `estimated_time` | ❌ 过度 | 说明书中仅有"2个月补电一次""15km/h限速"等时间表述，不作为维修时长 |
| `tool_required` | ⚠️ 少数场景有用 | 仅维护保养章节提到工具，且不精确。降级为可选字段 |
| `version` / `source` | ❌ 过度 | 单文档不需要版本追踪 |
| `synonyms` 同义词表 | ⚠️ 部分有用 | "充不进电↔无法充电↔充电失败"确实存在。改为放在 `llm_fields.question_kwd` 中由 LLM 自然覆盖 |

### 1.3 需要保留并简化

| 提案 | 简化后 | 理由 |
|------|--------|------|
| `system` → `part` | 说明书用 PART 组织，不是系统 | PART 6 电池充电、PART 9 故障排除 |
| `component` | ✅ 保留 | "控制器""电机""前照灯"等部件名是精确匹配的核心 |
| `fault_symptom` | ✅ 保留 | PART 9 故障表的自然字段 |
| `repair_action` | ✅ 保留 | 故障表的"故障原因及排除方法"列 |
| `repair_level` | ✅ 简化 | 从 4 级简化为 2 级：`self_help`（可自助）/ `service_center`（需送修） |
| `risk_level` | ✅ 保留 | 映射到三级警告：danger / warning / caution |
| `chunk_type` | ✅ 保留 | 7 种语义类型 |
| `doc_type` | ✅ 简化为 `part_type` | PART 分组已经隐含了文档类型信息 |

---

## 二、本说明书的 Metadata 应该长什么样

### 核心原则

```
只有一本说明书 → 去掉所有"跨文档"字段
目标是检索质量 → 每个字段必须能直接参与过滤或加权
不过度设计 → 说明书中没有的字段不硬造
```

### 2.1 不需要的字段（明确排除）

| 字段 | 排除原因 |
|------|----------|
| `vehicle_model` | 就一个 TDR2699Z，不需要过滤 |
| `model_year` | 就一个 2022 年 12 月第 2 版 |
| `applicable_models` | 没有跨车型适用性 |
| `system` | 说明书按 PART 组织，不按系统 |
| `fault_code` | 说明书中无故障码体系，只有描述性名称 |
| `estimated_time` | 说明书中不量化维修时长 |
| `version` | 单文档 |
| `source` | 单来源 |

### 2.2 需要的字段（三层结构）

```
┌─────────────────────────────────────────────────┐
│ 第一层：结构信息（从文档来，规则提取，不依赖LLM）  │
│   part, section, page, content_type              │
│                                                  │
│ 第二层：语义信息（LLM辅助提取，核心检索字段）      │
│   semantic_type, component, fault_symptom,       │
│   repair_action, repair_level, risk_level        │
│                                                  │
│ 第三层：检索增强（LLM后处理，提升召回）             │
│   important_kwd, question_kwd, domain_tags       │
└─────────────────────────────────────────────────┘
```

---

## 三、最终 Metadata Schema

```json
{
  "chunk_id": "PART9_F02_step",

  "// ── 第一层：结构信息（规则提取）──": "",
  "content_type": "table_row",
  "part": "PART 9",
  "section": "故障现象与排除",
  "page": 25,
  "parent_id": "PART9_01_parent",
  "child_ids": [],
  "mom_id": "PART9_故障表_full",

  "// ── 第二层：语义信息（LLM 辅助提取）──": "",
  "semantic_type": "故障诊断",
  "component": ["控制器", "电机"],
  "fault_symptom": "接通电源，仪表正常显示，调速失灵",
  "repair_action": "联系授权服务站检测维修",
  "repair_level": "service_center",
  "risk_level": null,

  "// ── 第三层：检索增强（LLM 后处理）──": "",
  "important_kwd": "调速失灵, 控制器故障, 电机故障",
  "question_kwd": "仪表正常但拧油门不走是什么原因？\n调速失灵怎么修？",
  "domain_tags": ["故障", "控制器", "电机"],

  "// ── 故障三元组（故障诊断类 chunk 专有）──": "",
  "fault_triplet": {
    "symptom": "接通电源，仪表正常显示，调速失灵",
    "component": "控制器",
    "cause": "控制器或电机故障",
    "action": "联系授权服务站检测维修",
    "can_diy": false
  }
}
```

---

## 四、字段定义与提取方式

### 4.1 第一层：结构信息（规则提取，无需 LLM）

| 字段 | 类型 | 提取方式 | 检索用途 |
|------|------|----------|----------|
| `content_type` | keyword | 分块 Phase 1 判定 | 过滤（只看表格/警告） |
| `part` | keyword | 从所属 `## PART N` 提取 | 限定搜索范围（只看 PART 6） |
| `section` | keyword | 从最近的 H2/H3 标题提取 | 来源展示、分类聚合 |
| `page` | int | 从 MinerU 的页码信息 | 页码范围过滤 |
| `parent_id` | keyword | 分块 Phase 5 分配 | 检索时拉取父块 |
| `child_ids` | list[keyword] | 同上 | 扩展兄弟上下文 |
| `mom_id` | keyword | children_delimiters 生成 | 故障码/参数对的母块聚合 |

### 4.2 第二层：语义信息（LLM 辅助提取）

| 字段 | 类型 | 提取方式 | 示例 | 检索用途 |
|------|------|----------|------|----------|
| `semantic_type` | keyword | 分块 Phase 3 规则判定 | `故障诊断` | 按语义类型加权/过滤 |
| `component` | list[keyword] | LLM 提取 + 规则匹配 | `["控制器","电机"]` | 部件名精确匹配 |
| `fault_symptom` | keyword | 故障表 symptom 列或 LLM | `接通电源调速失灵` | 匹配用户描述的故障 |
| `repair_action` | keyword | 故障表 action 列或 LLM | `联系授权服务站检测维修` | 提供修复方案 |
| `repair_level` | keyword | LLM 二分类 | `self_help` / `service_center` | "自己能修吗"类过滤 |
| `risk_level` | keyword | 从警告块提取 | `danger` / `warning` / `caution` | 安全查询优先召回 |

#### component 提取规则

```python
# 方式 A：从已知部件名列表匹配（规则，可靠）
KNOWN_COMPONENTS = [
    "控制器", "电机", "蓄电池", "充电器", "转换器",
    "仪表", "前照灯", "尾灯", "转向灯", "刹车灯",
    "调速转把", "刹把", "空气开关", "防盗器", "电门锁",
    "前叉", "后减震", "前碟刹盘", "后平叉", "保险丝",
]

def extract_components(text: str) -> list[str]:
    return [c for c in KNOWN_COMPONENTS if c in text]

# 方式 B：LLM 补充（规则没匹配到的才调 LLM）
# prompt: "这个文本提到了哪些电动车部件？只列出部件名。文本：{text}"
```

#### repair_level 判定规则

```python
REPAIR_LEVEL_RULES = {
    "self_help": ["自行检查", "用户自行", "可自行", "建议", "检查"],
    "service_center": ["联系授权", "送修", "服务站", "返厂", "专业维修"],
}

# 默认：含 self_help 关键词 → self_help
#       含 service_center 关键词 → service_center
#       两者都不含 → 不填（null）
```

### 4.3 第三层：检索增强（LLM 后处理，对应 RAGFlow 的 auto_keywords/auto_questions）

| 字段 | 类型 | 提取方式 | 示例 | RAGFlow 对应 |
|------|------|----------|------|-------------|
| `important_kwd` | keyword | LLM 提取 3 个关键词 | `控制器故障, 调速失灵` | `important_kwd` |
| `question_kwd` | keyword | LLM 生成 3 个预设问题 | `仪表正常但拧油门不走？` | `question_kwd` |
| `domain_tags` | list[keyword] | LLM 分类 | `["故障","控制器","电机"]` | 无对应，本方案新增 |

#### domain_tags 候选值

```python
DOMAIN_TAGS = [
    "安全", "充电", "电池", "电机", "控制器",
    "骑行", "保养", "故障", "安装", "保修",
    "参数", "电路", "仪表", "部件",
]
```

### 4.4 故障三元组（故障诊断类 chunk 专有）

| 字段 | 类型 | 提取方式 | 示例 |
|------|------|----------|------|
| `fault_triplet.symptom` | keyword | 表格 symptom 列 | "接通电源，仪表正常显示，调速失灵" |
| `fault_triplet.component` | keyword | LLM | "控制器" |
| `fault_triplet.cause` | keyword | 表格 cause 列 | "控制器或电机故障" |
| `fault_triplet.action` | keyword | 表格 action 列 | "联系授权服务站检测维修" |
| `fault_triplet.can_diy` | bool | LLM 判断 | false |

---

## 五、按语义类型的字段适用矩阵

并非所有字段对所有 chunk 都有值。以下矩阵定义每种语义类型**必须填充**和**可选填充**的字段：

| 字段 | 操作步骤 | 故障诊断 | 风险警告 | 参数查询 | 部件说明 | 电路拓扑 | 概述说明 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `semantic_type` | ● | ● | ● | ● | ● | ● | ● |
| `component` | ○ | ● | ○ | ● | ● | ● | — |
| `fault_symptom` | — | ● | — | — | — | ○ | — |
| `repair_action` | ○ | ● | — | — | — | — | — |
| `repair_level` | ○ | ● | — | — | — | — | — |
| `risk_level` | — | — | ● | — | — | — | — |
| `fault_triplet` | — | ● | — | — | — | ○ | — |
| `important_kwd` | ● | ● | ● | ● | ● | ● | — |
| `question_kwd` | ● | ● | ● | ○ | — | — | — |
| `domain_tags` | ● | ● | ● | ● | ● | ● | ○ |

> ● 必填　○ 选填　— 不适用

---

## 六、检索时 Metadata 如何参与

### 6.1 两种参与方式

```
方式 A：Qdrant Payload Filter（精确过滤）
  适合：content_type=warning, repair_level=self_help, risk_level=danger
  实现：qdrant.search(..., query_filter=Filter(must=[...]))

方式 B：多字段加权检索（语义融合）
  适合：component 匹配、fault_symptom 匹配、question_kwd 匹配
  实现：将 metadata 字段也做 embedding，检索时多向量融合
```

### 6.2 方式 A：Payload Filter 示例

```python
# 场景："充电有什么安全注意事项"
# → 只看风险警告类型
results = qdrant.search(
    collection_name="ebike_manual",
    query_vector=("dense", query_dense),
    query_sparse_vector=("sparse", query_sparse),
    query_filter=Filter(must=[
        FieldCondition(key="semantic_type", match=MatchValue(value="风险警告")),
    ]),
    limit=5,
)
```

### 6.3 方式 B：多字段加权检索

```python
# 查询编码时，不仅编码用户问题，也编码 component 名
# "控制器保修多久" → 编码 "控制器保修多久" + "控制器 保修 三包"
# 后者通过 domain_tags 和 component 字段增强

def encode_query_with_metadata(query: str, metadata_context: dict = None):
    """增强查询编码：用户问题 + metadata 上下文字段"""
    enhanced = query
    if metadata_context:
        parts = []
        if metadata_context.get("component"):
            parts.append("部件：" + " ".join(metadata_context["component"]))
        if metadata_context.get("domain_tags"):
            parts.append("领域：" + " ".join(metadata_context["domain_tags"]))
        enhanced = query + "\n" + "\n".join(parts)
    
    return model.encode([enhanced], return_dense=True, return_sparse=True)
```

---

## 七、与 RAGFlow Metadata 的对照

| RAGFlow 特性 | v6 对应 | 差异 |
|------|------|------|
| 文档级 `meta_fields` (dynamic JSON) | 无文档级（仅一本说明书） | 单文档场景不需要文档级过滤 |
| `important_kwd` | `important_kwd` | 一致 |
| `question_kwd` | `question_kwd` | 一致 |
| `auto_keywords` (LLM) | LLM 后处理 | 一致 |
| `auto_questions` (LLM) | LLM 后处理 | 一致 |
| `metadata_condition` (key-op-value) | Qdrant Payload Filter | 实现不同，效果等价 |
| `meta_data_filter` (manual/auto/semi_auto) | 方式 A（manual）+ LLM auto（可选） | auto 模式在本方案中是检索层逻辑，非 metadata 层 |
| `gen_meta_filter()` LLM 自动过滤 | 未实现（单文档不需要自动过滤） | — |
| Push-down 过滤 (ES) | Qdrant Payload Filter | 实现不同，效果等价 |
| `position_int` / `page_num_int` | `page` | 简化，单字段 |
| `doc_type_kwd` | `semantic_type` | 从格式类型升级为语义类型 |
| `mom_id` | `mom_id` | 一致 |
| 无 `component` | `component` | **本方案新增** |
| 无 `repair_level` | `repair_level` | **本方案新增** |
| 无 `fault_triplet` | `fault_triplet` | **本方案新增** |
| 无 `domain_tags` | `domain_tags` | **本方案新增** |

---

## 八、最终 Chunk 完整数据模型（合并 v6 分块 + v6 Metadata）

```json
{
  "chunk_id": "PART9_F02_step",
  "content_type": "table_row",
  "text": "[上下文] ## 故障现象与排除\n\n| 故障现象 | 故障原因及排除方法 |\n| 接通电源，仪表正常显示，调速失灵 | 控制器或电机故障：联系授权服务站检测维修。 |",
  "token_count": 85,
  "parent_id": "PART9_01_parent",
  "child_ids": [],
  "mom_id": "PART9_故障表_full",

  "semantic_type": "故障诊断",
  "component": ["控制器", "电机"],
  "fault_symptom": "接通电源，仪表正常显示，调速失灵",
  "repair_action": "联系授权服务站检测维修",
  "repair_level": "service_center",
  "risk_level": null,

  "fault_triplet": {
    "symptom": "接通电源，仪表正常显示，调速失灵",
    "component": "控制器",
    "cause": "控制器或电机故障",
    "action": "联系授权服务站检测维修",
    "can_diy": false
  },

  "important_kwd": "调速失灵, 控制器故障, 电机故障",
  "question_kwd": "仪表正常但拧油门不走是什么原因？\n调速失灵怎么修？",
  "domain_tags": ["故障", "控制器", "电机"],

  "metadata": {
    "part": "PART 9",
    "section": "故障现象与排除",
    "page": 25,
    "is_vlm_enhanced": false,
    "warning_level": null
  }
}
```

---

## 九、你的提案 → v6 的吸收路径

| 你的提案 | 吸收 | 调整 |
|----------|:---:|------|
| 文档级 metadata | — | 单说明书场景不需要跨文档字段 |
| `system` | → `part` | 说明书按 PART 组织 |
| `component` | ✅ | 核心字段，LLM + 规则双路提取 |
| `fault_symptom` / `fault_code` | ✅ 前者 | fault_code 改为描述性文本（本手册无编码体系） |
| `repair_action` / `repair_level` | ✅ 简化 | repair_level 从 4 级压缩为 2 级 |
| `risk_level` | ✅ | 直接映射警告标签 |
| `tool_required` | ⚠️ | 降级为可选，仅维护章节有用 |
| `important_keywords` / `generated_questions` / `synonyms` | ✅ 前三者 | synonyms 融入 question_kwd |
| Chunk 类型体系 | ✅ | 简化到 7 种 |
| 故障三元组 | ✅ | 已在 v6 分块方案 |
| Query→Metadata 对齐 | ⚠️ | 字段就位，评分逻辑在检索层 |
| 语义优先原则 | ✅ | 作为设计原则写入 |
| 可检索优先原则 | ✅ | 字段必须有检索用途 |
| 结构/语义分离原则 | ✅ | 三层结构体现了分离 |

---

*终版 v6，2026-06-18*
