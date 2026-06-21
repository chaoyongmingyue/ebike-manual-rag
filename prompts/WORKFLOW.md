# Prompt 升级工作流

## 修改 Prompt 的标准流程

### 1. 创建新版本
```
cd prompts/answer/
cp v1.yaml v2.yaml
# 编辑 v2.yaml，修改 prompt 字段
# 更新 version: v2
# 更新 description
```

### 2. 运行评测
```
cd tests/
python run_eval.bat
```
评测脚本会自动对比 v1 vs v2 的指标。

### 3. 对比结果
查看 `prompts/evaluation.yaml` 中 v1 和 v2 的评测指标：
- `recall_at_5` — 召回率
- `ndcg_at_5` — 排序质量
- `groundedness` — 答案忠实度（是否基于检索内容）
- `correctness` — 答案正确性

### 4. 切换默认版本
修改 `backend/search_server.py` 中的版本号：
```python
# 升级到 v2
ANSWER_PROMPT, _ = load_prompt("answer", version="v2")

# 或保持自动取最新（默认行为）
ANSWER_PROMPT, _ = load_prompt("answer")  # 自动取最大版本号
```

### 5. 提交
```bash
git add prompts/
git commit -m "prompt(answer): upgrade to v2 — improve formatting rules"
```
同时更新 `prompts/CHANGELOG.md`。

### 6. 回滚
如需回滚到旧版本：
```python
ANSWER_PROMPT, _ = load_prompt("answer", version="v1")
```
重启服务即可。无需修改 YAML 文件，Git 历史中也有完整记录。

## 目录结构
```
prompts/
├── query_expand/
│   ├── v1.yaml
│   └── v2.yaml          # 未来升级
├── rerank/
│   └── v1.yaml
├── answer/
│   └── v1.yaml
├── CHANGELOG.md          # 每个版本的变更记录
├── evaluation.yaml       # 评测指标追踪
└── WORKFLOW.md           # 本文件
```

## 变量约定
Prompt 模板使用 Python `.format()` 语法：
- `{变量名}` — 由代码填充
- `{{保留文字}}` — 渲染为 `{保留文字}`（LLM prompt 中的占位符）

每个 YAML 的 `variables` 字段列出所有需要填充的变量。
