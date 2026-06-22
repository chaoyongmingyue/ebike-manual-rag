"""
Retriever 评测脚本 v2 (高并发)
- 调用 POST /api/debug/context（跳过LLM答案生成，仅取检索结果）
- 回退: 若 debug 端点不可用则用 /api/chat
- asyncio.Semaphore 控制并发，默认5路并行
"""

import json
import os
import time
import asyncio
from datetime import datetime
from collections import defaultdict

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_CHAT = "http://localhost:8000/api/chat"
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "retriever_report.md")
REQUEST_TIMEOUT = 120.0
TOP_K = 10
CONCURRENCY = 5  # 并发数，可根据服务器负载调整

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(results: list[dict]) -> dict:
    n = len(results) or 1
    recall5 = sum(1 for r in results if r["recall@5"]) / n
    recall10 = sum(1 for r in results if r["recall@10"]) / n
    mrr = sum(r["mrr"] for r in results) / n
    prec5 = sum(r["precision@5"] for r in results) / n
    return {
        "total": len(results),
        "recall@5": round(recall5, 4),
        "recall@10": round(recall10, 4),
        "mrr": round(mrr, 4),
        "precision@5": round(prec5, 4),
    }


def status_icon(value: float, target: float) -> str:
    return "✅" if value >= target else "❌"


async def call_api_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    item: dict,
    total: int,
    progress: dict,
) -> dict:
    """Call /api/chat with skip_answer=true (full retrieval pipeline, no LLM generation)."""
    async with sem:
        try:
            resp = await client.post(
                API_CHAT,
                json={"query": item["question"], "top_k": TOP_K, "skip_answer": True},
                timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
            )
            resp.raise_for_status()
            data = resp.json()
            sources = data.get("sources", [])

            progress["done"] += 1
            elapsed = time.time() - progress["start"]
            eta = (elapsed / progress["done"]) * (total - progress["done"]) if progress["done"] > 0 else 0
            print(f"  [{progress['done']:02d}/{total}] {item['id']} ✓  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

            return {
                "source_ids": [s["chunk_id"] for s in sources],
                "source_semantic_types": [s.get("semantic_type", "") for s in sources],
                "source_previews": [s.get("text_preview", "")[:80] for s in sources],
                "error": None,
            }
        except Exception as e:
            progress["done"] += 1
            progress["errors"] += 1
            print(f"  [{progress['done']:02d}/{total}] {item['id']} ✗  {e}")
            return {
                "source_ids": [],
                "source_semantic_types": [],
                "source_previews": [],
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# Report writer (synchronous)
# ---------------------------------------------------------------------------
def write_report(results: list[dict], failures: list[dict]):
    metrics = compute_metrics(results)

    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_type[r["expected_semantic_type"]].append(r)

    by_diff: dict[str, list] = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)

    # Safety routing accuracy
    safety_items = [r for r in results if r.get("expected_route") == "safety"]
    safety_correct = 0
    for r in safety_items:
        stypes = r.get("actual_top5_semantic_types", [])
        if stypes:
            warn_ratio = sum(1 for t in stypes if t == "风险警告") / len(stypes)
            if warn_ratio >= 0.4:
                safety_correct += 1
    safety_acc = round(safety_correct / len(safety_items), 4) if safety_items else 1.0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_count = sum(1 for r in results if r.get("error"))

    lines = []
    lines.append("# Retriever 评测报告\n")
    lines.append(f"生成时间：{now}\n")
    lines.append(f"测试数据：{len(results)} 条  |  错误：{error_count} 条  |  并发：{CONCURRENCY} 路\n")

    # Overview
    lines.append("## 总览\n")
    lines.append("| 指标 | 值 | 目标 | 状态 |")
    lines.append("|------|----|------|------|")
    lines.append(f"| Recall@5 | {metrics['recall@5']:.2%} | ≥90% | {status_icon(metrics['recall@5'], 0.90)} |")
    lines.append(f"| Recall@10 | {metrics['recall@10']:.2%} | ≥95% | {status_icon(metrics['recall@10'], 0.95)} |")
    lines.append(f"| MRR | {metrics['mrr']:.3f} | ≥0.80 | {status_icon(metrics['mrr'], 0.80)} |")
    lines.append(f"| Precision@5 | {metrics['precision@5']:.2%} | — | — |")
    lines.append("")

    # By semantic type
    lines.append("## 按语义类型\n")
    lines.append("| 类型 | 条数 | Recall@5 | Recall@10 | MRR | P@5 |")
    lines.append("|------|------|----------|-----------|-----|-----|")
    for stype in ["故障诊断", "操作步骤", "参数查询", "风险警告", "部件说明", "电路拓扑", "概述说明"]:
        items = by_type.get(stype, [])
        if not items:
            continue
        m = compute_metrics(items)
        lines.append(f"| {stype} | {m['total']} | {m['recall@5']:.2%} | {m['recall@10']:.2%} | {m['mrr']:.3f} | {m['precision@5']:.2%} |")
    lines.append("")

    # By difficulty
    lines.append("## 按难度\n")
    lines.append("| 难度 | 条数 | Recall@5 | Recall@10 | MRR |")
    lines.append("|------|------|----------|-----------|-----|")
    for diff in ["easy", "hard"]:
        items = by_diff.get(diff, [])
        if not items:
            continue
        m = compute_metrics(items)
        lines.append(f"| {diff} | {m['total']} | {m['recall@5']:.2%} | {m['recall@10']:.2%} | {m['mrr']:.3f} |")
    lines.append("")

    # Safety routing
    lines.append("## 安全类路由准确率\n")
    lines.append(f"{safety_acc:.1%} ({safety_correct}/{len(safety_items)})\n")
    lines.append("> 判定标准：Top-5 sources 中 semantic_type='风险警告' 占比 ≥ 40%\n")

    # Failure cases
    lines.append("## 失败案例详情\n")
    if not failures:
        lines.append("✅ 全部通过，无失败案例。\n")
    else:
        lines.append(f"共 {len(failures)} 条 Recall@5=0 的案例：\n")
        lines.append("| ID | 问题 | 期望 Chunk | 实际 Top-5 Chunk | R@5 |")
        lines.append("|----|------|-----------|-----------------|-----|")
        for f in failures:
            expected = ", ".join(f["expected_chunk_ids"])
            actual = ", ".join(f["actual_top5_ids"])
            lines.append(f"| {f['id']} | {f['question']} | {expected} | {actual} | ❌ |")
        lines.append("")

        lines.append("### 失败案例详细分析\n")
        for f in failures:
            lines.append(f"#### {f['id']}: \"{f['question']}\"\n")
            lines.append(f"- **期望语义类型**: {f['expected_semantic_type']}")
            lines.append(f"- **期望 chunk**: {', '.join(f['expected_chunk_ids'])}")
            lines.append(f"- **难度**: {f['difficulty']}")
            if f.get("error"):
                lines.append(f"- **错误**: {f['error']}")
            lines.append(f"- **实际 Top-5 chunk 预览**:\n")
            for j, (cid, txt) in enumerate(zip(f["actual_top5_ids"], f["actual_top5_texts"])):
                lines.append(f"  {j+1}. `{cid}`: {txt}")
            lines.append("")

    # Optimization
    lines.append("## 优化方向\n")
    if metrics["recall@5"] < 0.90:
        lines.append("- **Recall@5 不达标**：检查 Query Expansion 对口语化查询是否生效；增加同义词映射；对 hard 类查询增加稀疏检索权重\n")
    if metrics["mrr"] < 0.80:
        lines.append("- **MRR 偏低**：检查 LLM Rerank 排序质量；Metadata Boost 权重调整\n")
    hard_items = by_diff.get("hard", [])
    easy_items = by_diff.get("easy", [])
    hard_m = compute_metrics(hard_items)
    easy_m = compute_metrics(easy_items)
    if hard_m["recall@5"] < easy_m["recall@5"] * 0.7:
        lines.append("- **hard/easy 差距大**：口语化表述与 chunk 原文不匹配，建议增加口语→书面语 query rewriting\n")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 报告: {REPORT_PATH}")
    print(f"   Recall@5: {metrics['recall@5']:.2%}  {status_icon(metrics['recall@5'], 0.90)}")
    print(f"   Recall@10: {metrics['recall@10']:.2%}  {status_icon(metrics['recall@10'], 0.95)}")
    print(f"   MRR: {metrics['mrr']:.3f}  {status_icon(metrics['mrr'], 0.80)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    test_set = load_test_set(TEST_SET_PATH)
    total = len(test_set)
    print(f"📂 加载 {total} 条测试数据  |  并发: {CONCURRENCY} 路\n")

    # Health check
    async with httpx.AsyncClient(proxy=None, trust_env=False) as client:
        try:
            h = await client.get("http://localhost:8000/api/health", timeout=httpx.Timeout(5.0))
            h.raise_for_status()
            health = h.json()
            print(f"🟢 服务器就绪 (qdrant: {health.get('qdrant_points', '?')} points)")
        except Exception as e:
            print(f"🔴 服务器未就绪: {e}")
        print(f"⚡ 调用 /api/chat + skip_answer=true (完整检索 pipeline，跳过LLM生成)\n")

    # Concurrent execution
    sem = asyncio.Semaphore(CONCURRENCY)
    progress = {"done": 0, "errors": 0, "start": time.time()}

    async with httpx.AsyncClient(proxy=None, trust_env=False) as client:
        tasks = [call_api_one(client, sem, item, total, progress) for item in test_set]
        api_results = await asyncio.gather(*tasks)

    # Process results
    results: list[dict] = []
    failures: list[dict] = []

    for item, api_r in zip(test_set, api_results):
        expected_ids = set(item["expected_chunk_ids"])
        source_ids = api_r["source_ids"]

        top5_ids = source_ids[:5]
        top10_ids = source_ids[:10]

        recall5 = 1 if expected_ids and (expected_ids & set(top5_ids)) else 0
        recall10 = 1 if expected_ids and (expected_ids & set(top10_ids)) else 0

        mrr = 0.0
        for rank, cid in enumerate(source_ids, start=1):
            if cid in expected_ids:
                mrr = 1.0 / rank
                break

        correct_in_5 = len(expected_ids & set(top5_ids))
        prec5 = correct_in_5 / 5.0 if top5_ids else 0.0

        record = {
            **item,
            "recall@5": bool(recall5),
            "recall@10": bool(recall10),
            "mrr": mrr,
            "precision@5": prec5,
            "actual_top5_ids": top5_ids,
            "actual_top5_texts": api_r["source_previews"][:5],
            "actual_top5_semantic_types": api_r["source_semantic_types"][:5],
            "error": api_r.get("error"),
        }
        results.append(record)
        if not recall5:
            failures.append(record)

    elapsed_total = time.time() - progress["start"]
    print(f"\n⏱ 总耗时: {elapsed_total:.0f}s  |  错误: {progress['errors']} 条\n")
    write_report(results, failures)


if __name__ == "__main__":
    asyncio.run(main())
