"""
Retriever 评测脚本
调用 POST /api/chat，评估检索命中率、MRR、Precision
"""

import json
import sys
import os
import time
from datetime import datetime
from collections import defaultdict
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/api/chat"
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "retriever_report.md")
REQUEST_TIMEOUT = 120.0
TOP_K = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def call_api(client: httpx.AsyncClient, query: str, top_k: int = TOP_K) -> dict | None:
    """Call POST /api/chat, return parsed JSON or None on failure."""
    try:
        resp = await client.post(
            API_URL,
            json={"query": query, "top_k": top_k},
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        print(f"  ❌ 无法连接服务器 — 请确认 search_server.py 在 localhost:8000 运行")
        return None
    except httpx.TimeoutException:
        print(f"  ⏱ 请求超时 ({REQUEST_TIMEOUT}s)")
        return None
    except httpx.HTTPStatusError as e:
        print(f"  ❌ HTTP {e.response.status_code}")
        return None
    except Exception as e:
        print(f"  ❌ 请求异常: {e}")
        return None


def compute_metrics(results: list[dict]) -> dict:
    """Aggregate Recall@5, Recall@10, MRR, Precision@5 from per-item results."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    test_set = load_test_set(TEST_SET_PATH)
    print(f"📂 加载 {len(test_set)} 条测试数据\n")

    results: list[dict] = []
    failures: list[dict] = []

    async with httpx.AsyncClient(proxy=None, trust_env=False) as client:
        # Quick health check
        try:
            h = await client.get("http://localhost:8000/api/health", timeout=httpx.Timeout(5.0))
            h.raise_for_status()
            health = h.json()
            print(f"🟢 服务器就绪 (qdrant points: {health.get('qdrant_points', '?')})\n")
        except Exception:
            print("🔴 服务器未启动 — 请先运行 search_server.py (localhost:8000)")
            print("   评测终止。")
            return

        for i, item in enumerate(test_set):
            qid = item["id"]
            question = item["question"]
            expected_ids = set(item["expected_chunk_ids"])
            expected_type = item["expected_semantic_type"]
            difficulty = item["difficulty"]
            expected_route = item.get("expected_route")

            print(f"[{i+1:02d}/{len(test_set)}] {qid}: {question}")

            resp = await call_api(client, question)
            if resp is None:
                # Server died mid-run
                failures.append({
                    **item,
                    "recall@5": False,
                    "recall@10": False,
                    "mrr": 0.0,
                    "precision@5": 0.0,
                    "actual_top5_ids": [],
                    "actual_top5_texts": [],
                    "error": "API call failed",
                })
                continue

            sources = resp.get("sources", [])
            # Extract chunk_ids from sources in rank order
            source_ids = [s["chunk_id"] for s in sources]

            # ---- Recall@K ----
            top5_ids = source_ids[:5]
            top10_ids = source_ids[:10]
            recall5 = 1 if expected_ids and (expected_ids & set(top5_ids)) else 0
            recall10 = 1 if expected_ids and (expected_ids & set(top10_ids)) else 0

            # ---- MRR ----
            mrr = 0.0
            for rank, cid in enumerate(source_ids, start=1):
                if cid in expected_ids:
                    mrr = 1.0 / rank
                    break

            # ---- Precision@5 ----
            correct_in_5 = len(expected_ids & set(top5_ids))
            prec5 = correct_in_5 / 5.0 if top5_ids else 0.0

            hit_status = "✅" if recall5 else "❌"
            print(f"    {hit_status} R@5={recall5} R@10={recall10} MRR={mrr:.3f} P@5={prec5:.2f}")

            record = {
                **item,
                "recall@5": bool(recall5),
                "recall@10": bool(recall10),
                "mrr": mrr,
                "precision@5": prec5,
                "actual_top5_ids": top5_ids,
                "actual_top5_texts": [s.get("text_preview", "")[:80] for s in sources[:5]],
                "actual_top5_semantic_types": [s.get("semantic_type", "") for s in sources[:5]],
                "answer": resp.get("answer", "")[:200],
            }
            results.append(record)
            if not recall5:
                failures.append(record)

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------
    metrics = compute_metrics(results)

    # Group by semantic_type
    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_type[r["expected_semantic_type"]].append(r)

    # Group by difficulty
    by_diff: dict[str, list] = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)

    # Safety routing accuracy
    safety_items = [r for r in results if r.get("expected_route") == "safety"]
    safety_correct = 0
    for r in safety_items:
        # At least 50% of top-5 have semantic_type == "风险警告"
        stypes = r.get("actual_top5_semantic_types", [])
        if stypes:
            warn_ratio = sum(1 for t in stypes if t == "风险警告") / len(stypes)
            if warn_ratio >= 0.4:
                safety_correct += 1
    safety_acc = round(safety_correct / len(safety_items), 4) if safety_items else 1.0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build report
    lines = []
    lines.append("# Retriever 评测报告\n")
    lines.append(f"生成时间：{now}\n")
    lines.append(f"测试数据：{len(test_set)} 条\n")

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

        # Detailed failure analysis
        lines.append("### 失败案例详细分析\n")
        for f in failures:
            lines.append(f"#### {f['id']}: \"{f['question']}\"\n")
            lines.append(f"- **期望语义类型**: {f['expected_semantic_type']}")
            lines.append(f"- **期望 chunk**: {', '.join(f['expected_chunk_ids'])}")
            lines.append(f"- **难度**: {f['difficulty']}")
            lines.append(f"- **实际 Top-5 chunk 预览**:\n")
            for j, (cid, txt) in enumerate(zip(f["actual_top5_ids"], f["actual_top5_texts"])):
                lines.append(f"  {j+1}. `{cid}`: {txt}")
            lines.append("")

    # Optimization suggestions
    lines.append("## 优化方向\n")
    if metrics["recall@5"] < 0.90:
        lines.append("- **Recall@5 不达标**：")
        lines.append("  - 检查 Query Expansion 是否对口语化查询生效")
        lines.append("  - 考虑增加同义词映射（如 '拧油门不走' → '调速失灵'）")
        lines.append("  - 对 hard 类查询增加稀疏检索权重")
        lines.append("")
    if metrics["mrr"] < 0.80:
        lines.append("- **MRR 偏低**：")
        lines.append("  - 检查 LLM Rerank 是否把正确 chunk 排到后面")
        lines.append("  - 考虑 Metadata Boost 权重调整")
        lines.append("")
    hard_items = by_diff.get("hard", [])
    hard_m = compute_metrics(hard_items)
    easy_items = by_diff.get("easy", [])
    easy_m = compute_metrics(easy_items)
    if hard_m["recall@5"] < easy_m["recall@5"] * 0.7:
        lines.append("- **hard/easy 差距大**：")
        lines.append("  - hard 查询的口语化表述与 chunk 原文不匹配")
        lines.append("  - 建议在 embedding 侧增加口语→书面语的 query rewriting")
        lines.append("")

    report = "\n".join(lines)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 报告已生成: {REPORT_PATH}")
    print(f"   Recall@5: {metrics['recall@5']:.2%}  {'✅' if metrics['recall@5'] >= 0.9 else '❌'}")
    print(f"   Recall@10: {metrics['recall@10']:.2%}  {'✅' if metrics['recall@10'] >= 0.95 else '❌'}")
    print(f"   MRR: {metrics['mrr']:.3f}  {'✅' if metrics['mrr'] >= 0.8 else '❌'}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
