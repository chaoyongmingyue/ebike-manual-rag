"""
Answer 评测脚本
评估 groundedness（答案可追溯性）、key fact coverage、rejection rate
"""

import json
import os
import re
import random
from datetime import datetime
from collections import defaultdict
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/api/chat"
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "answer_report.md")
REQUEST_TIMEOUT = 120.0
SAMPLE_SIZE = 20  # top-N easy items to evaluate
SPOT_CHECK_COUNT = 5  # items for manual review suggestion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_eval_items(test_set: list[dict], n: int = SAMPLE_SIZE) -> list[dict]:
    """Select up to n easy items + all rejection items."""
    easy = [t for t in test_set if t["difficulty"] == "easy"]
    selected = easy[:n]
    # Include all "说明书没有" items for rejection rate
    rejection_items = [t for t in test_set if t["expected_chunk_ids"] == []]
    for item in rejection_items:
        if item not in selected:
            selected.append(item)
    return selected


async def call_api(client: httpx.AsyncClient, query: str) -> dict | None:
    """Call POST /api/chat, return parsed JSON or None on failure."""
    try:
        resp = await client.post(
            API_URL,
            json={"query": query, "top_k": 10},
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        print(f"  ❌ 无法连接服务器")
        return None
    except httpx.TimeoutException:
        print(f"  ⏱ 请求超时")
        return None
    except Exception as e:
        print(f"  ❌ 请求异常: {e}")
        return None


def split_sentences(text: str) -> list[str]:
    """Split answer text into sentences by punctuation or newlines."""
    # Remove markdown formatting noise
    text = re.sub(r'\*\*|\[|\]|`|#{1,6}\s*', '', text)
    text = re.sub(r'【答案】[：:]?\s*', '', text)
    text = re.sub(r'引用来源[：:].*', '', text, flags=re.DOTALL)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)

    # Split on Chinese/English sentence boundaries
    raw = re.split(r'[。！？\n;；]', text)
    sentences = []
    for s in raw:
        s = s.strip()
        # Skip very short fragments, pure numbers, reference markers
        if len(s) >= 6 and not re.match(r'^[\d\s\.,，、·\-—\[\]\(\)]+$', s):
            sentences.append(s)
    return sentences


def check_sentence_grounded(sentence: str, source_texts: list[str]) -> bool:
    """Check if at least one 8-char substring of sentence appears in any source."""
    if len(sentence) < 8:
        # Short sentences: check if any 4-char chunk matches
        step = max(4, len(sentence) // 2)
    else:
        step = 8

    combined_sources = " ".join(source_texts)

    # Sliding window over sentence
    for i in range(0, len(sentence) - step + 1, max(1, step // 2)):
        substr = sentence[i:i + step]
        if substr in combined_sources:
            return True
    return False


def compute_groundedness(answer: str, source_texts: list[str]) -> dict:
    """Compute groundedness = fraction of sentences with source support."""
    sentences = split_sentences(answer)
    if not sentences:
        return {"grounded": 0.0, "total_sentences": 0, "grounded_sentences": 0, "ungrounded_examples": []}

    grounded_count = 0
    ungrounded = []
    for s in sentences:
        if check_sentence_grounded(s, source_texts):
            grounded_count += 1
        else:
            ungrounded.append(s[:100])

    return {
        "grounded": round(grounded_count / len(sentences), 4),
        "total_sentences": len(sentences),
        "grounded_sentences": grounded_count,
        "ungrounded_examples": ungrounded[:3],
    }


def compute_key_fact_coverage(answer: str, key_facts: list[str]) -> dict:
    """Check how many key_facts are present in the answer."""
    if not key_facts:
        return {"coverage": 1.0, "total": 0, "hit": 0, "missed": []}

    hit = 0
    missed = []
    for fact in key_facts:
        # Check if a significant portion of the fact appears in answer
        # Use longest token from fact as probe
        tokens = re.split(r'[，,、\s]+', fact)
        probes = [t for t in tokens if len(t) >= 3]
        if not probes:
            probes = [fact[:min(6, len(fact))]]

        found = any(p in answer for p in probes)
        if found:
            hit += 1
        else:
            missed.append(fact)

    return {
        "coverage": round(hit / len(key_facts), 4),
        "total": len(key_facts),
        "hit": hit,
        "missed": missed,
    }


def check_rejection(answer: str) -> bool:
    """Check if answer indicates the info is not available."""
    rejection_phrases = [
        "未提及", "没有相关信息", "未找到", "不包含", "无法找到",
        "知识库中未找到", "说明书未提及", "未提供", "没有提到",
        "抱歉", "无法回答",
    ]
    return any(phrase in answer for phrase in rejection_phrases)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    test_set = load_test_set(TEST_SET_PATH)
    eval_items = select_eval_items(test_set, SAMPLE_SIZE)
    print(f"📂 选取 {len(eval_items)} 条数据用于 Answer 评测")
    print(f"   其中 easy: {sum(1 for t in eval_items if t['difficulty'] == 'easy')}")
    print(f"   其中 rejection (chunk_ids=[]): {sum(1 for t in eval_items if t['expected_chunk_ids'] == [])}\n")

    results: list[dict] = []
    spot_candidates: list[dict] = []

    async with httpx.AsyncClient(proxy=None, trust_env=False) as client:
        # Health check
        try:
            h = await client.get("http://localhost:8000/api/health", timeout=httpx.Timeout(5.0))
            h.raise_for_status()
            print("🟢 服务器就绪\n")
        except Exception:
            print("🔴 服务器未启动 — 请先运行 search_server.py (localhost:8000)")
            return

        for i, item in enumerate(eval_items):
            qid = item["id"]
            question = item["question"]
            key_facts = item.get("key_facts", [])
            is_rejection_test = (item["expected_chunk_ids"] == [])

            tag = "🚫" if is_rejection_test else "📝"
            print(f"[{i+1:02d}/{len(eval_items)}] {tag} {qid}: {question}")

            resp = await call_api(client, question)
            if resp is None:
                results.append({
                    **item,
                    "answer": "",
                    "groundedness": 0.0,
                    "total_sentences": 0,
                    "grounded_sentences": 0,
                    "fact_coverage": 0.0,
                    "fact_total": len(key_facts),
                    "fact_hit": 0,
                    "fact_missed": key_facts,
                    "is_rejection": False,
                    "error": "API call failed",
                })
                continue

            answer = resp.get("answer", "")
            sources = resp.get("sources", [])
            source_texts = [s.get("text_full", "") for s in sources]

            # 1. Groundedness
            g = compute_groundedness(answer, source_texts)
            print(f"    Groundedness: {g['grounded']:.1%} ({g['grounded_sentences']}/{g['total_sentences']} sentences)")

            # 2. Key Fact Coverage
            k = compute_key_fact_coverage(answer, key_facts)
            if k["total"] > 0:
                print(f"    Key Fact Coverage: {k['coverage']:.1%} ({k['hit']}/{k['total']})")

            # 3. Rejection check
            is_rejection = check_rejection(answer)
            if is_rejection_test:
                print(f"    Rejection: {'✅' if is_rejection else '❌'} (expected rejection)")

            record = {
                **item,
                "answer": answer,
                "groundedness": g["grounded"],
                "total_sentences": g["total_sentences"],
                "grounded_sentences": g["grounded_sentences"],
                "ungrounded_examples": g.get("ungrounded_examples", []),
                "fact_coverage": k["coverage"],
                "fact_total": k["total"],
                "fact_hit": k["hit"],
                "fact_missed": k["missed"],
                "is_rejection": is_rejection,
            }
            results.append(record)

            # For spot-check: collect items with answer + sources
            if len(spot_candidates) < SPOT_CHECK_COUNT * 2:
                spot_candidates.append(record)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    valid = [r for r in results if r.get("groundedness", 0) > 0 or r["total_sentences"] > 0]
    valid_with_facts = [r for r in valid if r["fact_total"] > 0]
    rejection_items = [r for r in results if r.get("expected_chunk_ids") == []]

    avg_groundedness = round(sum(r["groundedness"] for r in valid) / max(len(valid), 1), 4)
    avg_fact_cov = round(sum(r["fact_coverage"] for r in valid_with_facts) / max(len(valid_with_facts), 1), 4)
    rejection_correct = sum(1 for r in rejection_items if r.get("is_rejection"))
    rejection_rate = round(rejection_correct / max(len(rejection_items), 1), 4)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Answer 评测报告\n")
    lines.append(f"生成时间：{now}\n")
    lines.append(f"评测数据：{len(eval_items)} 条 (easy ≤ {SAMPLE_SIZE} + rejection items)\n")

    # Overview
    def stat_icon(val, tgt):
        return "✅" if val >= tgt else "❌"

    lines.append("## 总览\n")
    lines.append("| 指标 | 值 | 目标 | 状态 |")
    lines.append("|------|----|------|------|")
    lines.append(f"| Groundedness | {avg_groundedness:.1%} | ≥90% | {stat_icon(avg_groundedness, 0.90)} |")
    lines.append(f"| Key Fact Coverage | {avg_fact_cov:.1%} | ≥80% | {stat_icon(avg_fact_cov, 0.80)} |")
    lines.append(f"| Rejection Rate | {rejection_rate:.1%} ({rejection_correct}/{len(rejection_items)}) | ≥80% | {stat_icon(rejection_rate, 0.80)} |")
    lines.append("")

    # Detailed results table
    lines.append("## 逐条结果\n")
    lines.append("| ID | 问题 | Groundedness | Fact Cov | Rejection | 备注 |")
    lines.append("|----|------|-------------|----------|-----------|------|")
    for r in results:
        g_str = f"{r['groundedness']:.1%}"
        f_str = f"{r['fact_coverage']:.1%}" if r['fact_total'] > 0 else "N/A"
        rej_str = "✅" if r.get("is_rejection") else ("—" if r.get("expected_chunk_ids") != [] else "❌")
        note = ""
        if r.get("error"):
            note = f"❌ {r['error']}"
        elif r.get("ungrounded_examples"):
            note = f"未落地: {r['ungrounded_examples'][0][:40]}..."
        lines.append(f"| {r['id']} | {r['question']} | {g_str} | {f_str} | {rej_str} | {note} |")
    lines.append("")

    # Ungrounded sentences analysis
    ungrounded_cases = [r for r in valid if r["groundedness"] < 0.5]
    if ungrounded_cases:
        lines.append("## 低落地率案例\n")
        for r in ungrounded_cases:
            lines.append(f"### {r['id']}: \"{r['question']}\"\n")
            lines.append(f"**Groundedness**: {r['groundedness']:.1%} ({r['grounded_sentences']}/{r['total_sentences']})\n")
            lines.append(f"**未落地句子示例**:\n")
            for ex in r.get("ungrounded_examples", [])[:3]:
                lines.append(f"- {ex}\n")
            lines.append(f"**完整回答**:\n> {r['answer'][:300]}\n")
            lines.append("")

    # Key fact misses
    fact_miss_cases = [r for r in valid if r["fact_coverage"] < 0.7 and r["fact_total"] > 0]
    if fact_miss_cases:
        lines.append("## Key Fact 遗漏案例\n")
        for r in fact_miss_cases:
            lines.append(f"### {r['id']}: \"{r['question']}\"\n")
            lines.append(f"**Fact Coverage**: {r['fact_coverage']:.1%} ({r['fact_hit']}/{r['fact_total']})\n")
            lines.append(f"**遗漏的 key facts**:\n")
            for m in r.get("fact_missed", []):
                lines.append(f"- {m}\n")
            lines.append("")

    # Spot check suggestions
    lines.append("## 人工评分建议\n")
    lines.append("> 抽查以下 5 条进行 1-5 分人工评分（1=完全错误/无关，5=完美）\n")
    random.seed(42)
    spot_items = random.sample(spot_candidates, min(SPOT_CHECK_COUNT, len(spot_candidates)))
    for idx, r in enumerate(spot_items, 1):
        lines.append(f"### 抽查 #{idx}: {r['id']} — \"{r['question']}\"\n")
        lines.append(f"- **语义类型**: {r['expected_semantic_type']}")
        lines.append(f"- **难度**: {r['difficulty']}")
        lines.append(f"- **期望 Key Facts**: {', '.join(r.get('key_facts', []))}\n")
        lines.append(f"**Answer**:\n\n{r['answer'][:600]}\n")
        src_previews = [s.get("text_preview", "")[:100] for s in []]  # We don't store per-item sources in results
        lines.append(f"**评分**: ___ / 5\n")
        lines.append(f"**评语**: \n")
        lines.append("")

    # Optimization suggestions
    lines.append("## 优化方向\n")
    if avg_groundedness < 0.90:
        lines.append("- **Groundedness 不达标**：")
        lines.append("  - 检查 LLM prompt 是否强调 '只能依据检索结果回答'")
        lines.append("  - 考虑降低 temperature 减少编造")
        lines.append("  - 对未落地句子分析是否来自 LLM 常识而非 retrieval")
        lines.append("")
    if avg_fact_cov < 0.80:
        lines.append("- **Key Fact Coverage 偏低**：")
        lines.append("  - 检查检索是否遗漏了包含 key facts 的 chunk")
        lines.append("  - 考虑增加 context window 或 top_k")
        lines.append("  - 检查 LLM 是否在摘要时丢失了关键细节")
        lines.append("")
    if rejection_rate < 0.80:
        lines.append("- **Rejection Rate 偏低**：")
        lines.append("  - LLM 可能在编造不存在的信息而非拒绝回答")
        lines.append("  - 强化 prompt 中的拒答指令")
        lines.append("  - 对 chunk_ids=[] 类查询增加前置过滤")
        lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 报告已生成: {REPORT_PATH}")
    print(f"   Groundedness: {avg_groundedness:.1%}  {stat_icon(avg_groundedness, 0.90)}")
    print(f"   Key Fact Coverage: {avg_fact_cov:.1%}  {stat_icon(avg_fact_cov, 0.80)}")
    print(f"   Rejection Rate: {rejection_rate:.1%}  {stat_icon(rejection_rate, 0.80)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
