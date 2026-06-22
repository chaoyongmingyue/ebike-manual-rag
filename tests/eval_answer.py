"""
Answer 评测脚本 v2 (并发)
- 调用 POST /api/chat 获取 full answer + sources
- asyncio.Semaphore(3) 控制并发（LLM生成较重，不宜太高）
- 评估 groundedness / key fact coverage / rejection rate
"""

import json
import os
import re
import time
import random
import asyncio
from datetime import datetime

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/api/chat"
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "answer_report.md")
REQUEST_TIMEOUT = 180.0  # LLM generation can be slow
SAMPLE_SIZE = 20
SPOT_CHECK_COUNT = 5
CONCURRENCY = 3  # 并发数，answer生成较重故设低

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_eval_items(test_set: list[dict], n: int = SAMPLE_SIZE) -> list[dict]:
    easy = [t for t in test_set if t["difficulty"] == "easy"]
    selected = easy[:n]
    rejection_items = [t for t in test_set if t["expected_chunk_ids"] == []]
    for item in rejection_items:
        if item not in selected:
            selected.append(item)
    return selected


def split_sentences(text: str) -> list[str]:
    text = re.sub(r'\*\*|\[|\]|`|#{1,6}\s*', '', text)
    text = re.sub(r'【答案】[：:]?\s*', '', text)
    text = re.sub(r'引用来源[：:].*', '', text, flags=re.DOTALL)
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    raw = re.split(r'[。！？\n;；]', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) >= 6 and not re.match(r'^[\d\s\.,，、·\-—\[\]\(\)]+$', s):
            sentences.append(s)
    return sentences


def check_sentence_grounded(sentence: str, source_texts: list[str]) -> bool:
    step = max(4, len(sentence) // 2) if len(sentence) < 8 else 8
    combined = " ".join(source_texts)
    for i in range(0, len(sentence) - step + 1, max(1, step // 2)):
        if sentence[i:i + step] in combined:
            return True
    return False


def compute_groundedness(answer: str, source_texts: list[str]) -> dict:
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
    if not key_facts:
        return {"coverage": 1.0, "total": 0, "hit": 0, "missed": []}
    hit = 0
    missed = []
    for fact in key_facts:
        tokens = re.split(r'[，,、\s]+', fact)
        probes = [t for t in tokens if len(t) >= 3]
        if not probes:
            probes = [fact[:min(6, len(fact))]]
        if any(p in answer for p in probes):
            hit += 1
        else:
            missed.append(fact)
    return {"coverage": round(hit / len(key_facts), 4), "total": len(key_facts), "hit": hit, "missed": missed}


def check_rejection(answer: str) -> bool:
    phrases = ["未提及", "没有相关信息", "未找到", "不包含", "无法找到",
               "知识库中未找到", "说明书未提及", "未提供", "没有提到", "抱歉", "无法回答"]
    return any(p in answer for p in phrases)


# ---------------------------------------------------------------------------
# Concurrent API call
# ---------------------------------------------------------------------------
async def call_api_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    item: dict,
    total: int,
    progress: dict,
) -> dict:
    async with sem:
        try:
            resp = await client.post(
                API_URL,
                json={"query": item["question"], "top_k": 10},
                timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "")
            sources = data.get("sources", [])
            source_texts = [s.get("text_full", "") for s in sources]

            g = compute_groundedness(answer, source_texts)
            k = compute_key_fact_coverage(answer, item.get("key_facts", []))
            is_rej = check_rejection(answer)

            progress["done"] += 1
            elapsed = time.time() - progress["start"]
            eta = (elapsed / progress["done"]) * (total - progress["done"]) if progress["done"] > 0 else 0
            tag = "R" if item.get("expected_chunk_ids") == [] else "A"
            print(f"  [{progress['done']:02d}/{total}] {item['id']} [{tag}]  "
                  f"G={g['grounded']:.0%}  K={k['coverage']:.0%}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

            return {
                "answer": answer,
                "groundedness": g["grounded"],
                "total_sentences": g["total_sentences"],
                "grounded_sentences": g["grounded_sentences"],
                "ungrounded_examples": g.get("ungrounded_examples", []),
                "fact_coverage": k["coverage"],
                "fact_total": k["total"],
                "fact_hit": k["hit"],
                "fact_missed": k["missed"],
                "is_rejection": is_rej,
                "error": None,
            }
        except Exception as e:
            progress["done"] += 1
            progress["errors"] += 1
            print(f"  [{progress['done']:02d}/{total}] {item['id']} ✗  {e}")
            return {
                "answer": "",
                "groundedness": 0.0,
                "total_sentences": 0,
                "grounded_sentences": 0,
                "ungrounded_examples": [],
                "fact_coverage": 0.0,
                "fact_total": len(item.get("key_facts", [])),
                "fact_hit": 0,
                "fact_missed": item.get("key_facts", []),
                "is_rejection": False,
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    test_set = load_test_set(TEST_SET_PATH)
    eval_items = select_eval_items(test_set, SAMPLE_SIZE)
    total = len(eval_items)
    print(f"📂 选取 {total} 条数据用于 Answer 评测  |  并发: {CONCURRENCY} 路")
    print(f"   其中 easy: {sum(1 for t in eval_items if t['difficulty'] == 'easy')}")
    print(f"   其中 rejection: {sum(1 for t in eval_items if t['expected_chunk_ids'] == [])}\n")

    # Health check
    async with httpx.AsyncClient(proxy=None, trust_env=False) as client:
        try:
            h = await client.get("http://localhost:8000/api/health", timeout=httpx.Timeout(5.0))
            h.raise_for_status()
            print("🟢 服务器就绪\n")
        except Exception as e:
            print(f"🔴 服务器未就绪: {e}\n")

    # Concurrent execution
    sem = asyncio.Semaphore(CONCURRENCY)
    progress = {"done": 0, "errors": 0, "start": time.time()}

    async with httpx.AsyncClient(proxy=None, trust_env=False) as client:
        tasks = [call_api_one(client, sem, item, total, progress) for item in eval_items]
        api_results = await asyncio.gather(*tasks)

    # Merge
    results = []
    for item, api_r in zip(eval_items, api_results):
        results.append({**item, **api_r})

    elapsed_total = time.time() - progress["start"]
    print(f"\n⏱ 总耗时: {elapsed_total:.0f}s  |  错误: {progress['errors']} 条\n")

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    valid = [r for r in results if r["total_sentences"] > 0]
    valid_with_facts = [r for r in valid if r["fact_total"] > 0]
    rejection_items = [r for r in results if r.get("expected_chunk_ids") == []]

    avg_groundedness = round(sum(r["groundedness"] for r in valid) / max(len(valid), 1), 4)
    avg_fact_cov = round(sum(r["fact_coverage"] for r in valid_with_facts) / max(len(valid_with_facts), 1), 4)
    rejection_correct = sum(1 for r in rejection_items if r.get("is_rejection"))
    rejection_rate = round(rejection_correct / max(len(rejection_items), 1), 4)

    def stat_icon(v, t):
        return "✅" if v >= t else "❌"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_count = sum(1 for r in results if r.get("error"))

    lines = []
    lines.append("# Answer 评测报告\n")
    lines.append(f"生成时间：{now}\n")
    lines.append(f"评测数据：{total} 条  |  错误：{error_count} 条  |  并发：{CONCURRENCY} 路\n")

    lines.append("## 总览\n")
    lines.append("| 指标 | 值 | 目标 | 状态 |")
    lines.append("|------|----|------|------|")
    lines.append(f"| Groundedness | {avg_groundedness:.1%} | ≥90% | {stat_icon(avg_groundedness, 0.90)} |")
    lines.append(f"| Key Fact Coverage | {avg_fact_cov:.1%} | ≥80% | {stat_icon(avg_fact_cov, 0.80)} |")
    lines.append(f"| Rejection Rate | {rejection_rate:.1%} ({rejection_correct}/{len(rejection_items)}) | ≥80% | {stat_icon(rejection_rate, 0.80)} |")
    lines.append("")

    # Per-item table
    lines.append("## 逐条结果\n")
    lines.append("| ID | 问题 | Groundedness | Fact Cov | Rejection | 备注 |")
    lines.append("|----|------|-------------|----------|-----------|------|")
    for r in results:
        g_str = f"{r['groundedness']:.1%}"
        f_str = f"{r['fact_coverage']:.1%}" if r['fact_total'] > 0 else "N/A"
        rej_str = "✅" if r.get("is_rejection") else ("—" if r.get("expected_chunk_ids") != [] else "❌")
        note = ""
        if r.get("error"):
            note = f"ERR: {r['error'][:40]}"
        elif r.get("ungrounded_examples"):
            note = f"未落地: {r['ungrounded_examples'][0][:40]}..."
        lines.append(f"| {r['id']} | {r['question']} | {g_str} | {f_str} | {rej_str} | {note} |")
    lines.append("")

    # Low groundedness cases
    ungrounded_cases = [r for r in valid if r["groundedness"] < 0.5]
    if ungrounded_cases:
        lines.append("## 低落地率案例\n")
        for r in ungrounded_cases:
            lines.append(f"### {r['id']}: \"{r['question']}\"\n")
            lines.append(f"**Groundedness**: {r['groundedness']:.1%} ({r['grounded_sentences']}/{r['total_sentences']})\n")
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
            for m in r.get("fact_missed", []):
                lines.append(f"- {m}\n")
            lines.append("")

    # Manual spot-check
    lines.append("## 人工评分建议\n")
    lines.append("> 抽查以下 5 条进行 1-5 分人工评分\n")
    random.seed(42)
    spot_pool = [r for r in results if r.get("answer")]
    spot_items = random.sample(spot_pool, min(SPOT_CHECK_COUNT, len(spot_pool)))
    for idx, r in enumerate(spot_items, 1):
        lines.append(f"### 抽查 #{idx}: {r['id']} — \"{r['question']}\"\n")
        lines.append(f"- **语义类型**: {r['expected_semantic_type']}")
        lines.append(f"- **难度**: {r['difficulty']}")
        lines.append(f"- **期望 Key Facts**: {', '.join(r.get('key_facts', []))}\n")
        lines.append(f"**Answer**:\n\n{r['answer'][:600]}\n")
        lines.append(f"**评分**: ___ / 5\n")
        lines.append(f"**评语**: \n")
        lines.append("")

    # Optimization
    lines.append("## 优化方向\n")
    if avg_groundedness < 0.90:
        lines.append("- **Groundedness 不达标**：检查 prompt 强调度；降 temperature；对未落地句子分析是否来自 LLM 常识\n")
    if avg_fact_cov < 0.80:
        lines.append("- **Key Fact Coverage 偏低**：检查检索是否遗漏关键 chunk；增 context window；LLM 摘要是否丢失细节\n")
    if rejection_rate < 0.80:
        lines.append("- **Rejection Rate 偏低**：LLM 在编造不存在的信息；强化 prompt 拒答指令；对 chunk_ids=[] 查询增加前置过滤\n")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 报告: {REPORT_PATH}")
    print(f"   Groundedness: {avg_groundedness:.1%}  {stat_icon(avg_groundedness, 0.90)}")
    print(f"   Key Fact Coverage: {avg_fact_cov:.1%}  {stat_icon(avg_fact_cov, 0.80)}")
    print(f"   Rejection Rate: {rejection_rate:.1%}  {stat_icon(rejection_rate, 0.80)}")


if __name__ == "__main__":
    asyncio.run(main())
