#!/usr/bin/env python3
"""
Step 1: LLM Post-processing of chunks_v6.json
Fills: important_kwd, question_kwd, domain_tags, fault_triplet, component, risk_level, repair_level
"""

import json, re, time, os, sys, requests
from pathlib import Path

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "data"
CHUNKS_PATH = str(DATA_DIR / "chunks_v6.json")
OUTPUT_PATH = str(DATA_DIR / "chunks_v6_llm.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
MAX_TEXT_LEN = 1500
DELAY = 0.2
MAX_RETRIES = 2

KNOWN_COMPONENTS = [
    "控制器","电机","蓄电池","充电器","转换器","仪表","前照灯","尾灯",
    "转向灯","刹车灯","调速转把","刹把","空气开关","防盗器","电门锁",
    "前叉","后减震","前碟刹盘","后平叉","保险丝","电池","充电孔",
    "喇叭","闪光器","座垫","后视镜"
]

DOMAIN_OPTIONS = "安全/充电/电池/电机/控制器/骑行/保养/故障/安装/保修/参数/电路/仪表/部件"

def call_llm(prompt, max_tokens=150):
    """Call Ollama API with retry logic"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens}
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=60)
            r.raise_for_status()
            result = r.json().get("response", "").strip()
            return result
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  Retry {attempt+1}: {e}")
                time.sleep(1)
            else:
                print(f"  FAILED after {MAX_RETRIES} retries: {e}")
                return ""
    return ""

def get_llm_text(text):
    """Truncate text for LLM input"""
    if len(text) > MAX_TEXT_LEN:
        return text[:MAX_TEXT_LEN] + "..."
    return text

def extract_keywords(text):
    """Extract 3 important keywords"""
    prompt = f"从这个文本中提取 3 个最重要的关键词，用逗号分隔。只输出关键词，不要解释。\n文本：{get_llm_text(text)}"
    result = call_llm(prompt, max_tokens=80)
    # Clean up
    result = result.replace('\n', ',').strip()
    parts = [p.strip() for p in result.split(',') if p.strip()]
    return ','.join(parts[:3]) if parts else ''

def extract_questions(text):
    """Extract 3 preset questions"""
    prompt = f"用户可能对这个文本提出哪 3 个问题？每行一个。只输出问题，不要编号和解释。\n文本：{get_llm_text(text)}"
    result = call_llm(prompt, max_tokens=120)
    lines = [l.strip().lstrip('0123456789.、- ') for l in result.split('\n') if l.strip()]
    return '\n'.join(lines[:3]) if lines else ''

def extract_domain_tags(text):
    """Extract domain tags"""
    prompt = f"这段内容属于哪些领域？从以下选择：{DOMAIN_OPTIONS}。最多3个，逗号分隔。只输出标签，不要解释。\n文本：{get_llm_text(text)}"
    result = call_llm(prompt, max_tokens=60)
    parts = [p.strip() for p in result.replace('\n', ',').split(',') if p.strip()]
    # Filter to valid options
    valid = [p for p in parts if p in DOMAIN_OPTIONS.split('/')]
    return valid[:3] if valid else []

def extract_fault_triplet(text):
    """Extract fault triplet from text for 故障诊断 chunks"""
    prompt = f"这段文本描述了一个故障。请提取：symptom(症状)、component(部件)、cause(原因)、action(处理方法)、can_diy(用户能自己处理吗? true/false)。输出JSON格式。\n文本：{get_llm_text(text)}"
    result = call_llm(prompt, max_tokens=200)
    # Try to parse JSON from result
    try:
        # Find JSON object in response
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            data = json.loads(json_match.group())
            return {
                'symptom': data.get('symptom', ''),
                'cause': data.get('cause', ''),
                'action': data.get('action', ''),
                'component': data.get('component', ''),
                'can_diy': data.get('can_diy', False)
            }
    except:
        pass
    return None

def match_components(text):
    """Rule-based component matching from known list"""
    found = []
    for comp in KNOWN_COMPONENTS:
        if comp in text and comp not in found:
            found.append(comp)
    return found

def process_chunk(chunk, idx, total):
    """Process a single chunk with LLM"""
    text = chunk.get('text', '')
    sem_type = chunk.get('semantic_type', '')
    content_type = chunk.get('content_type', '')
    token_count = chunk.get('token_count', 0)

    # Skip tiny 概述说明 chunks
    if sem_type == '概述说明' and token_count < 25:
        chunk['important_kwd'] = ''
        chunk['question_kwd'] = ''
        chunk['domain_tags'] = []
        # Still try rule-based component matching
        comps = match_components(text)
        if comps:
            chunk['component'] = list(set(chunk.get('component', []) + comps))
        return chunk

    prefix = f"[{idx+1}/{total}] {chunk['chunk_id']} ({sem_type})"

    # 1. important_kwd
    if not chunk.get('important_kwd'):
        chunk['important_kwd'] = extract_keywords(text)
        time.sleep(DELAY)

    # 2. question_kwd
    if not chunk.get('question_kwd'):
        chunk['question_kwd'] = extract_questions(text)
        time.sleep(DELAY)

    # 3. domain_tags
    if not chunk.get('domain_tags'):
        chunk['domain_tags'] = extract_domain_tags(text)
        time.sleep(DELAY)

    # 4. semantic-type specific processing
    if sem_type == '故障诊断':
        # Fill fault_triplet if empty
        existing_trips = chunk.get('fault_triplet')
        if not existing_trips or len(existing_trips) == 0:
            trip = extract_fault_triplet(text)
            if trip:
                chunk['fault_triplet'] = [trip]
                chunk['fault_symptom'] = chunk['fault_symptom'] or trip.get('symptom', '')
                chunk['repair_action'] = chunk['repair_action'] or trip.get('action', '')
                if trip.get('component'):
                    comps = chunk.get('component', [])
                    if trip['component'] not in comps:
                        comps.append(trip['component'])
                    chunk['component'] = comps
            time.sleep(DELAY)

        # Fill component if empty
        if not chunk.get('component'):
            comps = match_components(text)
            if not comps and existing_trips:
                # LLM didn't help, try from triplet
                for t in existing_trips:
                    comps.extend(match_components(t.get('symptom','') + t.get('action','')))
            chunk['component'] = list(set(comps))

        # Fill repair_level
        if not chunk.get('repair_level'):
            all_text = text
            if existing_trips:
                all_text += ' '.join(t.get('action','') for t in existing_trips)
            if any(kw in all_text for kw in ['授权','服务站','返厂','售后','送修']):
                chunk['repair_level'] = 'service_center'
            elif any(kw in all_text for kw in ['自行','用户可','检查','重新插拔','充足电','充气']):
                chunk['repair_level'] = 'self_help'

    elif sem_type == '风险警告':
        # Fill risk_level
        if not chunk.get('risk_level'):
            if any(kw in text for kw in ['危险','⛔']):
                chunk['risk_level'] = 'danger'
            elif any(kw in text for kw in ['警告','⚠']):
                chunk['risk_level'] = 'warning'
            elif any(kw in text for kw in ['注意','ℹ']):
                chunk['risk_level'] = 'caution'

    elif sem_type == '操作步骤':
        # Fill component
        if not chunk.get('component'):
            comps = match_components(text)
            chunk['component'] = list(set(comps))

        # Fill repair_level
        if not chunk.get('repair_level'):
            if any(kw in text for kw in ['自行']):
                chunk['repair_level'] = 'self_help'
            elif any(kw in text for kw in ['送修','服务站','授权']):
                chunk['repair_level'] = 'service_center'

    elif sem_type in ('参数查询', '部件说明', '电路拓扑'):
        # Fill component
        comps = match_components(text)
        if comps:
            existing = chunk.get('component', [])
            chunk['component'] = list(set(existing + comps))

    print(f"  {prefix} done - kwds={bool(chunk.get('important_kwd'))}, qs={bool(chunk.get('question_kwd'))}, tags={chunk.get('domain_tags')}")

    return chunk

def main():
    # Allow CLI override
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else CHUNKS_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH

    print("Loading chunks_v6.json...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data['chunks']
    stats = data.get('stats', {})
    total = len(chunks)
    print(f"Processing {total} chunks with Ollama ({MODEL})...")

    for i, chunk in enumerate(chunks):
        # Skip already-processed chunks
        if chunk.get('_llm_processed'):
            continue
        process_chunk(chunk, i, total)
        chunk['_llm_processed'] = True

    # Update stats
    fault_trip_chunks = sum(1 for c in chunks if c.get('fault_triplet') and len(c.get('fault_triplet', [])) > 0)
    risk_chunks = sum(1 for c in chunks if c.get('semantic_type') == '风险警告' and c.get('risk_level'))
    stats['llm_processed'] = True
    stats['fault_triplet_chunks'] = fault_trip_chunks
    stats['risk_level_coverage'] = f"{risk_chunks}/{sum(1 for c in chunks if c.get('semantic_type')=='风险警告')}"

    output = {'chunks': chunks, 'stats': stats}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"LLM Post-processing complete!")
    print(f"Output: {output_path}")
    print(f"Fault triplet chunks: {fault_trip_chunks}")
    print(f"Risk level coverage: {stats['risk_level_coverage']}")

    return output

if __name__ == '__main__':
    main()
