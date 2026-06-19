#!/usr/bin/env python3
# ebike manual chunking v6 - final implementation
import re, json, os, sys
from pathlib import Path
from collections import defaultdict

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "data"
INPUT = str(DATA_DIR / "电动自行车说明书_for_chunking.md")
OUTPUT = str(DATA_DIR / "chunks_v6.json")

CHARS_PER_TOKEN = 4
HARD_LIMIT = 768

COMPONENTS = [
    "控制器","电机","蓄电池","充电器","转换器","仪表",
    "前照灯","尾灯","转向灯","刹车灯","调速转把","刹把",
    "空气开关","防盗器","电门锁","前叉","后减震","前碟刹盘",
    "后平叉","保险丝","BMS","电池"
]
FAULT_KW = ["故障","异常","排除","原因","处理","维修方案"]
WARN_MAP = {"危险":"danger","警告":"warning","注意":"caution"}

def tok(text):
    return max(1, len(text) // CHARS_PER_TOKEN)

def clean(text):
    t = re.sub(r'\[VLM Error:.*?\]', '', text)
    t = re.sub(r'\n{4,}', '\n\n\n', t)
    return t.strip()

def extract_parts(text):
    return [c for c in COMPONENTS if c in text]

# ============================================================
# Phase 1+2: Structure Parsing
# ============================================================
def parse_document(lines):
    """Split file into PART sections"""
    sections = []
    part_starts = []
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r'^#{1,2}\s+PART\s+\d+', s):
            m = re.match(r'^#{1,2}\s+PART\s+(\d+)\s*(.*)', s)
            part_starts.append((i, m.group(1), m.group(2).strip(), s))
    for pi, (line_idx, pnum, ptitle, heading) in enumerate(part_starts):
        end = part_starts[pi+1][0] if pi+1 < len(part_starts) else len(lines)
        sections.append({
            'type':'part','part_num':pnum,'part_title':ptitle,
            'heading_line':heading,'lines':lines[line_idx:end]
        })
    return sections

def extract_content_blocks(section_lines):
    """Extract tables, steps, text from a PART section"""
    blocks = []
    i = 1; n = len(section_lines)
    text_accum = []

    def flush():
        nonlocal text_accum
        if not text_accum: return
        t = clean('\n'.join(text_accum))
        text_accum.clear()
        if not t: return
        wl = None
        fl = t.split('\n')[0] if t else ''
        for mk, lv in WARN_MAP.items():
            if "## " + mk in fl: wl = lv; break
        blocks.append({'type':'text','text':t,'warning_level':wl})

    while i < n:
        s = section_lines[i].strip()
        if not s:
            if text_accum: text_accum.append('')
            i += 1; continue
        # Skip images and VLM descriptions
        if (s.startswith('![') or s.startswith('images\\') or
            any(x in s for x in ['这张图片','这幅图片','该图片','这是一幅','这张图','该图']) or
            s == '---' or s.startswith('text_list') or s.startswith('[VLM Error')):
            if any(x in s for x in ['这张图片','这幅图片','该图片','这是一幅','这张图','该图']):
                while i < n and section_lines[i].strip(): i += 1
                continue
            i += 1; continue
        # Table
        if s.startswith('|'):
            tbl = []
            while i < n and section_lines[i].strip().startswith('|'): tbl.append(section_lines[i].strip()); i += 1
            if any(re.match(r'^\|\s*[-:]+\s*\|', l) for l in tbl) and len(tbl) >= 3:
                flush(); blocks.append({'type':'table','text':'\n'.join(tbl)})
            else:
                for tl in tbl: text_accum.append(tl)
            continue
        # Steps
        if re.match(r'^\d+\.\s', s):
            slines = []
            while i < n:
                ss = section_lines[i].strip()
                if re.match(r'^\d+\.\s', ss): slines.append(ss); i += 1
                elif ss == '': slines.append(''); i += 1
                else: break
            if sum(1 for x in slines if re.match(r'^\d+\.\s', x)) >= 3:
                flush(); blocks.append({'type':'steps','text':'\n'.join(slines).strip()})
            else:
                for x in slines:
                    if x: text_accum.append(x)
            continue
        text_accum.append(s); i += 1
    flush()
    return blocks

# ============================================================
# Phase 3: Semantic Classification
# ============================================================
def classify(block):
    text = block.get('text',''); btype = block.get('type',''); wl = block.get('warning_level')
    if wl: return '风险警告'
    if btype == 'table':
        if any(kw in text for kw in ['故障现象','故障原因','排除方法','问题现象','处理方法','检查项目','检查内容']): return '故障诊断'
        if re.search(r'\d+\s*(V|A|W|km|kg|mm|cm|r/min|Ah|kw|h|%|℃|月|年|个月|公里)', text): return '参数查询'
        if any(kw in text for kw in ['保修期限','三包','保修']): return '参数查询'
        return '概述说明'
    if btype == 'steps':
        if any(kw in text for kw in FAULT_KW): return '故障诊断'
        return '操作步骤'
    if btype == 'text':
        if any(kw in text for kw in FAULT_KW):
            if re.search(r'故障现象|故障原因|排除方法|问题现象|处理方法|维修方案', text): return '故障诊断'
            if '故障' in text and any(kw in text for kw in ['原因','维修','更换','处理']): return '故障诊断'
        for mk in WARN_MAP:
            if "## " + mk in text.split('\n')[0]: return '风险警告'
        if any(kw in text for kw in ['连接','供电','电路','拓扑','导线','接口']): return '电路拓扑'
        if sum(1 for l in text.split('\n') if re.match(r'^\d+\.\s', l.strip())) >= 3: return '操作步骤'
        if any(kw in text for kw in ['保养','维护','检查']): return '概述说明'
        if any(kw in text for kw in ['编号','标注','部件','结构']): return '部件说明'
        return '概述说明'
    return '概述说明'

# ============================================================
# Phase 4+5: Build chunks, split text, size control
# ============================================================
def split_text_at_h2(text):
    if not text.strip(): return [('',None,None)]
    text = text.strip()
    h2_matches = list(re.finditer(r'^##\s+(.+)', text, re.MULTILINE))
    if not h2_matches: return [(text,None,None)]

    sections = []
    for idx, m in enumerate(h2_matches):
        start = m.start(); heading = m.group(1).strip()
        next_start = h2_matches[idx+1].start() if idx+1 < len(h2_matches) else len(text)
        sub = text[start:next_start].strip()
        wl = None
        for mk, lv in WARN_MAP.items():
            if mk in heading: wl = lv; break
        sec = heading if not wl else None
        sections.append({'text':sub,'wl':wl,'section':sec,'chars':len(sub)})

    if h2_matches and h2_matches[0].start() > 0:
        pre = text[:h2_matches[0].start()].strip()
        if pre: sections.insert(0,{'text':pre,'wl':None,'section':None,'chars':len(pre)})

    result = []
    for s in sections:
        if s['chars'] > 450:
            h3s = list(re.finditer(r'^###\s+(.+)', s['text'], re.MULTILINE))
            if len(h3s) >= 2:
                parts = []
                for hi, hm in enumerate(h3s):
                    ns = h3s[hi+1].start() if hi+1 < len(h3s) else len(s['text'])
                    sub = s['text'][hm.start():ns].strip()
                    parts.append({'text':sub,'wl':s['wl'],'section':s['section'],'chars':len(sub)})
                if h3s[0].start() > 0:
                    pre = s['text'][:h3s[0].start()].strip()
                    if pre: parts.insert(0,{'text':pre,'wl':s['wl'],'section':s['section'],'chars':len(pre)})
                for p in parts:
                    if p['chars'] > 350: result.extend(split_by_paras(p))
                    else: result.append(p)
            else: result.extend(split_by_paras(s))
        elif s['chars'] > 350: result.extend(split_by_paras(s))
        else: result.append(s)

    return [(b['text'],b['wl'],b['section']) for b in result] if result else [(text,None,None)]

def split_by_paras(s):
    paras = s['text'].split('\n\n')
    if len(paras) <= 2: return [s]
    chunks = []; cur = []; cur_chars = 0
    for p in paras:
        if cur and cur_chars + len(p) > 900:
            chunks.append({'text':'\n\n'.join(cur).strip(),'wl':s['wl'],'section':s['section'],'chars':cur_chars})
            cur = []; cur_chars = 0
        cur.append(p); cur_chars += len(p) + 2
    if cur: chunks.append({'text':'\n\n'.join(cur).strip(),'wl':s['wl'],'section':s['section'],'chars':cur_chars})
    return chunks if chunks else [s]

def build_chunks(sections):
    chunks = []
    cid = [0]
    def nid(pref='c'): cid[0] += 1; return f"{pref}{cid[0]:03d}"

    for sec in sections:
        if sec['type'] != 'part': continue
        pnum = sec['part_num']; ptitle = sec['part_title']
        heading = sec['heading_line']; pid = nid('part')

        chunks.append({
            'chunk_id':pid,'semantic_type':'概述说明','content_type':'heading',
            'text':heading,'token_count':tok(heading),'parent_id':None,
            'child_ids':[],'mom_id':None,'component':[],'fault_symptom':None,
            'repair_action':None,'repair_level':None,'risk_level':None,'fault_triplet':None,
            'metadata':{'part':f"PART {pnum}",'section':ptitle,'page':None,'is_vlm_enhanced':False,'warning_level':None}
        })

        blocks = extract_content_blocks(sec['lines'])
        cur_section = ptitle; prior_text = ''

        for block in blocks:
            btype = block['type']; btext = block['text']; wl = block.get('warning_level')

            if btype == 'text':
                m = re.search(r'^##\s+(.+)', btext, re.MULTILINE)
                if m:
                    h = m.group(1).strip()
                    if not any(mk in h for mk in WARN_MAP): cur_section = h

            if btype == 'table':
                ctx = ''
                if prior_text:
                    ws = prior_text.split()[-50:]
                    if ws: ctx = ' '.join(ws)
                ft = btext
                if ctx and ctx not in btext: ft = f"[上下文] {ctx}\n\n{btext}"
                sem = classify(block)
                chunks.append({
                    'chunk_id':nid('tbl'),'semantic_type':sem,'content_type':'table',
                    'text':clean(ft),'token_count':tok(ft),'parent_id':pid,
                    'child_ids':[],'mom_id':None,'component':[],'fault_symptom':None,
                    'repair_action':None,'repair_level':None,'risk_level':None,'fault_triplet':None,
                    'metadata':{'part':f"PART {pnum}",'section':cur_section,'page':None,'is_vlm_enhanced':False,'warning_level':None}
                })
                prior_text = ''; continue

            if btype == 'steps':
                sem = classify(block)
                chunks.append({
                    'chunk_id':nid('stp'),'semantic_type':sem,'content_type':'steps',
                    'text':btext,'token_count':tok(btext),'parent_id':pid,
                    'child_ids':[],'mom_id':None,'component':[],'fault_symptom':None,
                    'repair_action':None,'repair_level':None,'risk_level':None,'fault_triplet':None,
                    'metadata':{'part':f"PART {pnum}",'section':cur_section,'page':None,'is_vlm_enhanced':False,'warning_level':None}
                })
                prior_text = ''; continue

            if btype == 'text':
                subs = split_text_at_h2(btext)
                for sub_text, sub_wl, sub_sec in subs:
                    if not sub_text.strip(): continue
                    sem = classify({'type':'text','text':sub_text,'warning_level':sub_wl})
                    sname = sub_sec if sub_sec else cur_section
                    chunks.append({
                        'chunk_id':nid('txt'),'semantic_type':sem,
                        'content_type':'warning' if sub_wl else 'text',
                        'text':clean(sub_text),'token_count':tok(sub_text),'parent_id':pid,
                        'child_ids':[],'mom_id':None,'component':[],'fault_symptom':None,
                        'repair_action':None,'repair_level':None,'risk_level':sub_wl,'fault_triplet':None,
                        'metadata':{'part':f"PART {pnum}",'section':sname,'page':None,'is_vlm_enhanced':False,'warning_level':sub_wl}
                    })
                    if sub_sec: cur_section = sub_sec
                prior_text = btext; continue
    return chunks

# ============================================================
# Post-processing
# ============================================================
def post_process(chunks):
    if not chunks: return chunks

    # 1. Split oversized
    all_c = []
    for c in chunks:
        t = c['text']
        if tok(t) > HARD_LIMIT:
            paras = t.split('\n\n')
            if len(paras) <= 1: all_c.append(c); continue
            cur = []; cur_chars = 0
            for p in paras:
                if cur and cur_chars + len(p) > HARD_LIMIT * CHARS_PER_TOKEN:
                    nc = dict(c); nc['text'] = '\n\n'.join(cur); nc['token_count'] = tok(nc['text'])
                    nc['chunk_id'] = f"{c['chunk_id']}s{len(all_c)}"; all_c.append(nc)
                    last = cur[-1] if cur else ''; overlap = last[-200:] if len(last) > 200 else last
                    cur = [overlap] if overlap.strip() else []; cur_chars = len(overlap) if overlap.strip() else 0
                cur.append(p); cur_chars += len(p) + 2
            if cur:
                nc = dict(c); nc['text'] = '\n\n'.join(cur); nc['token_count'] = tok(nc['text'])
                nc['chunk_id'] = f"{c['chunk_id']}s{len(all_c)}"; all_c.append(nc)
        else: all_c.append(c)

    # 2. Keep PART headers as-is (they're atomic heading chunks)
    pass1 = all_c

    # 3. Single-pass fragment merge per spec: < 50t fragments merge into prev non-atomic
    final = []
    for c in pass1:
        tc = c.get('token_count',0); ct = c.get('content_type','text')
        if ct in ('heading','table','steps'):
            final.append(c); continue
        # Backward merge only, with anti-cascade: don't absorb into already-large chunks
        if tc < 50 and final:
            p = final[-1]
            if (p.get('content_type') not in ('heading','table','steps') and
                p.get('token_count',0) < 120 and
                p.get('token_count',0) + tc <= HARD_LIMIT):  # Don't exceed hard limit
                p['text'] = p['text'] + '\n\n' + c['text']
                p['token_count'] = tok(p['text'])
                if c.get('risk_level') and not p.get('risk_level'):
                    p['risk_level'] = c['risk_level']
                    p['metadata']['warning_level'] = c['metadata'].get('warning_level')
                if c.get('semantic_type') != '概述说明' and p.get('semantic_type') == '概述说明':
                    p['semantic_type'] = c.get('semantic_type')
                continue
        final.append(c)

    return final

# ============================================================
# Phase 6: Fault Triplet Extraction
# ============================================================
def parse_table_triplets(text):
    lines = [l.strip() for l in text.split('\n') if l.strip().startswith('|')]
    if len(lines) < 3: return []
    sep = next((idx for idx,l in enumerate(lines) if re.match(r'^\|\s*[-:]+\s*\|',l)), None)
    if sep is None or sep == 0: return []
    hdr = lines[sep-1] if sep > 0 else lines[0]; rows = lines[sep+1:]
    hdr_cells = [re.sub(r'[*_#]','',c).strip() for c in hdr.split('|')[1:-1]]
    s_idx = c_idx = a_idx = -1
    for ci, col in enumerate(hdr_cells):
        if any(kw in col for kw in ['现象','问题','故障现象','检查项目','检查内容']): s_idx = ci
        elif any(kw in col for kw in ['原因','故障原因']): c_idx = ci
        elif any(kw in col for kw in ['处理','排除方法','处理方法','维修方案','排除','说明','质保','保修']): a_idx = ci
    if s_idx < 0 and a_idx < 0 and len(hdr_cells) >= 2: s_idx = 0; a_idx = 1
    trips = []
    for row in rows:
        cells = [re.sub(r'[*_#]','',c).strip() for c in row.split('|')[1:-1]]
        symptom = cells[s_idx] if 0 <= s_idx < len(cells) else ''
        cause = cells[c_idx] if 0 <= c_idx < len(cells) else ''
        action = cells[a_idx] if 0 <= a_idx < len(cells) else ''
        if symptom or cause or action: trips.append({'symptom':symptom,'cause':cause,'action':action})
    return trips

def extract_fault_triplets(chunks):
    total = 0
    for c in chunks:
        if c.get('semantic_type') != '故障诊断': continue
        text = c.get('text',''); ctype = c.get('content_type','')
        trips = []
        if ctype == 'table': trips = parse_table_triplets(text)
        else:
            for line in text.split('\n'):
                m = re.match(r'(.{2,30}(?:故障|异常|失灵|损坏|不通|不足|无法).{0,20})[：:,,，]\s*(.{2,80})', line)
                if m: trips.append({'symptom':m.group(1).strip(),'cause':'','action':m.group(2).strip()})
        if trips:
            c['fault_triplet'] = trips; total += len(trips)
            first = trips[0]
            c['fault_symptom'] = first.get('symptom','')
            c['repair_action'] = first.get('action','')
            all_text = ' '.join(f"{t.get('symptom','')} {t.get('cause','')} {t.get('action','')}" for t in trips)
            comps = extract_parts(all_text)
            if comps: ex = c.get('component',[]); ex.extend(cc for cc in comps if cc not in ex); c['component'] = ex
            all_actions = ' '.join(t.get('action','') for t in trips)
            if any(kw in all_actions for kw in ['授权','服务站','返厂','售后']): c['repair_level'] = '授权服务站'
            elif any(kw in all_actions for kw in ['自行','用户','重新插拔','充足电','充气']): c['repair_level'] = '用户可操作'
            else: c['repair_level'] = '经销商/用户'
        else: c['fault_triplet'] = []; comps = extract_parts(text)
        if comps: c['component'] = comps
    return chunks, total

# ============================================================
# Parent-child + Main
# ============================================================
def build_relations(chunks):
    idm = {c['chunk_id']:i for i,c in enumerate(chunks)}
    for c in chunks:
        pid = c.get('parent_id')
        if pid and pid in idm:
            parent = chunks[idm[pid]]
            if c['chunk_id'] not in parent['child_ids']: parent['child_ids'].append(c['chunk_id'])
    return chunks

def main():
    # Allow CLI override of input/output paths
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT

    with open(input_path, 'r', encoding='utf-8') as f: lines = f.read().split('\n')
    print(f"Input: {input_path} ({len(lines)} lines)")

    sections = parse_document(lines)
    part_sections = [s for s in sections if s['type'] == 'part']
    print(f"Phase 1+2: {len(part_sections)} PART sections")

    chunks = build_chunks(sections)
    print(f"Phase 3-5: {len(chunks)} initial chunks")

    chunks = post_process(chunks)
    print(f"Phase 5 (post): {len(chunks)} chunks after post-process")

    chunks, fault_count = extract_fault_triplets(chunks)
    print(f"Phase 6: {fault_count} fault triplets")

    chunks = build_relations(chunks)

    by_sem = defaultdict(int); frags = 0; total_t = 0
    for c in chunks:
        by_sem[c.get('semantic_type','null')] += 1
        t = c.get('token_count',0); total_t += t
        if t < 50 and c.get('content_type') not in ('heading','table','steps'): frags += 1

    total = len(chunks)
    stats = {
        'total_chunks':total,'by_semantic_type':dict(by_sem),
        'fault_triplet_count':fault_count,
        'avg_token_count':round(total_t/max(1,total),1),
        'fragment_count':frags
    }

    with open(output_path, 'w', encoding='utf-8') as f: json.dump({'chunks':chunks,'stats':stats}, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {output_path}")
    for k,v in stats.items(): print(f"  {k}: {v}")

    print(f"\nValidation:")
    print(f"  [{'PASS' if 90<=total<=120 else 'FAIL'}] total_chunks in 90-120: {total}")
    print(f"  [{'PASS' if len(by_sem)>=5 else 'FAIL'}] semantic types >= 5: {len(by_sem)}")
    print(f"  [{'PASS' if fault_count>=15 else 'FAIL'}] fault_triplet_count >= 15: {fault_count}")
    print(f"  [{'PASS' if frags<5 else 'FAIL'}] fragment_count < 5: {frags}")
    id_set = {c['chunk_id'] for c in chunks}
    bad = sum(1 for c in chunks if c.get('parent_id') and c['parent_id'] not in id_set)
    print(f"  [{'PASS' if bad==0 else 'FAIL'}] all parent_ids valid: {bad} invalid")
    return chunks

if __name__ == '__main__': main()
