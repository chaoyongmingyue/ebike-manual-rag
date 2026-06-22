"""
向量检索诊断脚本
停掉 search_server.py 后运行：
  python tests/diag_vectors.py
"""
import os, sys, json, time
import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

QDRANT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'qdrant_db'))
COLLECTION = 'ebike_manual'

# ---- Load model ----
from FlagEmbedding import BGEM3FlagModel
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'bge-m3'))
print(f'Loading model from {MODEL_PATH}...')
model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)
print('Model loaded.\n')

# ---- Load Qdrant ----
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, FusionQuery, Prefetch,
)
print(f'Opening Qdrant at {QDRANT_PATH}...')
client = QdrantClient(path=QDRANT_PATH)
info = client.get_collection(COLLECTION)
print(f'Points: {info.points_count}\n')

# ---- Test queries ----
queries = [
    ('调速失灵怎么修',       'tbl103'),
    ('车充不进电',           'tbl108'),
    ('大灯开关在哪里',       'txt037'),
    ('16岁以下能骑电动车吗', 'txt004'),
    ('续航',                 'tbl126'),
    ('怎么清洗电动车',       'stp085'),
]

for query, expected in queries:
    print(f'=== {query}  (expected: {expected}) ===')

    # Encode
    out = model.encode([query], return_dense=True, return_sparse=True, max_length=512)
    dense_vec = out['dense_vecs'][0].tolist()
    lex = out['lexical_weights'][0]
    sparse_vec = SparseVector(indices=list(lex.keys()), values=list(lex.values()))

    # 1. Dense only
    dr = client.query_points(
        collection_name=COLLECTION,
        query=dense_vec,
        using='dense',
        limit=5,
        with_payload=['chunk_id', 'text', 'semantic_type'],
    )
    print('  [Dense only]')
    for i, pt in enumerate(dr.points):
        cid = pt.payload.get('chunk_id', '?')
        st = pt.payload.get('semantic_type', '?')
        txt = (pt.payload.get('text', '') or '')[:50].replace('\n', ' ')
        hit = ' <-- HIT' if cid == expected else ''
        print(f'    {i+1}. {cid:<10} score={pt.score:.4f}  [{st}] {txt}{hit}')

    # 2. Sparse only
    sr = client.query_points(
        collection_name=COLLECTION,
        query=sparse_vec,
        using='sparse',
        limit=5,
        with_payload=['chunk_id', 'text', 'semantic_type'],
    )
    print('  [Sparse only]')
    for i, pt in enumerate(sr.points):
        cid = pt.payload.get('chunk_id', '?')
        st = pt.payload.get('semantic_type', '?')
        txt = (pt.payload.get('text', '') or '')[:50].replace('\n', ' ')
        hit = ' <-- HIT' if cid == expected else ''
        print(f'    {i+1}. {cid:<10} score={pt.score:.4f}  [{st}] {txt}{hit}')

    # 3. RRF fusion (same as production)
    rr = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using='dense', limit=40),
            Prefetch(query=sparse_vec, using='sparse', limit=40),
        ],
        query=FusionQuery(fusion='rrf'),
        limit=5,
        with_payload=['chunk_id', 'text', 'semantic_type'],
    )
    print('  [RRF fusion]')
    for i, pt in enumerate(rr.points):
        cid = pt.payload.get('chunk_id', '?')
        st = pt.payload.get('semantic_type', '?')
        txt = (pt.payload.get('text', '') or '')[:50].replace('\n', ' ')
        hit = ' <-- HIT' if cid == expected else ''
        print(f'    {i+1}. {cid:<10} score={pt.score:.4f}  [{st}] {txt}{hit}')

    print()

# ---- Check stored vectors ----
print('=== Stored vector sanity check ===')
pts, _ = client.scroll(collection_name=COLLECTION, limit=5, with_vectors=True)
for pt in pts:
    dv = pt.vector.get('dense')
    if dv is not None:
        arr = np.array(dv)
        print(f'  {pt.id}: norm={np.linalg.norm(arr):.4f}  zeros={(arr==0).sum()}/{len(arr)}  '
              f'range=[{arr.min():.4f}, {arr.max():.4f}]')
    else:
        print(f'  {pt.id}: NO dense vector!')
print()

# Check cosine sim between query and first 5 stored vectors
print('=== Cosine similarity: query "调速失灵" vs stored ===')
q_vec = np.array(dense_vec)
for pt in pts:
    dv = pt.vector.get('dense')
    if dv is not None:
        s_vec = np.array(dv)
        cos_sim = np.dot(q_vec, s_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(s_vec))
        cid = pt.payload.get('chunk_id', pt.id)
        print(f'  {cid}: cosine_sim={cos_sim:.4f}')

client.close()
print('\nDone.')
