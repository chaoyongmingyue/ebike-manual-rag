"""
Minimal sparse vector check — run while backend is STOPPED
"""
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

QDRANT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'qdrant_db'))
COLLECTION = 'ebike_manual'

from qdrant_client import QdrantClient

client = QdrantClient(path=QDRANT_PATH)

# Check collection config
info = client.get_collection(COLLECTION)
print(f'Points: {info.points_count}')
print(f'Vectors: {info.config.params.vectors}')
print(f'Sparse vectors config: {info.config.params.sparse_vectors}')
print()

# Scroll a few points to check sparse vectors exist
pts, next_offset = client.scroll(
    collection_name=COLLECTION,
    limit=5,
    with_vectors=True,
    with_payload=['chunk_id'],
)
print('=== First 5 points ===')
for pt in pts:
    has_dense = pt.vector and 'dense' in (pt.vector or {})
    has_sparse = pt.vector and 'sparse' in (pt.vector or {})

    # Check if sparse vector has data
    sv = None
    if has_sparse:
        sv = pt.vector['sparse']

    cid = pt.payload.get('chunk_id', '?') if pt.payload else '?'
    print(f'  {pt.id} -> {cid}')
    print(f'    dense: {has_dense}')
    print(f'    sparse: {has_sparse}', end='')
    if sv is not None:
        if isinstance(sv, dict):
            indices = sv.get('indices', [])
            values = sv.get('values', [])
            print(f'  indices={len(indices)}  values={len(values)}', end='')
            if indices:
                print(f'  sample={list(zip(indices[:3], values[:3]))}', end='')
        elif hasattr(sv, 'indices'):
            # SparseVector object
            print(f'  indices={len(sv.indices)}  values={len(sv.values)}', end='')
            if sv.indices:
                print(f'  sample={list(zip(sv.indices[:3], sv.values[:3]))}', end='')
        else:
            print(f'  type={type(sv).__name__}', end='')
    else:
        print(f'  EMPTY!', end='')
    print()
    print()

# Try a direct sparse search with explicit sparse vector
print('=== Direct sparse search test ===')
from FlagEmbedding import BGEM3FlagModel
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'bge-m3'))
model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)

query = '调速失灵怎么修'
out = model.encode([query], return_dense=True, return_sparse=True, max_length=512)
lex = out['lexical_weights'][0]
print(f'Query sparse tokens: {len(lex)}')
print(f'Sample: {list(lex.items())[:5]}')

# Try different sparse search formats
from qdrant_client.models import SparseVector

# Format 1: SparseVector object
sv = SparseVector(indices=list(lex.keys()), values=list(lex.values()))
print(f'\nSparseVector: indices={len(sv.indices)} values={len(sv.values)}')

try:
    r1 = client.query_points(
        collection_name=COLLECTION,
        query=sv,
        using='sparse',
        limit=3,
    )
    print(f'Format 1 (SparseVector): {len(r1.points)} results')
except Exception as e:
    print(f'Format 1 error: {e}')

# Format 2: dict
sv_dict = {'indices': list(lex.keys()), 'values': list(lex.values())}
try:
    r2 = client.query_points(
        collection_name=COLLECTION,
        query=sv_dict,
        using='sparse',
        limit=3,
    )
    print(f'Format 2 (dict): {len(r2.points)} results')
except Exception as e:
    print(f'Format 2 error: {e}')

# Format 3: NamedVector style
try:
    r3 = client.query_points(
        collection_name=COLLECTION,
        query=(COLLECTION, 'sparse', sv),
        limit=3,
    )
    print(f'Format 3 (tuple): {len(r3.points)} results')
except Exception as e:
    print(f'Format 3 error: {e}')

client.close()
print('\nDone.')
