#!/usr/bin/env python3
"""
Step 2+3: BGE-M3 encoding + Qdrant upsert
Reads chunks_v6_llm.json, encodes with BGE-M3, writes to Qdrant
"""

import json, time, os, sys, hashlib
from pathlib import Path

# Force offline mode for HuggingFace
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "data"
CHUNKS_PATH = str(DATA_DIR / "chunks_v6_llm.json")
QDRANT_PATH = str(DATA_DIR / "qdrant_db")

def main():
    # Allow CLI override
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else CHUNKS_PATH
    qdrant_path = sys.argv[2] if len(sys.argv) > 2 else QDRANT_PATH

    # Load LLM-processed chunks
    print(f"Loading {chunks_path}...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data['chunks']
    stats = data.get('stats', {})
    total = len(chunks)
    print(f"Loaded {total} chunks")

    # ============================================================
    # Step 2: BGE-M3 Encoding
    # ============================================================
    print("\n[Step 2] Loading BGE-M3 model...")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, local_files_only=True)
    print("Model loaded.")

    texts_to_encode = []
    for c in chunks:
        emb_text = c["text"]
        if c.get("important_kwd"):
            emb_text += "\n" + c["important_kwd"]
        if c.get("question_kwd"):
            emb_text += "\n" + c["question_kwd"]
        texts_to_encode.append(emb_text)

    print(f"Encoding {len(texts_to_encode)} texts...")
    t0 = time.time()
    output = model.encode(
        texts_to_encode,
        return_dense=True,
        return_sparse=True,
        max_length=8192,
        batch_size=12
    )
    elapsed = time.time() - t0
    print(f"Encoding complete in {elapsed:.1f}s ({len(texts_to_encode)/elapsed:.1f} texts/s)")

    # Attach vectors to chunks
    for i, c in enumerate(chunks):
        w = output["lexical_weights"][i]
        c["_dense"] = output["dense_vecs"][i].tolist()
        c["_sparse"] = {"indices": list(w.keys()), "values": list(w.values())}

    # Save intermediate result
    interm_path = chunks_path.replace('.json', '_encoded.json')
    with open(interm_path, 'w', encoding='utf-8') as f:
        json.dump({'chunks': chunks, 'stats': stats}, f, ensure_ascii=False, indent=2)
    print(f"Intermediate saved: {interm_path}")

    # ============================================================
    # Step 3: Qdrant Upsert
    # ============================================================
    print(f"\n[Step 3] Writing to Qdrant at {qdrant_path}...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, SparseVectorParams,
        PointStruct, PayloadSchemaType
    )

    client = QdrantClient(path=qdrant_path)

    # Delete and recreate collection
    try:
        client.delete_collection("ebike_manual")
        print("Deleted existing collection")
    except:
        pass

    client.create_collection(
        collection_name="ebike_manual",
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print("Created new collection")

    # Note: Payload indexes not supported in local mode - skip
    print("Skipping payload indexes (not supported in local mode)")

    # Prepare points
    points = []
    for c in chunks:
        meta = c.get('metadata', {})
        payload = {
            "chunk_id": c["chunk_id"],
            "semantic_type": c["semantic_type"],
            "content_type": c["content_type"],
            "text": c["text"],
            "token_count": c["token_count"],
            "parent_id": c.get("parent_id"),
            "child_ids": c.get("child_ids", []),
            "mom_id": c.get("mom_id"),
            "component": c.get("component", []),
            "fault_symptom": c.get("fault_symptom"),
            "repair_action": c.get("repair_action"),
            "repair_level": c.get("repair_level"),
            "risk_level": c.get("risk_level"),
            "fault_triplet": c.get("fault_triplet"),
            "important_kwd": c.get("important_kwd", ""),
            "question_kwd": c.get("question_kwd", ""),
            "domain_tags": c.get("domain_tags", []),
            "part": meta.get("part", ""),
            "section": meta.get("section", ""),
            "page": meta.get("page", 0),
            "is_vlm_enhanced": meta.get("is_vlm_enhanced", False),
            "warning_level": meta.get("warning_level"),
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        # Convert chunk_id to valid UUID using MD5 hash
        point_uuid = hashlib.md5(c["chunk_id"].encode()).hexdigest()
        points.append(PointStruct(
            id=point_uuid,
            vector={
                "dense": c["_dense"],
                "sparse": {
                    "indices": c["_sparse"]["indices"],
                    "values": c["_sparse"]["values"]
                }
            },
            payload=payload
        ))

    # Upsert in batches
    BATCH = 50
    for i in range(0, len(points), BATCH):
        batch = points[i:i+BATCH]
        client.upsert(collection_name="ebike_manual", points=batch)
        print(f"  Upserted {i+len(batch)}/{len(points)}")

    # Verify
    count = client.count("ebike_manual")
    print(f"\nQdrant count: {count.count} (expected {total})")

    # Clean up vector data
    for c in chunks:
        del c["_dense"]
        del c["_sparse"]

    # Save final output
    final_path = chunks_path.replace('.json', '_final.json')
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump({'chunks': chunks, 'stats': stats}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"Final output (without vectors): {final_path}")
    print(f"Qdrant DB: {qdrant_path}")
    print(f"Collection: ebike_manual, Points: {count.count}")

    return chunks, stats


if __name__ == '__main__':
    main()
