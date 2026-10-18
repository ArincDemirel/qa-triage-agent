"""Hybrid search: BM25 (exact words) + vectors (meaning), merged with Reciprocal Rank Fusion.

    python -m qa.search "TS-6002 upper limit"
    python -m qa.search "IIR coefficient change, offset fails in the cold" --mode vector

This is the function the agent's search tool calls.
"""
import re
import sys

from rank_bm25 import BM25Okapi

from qa.chunking import load_chunks
from qa.vectorstore import vector_search

RRF_K = 60          # standard constant: damps how much the very top ranks dominate
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text):
    """Lowercase words, keeping identifiers whole AND adding their parts:
    'TS-4471' -> ['ts-4471', 'ts', '4471'], so both 'TS-4471' and '4471' match."""
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        parts = re.split(r"[-_.]", tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


# Built once at import: 26 chunks, a few milliseconds.
CHUNKS = load_chunks(strategy="structured")
BY_ID = {c.chunk_id: c for c in CHUNKS}
BM25 = BM25Okapi([tokenize(c.text) for c in CHUNKS])


def _matches(chunk, test_id, doc_type):
    return (test_id is None or test_id in chunk.test_ids) and \
           (doc_type is None or chunk.doc_type == doc_type)


def bm25_search(query, k=10, test_id=None, doc_type=None):
    scores = BM25.get_scores(tokenize(query))
    ranked = sorted(((s, c) for s, c in zip(scores, CHUNKS)
                     if s > 0 and _matches(c, test_id, doc_type)),   # drop chunks with no word overlap
                    key=lambda x: -x[0])
    return [c.chunk_id for _, c in ranked[:k]]


def hybrid_search(query, k=5, test_id=None, doc_type=None, mode="hybrid", depth=20):
    """Return the top-k chunks as dicts: chunk_id, title, type, text (what the LLM reads)."""
    bm25_ids = bm25_search(query, depth, test_id, doc_type) if mode in ("hybrid", "bm25") else []
    vec_ids = ([cid for cid, _, _ in vector_search(query, depth, test_id, doc_type)]
               if mode in ("hybrid", "vector") else [])

    # Reciprocal Rank Fusion: combine by RANK, not by score. BM25 scores (e.g. 12.7) and
    # cosine similarities (e.g. 0.83) are on different scales and can't be added directly.
    fused = {}
    for ranking in (bm25_ids, vec_ids):
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    top = sorted(fused, key=lambda cid: -fused[cid])[:k]

    return [{"chunk_id": cid, "title": BY_ID[cid].title, "type": BY_ID[cid].doc_type,
             "score": round(fused[cid], 4), "text": BY_ID[cid].text} for cid in top]


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "hybrid"
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i + 1]
        args = args[:i] + args[i + 2:]
    for hit in hybrid_search(" ".join(args), mode=mode):
        print(f"{hit['score']:.4f}  {hit['chunk_id']}")