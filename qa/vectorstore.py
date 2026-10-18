"""Vector half of the search: embed every chunk once and store it in Chroma.

    python -m qa.vectorstore          # build (or rebuild) the index
    python -m qa.vectorstore "query"  # try a vector search

Embeddings come from the local Ollama server (nomic-embed-text). Chroma stores, per chunk:
  id        -> chunk_id, e.g. "RDAR-118204#s1" (what the LLM cites)
  embedding -> the vector, searched with an HNSW index (cosine distance)
  document  -> the chunk text (what the LLM reads)
  metadata  -> doc_id, title, type, section, and one flag per test ID (for filtering)
"""
import sys

import chromadb
from langchain_ollama import OllamaEmbeddings

from qa.chunking import load_chunks

DB_DIR = "data/chroma"
COLLECTION = "corpus_nomic"          # name the collection after the model that made it
EMBED_MODEL = "nomic-embed-text"

# nomic-embed-text was trained with task prefixes: documents and queries are embedded
# differently. Forgetting them silently lowers retrieval quality.
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


def _embedder():
    return OllamaEmbeddings(model=EMBED_MODEL)


def embed_documents(texts):
    return _embedder().embed_documents([DOC_PREFIX + t for t in texts])


def embed_query(text):
    return _embedder().embed_query(QUERY_PREFIX + text)


def _metadata(chunk):
    meta = {"doc_id": chunk.doc_id, "title": chunk.title, "type": chunk.doc_type,
            "section": chunk.section}
    # Chroma metadata values must be single values, not lists. So each test ID becomes
    # its own flag, which makes "only chunks about TS-6002" a filter: {"TS-6002": True}.
    for tid in chunk.test_ids:
        meta[tid] = True
    return meta


def _client():
    return chromadb.PersistentClient(path=DB_DIR)     # saved to disk, reloads instantly


def build_index(embed=embed_documents):
    """Embed all chunks and (re)write the collection. Run whenever the corpus changes."""
    chunks = load_chunks(strategy="structured")
    # Embed FIRST. If the model server is down, this fails before anything is deleted,
    # so the old index stays intact instead of being replaced by an empty one.
    vectors = embed([c.text for c in chunks])
    client = _client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)          # rebuild from scratch: no stale chunks
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[_metadata(c) for c in chunks],
    )
    print(f"Indexed {col.count()} chunks into {DB_DIR}/{COLLECTION}")
    return col


def vector_search(query, k=5, test_id=None, doc_type=None, embed=embed_query):
    """Return [(chunk_id, similarity, text)] for the k chunks closest in meaning."""
    col = _client().get_collection(COLLECTION)
    if col.count() == 0:
        raise RuntimeError("Vector index is empty. Build it first: python -m qa.vectorstore")
    filters = [f for f in ({test_id: True} if test_id else None,
                           {"type": doc_type} if doc_type else None) if f]
    where = None if not filters else filters[0] if len(filters) == 1 else {"$and": filters}
    res = col.query(query_embeddings=[embed(query)], n_results=min(k, col.count()),
                    where=where)
    # Chroma returns cosine *distance* (0 = identical); similarity = 1 - distance.
    return [(cid, round(1 - dist, 3), doc) for cid, dist, doc in
            zip(res["ids"][0], res["distances"][0], res["documents"][0])]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for cid, sim, _ in vector_search(" ".join(sys.argv[1:])):
            print(f"{sim:.3f}  {cid}")
    else:
        build_index()