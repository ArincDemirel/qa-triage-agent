"""Turn the markdown corpus into searchable chunks.

Two strategies, so the choice can be measured instead of assumed:
  structured (default): one chunk per section, one chunk per table row (with headers),
                        and every chunk prefixed with "doc_id | title > section".
  fixed (baseline):     every N words, ignoring structure.

Every chunk carries the document's metadata (doc_id, type, test_ids), which later
lets search filter by test and links documents to the numbers path.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str          # e.g. "RDAR-118204#s1"  (doc + position, for citations)
    doc_id: str
    title: str
    doc_type: str          # wiki | radar
    section: str
    text: str              # what gets searched and shown to the LLM
    test_ids: list = field(default_factory=list)


def parse_doc(raw: str):
    """Split the '---' front matter from the markdown body."""
    _, front, body = raw.split("---\n", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    meta["test_ids"] = [t.strip() for t in meta.get("test_ids", "").split(",") if t.strip()]
    return meta, body


def split_sections(body: str):
    """Return (heading, text) for each '## ' section. The '# ' title line is dropped."""
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def chunk_structured(meta, body):
    chunks = []
    for si, (heading, text) in enumerate(split_sections(body)):
        # Context prefix: makes every chunk understandable on its own, and keeps
        # identifiers that only appear in the title (like the Radar number).
        context = f"{meta['doc_id']} | {meta['title']} > {heading}"
        lines = text.splitlines()
        table = [ln for ln in lines if ln.strip().startswith("|")]
        prose = "\n".join(ln for ln in lines if not ln.strip().startswith("|")).strip()

        def add(suffix, chunk_text):
            ids = sorted(set(re.findall(r"TS-\d{4}", chunk_text))) or meta["test_ids"]
            chunks.append(Chunk(f"{meta['doc_id']}#{suffix}", meta["doc_id"], meta["title"],
                                meta["type"], heading, chunk_text, ids))

        if prose:
            add(f"s{si}", f"{context}\n{prose}")
        if len(table) >= 3:                        # header, separator, then rows
            header, rows = table[0], table[2:]
            for ri, row in enumerate(rows):
                add(f"s{si}r{ri}", f"{context}\n{header}\n{row}")   # row keeps its headers
    return chunks


def chunk_fixed(meta, body, size=60, overlap=15):
    words = body.split()
    chunks = []
    for i, start in enumerate(range(0, max(len(words) - overlap, 1), size - overlap)):
        chunks.append(Chunk(f"{meta['doc_id']}#w{i}", meta["doc_id"], meta["title"],
                            meta["type"], "", " ".join(words[start:start + size]),
                            meta["test_ids"]))
    return chunks


def load_chunks(corpus_dir="corpus", strategy="structured"):
    chunker = {"structured": chunk_structured, "fixed": chunk_fixed}[strategy]
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        meta, body = parse_doc(path.read_text())
        chunks.extend(chunker(meta, body))
    return chunks


if __name__ == "__main__":
    for strategy in ("fixed", "structured"):
        chunks = load_chunks(strategy=strategy)
        print(f"{strategy}: {len(chunks)} chunks")
    for c in load_chunks():
        if c.doc_id in ("WIKI-TESTPLAN", "RDAR-118204"):
            print(f"\n--- {c.chunk_id}  tests={c.test_ids}\n{c.text}")