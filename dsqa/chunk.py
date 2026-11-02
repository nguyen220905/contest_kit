"""Chunking: D1's _chunk_markdown packing, but page-aware."""
import re


def _title(pages):
    if not pages:
        return ""
    first = pages[0]["text"] or ""
    for line in first.splitlines():
        line = line.strip().lstrip("#").strip()
        if len(line) >= 20:
            return line[:200]
    return ""


def _units(text, size):
    """Paragraphs split on blank lines; a paragraph longer than `size` is split on single
    newlines (pymupdf text has few blank lines), and finally hard-cut."""
    out = []
    for p in re.split(r"\n\s*\n", text or ""):
        p = p.strip()
        if not p:
            continue
        if len(p) <= size:
            out.append(p)
            continue
        for line in p.split("\n"):
            line = line.strip()
            while len(line) > size:
                out.append(line[:size])
                line = line[size:]
            if line:
                out.append(line)
    return out


def chunk_pages(doc_id, pages, size=800, overlap=150, prefix_title=False) -> list:
    """-> [{doc, page, text, etext}]. `text` goes into the prompt; `etext` (= text, or
    '[title]\\n' + text when prefix_title) is what gets embedded / BM25-indexed.
    page = page of the paragraph that started the chunk."""
    title = _title(pages) if prefix_title else ""
    chunks = []
    cur, cur_page = "", None

    def emit():
        etext = f"[{title}]\n{cur}" if title else cur
        chunks.append({"doc": doc_id, "page": cur_page, "text": cur, "etext": etext})

    for pg in pages:
        for para in _units(pg["text"], size):
            if cur and len(cur) + len(para) + 2 > size:
                emit()
                cur = cur[-overlap:] + "\n\n" + para if overlap else para
                cur_page = pg["page"]
            else:
                if not cur:
                    cur_page = pg["page"]
                cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        emit()
    return chunks
