"""PDF parsing -> list[{page, text}]. Shared lib (copied into submissions later)."""
import re
from pathlib import Path


def find_pdfs(corpus_dir):
    return sorted(p for p in Path(corpus_dir).rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def _fix_hyphens(text: str) -> str:
    # "inter-\nnational" -> "international" (only lowercase continuation, keeps "x-\nRay"-like cases rare)
    return re.sub(r"(\w)-\n([a-z])", r"\1\2", text)


def parse_pymupdf(pdf_path) -> list:
    import fitz  # pymupdf
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, 1):
            pages.append({"page": i, "text": _fix_hyphens(page.get_text("text"))})
    return pages


_docling_conv = {}


def parse_docling(pdf_path, artifacts_path=None) -> list:
    """Same options as D1 (no OCR, no table structure). Whole doc as one element, page=None
    (export_to_markdown() carries no page numbers)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    key = str(artifacts_path)
    if key not in _docling_conv:
        kw = dict(do_ocr=False, do_table_structure=False)
        if artifacts_path:
            kw["artifacts_path"] = str(artifacts_path)
        opts = PdfPipelineOptions(**kw)
        _docling_conv[key] = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    result = _docling_conv[key].convert(str(pdf_path))
    return [{"page": None, "text": result.document.export_to_markdown()}]
