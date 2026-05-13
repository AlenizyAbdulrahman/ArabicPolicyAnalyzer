from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT_DIR / "documents"
INDEX_DIR = ROOT_DIR / "indexes" / "policy_index"
REGISTRY_PATH = ROOT_DIR / "documents_metadata.json"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL = "text-embedding-3-small"

GENERIC_SEPARATORS = [
    "\n\n",
    "\n",
    ".",
    "\u061f",
    "?",
    "!",
    "\u061b",
    ";",
    "\u060c",
    ",",
    " ",
    "",
]

NOISE_PATTERNS = [
    r"Document Classification:\s*Public",
    r"Public Classification:\s*Document",
    r"Version\s+\d+(\.\d+)?",
    r"^\s*\d+\s*$",
    r"SDAIA",
    r"Saudi Data\s*&\s*AI Authority",
    r"Saudi Data and AI Authority",
    r"National Cybersecurity Authority",
    r"\u0627\u0644\u0647\u064a\u0626\u0629 \u0627\u0644\u0633\u0639\u0648\u062f\u064a\u0629 \u0644\u0644\u0628\u064a\u0627\u0646\u0627\u062a \u0648\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a",
    r"\u0627\u0644\u0647\u064a\u0626\u0629 \u0627\u0644\u0648\u0637\u0646\u064a\u0629 \u0644\u0644\u0623\u0645\u0646 \u0627\u0644\u0633\u064a\u0628\u0631\u0627\u0646\u064a",
]

DOMAIN_RULES = [
    (
        "data_sharing",
        "Data Sharing",
        [
            "datasharing",
            "data sharing",
            "\u0645\u0634\u0627\u0631\u0643\u0629 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a",
            "sharing policy",
        ],
    ),
    (
        "critical_systems_cybersecurity",
        "Critical Systems Cybersecurity",
        [
            "criticalsystems",
            "critical systems",
            "\u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u0633\u0627\u0633\u0629",
            "critical",
        ],
    ),
    (
        "cybersecurity",
        "Cybersecurity Controls",
        [
            "cybersecurity",
            "nca",
            "\u0627\u0644\u0623\u0645\u0646 \u0627\u0644\u0633\u064a\u0628\u0631\u0627\u0646\u064a",
            "cyber security",
        ],
    ),
    (
        "personal_data_protection",
        "Personal Data Protection",
        [
            "personaldata",
            "personal data",
            "pdpl",
            "privacy",
            "protection law",
            "\u062d\u0645\u0627\u064a\u0629 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a \u0627\u0644\u0634\u062e\u0635\u064a\u0629",
        ],
    ),
]


def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment or Streamlit secrets before ingesting."
        )


def document_id_for(file_name: str) -> str:
    digest = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^a-z0-9]+", "-", Path(file_name).stem.lower()).strip("-")
    return f"{stem or 'document'}-{digest}"


def clean_page_content(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def infer_language(text: str, file_name: str) -> str:
    sample = f"{file_name}\n{text[:4000]}"
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", sample))
    latin_chars = len(re.findall(r"[A-Za-z]", sample))
    return "ar" if arabic_chars > latin_chars else "en"


def infer_domain(file_name: str, text: str) -> tuple[str, str]:
    haystack = f"{file_name} {text[:3000]}".lower()
    for domain, label, needles in DOMAIN_RULES:
        if any(needle.lower() in haystack for needle in needles):
            return domain, label
    return "general", "General"


def infer_authority(file_name: str, text: str) -> str:
    haystack = f"{file_name} {text[:3000]}".lower()
    if (
        "nca" in haystack
        or "cybersecurity" in haystack
        or "\u0627\u0644\u0623\u0645\u0646 \u0627\u0644\u0633\u064a\u0628\u0631\u0627\u0646\u064a" in haystack
    ):
        return "NCA"
    if "sdaia" in haystack or "data" in haystack or "\u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a" in haystack:
        return "SDAIA"
    return "Unknown"


def title_from_file(file_name: str) -> str:
    title = Path(file_name).stem
    title = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)
    title = title.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", title).strip()


def load_registry_overrides(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {item["file_name"]: item for item in existing.get("documents", []) if "file_name" in item}


def splitter_for() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=GENERIC_SEPARATORS,
        chunk_size=1800,
        chunk_overlap=250,
        keep_separator=True,
    )


def process_pdf(pdf_path: Path, overrides: dict[str, Any] | None = None) -> tuple[list[Document], dict[str, Any]]:
    pdf_path = pdf_path.resolve()
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    cleaned_pages = [clean_page_content(page.page_content) for page in pages]
    sample_text = "\n".join(cleaned_pages[:3])
    file_name = pdf_path.name
    overrides = overrides or {}

    language = overrides.get("language") or infer_language(sample_text, file_name)
    inferred_domain, inferred_label = infer_domain(file_name, sample_text)
    document_id = overrides.get("document_id") or document_id_for(file_name)
    domain = overrides.get("domain") or inferred_domain
    domain_label = overrides.get("domain_label") or inferred_label

    metadata = {
        "document_id": document_id,
        "file_name": file_name,
        "title": overrides.get("title") or title_from_file(file_name),
        "domain": domain,
        "domain_label": domain_label,
        "language": language,
        "authority": overrides.get("authority") or infer_authority(file_name, sample_text),
        "document_type": overrides.get("document_type") or "policy_or_regulation",
        "source_path": str(pdf_path.relative_to(ROOT_DIR)),
        "page_count": len(pages),
    }

    page_documents: list[Document] = []
    for page_number, text in enumerate(cleaned_pages, start=1):
        if not text:
            continue
        page_documents.append(Document(page_content=text, metadata={**metadata, "page": page_number}))

    chunks = splitter_for().split_documents(page_documents)
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"{document_id}-{idx:04d}"
        chunk.metadata["chunk_index"] = idx

    metadata["chunk_count"] = len(chunks)
    return chunks, metadata


def build_index(
    documents_dir: Path = DOCUMENTS_DIR,
    index_dir: Path = INDEX_DIR,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    require_openai_key()
    documents_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(documents_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {documents_dir}.")

    overrides = load_registry_overrides(registry_path)
    all_chunks: list[Document] = []
    registry: list[dict[str, Any]] = []

    for pdf_path in pdfs:
        chunks, metadata = process_pdf(pdf_path, overrides.get(pdf_path.name))
        all_chunks.extend(chunks)
        registry.append(metadata)

    if not all_chunks:
        raise RuntimeError("No text chunks were extracted from the PDFs.")

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    vector_store.save_local(str(index_dir))

    chunks_payload = [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in all_chunks]
    chunks_path = index_dir / "chunks.json"
    chunks_path.write_text(
        json.dumps({"chunks": chunks_payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps({"documents": registry}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "document_count": len(registry),
        "chunk_count": len(all_chunks),
        "index_dir": str(index_dir),
        "registry_path": str(registry_path),
        "chunks_path": str(chunks_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the multi-document RAG index.")
    parser.add_argument("--documents-dir", default=str(DOCUMENTS_DIR))
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    result = build_index(Path(args.documents_dir), Path(args.index_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
