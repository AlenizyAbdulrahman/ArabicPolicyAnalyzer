from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ingest import CHUNKS_PATH, EMBEDDING_MODEL, INDEX_DIR, REGISTRY_PATH


ROUTER_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o-mini"
RERANKER_MODEL = "BAAI/bge-reranker-base"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u0600-\u06FF]+", (text or "").lower())


def normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    replacements = {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0649": "\u064a",
        "\u0629": "\u0647",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    return text


def query_language(text: str) -> str:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    latin_chars = len(re.findall(r"[A-Za-z]", text or ""))
    return "ar" if arabic_chars > latin_chars else "en"


def load_registry(path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("documents", [])


def load_chunk_documents(path: Path = CHUNKS_PATH) -> list[Document]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in payload.get("chunks", [])
    ]


def matches_filters(
    metadata: dict[str, Any],
    domains: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> bool:
    if domains and metadata.get("domain") not in domains:
        return False
    if document_ids and metadata.get("document_id") not in document_ids:
        return False
    return True


class SimpleBM25:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.doc_tokens = [tokenize(doc.page_content) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.df: Counter[str] = Counter()

        for tokens in self.doc_tokens:
            self.df.update(set(tokens))

    def search(
        self,
        query: str,
        domains: list[str] | None = None,
        document_ids: list[str] | None = None,
        k: int = 30,
    ) -> list[tuple[Document, float]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        n_docs = max(len(self.documents), 1)
        k1 = 1.5
        b = 0.75
        scores: list[tuple[Document, float]] = []

        for doc, tokens, doc_len in zip(self.documents, self.doc_tokens, self.doc_lengths):
            if not matches_filters(doc.metadata, domains, document_ids):
                continue

            term_counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in term_counts:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + ((n_docs - df + 0.5) / (df + 0.5)))
                tf = term_counts[term]
                denom = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1))
                score += idf * ((tf * (k1 + 1)) / denom)

            if score > 0:
                scores.append((doc, score))

        return sorted(scores, key=lambda item: item[1], reverse=True)[:k]


@dataclass
class Route:
    task: str
    domains: list[str]
    document_ids: list[str]
    reason: str


class MultiDocumentRAGAgent:
    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        registry_path: Path = REGISTRY_PATH,
        chunks_path: Path = CHUNKS_PATH,
    ) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.registry = load_registry(registry_path)
        self.chunks = load_chunk_documents(chunks_path)
        if not self.registry or not self.chunks:
            raise RuntimeError("The RAG index is missing. Run ingestion first.")

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        self.bm25 = SimpleBM25(self.chunks)
        self.reranker: HuggingFaceCrossEncoder | None = None
        self.answer_llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0.1)
        self.router_llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0)

    def _domain_keywords_route(self, query: str) -> list[str]:
        q = normalize_for_match(query)
        route_rules = {
            "data_sharing": [
                "data sharing",
                "sharing",
                "share",
                "\u0645\u0634\u0627\u0631\u0643\u0629",
                "\u0645\u0634\u0627\u0631\u0643\u0647",
                "\u0633\u064a\u0627\u0633\u0629 \u0645\u0634\u0627\u0631\u0643\u0629 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a",
                "\u0633\u064a\u0627\u0633\u0647 \u0645\u0634\u0627\u0631\u0643\u0647 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a",
                "\u0637\u0644\u0628 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a",
                "\u0627\u0644\u062c\u0647\u0629 \u0627\u0644\u0645\u0635\u062f\u0631",
                "\u0627\u0644\u062c\u0647\u0647 \u0627\u0644\u0645\u0635\u062f\u0631",
            ],
            "critical_systems_cybersecurity": [
                "critical",
                "critical systems",
                "\u0627\u0644\u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u062d\u0633\u0627\u0633\u0629",
            ],
            "cybersecurity": [
                "cyber",
                "security controls",
                "nca",
                "\u0633\u064a\u0628\u0631\u0627\u0646\u064a",
                "\u0627\u0644\u0623\u0645\u0646",
            ],
            "personal_data_protection": [
                "personal data",
                "privacy",
                "pdpl",
                "\u062d\u0645\u0627\u064a\u0629 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a",
                "\u0627\u0644\u0634\u062e\u0635\u064a\u0629",
            ],
        }
        return [
            domain
            for domain, needles in route_rules.items()
            if any(normalize_for_match(needle) in q for needle in needles)
        ]

    def route_query(self, query: str) -> Route:
        deterministic_domains = self._domain_keywords_route(query)
        task = self._task_from_query(query)

        try:
            docs_for_prompt = [
                {
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "domain": doc["domain"],
                    "domain_label": doc["domain_label"],
                    "language": doc["language"],
                }
                for doc in self.registry
            ]
            prompt = f"""
You are a routing agent for a multi-document Arabic/English policy RAG system.
Choose the best task and filters for the user's question.

Allowed tasks: answer, compare, summarize.
Available documents:
{json.dumps(docs_for_prompt, ensure_ascii=False)}

Return only JSON with this shape:
{{"task": "answer|compare|summarize", "domains": [], "document_ids": [], "reason": "..."}}

User question: {query}
"""
            raw = self.router_llm.invoke(prompt).content.strip()
            payload = json.loads(raw.replace("```json", "").replace("```", "").strip())
            valid_domains = {doc["domain"] for doc in self.registry}
            valid_doc_ids = {doc["document_id"] for doc in self.registry}
            routed_task = payload.get("task", task)

            return Route(
                task=routed_task if routed_task in {"answer", "compare", "summarize"} else task,
                domains=[domain for domain in payload.get("domains", []) if domain in valid_domains]
                or deterministic_domains,
                document_ids=[
                    document_id for document_id in payload.get("document_ids", []) if document_id in valid_doc_ids
                ],
                reason=payload.get("reason", "LLM router selected retrieval filters."),
            )
        except Exception:
            return Route(task, deterministic_domains, [], "Heuristic router selected retrieval filters.")

    def _task_from_query(self, query: str) -> str:
        q = query.lower()
        if any(word in q for word in ["compare", "difference", "\u0642\u0627\u0631\u0646", "\u0627\u0644\u0641\u0631\u0642", "\u0645\u0642\u0627\u0631\u0646\u0629"]):
            return "compare"
        if any(word in q for word in ["summarize", "summary", "\u0644\u062e\u0635", "\u0645\u0644\u062e\u0635"]):
            return "summarize"
        return "answer"

    def _dense_search(
        self,
        query: str,
        domains: list[str] | None,
        document_ids: list[str] | None,
        k: int,
    ) -> list[tuple[Document, float]]:
        fetch_k = len(self.chunks) if domains or document_ids else max(k * 4, 40)
        raw_results = self.vector_store.similarity_search_with_score(query, k=fetch_k)
        filtered = []
        for doc, distance in raw_results:
            if matches_filters(doc.metadata, domains, document_ids):
                filtered.append((doc, 1.0 / (1.0 + float(distance))))
        return filtered[:k]

    def _get_reranker(self) -> HuggingFaceCrossEncoder:
        if self.reranker is None:
            self.reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        return self.reranker

    def search_documents(
        self,
        query: str,
        domains: list[str] | None = None,
        document_ids: list[str] | None = None,
        k: int = 8,
    ) -> list[Document]:
        dense_results = self._dense_search(query, domains, document_ids, k=30)
        keyword_results = self.bm25.search(query, domains, document_ids, k=30)
        candidates: dict[str, dict[str, Any]] = {}

        for doc, score in dense_results:
            chunk_id = doc.metadata["chunk_id"]
            candidates.setdefault(chunk_id, {"doc": doc, "dense": 0.0, "keyword": 0.0})
            candidates[chunk_id]["dense"] = max(candidates[chunk_id]["dense"], score)

        for doc, score in keyword_results:
            chunk_id = doc.metadata["chunk_id"]
            candidates.setdefault(chunk_id, {"doc": doc, "dense": 0.0, "keyword": 0.0})
            candidates[chunk_id]["keyword"] = max(candidates[chunk_id]["keyword"], score)

        if not candidates:
            return []

        max_dense = max(item["dense"] for item in candidates.values()) or 1.0
        max_keyword = max(item["keyword"] for item in candidates.values()) or 1.0
        ranked = sorted(
            candidates.values(),
            key=lambda item: 0.55 * (item["dense"] / max_dense) + 0.45 * (item["keyword"] / max_keyword),
            reverse=True,
        )
        docs = [item["doc"] for item in ranked[:20]]

        try:
            scores = self._get_reranker().score([(query, doc.page_content) for doc in docs])
            reranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
            return [doc for doc, _ in reranked[:k]]
        except Exception:
            return docs[:k]

    def retrieve_sources(self, query: str, route: Route, k: int) -> list[Document]:
        routed_sources = self.search_documents(query, route.domains, route.document_ids, k=k)

        if not route.domains and not route.document_ids:
            return routed_sources
        if routed_sources:
            return routed_sources

        global_sources = self.search_documents(query, None, None, k=k)
        merged: list[Document] = []
        seen_chunk_ids: set[str] = set()

        for doc in routed_sources + global_sources:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            merged.append(doc)
            if len(merged) >= k:
                break

        return merged

    def get_document_summary_sources(self, document_ids: list[str], query: str = "") -> list[Document]:
        selected = [doc for doc in self.chunks if doc.metadata.get("document_id") in document_ids]
        if not selected:
            return []

        if query:
            ranked = self.bm25.search(query, document_ids=document_ids, k=8)
            if ranked:
                return [doc for doc, _ in ranked[:8]]

        by_doc: dict[str, list[Document]] = defaultdict(list)
        for doc in selected:
            by_doc[doc.metadata["document_id"]].append(doc)

        samples: list[Document] = []
        for docs in by_doc.values():
            samples.extend(docs[:3])
            if len(docs) > 6:
                middle = len(docs) // 2
                samples.extend(docs[middle : middle + 2])
            samples.extend(docs[-2:])

        return samples[:10]

    def _context_from_sources(self, sources: list[Document]) -> str:
        blocks = []
        for idx, doc in enumerate(sources, start=1):
            metadata = doc.metadata
            text = doc.page_content.strip()
            if len(text) > 1700:
                text = text[:1700].rsplit(" ", 1)[0] + "..."
            blocks.append(
                "\n".join(
                    [
                        f"[S{idx}]",
                        f"Title: {metadata.get('title')}",
                        f"Domain: {metadata.get('domain_label')}",
                        f"File: {metadata.get('file_name')}",
                        f"Page: {metadata.get('page')}",
                        f"Chunk: {metadata.get('chunk_id')}",
                        f"Content: {text}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:
        route = self.route_query(question)

        if route.task == "summarize" and route.document_ids:
            sources = self.get_document_summary_sources(route.document_ids, question)
        elif route.task == "compare":
            sources = self.retrieve_sources(question, route, k=10)
        else:
            sources = self.retrieve_sources(question, route, k=8)

        if not sources:
            answer = (
                "\u0644\u0645 \u0623\u062c\u062f \u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0643\u0627\u0641\u064a\u0629 \u0641\u064a "
                "\u0627\u0644\u0645\u0633\u062a\u0646\u062f\u0627\u062a \u0627\u0644\u0645\u062a\u0627\u062d\u0629 \u0644\u0644\u0625\u062c\u0627\u0628\u0629 \u0628\u062b\u0642\u0629."
                if query_language(question) == "ar"
                else "I could not find enough information in the available documents to answer confidently."
            )
            return {"answer": answer, "sources": [], "route": route.__dict__}

        language_instruction = (
            "Answer in Arabic unless the user explicitly asks for English."
            if query_language(question) == "ar"
            else "Answer in English unless the user explicitly asks for Arabic."
        )
        prompt = f"""
You are a data governance and cybersecurity policy assistant using agentic RAG.
{language_instruction}

Rules:
1. Use only the provided sources.
2. If the answer is not supported by the sources, say that clearly.
3. Cite every important claim using source markers like [S1] or [S2].
4. For procedures, use clear steps.
5. For comparisons, organize by document/domain and highlight differences.
6. Do not provide legal advice; frame the response as policy assistance.

Agent route:
{json.dumps(route.__dict__, ensure_ascii=False)}

Sources:
{self._context_from_sources(sources)}

Current question:
{question}

Answer:
"""
        response = self.answer_llm.invoke(prompt).content
        source_items = self._source_items(sources)
        return {
            "answer": response,
            "sources": source_items,
            "route": route.__dict__,
        }

    def _source_items(self, sources: list[Document]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str | None]] = set()

        for idx, doc in enumerate(sources, start=1):
            key = (doc.metadata.get("domain"), doc.metadata.get("file_name"))
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "source_id": f"S{idx}",
                    "file_name": doc.metadata.get("file_name"),
                    "title": doc.metadata.get("title"),
                    "domain": doc.metadata.get("domain"),
                    "domain_label": doc.metadata.get("domain_label"),
                }
            )

        return items
