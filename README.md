# Arabic Policy Analyzer

## Overview

Arabic Policy Analyzer is a Streamlit-based multi-document Agentic RAG system for Arabic and English policy and regulation PDFs. The system allows users to ask questions across multiple document domains, automatically routes each question to the most relevant domain or document, retrieves supporting text chunks, re-ranks the retrieved evidence, and generates a grounded answer using an LLM.

The project is designed for formal policy and regulatory documents, where answers need to be traceable to source material rather than generated from general knowledge.

## Supported Document Domains

The current system supports policy documents from the following domains:

- Data Sharing
- Cybersecurity
- Critical Systems Cybersecurity
- Personal Data Protection

## System Architecture

The system follows a four-stage Agentic RAG pipeline:

1. **Knowledge Base Ingestion**
   - Reads PDF files from the `documents/` folder.
   - Extracts text using `PyPDFLoader`.
   - Cleans repeated noise such as headers, footers, page numbers, labels, and authority names.
   - Detects document language.
   - Infers document domain.
   - Splits text into chunks.
   - Adds metadata to each chunk.

2. **Indexing**
   - Creates embeddings using OpenAI `text-embedding-3-small`.
   - Stores embeddings in a FAISS vector index.
   - Saves readable chunks and metadata in `chunks.json`.
   - Saves document-level metadata in `documents_metadata.json`.

3. **Hybrid Retrieval and Re-ranking**
   - Routes the user question using Arabic/English keyword heuristics and an LLM router.
   - Detects the task type: answer, comparison, or summary.
   - Retrieves relevant chunks using both FAISS semantic search and BM25 keyword search.
   - Merges retrieved candidates.
   - Re-ranks candidates using `BAAI/bge-reranker-base`.
   - Selects the top source chunks for answer generation.

4. **Grounded Answer Generation**
   - Sends selected source chunks to `gpt-4o-mini`.
   - Instructs the model to answer only from retrieved sources.
   - Returns the final answer with simplified source information.
   - Does not include previous chat history in the final answer prompt.

## Project Structure

```text
.
├── documents/
│   └── PDF policy and regulation files
│
├── indexes/
│   └── policy_index/
│       ├── FAISS vector index files
│       └── chunks.json
│
├── ingest.py
├── rag_agent.py
├── second-app.py
├── documents_metadata.json
└── README.md
```

## Main Files

### `ingest.py`

Builds the knowledge base from the PDF documents.

Main responsibilities:

- Load PDFs from the `documents/` folder.
- Extract text using `PyPDFLoader`.
- Clean repeated document noise.
- Detect language.
- Infer document domain.
- Split text into chunks.
- Add metadata to every chunk.
- Create embeddings.
- Build and save the FAISS index.
- Save `chunks.json` and `documents_metadata.json`.

### `rag_agent.py`

Contains the main RAG agent logic.

Main responsibilities:

- Load the FAISS index.
- Load `chunks.json` as LangChain Document objects.
- Load `documents_metadata.json`.
- Build a BM25 keyword retriever.
- Route questions to the most relevant domain or document.
- Detect task type.
- Perform hybrid retrieval.
- Re-rank candidate chunks.
- Generate grounded answers using `gpt-4o-mini`.

### `second-app.py`

Provides the Streamlit user interface.

Main responsibilities:

- Load the OpenAI API key from Streamlit secrets or environment variables.
- Check if the knowledge base exists.
- Provide a button to rebuild the knowledge base.
- Display indexed documents in the sidebar.
- Provide a chat interface.
- Send user questions to `MultiDocumentRAGAgent.answer()`.
- Display answers and simplified sources.

## Generated Files

### `indexes/policy_index/`

Stores the FAISS vector index used for semantic retrieval.

### `indexes/policy_index/chunks.json`

Stores readable text chunks and metadata. It is used for:

- BM25 keyword retrieval
- Answer context
- Source metadata
- Debugging and inspection

### `documents_metadata.json`

Stores document-level metadata used by the sidebar and routing logic.

## Why FAISS and `chunks.json` Are Both Needed

FAISS is used for fast semantic search over embeddings. It helps the system retrieve chunks that are similar in meaning to the user question.

`chunks.json` stores the original readable text and metadata. It is needed for BM25 keyword search, source tracking, metadata filtering, and answer generation.

In simple terms:

```text
FAISS = semantic search over vectors
chunks.json = readable text + metadata
```

## Installation

Create a virtual environment and install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Environment Variables

Set your OpenAI API key using an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

For Streamlit Cloud or local Streamlit secrets, you can also store the key in:

```text
.streamlit/secrets.toml
```

Example:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

## How to Run

### 1. Add Documents

Place the PDF files inside the `documents/` folder.

```text
documents/
├── data_sharing_policy.pdf
├── cybersecurity_policy.pdf
└── personal_data_protection.pdf
```

### 2. Build the Knowledge Base

Run:

```bash
python ingest.py
```

This creates the FAISS index, `chunks.json`, and `documents_metadata.json`.

### 3. Start the Streamlit App

Run:

```bash
streamlit run second-app.py
```

### 4. Ask Questions

Use the chat interface to ask questions about the indexed policy documents.

Example questions:

```text
What are the main requirements for data sharing?
```

```text
ما هي الضوابط المتعلقة بمشاركة البيانات؟
```

```text
Compare the data sharing requirements with personal data protection requirements.
```

## Rebuilding the Knowledge Base

If PDFs are added, removed, or updated, the knowledge base should be rebuilt.

You can rebuild it by:

```bash
python ingest.py
```

or by using the rebuild button in the Streamlit sidebar.

## Key Design Decisions

### No Previous Chat History in Final Answer Prompt

The final answer prompt does not include previous chat history. This reduces the chance of the model repeating an old answer or mixing previous context with a new question.

Each answer is generated using only:

- The current user question
- The retrieved source chunks

### Hybrid Retrieval

The system uses both FAISS and BM25 because policy documents require both semantic understanding and exact keyword matching.

FAISS is useful for meaning-based retrieval, while BM25 is useful for exact terms, article names, definitions, and regulatory expressions.

### Re-ranking

The cross-encoder re-ranker improves retrieval precision by scoring candidate chunks directly against the user question.


## Technologies Used

- Python
- Streamlit
- LangChain
- PyPDFLoader
- FAISS
- BM25
- OpenAI `text-embedding-3-small`
- OpenAI `gpt-4o-mini`
- `BAAI/bge-reranker-base`
