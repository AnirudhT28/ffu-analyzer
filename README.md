# 🏗️ FFU Analyzer
### *AI-Powered Intelligence for Construction Estimators*

Implemented an advanced **Retrieval-Augmented Generation (RAG)** platform specifically engineered for the construction industry. This system autonomously parses, indexes, and analyzes complex "Förfrågningsunderlag" (FFU) documents—transforming dense technical descriptions, AMA standards, and Excel-based quantity lists (`mängdförteckningar`) into an interactive, searchable knowledge base.

---

## 🛠️ The Tech Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | React + Vite + TailwindCSS | High-performance SPA with citation-aware UI. |
| **Backend** | FastAPI + Python | Asynchronous API handling RAG orchestration. |
| **Vector Engine** | SQLite + NumPy | Custom cosine similarity engine for zero-dependency retrieval. |
| **LLM Engine** | OpenAI `gpt-4o-mini` | Reasoning, query expansion, and Swedish synthesis. |
| **Embeddings** | `text-embedding-3-small` | High-efficiency vector representation. |
| **Parsing** | LlamaParse + Pandas | Multi-modal parsing (PDF & Excel). |

---

## Key Features & Implementation

### 1. High-Precision RAG Pipeline
To keep the system lightweight and fast, I bypassed heavy vector-database frameworks in favor of a NumPy-powered similarity engine. 
* **Native Retrieval:** We calculate 1:1 question-to-chunk cosine similarity directly against a SQLite-stored vector array.
* **Contextual Grounding:** The system retrieves the **Top 7** most relevant context chunks, strictly forcing the LLM to answer using only the provided data to eliminate hallucinations.

### 2. Deep Structural Parsing
Construction FFUs are notoriously difficult to parse due to complex layouts. Specially with all the tables in pdf form which are tough to parse through regular text parsers.
* **PDF Intelligence:** Powered by **LlamaParse**, the pipeline converts multi-column layouts and dense AMA tables into pristine Markdown, preserving headers and nested relationships.
* **Excel (`.xlsx` / `.xls`) Logic:** Using **Pandas**, the system iterates through every sheet ("Flik"), explicitly tagging chunks with sheet names to maintain context during retrieval.

### 3. Hyper-Accurate "Citation Pills"
Trust and accuracy is extremly important in construction estimation. The UI renders interactive citation refrences to source documents for every claim made by the AI.
* **Excel Citations:** Matches chunks against `### Flik: <Name>` markers.
* **PDF Citations:** Matches chunks against Markdown page markers (`Page X`).
* **UX:** Hovering over a pill reveals the exact source filename and page/sheet location.

### 4. Empirical Optimization (LLM-as-a-Judge)
Created a custom evaluation suite (`evaluate_rag.py`) and a "base Dataset," by giving a mized sample of different document types to the vision based gpt and asking to create a sample set of questions which was then used `gpt-4o-mini` to grade retrieval accuracy.
> **The Result:** discovered that standard large chunks (1500 tokens) diluted technical context. By optimizing to **Size 512 / Overlap 50**, it boosted the average Retrieval Score from **3.8/5.0 to 4.2/5.0**. (not near 5 due to misses on the drawing document questions)

---

## Future Roadmap

### 1. Scaling & Dynamic Ingestion
* **Cloud Vector Store:** Move from static SQLite shards to a managed store like **Pinecone** or **pgvector** to support millions of documents.
* **Async Workers:** Implement **Celery/Redis** to allow users to upload new FFUs via the UI without interrupting the main server process.

### 2. Multimodal Support: The "Visual Gap"
Construction documents are inherently visual (blueprints, site plans, diagrams).
* **The Goal:** Transition to **Multimodal RAG** using Vision LLMs to describe diagrams during indexing and CLIP-based embeddings to allow the retriever to "see" spatial information.

### 3. Hybrid Search (Semantic + Keyword)
Swedish tender documents use highly specific technical codes (e.g., *AMA BCB.11*).
* **The Goal:** Combine Vector Similarity with **BM25 Keyword Matching** to ensure exact technical codes are prioritized alongside natural language intent.

### 4. LlamaParse ROI Analysis
While LlamaParse adds a per-page API cost, it significantly reduces the "failure rate" of the RAG system.
* **Hypothesis:** Cleaner Markdown leads to smaller, high-density chunks, which reduces the total token count sent to GPT-4o, potentially offsetting the parsing cost by lowering the final inference bill.

---

## Deployment & Development

### 📦 The "Shard" Strategy
To bypass GitHub's 100MB file limit while maintaining a zero-configuration demo, the database is stored in **40MB shards** (`ffu.db.part*`). 
* **On Boot:** The backend automatically detects these shards and reconstructs the full 130MB+ SQLite database in the cloud container's memory before the server starts.

### *Try it out here!*

https://ffu-analyzer-sigma.vercel.app/

