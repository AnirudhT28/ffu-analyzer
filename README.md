# FFU Analyzer

You should already have received:
- the assignment brief by email
- your personal OpenAI API key
- a zipped FFU document set

Requirements: Python 3.12+ and Node 24+.

## Getting Started

1. Put your API key in [`.env`](/ffu-analyzer/.env):

```env
OPENAI_API_KEY=your-key-here
```

2. Unzip the FFU files into [backend/data](/ffu-analyzer/backend/data).
3. Start the backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

4. In a second terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

5. Open `http://localhost:5173`, click `Process FFU`, then start chatting.


# 🏗️ FFU-Analyzer: Architectural Overview & RAG Optimization

## Current Status
**System Evaluation Score:** 4.17 / 5.00  
**Status:** Technical Backend & Pipeline Optimization Complete.

## Core Objective
High-fidelity extraction and Q&A for complex Swedish construction tender documents (FFU), specifically handling AMA/MER standards, technical account codes (e.g., BFB.1), and tabular pricing data.

## 1. The "Split Ingestion" Architecture
To handle the highly variable nature of construction documents without destroying our LLM context window with token bloat, the ingestion layer is bifurcated based on file type:

* **PDF Documents (Technical Specs, PMs, ESA):** * Routed through **LlamaParse** (EU-endpoint) with custom parsing instructions. 
  * Outputs structurally aware Markdown to ensure AMA/MER tables remain intact and readable by the retriever.
* **Excel Documents (Anbudsformulär / Pricing):** * Routed through a custom **Pandas "Scorched Earth" Pipeline**. 
  * **Why:** LlamaParse hallucinates massive visual Markdown grids for empty Excel cells, causing severe token bloat. 
  * **How:** Pandas loads the sheets, aggressively scrubs invisible whitespaces into `NaN`, drops all empty columns/rows, and converts the dense data into **TSV (Tab-Separated Values)**. This reduces token consumption by ~60% and allows the LLM to easily read roles, contacts, and requirements.

## 2. RAG Hyperparameters (The Golden Config)
Through empirical testing via `evaluate_rag.py`, the following parameters achieved peak accuracy:
* **Text Splitter:** `RecursiveCharacterTextSplitter`
* **Chunk Size:** 512 tokens
* **Chunk Overlap:** 50 tokens
* **Context Window:** `k=7` (Expanded to ensure fragmented table data stays connected to its headers).
* **LLM Persona:** Prompted as a *"Senior Swedish Construction Estimator"* to force domain-specific interpretation of technical codes.

## 3. Infrastructure & Cost Control
* **File Versioning:** Pre-ingestion regex filters ensure only the latest revision of a document (e.g., `rev. 2025-05-13`) is embedded, preventing vector collisions ("Evil Twins").
* **Local Caching:** Parsed Markdown/TSV is cached locally to bypass LlamaParse API costs and latency during iterative development.