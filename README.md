# FFU Analyzer - AI Agentish Platform for Construction Estimators

An advanced Retrieval-Augmented Generation (RAG) platform tailored for the Swedish construction industry. The system autonomously reads, analyzes, and answers questions about complex "Förfrågningsunderlag" (FFU) documents, including technical descriptions, AMA standards, and Excel-based quantity lists (`mängdförteckningar`).

## 🏗️ Architecture

- **Frontend:** React + Vite + TailwindCSS (Deployed on Vercel)
- **Backend:** FastAPI + Python (Deployed on Koyeb)
- **Vector Store:** Raw SQLite (`ffu.db`) using `numpy` cosine similarity for blazingly fast retrieval without heavy vector-database dependencies.
- **LLM Engine:** OpenAI `gpt-4o-mini` & `text-embedding-3-small`

---

## 🚀 Key Features & Engineering Innovations

### 1. The RAG Pipeline
We deliberately avoided bloated frameworks (like LangChain's chains) for the chat endpoint, opting for a 1:1 question-to-chunk cosine similarity calculation natively using `numpy`. The pipeline embeds the user's query, fetches all indexed chunks from SQLite, calculates similarity, and retrieves the Top 7 most relevant contexts before restricting the system prompt to answer strictly based on those chunks.

### 2. Deep Parsing of PDFs & Excel
Construction FFU documents are notoriously difficult to parse due to multi-column PDF layouts, dense AMA tables, and Excel sheets.
- **PDFs:** Processed via **LlamaParse**. LlamaParse accurately understands and extracts complex document structures (tables, nested paragraphs, headers) and converts them into pristine Markdown.
- **Excel (`.xlsx` / `.xls`):** Processed via **Pandas**. The ingestion pipeline loops through each sheet ("Flik") and extracts rows, explicitly tagging every chunk with the exact sheet name.

### 3. Hyper-Accurate Citations
Instead of returning raw filenames, the retrieval pipeline utilizes regex to parse the chunk's content.
- If it's an Excel file, the chunk is matched against `### Flik: <Name>` to cite the exact sheet.
- If it's a PDF, the chunk is matched against markdown page markers (`Page X`) to cite the exact page.
The UI renders these as text-based citation pills that display `[Filename] (Sida: X)` or `[Filename] (Flik: X)` on hover.

### 4. LLM-as-a-Judge & Empirical Chunk Selection
To ensure absolute accuracy, an automated evaluation pipeline (`evaluate_rag.py`) loops through a golden dataset of Q&A pairs (`eval_data.json`). It uses `gpt-4o-mini` as a judge to grade the system's "Faithfulness" to ground truth on a scale from 1-to-5.

Through iterative benchmarking via this script, we empirically found that the standard large chunking parameters (Size 1500 / Overlap 300) resulted in diluted context. Optimizing our `RecursiveCharacterTextSplitter` to **Size 512 / Overlap 50** skyrocketed our average Retrieval Score from 3.8/5.0 to **4.8/5.0**.

### 5. Serverless SQLite Sharding (Bypassing Git & Koyeb Limits)
Because Koyeb's ephemeral filesystem wipes databases on restart, and GitHub strictly rejects files larger than 100MB, the monolithic 130MB SQLite database is handled uniquely:
- The database is split into 50MB shards (`ffu.db.partaa`, `ffu.db.partab`, etc.) to bypass Git LFS and GitHub's hard limits.
- On startup, the FastAPI `@asynccontextmanager lifespan` automatically detects missing databases, combines the shards back together in milliseconds, and reconstructs the fully populated 130MB SQLite database directly in the server's memory.

---

## 🛠️ Local Development

### 1. Database Reconstruction
If you pull the repo fresh, the SQLite database is stored in `backend/ffu.db.part*`. Simply run the FastAPI server, and it will reconstruct itself:
```bash
cd backend
uvicorn main:app --reload
```

### 2. Run the Evaluator
To benchmark the RAG pipeline or test new parsing techniques:
```bash
cd backend
python evaluate_rag.py
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🌍 Environment Variables

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=sk-...
LLAMA_CLOUD_API_KEY=llx-...
```

If deploying the Vite frontend to a different domain than the backend, add `VITE_API_URL` to your Vercel/Netlify environment variables:
```env
VITE_API_URL=https://my-koyeb-app.koyeb.app
```