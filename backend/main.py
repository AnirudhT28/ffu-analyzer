import os
import re
import time
import json
import logging
import sqlite3
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_core.documents import Document
from llama_parse import LlamaParse

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
data_dir = Path("data")
cache_dir = data_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

PERSIST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

db = None  # Initialized in lifespan after database reconstruction


def filter_latest_revisions(file_paths: list[str]) -> list[str]:
    grouped_files = {}
    pattern = re.compile(r"^(.*?)(?:\s*rev\.?\s*\d{4}-\d{2}-\d{2})?(\.[a-zA-Z0-9]+)$", re.IGNORECASE)
    for path in file_paths:
        path_obj = Path(path)
        filename = path_obj.name
        match = pattern.match(filename)
        if match:
            base_name, ext = match.group(1).strip(), match.group(2)
            key = f"{base_name}{ext}".lower()
        else:
            key = filename.lower()
        if key not in grouped_files:
            grouped_files[key] = path
        else:
            if "rev" in filename.lower() and "rev" not in Path(grouped_files[key]).name.lower():
                grouped_files[key] = path
            elif "rev" in filename.lower() and "rev" in Path(grouped_files[key]).name.lower():
                if filename > Path(grouped_files[key]).name:
                    grouped_files[key] = path
    return list(grouped_files.values())


def extract(path):
    cache_path = cache_dir / f"{path.stem}.md"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()
            return Document(page_content=text, metadata={"source": path.name})

    if path.suffix.lower() == ".pdf":
        parser = LlamaParse(
            api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
            base_url="https://api.cloud.eu.llamaindex.ai",
            result_type="markdown",
            parsing_instruction="This document is a Swedish construction tender (FFU). Extract tables clearly. Ignore and remove completely empty columns or rows in Excel files. Do not output strings of empty markdown pipes. Maintain strict association between headers and row data."
        )
        documents = parser.load_data(str(path))
        text = "\n\n".join([doc.text for doc in documents])
    elif path.suffix.lower() == ".xlsx" or path.suffix.lower() == ".xls":
        import pandas as pd
        import numpy as np
        sheets = pd.read_excel(path, sheet_name=None)
        text = ""
        for sheet_name, df in sheets.items():
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
            df = df.fillna("")
            tsv_table = df.to_csv(index=False, sep='\t')
            text += f"\n\n### Flik: {sheet_name}\n{tsv_table}\n\n"
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if not text or not text.strip():
        logger.error(f"Extraction failed or returned empty text for {path.name}")
        return None

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)

    return Document(page_content=text, metadata={"source": path.name})


splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)


def reconstruct_database():
    db_path = os.path.join(PERSIST_DIRECTORY, "ffu.db")
    parts = sorted(glob.glob(os.path.join(PERSIST_DIRECTORY, "ffu.db.part*")))
    if not os.path.exists(db_path) or os.path.getsize(db_path) < 1000000:
        if parts:
            print(f"DEBUG: Shards found: {parts}")
            print("DEBUG: Reconstructing...")
            with open(db_path, "wb") as outfile:
                for part in parts:
                    with open(part, "rb") as infile:
                        outfile.write(infile.read())
            print(f"DEBUG: Final DB size: {os.path.getsize(db_path) / (1024*1024):.2f} MB")
        else:
            print("DEBUG: No shards found. Cannot reconstruct database.")


@asynccontextmanager
async def lifespan(app):
    global db

    # Step 1: Reconstruct the database from shards FIRST
    reconstruct_database()

    # Step 2: ONLY NOW connect to the fully reconstructed database
    db_path = os.path.join(PERSIST_DIRECTORY, "ffu.db")
    db = sqlite3.connect(db_path, check_same_thread=False)

    db.execute("CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, filename TEXT, chunk_text TEXT, embedding TEXT)")
    db.commit()

    print(f"DEBUG: Absolute DB Path: {db_path}")
    print(f"DEBUG: Checking for database at {db_path}. Exists: {os.path.exists(db_path)}")

    if os.path.exists(db_path):
        print(f"DEBUG: Database size: {os.path.getsize(db_path) / (1024*1024):.2f} MB")
        row_count = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        print(f"DEBUG: Vector Document Count: {row_count}")

    print(f"DEBUG: Contents of DB directory: {os.listdir(PERSIST_DIRECTORY)}")

    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"DEBUG: OpenAI API Key exists in environment: {bool(api_key)}")
    print("DEBUG: Backend is fully armed and ready.")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
def process():
    logger.info("Processing documents...")
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    db.execute("DELETE FROM documents")
    db.commit()

    raw_paths = [str(p) for p in list(data_dir.rglob("*.pdf")) + list(data_dir.rglob("*.xlsx")) if not p.name.startswith("._")]
    filtered_strings = filter_latest_revisions(raw_paths)
    paths = sorted([Path(p) for p in filtered_strings])

    all_chunks = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(extract, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                doc = future.result()
            except Exception as e:
                print(f"Failed to read {path.name}: {e}")
                continue

            if not doc:
                continue

            print(f"Extracted document from {path.name}")
            doc_chunks = splitter.split_documents([doc])
            print(f"Created {len(doc_chunks)} chunks for {path.name}")

            for chunk in doc_chunks:
                all_chunks.append((chunk.metadata.get("source", path.name), chunk.page_content))
            logger.info(f"Processed {path.name}")

    if all_chunks:
        all_chunks = [(fn, text) for fn, text in all_chunks if text.strip()]
        print(f"Batch embedding {len(all_chunks)} chunks...")
        batch_size = 25
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            texts = [b[1] for b in batch]
            response = client.embeddings.create(input=texts, model="text-embedding-3-small")

            records = []
            for j, data in enumerate(response.data):
                filename, chunk_text = batch[j]
                embedding_json = json.dumps(data.embedding)
                records.append((filename, chunk_text, embedding_json))

            db.executemany("INSERT INTO documents(filename, chunk_text, embedding) VALUES(?, ?, ?)", records)
            time.sleep(0.5)
        db.commit()

    db_path = os.path.join(PERSIST_DIRECTORY, "ffu.db")
    parts = sorted(glob.glob(os.path.join(PERSIST_DIRECTORY, "ffu.db.part*")))
    part_names = [os.path.basename(p) for p in parts]

    exists = os.path.exists(db_path)
    size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2) if exists else 0.0

    return {
        "status": "ok",
        "count": len(paths),
        "db_exists": exists,
        "db_size_mb": size_mb,
        "parts_found": part_names
    }


@app.get("/debug")
def debug():
    db_path = os.path.join(PERSIST_DIRECTORY, "ffu.db")
    parts = sorted(glob.glob(os.path.join(PERSIST_DIRECTORY, "ffu.db.part*")))
    part_names = [os.path.basename(p) for p in parts]

    exists = os.path.exists(db_path)
    size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2) if exists else 0.0
    row_count = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] if db and exists else 0

    return {
        "db_exists": exists,
        "db_size_mb": size_mb,
        "parts_present": part_names,
        "row_count": row_count
    }


@app.post("/chat")
def chat(body: dict):
    start_time = time.time()

    req_messages = body.get("messages")
    if req_messages and len(req_messages) > 0:
        user_query = req_messages[-1].get("content", "")
    else:
        user_query = body.get("message", "")

    print(f"DEBUG: User Query: {user_query}")

    try:
        # 1. Embed query
        response = client.embeddings.create(input=user_query, model="text-embedding-3-small")
        query_vector = np.array(response.data[0].embedding)

        # 2. Fetch docs & calculate similarity
        rows = db.execute("SELECT filename, chunk_text, embedding FROM documents").fetchall()
        print(f"DEBUG: Retrieved {len(rows)} chunks from the database.")

        similarities = []
        for row in rows:
            filename, chunk_text, embedding_json = row
            doc_vector = np.array(json.loads(embedding_json))
            cos_sim = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            similarities.append((cos_sim, filename, chunk_text))

        # 3. Top 7 chunks
        top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:7]

        for i, (score, filename, chunk) in enumerate(top_chunks):
            preview = chunk[:150].replace('\n', ' ')
            print(f"DEBUG: Chunk Source: {filename} | Score: {score:.4f}")
            print(f"DEBUG: Preview: {preview}...")

        context_text = "\n\n---\n\n".join([f"Source: {filename}\n{chunk}" for _, filename, chunk in top_chunks])

        # Extract sources from chunks
        sources_set = set()
        for _, filename, chunk in top_chunks:
            if filename.lower().endswith((".xlsx", ".xls")):
                match = re.search(r"### Flik:\s*(.*?)\n", chunk)
                if match:
                    sheet_name = match.group(1).strip()
                    sources_set.add(f"{filename} (Flik: {sheet_name})")
                else:
                    sources_set.add(filename)
            else:
                page_match = re.search(r"(?i)(?:page|sida)\s+(\d+)", chunk)
                if page_match:
                    page_num = page_match.group(1)
                    sources_set.add(f"{filename} (Sida: {page_num})")
                else:
                    sources_set.add(filename)
        unique_sources = list(sources_set)

        # 4. Construct strict system prompt
        system_prompt = f"""You are a Senior Swedish Construction Estimator (Kalkylator). You are analyzing 'Förfrågningsunderlag' (FFU) documents. 
- If the context contains a table header but the data seems to be in a surrounding chunk, look across all provided chunks to piece the information together.
- Pay close attention to 'AMA' or 'MER' standards and specific 'Konto/Kod' identifiers (like BFB.1).
- If the answer is not in the context, say you don't know. 
Context: {context_text}"""

        system_msg = {"role": "system", "content": system_prompt}
        if req_messages and len(req_messages) > 0:
            messages = [system_msg] + req_messages
        else:
            messages = [system_msg] + body.get("history", []) + [{"role": "user", "content": user_query}]

        # 5. Send to LLM
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return {
            "answer": resp.choices[0].message.content or "",
            "sources": unique_sources,
            "time_taken_seconds": round(time.time() - start_time, 2)
        }
    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "sources": [],
            "time_taken_seconds": round(time.time() - start_time, 2)
        }