import os
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import pymupdf4llm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
db = sqlite3.connect(Path(__file__).with_name("ffu.db"), check_same_thread=False)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
data_dir = Path("data")
extract = lambda path: pymupdf4llm.to_markdown(str(path), ignore_images=True, ignore_graphics=True)
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)


@asynccontextmanager
async def lifespan(app):
    db.execute("CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, filename TEXT, chunk_text TEXT, embedding TEXT)")
    db.commit()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/process")
def process():
    logger.info("Processing documents...")
    db.execute("DELETE FROM documents"); db.commit()
    paths = sorted([p for p in data_dir.rglob("*.pdf") if not p.name.startswith("._")])
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(extract, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                text = future.result()
            except Exception as e:
                print(f"Failed to read {path.name}: {e}")
                continue
            print(f"Extracted {len(text)} characters from {path.name}")
            chunks = splitter.split_text(text)
            print(f"Created {len(chunks)} chunks for {path.name}")
            for chunk in chunks:
                response = client.embeddings.create(input=chunk, model="text-embedding-3-small")
                embedding_json = json.dumps(response.data[0].embedding)
                db.execute("INSERT INTO documents(filename, chunk_text, embedding) VALUES(?, ?, ?)", (path.name, chunk, embedding_json))
            db.commit()
            logger.info(f"Processed {path.name}")
    return {"status": "ok", "count": len(paths)}


@app.post("/chat")
def chat(body: dict):
    import time
    import numpy as np
    start_time = time.time()
    
    req_messages = body.get("messages")
    if req_messages and len(req_messages) > 0:
        user_query = req_messages[-1].get("content", "")
    else:
        user_query = body.get("message", "")
    
    try:
        # 1. Embed query
        response = client.embeddings.create(input=user_query, model="text-embedding-3-small")
        query_vector = np.array(response.data[0].embedding)
        
        # 2. Fetch docs & calculate similarity
        rows = db.execute("SELECT filename, chunk_text, embedding FROM documents").fetchall()
        
        similarities = []
        for row in rows:
            filename, chunk_text, embedding_json = row
            doc_vector = np.array(json.loads(embedding_json))
            cos_sim = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            similarities.append((cos_sim, filename, chunk_text))
            
        # 3. Top 5 chunks
        top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]
        context_text = "\n\n---\n\n".join([f"Source: {filename}\n{chunk}" for _, filename, chunk in top_chunks])
        
        # 4. Construct strict system prompt
        system_prompt = f"""You are an expert assistant analyzing Swedish tender documents (FFUs). For specific project details, use ONLY the following context chunks. If the project detail is not in the context, clearly say you don't know. You may use your general knowledge to define construction terms or explain industry concepts.\nContext:\n{context_text}"""
        
        system_msg = {"role": "system", "content": system_prompt}
        if req_messages and len(req_messages) > 0:
            messages = [system_msg] + req_messages
        else:
            messages = [system_msg] + body.get("history", []) + [{"role": "user", "content": user_query}]
        
        # 5. Send to LLM
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return {"response": resp.choices[0].message.content or "", "time_taken_seconds": round(time.time() - start_time, 2)}
    except Exception as e:
        return {"response": f"Error: {e}", "time_taken_seconds": round(time.time() - start_time, 2)}
