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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
db = sqlite3.connect(Path(__file__).with_name("ffu.db"), check_same_thread=False)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
data_dir = Path("data")
extract = lambda path: pymupdf4llm.to_markdown(str(path), ignore_images=True, ignore_graphics=True)


@asynccontextmanager
async def lifespan(app):
    db.execute("CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, filename TEXT, content TEXT)")
    db.commit()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/process")
def process():
    logger.info("Processing documents...")
    db.execute("DELETE FROM documents"); db.commit()
    paths = sorted(data_dir.rglob("*.pdf"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(extract, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            db.execute("INSERT INTO documents(filename, content) VALUES(?, ?)", (path.name, future.result())); db.commit()
            logger.info(f"Processed {path.name}")
    return {"status": "ok", "count": len(paths)}


@app.post("/chat")
def chat(body: dict):
    import time
    start_time = time.time()
    docs = db.execute("SELECT id, filename FROM documents ORDER BY id").fetchall()
    system = {"role": "system", "content": "You are an FFU document analyst for Swedish construction tender documents. Available documents:\n" + "\n".join(f"{doc_id}: {name}" for doc_id, name in docs) + "\nUse read_document when you need the full content of a document."}
    messages = [system, *body.get("history", []), {"role": "user", "content": body.get("message", "")}]
    tools = [{
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read one FFU document by database id.",
            "parameters": {
                "type": "object",
                "properties": {"document_id": {"type": "integer"}},
                "required": ["document_id"],
            },
        },
    }]
    try:
        for _ in range(10):
            resp = client.chat.completions.create(model="gpt-5.4", messages=messages, tools=tools, tool_choice="auto")
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return {"response": msg.content or "", "time_taken_seconds": round(time.time() - start_time, 2)}
            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                row = db.execute("SELECT content FROM documents WHERE id = ?", (args["document_id"],)).fetchone()
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": row[0] if row else "Document not found.",
                })
        return {"response": "Stopped after 10 tool iterations.", "time_taken_seconds": round(time.time() - start_time, 2)}
    except Exception as e:
        return {"response": f"Error: {e}", "time_taken_seconds": round(time.time() - start_time, 2)}
