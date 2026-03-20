import os
import time
import json
import sqlite3
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def main():
    print("Loading environment and database...")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    db_path = Path(__file__).with_name("ffu.db")
    db = sqlite3.connect(db_path)
    
    print("Loading embeddings into memory...")
    try:
        rows = db.execute("SELECT filename, chunk_text, embedding FROM documents").fetchall()
    except Exception as e:
        print(f"Database error: {e}")
        return
        
    documents = []
    for row in rows:
        filename, chunk_text, embedding_json = row
        doc_vector = np.array(json.loads(embedding_json))
        documents.append({
            "filename": filename,
            "chunk_text": chunk_text,
            "doc_vector": doc_vector
        })
        
    print(f"Loaded {len(documents)} chunks.")
    
    eval_path = Path(__file__).with_name("eval_data.json")
    if not eval_path.exists():
        print(f"Could not find {eval_path}")
        return
        
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    total_score = 0
    
    print("\nStarting Evaluation...\n" + "="*60)
    
    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        start_t = time.time()
        
        # 1. Embed Question
        res = client.embeddings.create(input=question, model="text-embedding-3-small")
        query_vector = np.array(res.data[0].embedding)
        
        # 2. Top 3 chunks
        similarities = []
        for doc in documents:
            cos_sim = np.dot(query_vector, doc["doc_vector"]) / (np.linalg.norm(query_vector) * np.linalg.norm(doc["doc_vector"]))
            similarities.append((cos_sim, doc["filename"], doc["chunk_text"]))
            
        top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
        context_text = "\n\n---\n\n".join([f"Source: {chunk[1]}\n{chunk[2]}" for chunk in top_chunks])
        
        # 3. Answer Generation
        system_prompt = f"""You are a precise assistant analyzing Swedish tender documents (FFUs). Answer the user's question using ONLY the following context chunks. If the answer is not in the context, say you don't know.\n\nContext:\n{context_text}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        gen_res = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        system_answer = gen_res.choices[0].message.content
        
        # 4. LLM Judge
        judge_prompt = "You are an expert grader. Compare the System Answer to the Ground Truth. Score the System Answer strictly from 1 to 5 on factual accuracy and 'Faithfulness' to the ground truth. Return ONLY the integer."
        judge_messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": f"Ground Truth: {ground_truth}\n\nSystem Answer: {system_answer}"}
        ]
        
        judge_res = client.chat.completions.create(model="gpt-4o-mini", messages=judge_messages)
        try:
            # Strip any punctuation/spaces
            response_text = judge_res.choices[0].message.content.strip()
            score = int("".join(filter(str.isdigit, response_text)))
            # Clamp to 1-5
            score = max(1, min(5, score))
        except Exception:
            score = 1 # Fallback
            
        latency = time.time() - start_t
        total_score += score
        
        print(f"[{i}/{len(eval_data)}] Question: {question}")
        print(f"Latency: {latency:.2f}s | Score: {score}/5")
        print("-" * 60)
        
    avg_score = total_score / len(eval_data)
    print(f"Average Score: {avg_score:.2f} / 5.00")
    print("="*60)

if __name__ == "__main__":
    main()
