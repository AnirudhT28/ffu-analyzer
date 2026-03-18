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
uvicorn main:app --reload --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 0
```

4. In a second terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

5. Open `http://localhost:5173`, click `Process FFU`, then start chatting.
