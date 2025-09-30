# BizTalk to Boomi PDF Import

A full-stack application to convert a BizTalk project analysis PDF into:

1. A ready-to-use Boomi AI prompt
2. Source-to-target field mappings (Markdown table + Excel workbook)
3. Connection objects listing with masked authentication details

Technologies:
- Frontend: React + Vite + Tailwind CSS
- Backend: FastAPI + LangChain/LangGraph + Azure OpenAI

## Project Structure
```
/frontend   # React client
/backend    # FastAPI server
.env.example
README.md
```

## Prerequisites
- Node.js 18+
- Python 3.11+
- Azure OpenAI resource with a deployed chat model (variable name matches `AZURE_OPENAI_MODEL`)

## Environment Variables
Copy `.env.example` to `.env` inside `backend/` and fill in:
```
AZURE_OPENAI_ENDPOINT= https://<your-endpoint>.openai.azure.com/
AZURE_OPENAI_API_KEY= <your-key>
AZURE_OPENAI_MODEL= gpt-5-chat
```

## Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Run with Docker (optional)

### Prereqs
- Docker Desktop or a compatible Docker engine

### Using Docker Compose
```bash
# From repo root
docker compose build
docker compose up -d
```

Services:
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

Notes:
- Backend reads environment from `backend/.env` (mounted via `env_file` in `docker-compose.yml`).
- Update the frontend to point at `http://localhost:8000` (default already expects this).
- To view logs:
```bash
docker compose logs -f backend
docker compose logs -f frontend
```
- To stop/remove:
```bash
docker compose down
```

### API
- `POST /process-pdf` multipart form-data with field `file` (PDF). Returns JSON: `{ job_id, download_url }`
- `GET /download/{job_id}` returns a ZIP (application/zip) containing:
   - `<job_id>.md` — the Markdown report with Boomi prompt, mapping table, and masked connections
   - `<job_id>_mapping.xlsx` — the Excel workbook (sheet: "Mapping") generated from the mapping table
- `GET /health` health check

## Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Visit: http://localhost:5173

The frontend expects backend at `http://localhost:8000`.

## Usage Flow
1. Start backend (`uvicorn ...`)
2. Start frontend (`npm run dev`)
3. Upload the BizTalk analysis PDF
4. When processing completes, click the download link to obtain a ZIP containing:
   - Markdown report (.md) with the Boomi AI prompt, mapping table, and masked connections
   - Excel workbook (.xlsx) of the source↔target mapping (sheet name: "Mapping")

## Notes
- PDF parsing uses `pypdf` (basic text extraction). Improve with `pdfplumber` if layout issues arise.
- LangGraph orchestrates nodes: parse → boomi_prompt (LLM refined) → mapping → connections → assemble.
- Secrets in connections are masked with `***` via LLM instruction; validate outputs manually for compliance.
- No persistence layer; results stored in-memory and lost on server restart.
- CORS is enabled for `http://localhost:5173` by default. Add more via `CORS_EXTRA_ORIGINS` (comma-separated) env var.
- Boomi prompt generation is logged (INFO level) truncated to 800 chars; adjust with `LOG_LEVEL` env var.
- The Excel file is created from the Markdown mapping table using `openpyxl` and included in the ZIP returned by `/download/{job_id}`.

## Logging
- Detailed LLM prompt/response logs are written to `backend/logs/llm.log` with rotation (5 MB per file, 5 backups).
- Each LLM call is logged with:
   - `job_id` (to correlate with your download URL)
   - `label` (`boomi_prompt`, `mapping_table`, `connections_table`)
   - redacted `PROMPT` and `RESPONSE` bodies
- Redaction masks common secrets like passwords, API keys, Basic auth tokens. Avoid placing real secrets in PDFs.
- Control verbosity via env var: `LOG_LEVEL=DEBUG|INFO|WARN|ERROR`.

Quick view commands (optional):
```bash
# Tail logs
tail -f backend/logs/llm.log

# Show last 100 lines
tail -n 100 backend/logs/llm.log
```

## Extending
- Add authentication & rate limiting
- Persist artifacts in object storage (S3/Azure Blob)
- Add retry logic and better PDF structural parsing
- Integrate real LangGraph state machine for multi-step reasoning

## License
MIT
