# SFL BRE Portal — Backend API (FastAPI)

Python/FastAPI REST API for the [Frontend](../Frontend) React app (Satin Finserv Limited BRE Portal). Covers every feature area of the app — Products/Data Sources, Model Hub (pipeline + training), Model Testing (inference), AI Architecture, and Underwriting Settings — with real server-side computation, including genuine bank-statement parsing (CSV/TXT column parsing, PDF text extraction via `pypdf`) instead of hardcoded fixtures.

State is kept in-memory (no database) and resets on server restart. The previous Node/Express implementation is preserved at `../Backend-node-legacy` for reference.

## Getting Started

```bash
cd Backend
python -m venv venv
./venv/Scripts/activate        # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env           # optional: change PORT / CORS_ORIGIN
uvicorn app.main:app --reload --port 4000
```

Interactive API docs are auto-generated at `http://localhost:4000/docs` (Swagger UI) once running.

The Frontend dev server runs on `http://localhost:5173` by default and proxies `/api/*` to `http://localhost:4000` (see `Frontend/vite.config.js`) — no frontend changes are needed to use this backend; the JSON contract is identical to the previous Node implementation.

## Project Structure

```text
Backend/
├── app/
│   ├── main.py                # FastAPI app + router mounting
│   ├── config.py              # env var loading
│   ├── data/                  # static reference data (data sources, rules, model catalog)
│   ├── state/                 # in-memory mutable state (selection, uploads, models, settings)
│   ├── services/              # computation engines (pipeline simulation, statement parsing, inference scoring)
│   └── routers/                # FastAPI routers, mounted under /api
├── requirements.txt
└── venv/                      # local virtualenv (gitignored)
```

## API Overview

All routes are prefixed with `/api`. Full interactive reference: `/docs`.

| Area | Routes |
| --- | --- |
| Auth | `GET /auth/captcha`, `POST /auth/login`, `POST /auth/quick-login` |
| Data Sources | `GET /data-sources`, `GET /data-sources/{id}`, `POST /data-sources`, `GET /data-sources/presets`, `GET\|PUT /data-sources/selection` |
| Pipeline (Model Hub) | `GET /pipeline/uploads`, `POST /pipeline/uploads` (multipart file), `POST /pipeline/uploads/autofill`, `POST /pipeline/run`, `GET /pipeline/status` |
| Models | `GET /models/algorithms`, `POST /models/train`, `GET /models`, `PUT /models/{modelId}/version`, `POST /models/{modelId}/deploy` |
| Inference (Model Testing) | `GET /inference/deployed-models`, `POST /inference/run`, `POST /inference/evaluate/{modelId}`, `GET /inference/history` |
| AI Architecture | `GET /ai-architecture`, `PUT /ai-architecture/llm`, `POST /ai-architecture/extract`, `PUT /ai-architecture/cleanliness`, `PUT /ai-architecture/vllm`, `POST /ai-architecture/vllm/test` |
| Settings | `GET /settings/rules`, `PUT /settings/rules/{ruleId}/toggle`, `POST /settings/rules/set-all`, `POST /settings/rules/reset`, `POST /settings/save`, `GET\|PUT /settings/general` |
| Dashboard | `GET /dashboard/kpis`, `GET /dashboard/charts`, `GET /dashboard/recent-statements` |
| System | `GET /health`, `POST /reset` |

### Notable behavior

- **`POST /pipeline/uploads`** accepts a real multipart file. `services/file_analysis.py` scans the actual bytes for a cleanliness score (missing-cell ratio for CSV/JSON, Shannon entropy for binary formats like PDF), and `services/statement_parser.py` extracts real transactions — CSV/TSV/TXT parsed by column, PDF parsed via `pypdf` text extraction plus regex heuristics (date-prefixed lines, amount+balance pairs, debit/credit inferred from balance direction).
- **`POST /inference/run`** uses that real parsed statement (via `sourceId`) when one is available for the selected source — real transactions, and a credit score/feature vector computed deterministically from real ratios (DSCR, cash withdrawal ratio, income stability, NACH bounces detected by narration keywords) rather than randomly generated. Falls back to a `customId`-seeded synthetic profile otherwise. The response's `dataSource` field (`"UPLOADED_STATEMENT"` vs `"SIMULATED"`) tells you which path was used.
- **`POST /pipeline/run`** computes noise% from the real cleanliness of each selected source's uploaded file, and activates the "LLM cleaning" flag above the same 40%-noise / 60%-cleanliness threshold the platform documents.
- Each inference run is recorded into an in-memory history that feeds `GET /dashboard/recent-statements` and nudges the `analyzed` KPI.
