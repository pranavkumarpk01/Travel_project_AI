# AI Travel Planner — LangGraph Teaching Project

A small, production-inspired AI travel assistant built to **teach LangGraph**.
It's a chat app that gathers trip requirements, searches real/mock travel
options, compares flight/train/bus, recommends the best one, pauses for
human approval before booking (Human-in-the-Loop), books it, stores it in a
database, and emails a confirmation.

Every LangGraph concept in the brief — StateGraph, nodes, edges, conditional
routing, state, memory, checkpointing, interrupts/resume, and tool calling —
is implemented in a small, readable way. See `docs/TEACHING_SCRIPT.md` for a
full classroom walkthrough and `docs/ARCHITECTURE.md` for diagrams.

## Folder structure

```
travel-planner/
├── app.py                  # FastAPI backend (thin HTTP layer over the graph)
├── streamlit_app.py        # Minimal Streamlit chat UI
├── graph/
│   ├── state.py             # TravelState TypedDict + required fields
│   ├── nodes.py             # One function per graph node
│   └── graph.py             # StateGraph wiring + checkpointer + run/resume helpers
├── llm/
│   └── model.py             # get_llm() — swaps between Ollama and Gemini
├── tools/
│   ├── tavily_tool.py       # Real (or mock) flight/train/bus search
│   ├── booking_tool.py      # Simulated booking creation
│   ├── email_tool.py        # Gmail SMTP confirmation email
│   └── database_tool.py     # Save / fetch bookings (SQLAlchemy)
├── database/
│   ├── models.py             # Booking table
│   └── db.py                 # Engine/session (SQLite by default, Postgres via Docker)
├── checkpoints/              # SqliteSaver checkpoint DB lives here
├── memory/                   # Notes on how LangGraph memory works in this app
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   └── Dockerfile.streamlit
├── docs/
│   ├── ARCHITECTURE.md       # Architecture, workflow, and sequence diagrams
│   └── TEACHING_SCRIPT.md    # Full classroom script
├── requirements.txt
└── .env.example
```

## Quick start (no Docker)

Requires Python 3.12 and [Ollama](https://ollama.com) running locally.

```bash
# 1. Pull the default model
ollama pull llama3.2:3b

# 2. Install dependencies
cd travel-planner
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env if you want Tavily search or Gmail SMTP — both are optional,
# the app falls back to mock data / console-printed emails otherwise

# 4. Run the backend
uvicorn app:app --reload --port 8000

# 5. In a second terminal, run the UI
streamlit run streamlit_app.py
```

Open http://localhost:8501 and try:

> "I want to travel from Bangalore to Goa"

## Quick start (Docker)

Runs the API + Streamlit UI + Postgres. Ollama still runs on your host
machine (Docker reaches it via `host.docker.internal`).

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml --env-file .env up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501
- Postgres: localhost:5432 (`travel_user` / `travel_pass` / `travel_planner`)

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_PROVIDER` | `ollama` or `gemini` | `ollama` |
| `MODEL_NAME` | Model id for the chosen provider | `llama3.2:3b` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `GOOGLE_API_KEY` | Required if `MODEL_PROVIDER=gemini` | — |
| `TAVILY_API_KEY` | Enables live travel search; falls back to mock data if blank | — |
| `DATABASE_URL` | SQLAlchemy URL (SQLite by default, Postgres in Docker) | `sqlite:///./database/travel_planner.db` |
| `SMTP_HOST` / `SMTP_PORT` | Gmail SMTP endpoint | `smtp.gmail.com` / `587` |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | Gmail address + [App Password](https://myaccount.google.com/apppasswords). Blank = email is printed to console instead of sent | — |
| `API_BASE_URL` | Where Streamlit finds the FastAPI backend | `http://localhost:8000` |
| `CHECKPOINT_DB_PATH` | Where LangGraph's SqliteSaver stores checkpoints | `./checkpoints/checkpoints.sqlite` |

### Switching to Gemini

```
MODEL_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
GOOGLE_API_KEY=your-key-here
```

No code changes needed — `llm/model.py` reads these at runtime.

## Demo walkthrough

1. **"I want to travel from Bangalore to Goa"** → intent detection routes to
   the trip-planning branch; info extraction pulls `source`/`destination`;
   missing-info check finds the rest are empty → assistant asks for travel
   date.
2. Answer each follow-up question (date, travellers, budget, transport
   preference, round trip?, hotel?). Watch how earlier answers are never
   re-asked — that's the checkpointer acting as memory.
3. Once complete, the graph searches travel options, compares them, and
   posts a recommendation.
4. The graph **pauses** (`interrupt()`) and the UI shows a Booking Summary
   card with Yes/No buttons — this is Human-in-the-Loop.
5. Click **Yes** → booking is created, saved to the database, a
   confirmation email is sent (or printed to console), and a confirmation
   message appears.
6. Type **"show my previous bookings"** → a separate graph branch reads
   straight from the database.
7. Restart the FastAPI process mid-conversation and send another message
   with the same `thread_id` — the graph resumes exactly where it left off.
   This demonstrates checkpointing/persistence.

## Notes for teaching

See `docs/TEACHING_SCRIPT.md` for a full script (introduction → LangGraph
fundamentals → live demo) and `docs/ARCHITECTURE.md` for the architecture,
LangGraph workflow, and sequence diagrams referenced throughout the script.
