# Teaching Script — AI Travel Planner with LangGraph

A classroom-ready script for walking through this project start to finish.
Written for an audience that knows Python and has used an LLM API before,
but has never used LangGraph. Total runtime: roughly 60–90 minutes including
the live demo.

---

## 1. Introduction (5 min)

> "Today we're building an AI travel assistant — the kind of thing you'd see
> as a chatbot on a travel site. You tell it where you want to go, it asks
> follow-up questions, searches for flights/trains/buses, recommends one,
> asks you to confirm before booking, books it, and emails you a
> confirmation.
>
> The interesting part isn't the travel logic — it's *how we structure the
> agent's decision-making*. We're going to use **LangGraph**, a library from
> the LangChain team for building agents as explicit graphs instead of one
> giant prompt or a tangle of if/else statements."

Ask the class: *"What's hard about building a multi-turn assistant with a
plain while-loop and if/else logic?"* Let a few answers land (state gets
messy, hard to pause/resume, hard to visualize, hard to test one step at a
time) before revealing that LangGraph solves exactly these problems.

## 2. Project overview (5 min)

Show `docs/ARCHITECTURE.md` diagram 1 (system architecture). Walk through
the boxes: Streamlit UI → FastAPI → LangGraph → (LLM, Tavily, DB, SMTP).

> "FastAPI and Streamlit are just plumbing. The actual intelligence — the
> thing we're here to learn — lives entirely in the `graph/` folder."

## 3. Folder structure (5 min)

Open the repo tree from `README.md`. Emphasize the mapping:

- `graph/state.py` → **State** definition
- `graph/nodes.py` → **Nodes** (one function each)
- `graph/graph.py` → **Edges** + graph compilation
- `llm/model.py` → swappable LLM provider (not LangGraph-specific, just good
  practice)
- `tools/` → **Tool calling** targets
- `database/` → persistence, used by one of the tools

> "If you can explain what each of these five folders does, you basically
> understand this codebase."

## 4. LangGraph fundamentals

### 4.1 State (10 min)

Open `graph/state.py`.

> "Every LangGraph app starts with one question: what data flows through my
> agent? We define it once, as a `TypedDict`, and every node reads and
> writes to it."

Point out:
- `messages: Annotated[list, add_messages]` — the **reducer** pattern. Most
  fields get *overwritten* by a node's return value; `messages` gets
  *appended to* because of the `add_messages` reducer. This is the first
  "aha" moment for students — LangGraph doesn't just merge dicts naively,
  each field can have custom merge behaviour.
- The trip fields (`source`, `budget`, etc.) are just plain optional values.
  Nothing fancy — this is intentionally boring so the graph logic itself
  stays interesting.

### 4.2 Nodes (10 min)

Open `graph/nodes.py`. Pick 3 nodes to read together:

1. `check_missing_info_node` — pure Python, no LLM, no I/O. Show that nodes
   don't have to call an LLM at all.
2. `extract_info_node` — calls `llm.with_structured_output(TripInfo)`. This
   is "structured output" — a lightweight form of tool calling where the
   LLM's job is just to fill in a Pydantic schema.
3. `search_travel_options_node` — calls `search_travel_options_tool.invoke(...)`.
   This is explicit tool calling: a plain Python function decorated with
   `@tool`, invoked directly.

> "Every node has the same shape: `def node(state) -> dict`. It reads
> whatever fields it needs from `state`, does its one job, and returns only
> the fields it changed. That's the entire mental model."

### 4.3 Edges & conditional routing (10 min)

Open `graph/graph.py`. Trace one conditional edge live:

```python
graph.add_conditional_edges(
    "check_missing_info",
    nodes.route_after_missing_check,
    {"ask_questions": "ask_questions", "search_travel_options": "search_travel_options"},
)
```

Then show `route_after_missing_check` in `nodes.py` — it's a plain function
returning a string. LangGraph calls it after `check_missing_info` runs and
uses the returned string to decide which node runs next.

> "A normal edge (`add_edge`) always goes to the same next node. A
> conditional edge asks a routing function 'where do I go next?' after
> every run. That's how we get branching logic without any if/else soup in
> the main flow."

Show diagram 2 (LangGraph workflow) side-by-side with the code — every arrow
on the diagram is one line of `add_edge`/`add_conditional_edges`.

### 4.4 START and END (2 min)

> "`START` and `END` are LangGraph's built-in sentinel nodes.
> `graph.add_edge(START, "detect_intent")` says 'this is where every
> invocation begins.' Several nodes point to `END` — that's a normal turn
> finishing. Notice `ask_questions` also points to `END`: the graph doesn't
> loop internally waiting for the next answer, it just *ends the turn*.
> The next user message starts a brand-new invocation — but thanks to the
> checkpointer, it picks up the same state."

### 4.5 Memory (10 min)

> "Ask yourself: when the user says 'my budget is ₹5000' and three
> messages later asks 'what transport should I choose?' — how does the
> assistant remember the budget?"

Open `graph/graph.py`, the `build_graph()` checkpointer section:

```python
checkpointer = SqliteSaver(conn)
return graph.compile(checkpointer=checkpointer)
```

> "That's it. That one line gives us memory. Every node's output gets
> written to this SQLite file, keyed by `thread_id`. On the next call,
> LangGraph loads the last checkpoint for that thread_id *before* running
> any node. Nothing about our node functions needs to know memory exists —
> it's transparent."

**Live demo moment:** open the Streamlit "Memory Demo" page and narrate the
budget example live.

### 4.6 Checkpointing & persistence (10 min)

> "Memory and checkpointing are the same mechanism here, which is worth
> calling out explicitly: memory is *recalling facts across turns*,
> checkpointing is *being able to resume a stopped process*. Both come from
> the same SqliteSaver."

**Live demo moment:**
1. Start a conversation, answer 2-3 questions.
2. Kill the `uvicorn` process (Ctrl+C).
3. Restart it (`uvicorn app:app --reload --port 8000`).
4. Send another message with the same `thread_id` from Streamlit.
5. Show that the assistant still remembers everything and continues asking
   the next missing field — it didn't restart the conversation.

Open `checkpoints/checkpoints.sqlite` with any SQLite browser (or
`sqlite3 checkpoints/checkpoints.sqlite ".tables"`) to show students it's
just a real database file they can inspect.

### 4.7 Human-in-the-loop & interrupts (15 min — the highlight)

Open `graph/nodes.py::human_approval_node`:

```python
decision = interrupt(summary)
```

> "`interrupt()` is one of LangGraph's most powerful primitives. When a
> node calls it, the *entire graph execution pauses right there* — not just
> logically, but literally: `app.invoke()` returns control to our FastAPI
> code immediately, with no answer yet."

Open `graph/graph.py::_format_result`:

```python
state_snapshot = app.get_state(config)
if state_snapshot.next:
    ...
```

> "`app.get_state(config).next` tells us the graph didn't reach END — it's
> paused, waiting. We surface that to the UI as a Booking Summary card with
> Yes/No buttons."

Open `graph/graph.py::resume_turn`:

```python
result = app.invoke(Command(resume=approved), config=config)
```

> "`Command(resume=...)` is how we un-pause the graph. Whatever value we
> pass becomes the return value of the `interrupt()` call inside
> `human_approval_node` — as if the function had just been waiting on
> `input()`. Execution then continues to `booking` or `cancelled` depending
> on `route_after_approval`."

**Live demo moment:** complete a full booking flow in the Streamlit UI,
pausing on the approval card to ask the class "what do you think happens
if I click No instead?" before clicking Yes.

### 4.8 Tool calling (10 min)

Open `tools/tavily_tool.py`, `tools/booking_tool.py`,
`tools/database_tool.py`, `tools/email_tool.py`.

> "A LangChain 'tool' is just a Python function with `@tool` and a
> docstring. The docstring matters — if we ever let the LLM choose which
> tool to call (`llm.bind_tools([...])`), the LLM reads these docstrings to
> decide. In this project we call tools explicitly from nodes for
> clarity, but the exact same `@tool` functions would work in an
> LLM-driven tool-calling loop too."

Show the fallback pattern in `tavily_tool.py` — if `TAVILY_API_KEY` is
blank, it returns deterministic mock data instead of failing. Useful
lesson: **always design for the "no API key" classroom scenario.**

## 5. Database integration (5 min)

Open `database/models.py` and `database/db.py`.

> "One SQLAlchemy model, one table: `bookings`. LangGraph doesn't care what
> your storage layer is — `store_booking_node` just calls
> `save_booking_tool.invoke(...)`, and that tool happens to use SQLAlchemy.
> Swap SQLite for Postgres by changing one environment variable —
> `DATABASE_URL` — no code changes."

## 6. API integration (5 min)

Revisit `tools/tavily_tool.py`'s live-search branch. Explain Tavily as a
search API built for LLM agents (returns clean text snippets instead of raw
HTML). Mention the brief's other optional APIs (Google Maps, IRCTC/OpenRail)
as extension exercises for advanced students.

## 7. Email notifications (5 min)

Open `tools/email_tool.py`. Point out the console fallback again — same
philosophy as the Tavily fallback: **never let a missing API key break the
live demo.**

## 8. Running the project (5 min)

Walk through `README.md`'s Quick Start section live: `ollama pull`, `pip
install -r requirements.txt`, `cp .env.example .env`, `uvicorn app:app
--reload`, `streamlit run streamlit_app.py`.

## 9. Live demo flow (10–15 min)

Follow the "Demo walkthrough" section of `README.md` end-to-end in front of
the class:

1. "I want to travel from Bangalore to Goa" → watch intent detection +
   missing-info questions.
2. Answer each question — call out that no field is ever asked twice.
3. See the comparison + recommendation.
4. Approve the booking on the interrupt card.
5. Show the confirmation message, the console/inbox email, and the row in
   the `bookings` table (`sqlite3 database/travel_planner.db "select * from
   bookings;"` or a Postgres client if running via Docker).
6. Ask "show my previous bookings" to demonstrate the second graph branch.
7. Restart the API mid-conversation to demonstrate checkpointing one more
   time, this time with the class predicting what will happen first.

## 10. Wrap-up discussion prompts

- "Which part of this would be hardest to build without LangGraph?"
  (Usually: pausing/resuming mid-conversation, i.e. Human-in-the-Loop.)
- "What would you add as a new node?" (e.g. a cancellation-policy check, a
  loyalty-points lookup, a multi-city itinerary loop.)
- "Where would you add a loop (a node whose conditional edge can point back
  to itself or an earlier node)?" — good bridge to more advanced LangGraph
  patterns like agentic retry loops, which this project deliberately keeps
  out of scope.

---

### Appendix: concept → code map

| LangGraph concept | Where to find it |
|---|---|
| StateGraph | `graph/graph.py: StateGraph(TravelState)` |
| State | `graph/state.py: TravelState` |
| Nodes | `graph/nodes.py`, registered in `graph/graph.py` via `add_node` |
| Edges | `graph/graph.py: add_edge(...)` |
| Conditional edges | `graph/graph.py: add_conditional_edges(...)` |
| START / END | `graph/graph.py` (imported from `langgraph.graph`) |
| Tool calling | `tools/*.py` (`@tool`), invoked in `graph/nodes.py` |
| Memory | `SqliteSaver` checkpointer in `graph/graph.py` |
| Checkpointing / Persistence | Same `SqliteSaver`, file at `checkpoints/checkpoints.sqlite` |
| Human-in-the-Loop | `graph/nodes.py: human_approval_node` (`interrupt()`) |
| Resume | `graph/graph.py: resume_turn` (`Command(resume=...)`) |
