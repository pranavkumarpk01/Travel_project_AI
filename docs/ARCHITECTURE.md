# Architecture

## 1. System architecture

```mermaid
flowchart LR
    subgraph Client
        UI[Streamlit UI]
    end

    subgraph Backend
        API[FastAPI app.py]
        GRAPH[LangGraph StateGraph]
        CKPT[(SQLite\ncheckpoints.sqlite)]
    end

    subgraph External
        LLM[Ollama / Gemini]
        TAVILY[Tavily Search API]
        SMTP[Gmail SMTP]
        DB[(SQLite / Postgres\nbookings table)]
    end

    UI -- HTTP /chat, /chat/resume --> API
    API --> GRAPH
    GRAPH <-- read/write state --> CKPT
    GRAPH -- prompts --> LLM
    GRAPH -- search --> TAVILY
    GRAPH -- save/fetch --> DB
    GRAPH -- send --> SMTP
    UI -- GET /bookings/email --> API
```

## 2. LangGraph workflow

```mermaid
flowchart TD
    START([START]) --> INTENT[detect_intent]

    INTENT -- new_trip --> EXTRACT[extract_info]
    INTENT -- show_bookings --> FETCH[fetch_previous_bookings]
    INTENT -- chit_chat --> CHAT[chit_chat]

    EXTRACT --> MISSING[check_missing_info]
    MISSING -- fields missing --> ASK[ask_questions]
    MISSING -- all collected --> SEARCH[search_travel_options]

    ASK --> ENDA([END - wait for next user message])

    SEARCH --> COMPARE[compare_options]
    COMPARE --> RECOMMEND[recommend]
    RECOMMEND --> APPROVAL{{human_approval\ninterrupt()}}

    APPROVAL -- approved --> BOOK[create_booking]
    APPROVAL -- rejected --> CANCEL[cancelled]

    BOOK --> STORE[store_booking]
    STORE --> EMAIL[send_email]
    EMAIL --> CONFIRM[show_confirmation]
    CONFIRM --> ENDB([END])
    CANCEL --> ENDC([END])

    FETCH --> ENDD([END])
    CHAT --> ENDE([END])
```

## 3. Sequence diagram — full booking flow with Human-in-the-Loop

```mermaid
sequenceDiagram
    participant U as User (Streamlit)
    participant A as FastAPI
    participant G as LangGraph
    participant L as LLM
    participant T as Tavily/Mock
    participant D as Database
    participant M as Gmail SMTP

    U->>A: POST /chat "Bangalore to Goa"
    A->>G: invoke(state, thread_id)
    G->>L: detect_intent + extract_info
    L-->>G: intent=new_trip, source/destination filled
    G-->>A: missing_fields -> ask_questions
    A-->>U: "What date would you like to travel?"

    U->>A: POST /chat "20 July, 2 travellers, budget 6000..."
    A->>G: invoke(state, thread_id)
    G->>L: extract_info (merge into state)
    G->>G: check_missing_info -> none missing
    G->>T: search_travel_options(source, destination, date)
    T-->>G: flight/train/bus options
    G->>L: compare_options
    G->>G: recommend
    G->>G: human_approval -> interrupt()
    G-->>A: status=interrupted, booking summary
    A-->>U: Booking Summary card (Yes/No)

    U->>A: POST /chat/resume approved=true
    A->>G: invoke(Command(resume=true), thread_id)
    G->>G: booking node creates booking record
    G->>D: save_booking_tool(booking)
    D-->>G: persisted booking (with id)
    G->>M: send_booking_email_tool(booking)
    M-->>G: sent / printed to console
    G-->>A: status=ok, confirmation message
    A-->>U: "Booking confirmed! ID: TRV-XXXX..."
```

## Design notes

- **State** (`graph/state.py`) is a single `TypedDict`. `messages` uses the
  `add_messages` reducer so chat history appends instead of overwriting;
  every other field is a plain overwrite, which is enough for this app and
  easy to explain.
- **Memory = Checkpointing.** There's no separate "memory store" — the
  `SqliteSaver` checkpointer *is* the memory. Every node's return value is
  merged into state and persisted to `checkpoints/checkpoints.sqlite`,
  keyed by `thread_id`. Restoring memory and resuming after a crash are the
  same mechanism.
- **Human-in-the-loop** uses LangGraph's native `interrupt()` (see
  `graph/nodes.py::human_approval_node`). The graph execution genuinely
  pauses; `app.get_state(config).next` is non-empty until a caller resumes
  it with `Command(resume=...)`.
- **Tool calling** is deliberately explicit (`some_tool.invoke({...})`)
  rather than hidden behind an LLM tool-calling loop, so students can see
  exactly which node calls which tool with which arguments.
