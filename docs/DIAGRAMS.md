# Travel Planner — Diagrams

Copy any block below into [mermaid.live](https://mermaid.live), Excalidraw, Notion,
or a Mermaid-aware whiteboard. All diagrams reflect the current code in
`graph/graph.py`, `graph/nodes.py`, and `graph/state.py`.

---

## 1. Graph flow (LangGraph wiring)

```mermaid
flowchart TD
    START([START]) --> DI[detect_intent]

    DI -->|new_trip| EI[extract_info]
    DI -->|show_bookings| FPB[fetch_previous_bookings]
    DI -->|chit_chat| CC[chit_chat]

    EI --> CMI[check_missing_info]
    CMI -->|missing fields| AQ[ask_questions]
    CMI -->|all collected| STO[search_travel_options]
    AQ --> E1([END])

    STO --> CO[compare_options]
    CO --> REC[recommend]
    REC --> HA{{human_approval\n⏸ interrupt}}

    HA -->|approved| CB[create_booking]
    HA -->|rejected| CAN[cancelled]

    CB --> SB[store_booking]
    SB --> SE[send_email]
    SE --> SC[show_confirmation]

    SC --> E2([END])
    CAN --> E3([END])
    FPB --> E4([END])
    CC --> E5([END])
```

---

## 2. Conversation lifecycle (state machine)

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Collecting: new_trip intent
    Idle --> ShowBookings: show_bookings
    Idle --> ChitChat: chit_chat

    Collecting --> Collecting: ask a question /\nuser answers OR corrects a slot
    Collecting --> Searching: all required fields present

    Searching --> AwaitingApproval: recommendation ready
    AwaitingApproval --> Booked: user says yes
    AwaitingApproval --> Idle: user says no\n(cancelled, fields reset)

    Booked --> TweakOrNew: booking_complete = true\n(trip kept in memory)
    TweakOrNew --> Searching: tweak same route\n(change budget / transport)
    TweakOrNew --> Collecting: brand-new route\n(fresh trip, stale fields cleared)

    ShowBookings --> Idle
    ChitChat --> Idle
```

---

## 3. State shape (TravelState)

```mermaid
classDiagram
    class TravelState {
      +list messages
      +str user_name
      +str user_email
      +str intent
      %% --- trip slots ---
      +str source
      +str destination
      +str travel_date
      +str return_date
      +bool return_trip
      +int num_travellers
      +float budget
      +str transport_pref
      +bool hotel_required
      +bool flexible_dates
      +str seat_pref
      +list traveller_names
      %% --- memory / control ---
      +list missing_fields
      +str last_asked_field
      +str last_answered_field
      +bool booking_complete
      +str correction_ack
      %% --- results ---
      +dict search_results
      +dict comparison
      +dict recommendation
      +bool approved
      +dict booking
      +list previous_bookings
    }
```

---

## 4. Node responsibilities

```mermaid
flowchart LR
    subgraph Understand
        A[detect_intent\nnew_trip / show_bookings / chit_chat]
        B[extract_info\nslot-fill + resolve corrections]
        C[check_missing_info\nwhich required fields remain]
    end
    subgraph Collect
        D[ask_questions\nask next slot + correction ack]
    end
    subgraph Plan
        E[search_travel_options\nTavily / mock provider]
        F[compare_options\nLLM price/time summary]
        G[recommend\npick best within budget]
    end
    subgraph Commit
        H[human_approval\npause for yes/no]
        I[create_booking]
        J[store_booking\nSQLite]
        K[send_email\nSMTP]
        L[show_confirmation\nkeep trip in memory]
    end
    subgraph SideFlows
        M[fetch_previous_bookings]
        N[chit_chat]
    end

    A --> B --> C --> D
    C --> E --> F --> G --> H --> I --> J --> K --> L
```

| Node | Responsibility |
|------|----------------|
| `detect_intent` | Classify message: new_trip / show_bookings / chit_chat |
| `extract_info` | Fill slots from the message; resolve corrections (incl. "change that to X") |
| `check_missing_info` | Compute `missing_fields` (adds `traveller_names` when >1 traveller) |
| `ask_questions` | Ask the next missing field; prepend correction acknowledgement |
| `search_travel_options` | Fetch flight/train/bus options (Tavily or mock) |
| `compare_options` | LLM summary of price, duration, convenience |
| `recommend` | Choose cheapest in-budget option, honoring transport preference |
| `human_approval` | `interrupt()` — pause until user approves/rejects |
| `create_booking` → `store_booking` → `send_email` → `show_confirmation` | Book, persist, email, confirm — then retain trip in memory |
| `cancelled` | Reset trip fields, end politely |
| `fetch_previous_bookings` | Read bookings from DB (no LLM) |
| `chit_chat` | Friendly reply nudging toward planning |

---

## 5. Memory & correction sub-flow (the fix)

```mermaid
flowchart TD
    U[User message] --> LLM[LLM structured extraction\nbest-effort]
    U --> DET[Deterministic layer]

    DET --> RC{correction?\nverb / marker / pronoun}
    RC -->|keyword: budget, travellers...| TF[target = that field]
    RC -->|pronoun: that / it / this| LA[target = last_answered_field]
    RC -->|no| PA[treat as answer to pending question]

    TF --> APPLY[apply value + set correction_ack]
    LA --> APPLY
    LLM -.merged, deterministic wins.-> APPLY
    PA --> SET[fill pending field]

    APPLY --> NEXT[re-ask pending field with ack]
    SET --> NEXT
```
