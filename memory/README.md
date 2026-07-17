# Memory in this project

This folder is a placeholder to make the concept concrete for the classroom, but the
actual memory mechanism lives in LangGraph's **checkpointer**
(`checkpoints/checkpoints.sqlite`).

Every graph turn is saved against a `thread_id` (one per user conversation). Because the
full `State` — including `budget`, `source`, `destination`, extracted preferences, and the
message history — is written to that checkpoint after every node, the graph can:

1. **Recall facts across turns** (conversational memory) — e.g. remember the budget the
   user mentioned three messages ago without re-asking.
2. **Resume an interrupted conversation** (checkpointing) — e.g. if the app restarts mid
   booking, replaying the same `thread_id` continues exactly where it left off, including
   paused `interrupt()` calls waiting on human approval.

See `docs/TEACHING_SCRIPT.md` for how to demo both live in class.
