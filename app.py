import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from database.db import init_db
from graph.graph import resume_turn, run_turn
from tools.database_tool import fetch_previous_bookings_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_planner")

app = FastAPI(title="AI Travel Planner", version="1.0")


@app.on_event("startup")
def on_startup():
    init_db()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error("Unhandled error on %s:\n%s", request.url.path, tb)
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})


class ChatRequest(BaseModel):
    thread_id: str
    message: str
    user_name: str | None = None
    user_email: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    return run_turn(
        thread_id=req.thread_id,
        user_message=req.message,
        user_name=req.user_name,
        user_email=req.user_email,
    )


@app.post("/chat/resume")
def chat_resume(req: ResumeRequest):
    return resume_turn(thread_id=req.thread_id, approved=req.approved)


@app.get("/bookings/{email}")
def list_bookings(email: str):
    return fetch_previous_bookings_tool.invoke({"email": email})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
