import os
import sqlite3

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from graph import nodes
from graph.state import TravelState

load_dotenv()

CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "./checkpoints/checkpoints.sqlite")


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("detect_intent", nodes.detect_intent_node)
    graph.add_node("extract_info", nodes.extract_info_node)
    graph.add_node("check_missing_info", nodes.check_missing_info_node)
    graph.add_node("ask_questions", nodes.ask_questions_node)
    graph.add_node("search_travel_options", nodes.search_travel_options_node)
    graph.add_node("compare_options", nodes.compare_options_node)
    graph.add_node("recommend", nodes.recommend_node)
    graph.add_node("human_approval", nodes.human_approval_node)
    # Node id is "create_booking" (not "booking") because State already has a
    # "booking" field, and LangGraph doesn't allow a node name to match a
    # state key.
    graph.add_node("create_booking", nodes.booking_node)
    graph.add_node("cancelled", nodes.cancelled_node)
    graph.add_node("store_booking", nodes.store_booking_node)
    graph.add_node("send_email", nodes.send_email_node)
    graph.add_node("show_confirmation", nodes.show_confirmation_node)
    graph.add_node("fetch_previous_bookings", nodes.fetch_previous_bookings_node)
    graph.add_node("chit_chat", nodes.chit_chat_node)

    graph.add_edge(START, "detect_intent")

    graph.add_conditional_edges(
        "detect_intent",
        nodes.route_after_intent,
        {
            "extract_info": "extract_info",
            "fetch_previous_bookings": "fetch_previous_bookings",
            "chit_chat": "chit_chat",
        },
    )

    graph.add_edge("extract_info", "check_missing_info")

    graph.add_conditional_edges(
        "check_missing_info",
        nodes.route_after_missing_check,
        {
            "ask_questions": "ask_questions",
            "search_travel_options": "search_travel_options",
        },
    )

    graph.add_edge("ask_questions", END)

    graph.add_edge("search_travel_options", "compare_options")
    graph.add_edge("compare_options", "recommend")
    graph.add_edge("recommend", "human_approval")

    graph.add_conditional_edges(
        "human_approval",
        nodes.route_after_approval,
        {
            "booking": "create_booking",
            "cancelled": "cancelled",
        },
    )

    graph.add_edge("create_booking", "store_booking")
    graph.add_edge("store_booking", "send_email")
    graph.add_edge("send_email", "show_confirmation")
    graph.add_edge("show_confirmation", END)
    graph.add_edge("cancelled", END)

    graph.add_edge("fetch_previous_bookings", END)
    graph.add_edge("chit_chat", END)

    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)


compiled_graph = None


def get_graph():
    global compiled_graph
    if compiled_graph is None:
        compiled_graph = build_graph()
    return compiled_graph


def run_turn(thread_id: str, user_message: str, user_name: str = None, user_email: str = None) -> dict:
    from langchain_core.messages import HumanMessage

    app = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {"messages": [HumanMessage(content=user_message)]}
    if user_name:
        input_state["user_name"] = user_name
    if user_email:
        input_state["user_email"] = user_email

    result = app.invoke(input_state, config=config)
    return _format_result(app, config, result)


def resume_turn(thread_id: str, approved: bool) -> dict:
    app = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(Command(resume=approved), config=config)
    return _format_result(app, config, result)


def _format_result(app, config, result) -> dict:
    state_snapshot = app.get_state(config)
    if state_snapshot.next:
        for task in state_snapshot.tasks:
            if task.interrupts:
                return {
                    "status": "interrupted",
                    "interrupt": task.interrupts[0].value,
                    "thread_id": config["configurable"]["thread_id"],
                }

    messages = result.get("messages", [])
    last_ai_messages = [m.content for m in messages if m.__class__.__name__ == "AIMessage"]
    return {
        "status": "ok",
        "reply": last_ai_messages[-1] if last_ai_messages else "",
        "state": {k: v for k, v in result.items() if k != "messages"},
        "thread_id": config["configurable"]["thread_id"],
    }
