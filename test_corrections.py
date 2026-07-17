"""Standalone tests for the deterministic correction logic (no LLM calls)."""
import sys
import types

sys.path.insert(0, ".")

# --- Stub the heavy external deps so we can import graph.nodes standalone ---
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _Dummy:  # stand-in for message/tool classes
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, _):
        return _Dummy()

    def __call__(self, *a, **k):
        return _Dummy()


class _Msg:  # message stub that actually stores .content
    def __init__(self, content="", *a, **k):
        self.content = content


class _BaseModel:
    pass


_stub("langchain_core")
_stub("langchain_core.messages", AIMessage=_Msg, HumanMessage=_Msg)
_stub("langgraph")
_stub("langgraph.types", interrupt=lambda *a, **k: None)
_stub("langgraph.graph")
_stub("langgraph.graph.message", add_messages=lambda *a, **k: None)
_stub("pydantic", BaseModel=_BaseModel)
_stub("llm")
_stub("llm.model", get_shared_llm=lambda *a, **k: _Dummy())
_stub("tools")
_stub("tools.booking_tool", create_booking_tool=_Dummy())
_stub("tools.database_tool", fetch_previous_bookings_tool=_Dummy(), save_booking_tool=_Dummy())
_stub("tools.email_tool", send_booking_email_tool=_Dummy())
_stub("tools.tavily_tool", search_travel_options_tool=_Dummy())

from graph.nodes import (
    _resolve_correction,
    _detect_name_correction,
    _looks_like_direct_answer,
    _correction_ack,
)

PASS, FAIL = 0, 0


def check(label, got, expected):
    global PASS, FAIL
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}\n       got={got!r} expected={expected!r}")
    PASS += ok
    FAIL += not ok


# --- The headline bug: "can u change that to vijay" while budget was pending ---
state = {"last_answered_field": "traveller_names"}
check(
    "pronoun correction -> last answered field (varun->vijay)",
    _resolve_correction("can u change that to vijay", state),
    ("traveller_names", ["Vijay"]),
)

check(
    "correction phrase is not a direct answer",
    _looks_like_direct_answer("can u change that to vijay"),
    False,
)

# --- Keyworded corrections ---
check("change the budget to 3000", _resolve_correction("change the budget to 3000", {}), ("budget", 3000.0))
check("change travellers to 1", _resolve_correction("change travellers to 1", {}), ("num_travellers", 1))
check(
    "make it one-way (pronoun -> return_trip)",
    _resolve_correction("make it one-way", {"last_answered_field": "return_trip"}),
    ("return_trip", False),
)
check(
    "actually make it a round trip",
    _resolve_correction("actually make it a round trip", {"last_answered_field": "return_trip"}),
    ("return_trip", True),
)

# --- Name correction 'X not Y' shape ---
check("his name is Sharan not Ramu", _detect_name_correction("his name is Sharan not Ramu"), ["Sharan"])

# --- Non-corrections must return None ---
check("plain budget answer '5000'", _resolve_correction("5000", {}), None)
check("plain name answer 'varun'", _resolve_correction("varun", {}), None)
check("route message", _resolve_correction("banglore to mumbai", {}), None)
check("pronoun with no answer history", _resolve_correction("change that to vijay", {}), None)

# --- Acknowledgement text ---
check("ack for traveller names", _correction_ack("traveller_names", ["Vijay"]), "Got it — updated traveller name(s) to Vijay.")
check("ack for budget", _correction_ack("budget", 3000.0), "Got it — updated budget to INR 3000.")

# --- Integration: full extract_info_node with the LLM contributing nothing ---
from graph.nodes import extract_info_node
from langchain_core.messages import HumanMessage


def run_extract(text, state):
    st = dict(state)
    st["messages"] = [HumanMessage(text)]
    return extract_info_node(st)


# Headline scenario: budget was pending, user corrects the traveller name.
upd = run_extract(
    "can u change that to vijay",
    {
        "traveller_names": ["Varun"],
        "num_travellers": 2,
        "last_answered_field": "traveller_names",
        "last_asked_field": "budget",
    },
)
check("extract: name corrected to Vijay", upd.get("traveller_names"), ["Vijay"])
check("extract: budget NOT set by correction", upd.get("budget"), None)
check("extract: acknowledgement emitted", upd.get("correction_ack"), "Got it — updated traveller name(s) to Vijay.")

# Direct answer still works (budget pending, user gives a number).
upd = run_extract("5000", {"last_asked_field": "budget"})
check("extract: plain budget answer", upd.get("budget"), 5000.0)
check("extract: records last_answered_field", upd.get("last_answered_field"), "budget")

# Post-booking tweak: retained route, change budget + pick a train.
upd = run_extract(
    "change the budget to 3000 and help me find a train",
    {
        "source": "Banglore", "destination": "Cochin", "travel_date": "2026-07-21",
        "num_travellers": 2, "budget": 5000.0, "transport_pref": "flight",
        "return_trip": False, "traveller_names": ["Vijay"], "booking_complete": True,
    },
)
check("extract: budget tweaked post-booking", upd.get("budget"), 3000.0)
check("extract: transport switched to train", upd.get("transport_pref"), "train")
check("extract: route retained (no reset)", upd.get("source"), None)  # unchanged => not in update
check("extract: booking_complete cleared", upd.get("booking_complete"), False)

# Post-booking new route starts a fresh trip (clears stale date/budget).
upd = run_extract(
    "banglore to mumbai",
    {
        "source": "Banglore", "destination": "Cochin", "travel_date": "2026-07-21",
        "num_travellers": 2, "budget": 3000.0, "transport_pref": "train",
        "booking_complete": True,
    },
)
check("extract: new destination", upd.get("destination"), "Mumbai")
check("extract: stale date wiped for fresh trip", upd.get("travel_date"), None)
check("extract: stale budget wiped for fresh trip", upd.get("budget"), None)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
