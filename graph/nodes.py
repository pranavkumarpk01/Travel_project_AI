import difflib
import json
import re
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel

from graph.state import FIELD_QUESTIONS, REQUIRED_FIELDS, TravelState
from llm.model import get_shared_llm
from tools.booking_tool import create_booking_tool
from tools.database_tool import fetch_previous_bookings_tool, save_booking_tool
from tools.email_tool import send_booking_email_tool
from tools.tavily_tool import search_travel_options_tool


def _last_human_text(state: TravelState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _reset_trip_fields() -> dict:
    return {
        "source": None,
        "destination": None,
        "travel_date": None,
        "return_date": None,
        "return_trip": None,
        "num_travellers": None,
        "budget": None,
        "transport_pref": None,
        "hotel_required": None,
        "flexible_dates": None,
        "seat_pref": None,
        "traveller_names": None,
        "missing_fields": [],
        "search_results": None,
        "comparison": None,
        "recommendation": None,
        "approved": None,
        "last_asked_field": None,
    }


_ROUTE_PATTERN_FROM = re.compile(r"\bfrom\s+([a-zA-Z]+(?:\s[a-zA-Z]+)?)\s+to\s+([a-zA-Z]+(?:\s[a-zA-Z]+)?)\b", re.I)
_ROUTE_PATTERN_BARE = re.compile(r"^\s*([a-zA-Z]+(?:\s[a-zA-Z]+)?)\s+to\s+([a-zA-Z]+(?:\s[a-zA-Z]+)?)\s*$", re.I)


def _extract_route(text: str):
    m = _ROUTE_PATTERN_FROM.search(text) or _ROUTE_PATTERN_BARE.match(text)
    if not m:
        return None, None
    return m.group(1).strip().title(), m.group(2).strip().title()


def _parse_traveller_names(text: str):
    parts = re.split(r",|\band\b|&", text, flags=re.I)
    names = [p.strip().title() for p in parts if p.strip()]
    return names or None


def _sanitize_extracted(update: dict) -> dict:
    clean = {}
    for key, value in update.items():
        if key == "traveller_names":
            if isinstance(value, list) and all(isinstance(n, str) for n in value):
                names = [n.strip().title() for n in value if n.strip()]
                if names:
                    clean[key] = names
            continue
        if not isinstance(value, (dict, list)):
            clean[key] = value
    return clean


_CORRECTION_MARKERS = (
    "change", "actually", "instead", "correct", "wait", "i mean", "meant", "sorry", "make it", "not "
)


def _looks_like_direct_answer(text: str) -> bool:
    lower = text.lower()
    if any(m in lower for m in _CORRECTION_MARKERS):
        return False
    return len(text.split()) <= 6


def _coerce_answer(field: str, text: str):
    text = text.strip()
    if not text:
        return None
    lower = text.lower()

    if field == "num_travellers":
        m = re.search(r"\d+", text)
        return int(m.group()) if m else None

    if field == "budget":
        m = re.search(r"[\d,]+(?:\.\d+)?", text)
        return float(m.group().replace(",", "")) if m else None

    if field in ("return_trip", "hotel_required", "flexible_dates"):
        negative = any(w in lower for w in ["no", "not", "one way", "n/a", "none", "don't", "dont"])
        positive = any(w in lower for w in ["yes", "yeah", "yep", "sure", "round trip", "need"])
        if positive and not negative:
            return True
        if negative and not positive:
            return False
        return None

    if field == "transport_pref":
        for opt in ("flight", "train", "bus", "any"):
            if opt in lower:
                return opt
        if "no preference" in lower or "doesn't matter" in lower or "either" in lower:
            return "any"
        return text

    if field in ("source", "destination"):
        return text.title()

    if field == "traveller_names":
        return _parse_traveller_names(text)

    return text


_DATE_HINT_RE = re.compile(
    r"\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?"
    r"|\b\d{1,2}(st|nd|rd|th)\b"
    r"|\b(19|20)\d{2}\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\btoday\b|\btomorrow\b|\btonight\b"
    r"|\bnext\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\bthis\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\bin\s+\d+\s+(day|days|week|weeks)\b",
    re.I,
)


def _has_date_hint(text: str) -> bool:
    return bool(_DATE_HINT_RE.search(text))


_FIELD_CORRECTION_KEYWORDS = {
    "num_travellers": ("traveller", "travellers", "traveler", "travelers", "people", "passenger", "passengers", "person", "pax"),
    "budget": ("budget", "price", "rupees", "inr", "rs."),
    "transport_pref": ("transport", "mode of travel", "flight", "train", "bus"),
    "return_trip": ("round trip", "one way", "oneway", "return trip"),
}
_CORRECTION_VERBS = ("change", "update", "set ", "make it", "correct", "fix")


def _keyword_present(lower_text: str, words: list, keyword: str) -> bool:
    if " " in keyword:
        return keyword in lower_text
    if keyword in lower_text:
        return True
    return any(
        len(word) >= 4 and difflib.SequenceMatcher(None, word, keyword).ratio() >= 0.82
        for word in words
    )


def _detect_field_correction(text: str):
    lower = text.lower()
    if not any(v in lower for v in _CORRECTION_VERBS):
        return None
    words = re.findall(r"[a-z]+", lower)
    for field, keywords in _FIELD_CORRECTION_KEYWORDS.items():
        if any(_keyword_present(lower, words, kw) for kw in keywords):
            value = _coerce_answer(field, text)
            if value is not None:
                return field, value
    return None


class IntentResult(BaseModel):
    intent: str


def detect_intent_node(state: TravelState) -> dict:
    if state.get("missing_fields"):
        return {"intent": "new_trip"}
    if state.get("recommendation") and state.get("approved") is None:
        return {"intent": "new_trip"}

    llm = get_shared_llm()
    text = _last_human_text(state)
    prompt = (
        "Classify the user's message into exactly one category:\n"
        "- new_trip: wants to plan/search/book travel\n"
        "- show_bookings: wants to see their previous bookings\n"
        "- chit_chat: greeting or anything unrelated to travel\n\n"
        f"Message: \"{text}\"\n\n"
        "Respond with only one word: new_trip, show_bookings, or chit_chat."
    )
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip().lower()
    except Exception:
        raw = "new_trip"

    if "show_bookings" in raw or "booking" in raw and "show" in raw:
        intent = "show_bookings"
    elif "chit_chat" in raw or "chitchat" in raw:
        intent = "chit_chat"
    else:
        intent = "new_trip"

    return {"intent": intent}


def route_after_intent(state: TravelState) -> str:
    intent = state.get("intent", "new_trip")
    if intent == "show_bookings":
        return "fetch_previous_bookings"
    if intent == "chit_chat":
        return "chit_chat"
    return "extract_info"


class TripInfo(BaseModel):
    source: Optional[str] = None
    destination: Optional[str] = None
    travel_date: Optional[str] = None
    return_date: Optional[str] = None
    return_trip: Optional[bool] = None
    num_travellers: Optional[int] = None
    budget: Optional[float] = None
    transport_pref: Optional[str] = None
    hotel_required: Optional[bool] = None
    flexible_dates: Optional[bool] = None
    seat_pref: Optional[str] = None
    traveller_names: Optional[List[str]] = None


def extract_info_node(state: TravelState) -> dict:
    llm = get_shared_llm()
    text = _last_human_text(state)
    pending_field = state.get("last_asked_field")

    known_fields = REQUIRED_FIELDS + ["return_date", "flexible_dates", "seat_pref", "traveller_names"]
    known = {f: state.get(f) for f in known_fields if state.get(f) not in (None, "", [])}

    context_lines = []
    if known:
        context_lines.append(f"Trip details already collected: {json.dumps(known, default=str)}")
    if pending_field:
        question = FIELD_QUESTIONS.get(pending_field, pending_field)
        context_lines.append(f'You just asked the user: "{question}" (field: \'{pending_field}\').')
    context = ("\n" + "\n".join(context_lines) + "\n") if context_lines else ""

    structured_llm = llm.with_structured_output(TripInfo)
    prompt = (
        "You are updating a travel booking form based on the user's latest message.\n"
        f"{context}\n"
        f'User\'s latest message: "{text}"\n\n'
        "Decide what to change:\n"
        "- If the message answers the pending question above, fill that field.\n"
        "- If the message corrects a detail already collected (e.g. \"change travellers "
        "to 1\", \"actually make it one-way\", \"his name is Sharan not Ramu\"), update ONLY "
        "that specific field — use the already-collected details above to figure out which "
        "field a correction refers to. For traveller_names, return the full corrected list.\n"
        "- If the message states other new details, fill those too.\n"
        "Return ONLY the fields that should be set or changed. traveller_names must be a list "
        "of plain name strings; every other field must be a plain value (text, number, or "
        "true/false), never a nested object. Leave anything unaffected by this message as null."
    )
    try:
        extracted: TripInfo = structured_llm.invoke(prompt)
        update = _sanitize_extracted(extracted.model_dump(exclude_none=True))
    except Exception:
        update = {}

    if "travel_date" in update and not _has_date_hint(text):
        del update["travel_date"]
    if "return_date" in update and not _has_date_hint(text):
        del update["return_date"]

    if "traveller_names" in update:
        text_lower = text.lower()
        if not all(name.lower() in text_lower for name in update["traveller_names"]):
            del update["traveller_names"]

    if "num_travellers" in update and not re.search(rf"\b{update['num_travellers']}\b", text):
        del update["num_travellers"]

    src, dst = _extract_route(text)
    if src:
        update["source"] = src
    if dst:
        update["destination"] = dst

    if pending_field and pending_field not in update and _looks_like_direct_answer(text):
        coerced = _coerce_answer(pending_field, text)
        if coerced is not None:
            update[pending_field] = coerced

    if not pending_field or pending_field not in update:
        correction = _detect_field_correction(text)
        if correction:
            field, value = correction
            if field not in update:
                update[field] = value

    if update.get("num_travellers") is not None:
        try:
            if int(update["num_travellers"]) <= 1:
                update["traveller_names"] = None
        except (TypeError, ValueError):
            pass

    update["last_asked_field"] = None
    return update


def check_missing_info_node(state: TravelState) -> dict:
    missing = []
    for field in REQUIRED_FIELDS:
        if state.get(field) in (None, "", []):
            missing.append(field)
        if field == "num_travellers":
            n = state.get("num_travellers")
            if n and int(n) > 1 and not state.get("traveller_names"):
                missing.append("traveller_names")
    return {"missing_fields": missing}


def route_after_missing_check(state: TravelState) -> str:
    return "ask_questions" if state.get("missing_fields") else "search_travel_options"


def ask_questions_node(state: TravelState) -> dict:
    missing = state.get("missing_fields", [])
    field = missing[0]
    question = FIELD_QUESTIONS.get(field, f"Could you share your {field.replace('_', ' ')}?")
    return {"messages": [AIMessage(content=question)], "last_asked_field": field}


def search_travel_options_node(state: TravelState) -> dict:
    results = search_travel_options_tool.invoke(
        {
            "source": state["source"],
            "destination": state["destination"],
            "date": state["travel_date"],
        }
    )
    return {"search_results": results}


def compare_options_node(state: TravelState) -> dict:
    llm = get_shared_llm()
    results = state.get("search_results", {})
    prompt = (
        "You are a travel assistant. Given these raw search results for "
        "flight/train/bus options (JSON below), produce a short comparison "
        "covering price, duration, and convenience for each mode. "
        "Keep it to 3-4 sentences total.\n\n"
        f"Preferences: budget=INR {state.get('budget')}, "
        f"transport_pref={state.get('transport_pref')}, "
        f"travellers={state.get('num_travellers')}\n\n"
        f"Search results:\n{json.dumps(results, default=str)[:3000]}"
    )
    try:
        response = llm.invoke(prompt)
        comparison_text = response.content
    except Exception as exc:
        comparison_text = f"(Comparison unavailable: {exc})"

    return {"comparison": {"summary": comparison_text}}


def recommend_node(state: TravelState) -> dict:
    results = state.get("search_results", {})

    def cheapest(mode: str):
        options = results.get(mode) or []
        priced = [o for o in options if isinstance(o.get("price"), (int, float))]
        return min(priced, key=lambda o: o["price"]) if priced else (options[0] if options else None)

    pref = (state.get("transport_pref") or "any").lower()
    candidates = {mode: cheapest(mode) for mode in ["flight", "train", "bus"]}
    candidates = {k: v for k, v in candidates.items() if v}

    if pref in candidates:
        chosen_mode, chosen = pref, candidates[pref]
    elif candidates:
        budget = state.get("budget")
        in_budget = {
            m: o for m, o in candidates.items() if budget is None or o.get("price", 0) <= budget
        }
        pool = in_budget or candidates
        chosen_mode, chosen = min(pool.items(), key=lambda kv: kv[1].get("price", float("inf")))
    else:
        chosen_mode, chosen = "flight", {"price": 0, "duration_hours": None}

    recommendation = {
        "mode": chosen_mode,
        "price": chosen.get("price"),
        "duration_hours": chosen.get("duration_hours"),
        "provider": chosen.get("provider", chosen.get("title", "N/A")),
    }

    summary = (
        f"Here's my recommendation:\n\n"
        f"**{state.get('source')} -> {state.get('destination')}**\n"
        f"Mode: {chosen_mode.title()}\n"
        f"Provider: {recommendation['provider']}\n"
        f"Price: INR {recommendation['price']}\n"
        f"Duration: {recommendation['duration_hours']} hrs\n"
        f"Date: {state.get('travel_date')}\n\n"
        f"{state.get('comparison', {}).get('summary', '')}"
    )

    return {
        "recommendation": recommendation,
        "messages": [AIMessage(content=summary)],
    }


def human_approval_node(state: TravelState) -> dict:
    rec = state.get("recommendation", {})
    summary = {
        "type": "booking_approval",
        "route": f"{state.get('source')} -> {state.get('destination')}",
        "mode": rec.get("mode"),
        "price": rec.get("price"),
        "date": state.get("travel_date"),
        "question": "Proceed with this booking? (yes/no)",
    }

    decision = interrupt(summary)

    approved = bool(decision) if isinstance(decision, bool) else str(decision).strip().lower() in (
        "yes",
        "y",
        "true",
        "approve",
        "approved",
    )
    return {"approved": approved}


def route_after_approval(state: TravelState) -> str:
    return "booking" if state.get("approved") else "cancelled"


def booking_node(state: TravelState) -> dict:
    rec = state.get("recommendation", {})
    booking = create_booking_tool.invoke(
        {
            "user_name": state.get("user_name") or "Guest",
            "email": state.get("user_email") or "guest@example.com",
            "source": state["source"],
            "destination": state["destination"],
            "travel_date": state["travel_date"],
            "transport_type": rec.get("mode", "flight"),
            "provider": rec.get("provider", "N/A"),
            "amount": float(rec.get("price") or 0),
            "traveller_names": state.get("traveller_names") or [],
        }
    )
    return {"booking": booking}


def cancelled_node(state: TravelState) -> dict:
    update = {
        "messages": [AIMessage(content="No problem — I've cancelled this booking. Let me know if you'd like to search again.")],
        "booking": None,
    }
    update.update(_reset_trip_fields())
    return update


def store_booking_node(state: TravelState) -> dict:
    saved = save_booking_tool.invoke({"booking": state["booking"]})
    return {"booking": saved}


def send_email_node(state: TravelState) -> dict:
    result = send_booking_email_tool.invoke({"booking": state["booking"]})
    booking = dict(state["booking"])
    booking["email_status"] = result
    return {"booking": booking}


def show_confirmation_node(state: TravelState) -> dict:
    b = state["booking"]
    email_result = b.get("email_status", {})
    passengers = [b["user_name"]] + list(b.get("traveller_names") or [])
    message = (
        f"Booking confirmed!\n\n"
        f"Booking ID: {b['booking_id']}\n"
        f"Passenger(s): {', '.join(passengers)}\n"
        f"{b['source']} -> {b['destination']} on {b['travel_date']}\n"
        f"Transport: {b['transport_type'].title()}\n"
        f"Provider: {b.get('provider', 'N/A')}\n"
        f"Amount: INR {b['amount']}\n"
        f"Status: {b['status']}\n\n"
        f"Confirmation email: {email_result.get('detail', 'not sent')}"
    )
    update = {"messages": [AIMessage(content=message)]}
    update.update(_reset_trip_fields())
    return update


def fetch_previous_bookings_node(state: TravelState) -> dict:
    email = state.get("user_email") or "guest@example.com"
    bookings = fetch_previous_bookings_tool.invoke({"email": email})

    if not bookings:
        text = "You don't have any previous bookings yet."
    else:
        lines = [
            f"- {b['booking_id']}: {b['source']} -> {b['destination']} on {b['travel_date']} "
            f"({b['transport_type']} via {b.get('provider', 'N/A')}, INR {b['amount']}, {b['status']})"
            for b in bookings
        ]
        text = "Here are your previous bookings:\n" + "\n".join(lines)

    return {"previous_bookings": bookings, "messages": [AIMessage(content=text)]}


def chit_chat_node(state: TravelState) -> dict:
    llm = get_shared_llm()
    text = _last_human_text(state)
    try:
        response = llm.invoke(
            "You are a friendly travel assistant. Reply briefly (1-2 sentences) "
            f"to this message and nudge the user toward planning a trip: \"{text}\""
        )
        reply = response.content
    except Exception:
        reply = "Hi! Tell me where you'd like to travel from and to, and I'll help you plan it."
    return {"messages": [AIMessage(content=reply)]}
