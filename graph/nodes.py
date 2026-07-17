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
        "last_answered_field": None,
        "booking_complete": None,
        "correction_ack": None,
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


_NAME_CORRECTION_RE = re.compile(
    r"\bnames?(?:'s)?\b\s*(?:is|are|to|as)?\s*([a-z]+(?:\s(?!not\b)[a-z]+){0,2})(?=\s+not\b|,|$)",
    re.I,
)


def _detect_name_correction(text: str):
    m = _NAME_CORRECTION_RE.search(text)
    if not m:
        return None
    return _parse_traveller_names(m.group(1))


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
        norm = lower.replace("-", " ")
        negative = any(w in norm for w in ["no", "not", "one way", "oneway", "n/a", "none", "don't", "dont"])
        positive = any(w in norm for w in ["yes", "yeah", "yep", "sure", "round trip", "roundtrip", "return", "need"])
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
    "budget": ("budget", "price", "rupees", "inr", "rs.", "cost"),
    "transport_pref": ("transport", "mode of travel", "flight", "train", "bus"),
    "return_trip": ("round trip", "one way", "oneway", "return trip"),
    "traveller_names": ("name", "names"),
    "travel_date": ("date", "day"),
    "source": ("origin", "leaving from", "departure city", "starting"),
    "destination": ("destination", "going to"),
}
_CORRECTION_VERBS = ("change", "update", "set ", "make it", "make that", "correct", "fix", "rename")
_CORRECTION_MARKER_WORDS = ("actually", "instead", "i mean", "meant", "sorry", "no,", "not ")

# Pronouns the user might use to refer back to their previous answer, e.g.
# "change that to Vijay" or "make it one-way".
_PRONOUN_RE = re.compile(
    r"\b(that|it|this|the last one|the previous one|the last answer|the previous answer)\b",
    re.I,
)


def _keyword_present(lower_text: str, words: list, keyword: str) -> bool:
    if " " in keyword:
        return keyword in lower_text
    if keyword in lower_text:
        return True
    return any(
        len(word) >= 4 and difflib.SequenceMatcher(None, word, keyword).ratio() >= 0.82
        for word in words
    )


def _looks_like_correction(lower_text: str) -> bool:
    return any(v in lower_text for v in _CORRECTION_VERBS) or any(
        m in lower_text for m in _CORRECTION_MARKER_WORDS
    )


def _extract_correction_value(text: str):
    """Pull the new value out of a correction phrase.

    "change that to Vijay"      -> "Vijay"
    "change the budget to 3000" -> "3000"
    "make it one-way"           -> "one-way"
    "actually vijay"            -> "vijay"
    """
    m = re.search(r"\b(?:to|as|into|=|:)\s+(.+)$", text, re.I)
    if m:
        return m.group(1).strip(" .!?")
    m = re.search(
        r"\b(?:it|that|this|actually|instead)\s+(.+)$", text, re.I
    )
    if m:
        return m.group(1).strip(" .!?")
    return None


def _target_correction_field(lower_text: str, state) -> Optional[str]:
    words = re.findall(r"[a-z]+", lower_text)
    for field, keywords in _FIELD_CORRECTION_KEYWORDS.items():
        if any(_keyword_present(lower_text, words, kw) for kw in keywords):
            return field
    # No explicit field mentioned — a pronoun ("that", "it") refers back to
    # whatever the user answered most recently.
    if _PRONOUN_RE.search(lower_text):
        return state.get("last_answered_field")
    return None


def _resolve_correction(text: str, state):
    """Deterministically resolve a correction to (field, value).

    Handles both keyworded corrections ("change the budget to 3000") and vague
    pronoun corrections ("change that to Vijay") by resolving the pronoun to the
    field the user answered last.
    """
    lower = text.lower()
    if not _looks_like_correction(lower):
        return None
    field = _target_correction_field(lower, state)
    if not field:
        return None
    value_phrase = _extract_correction_value(text)
    # Fall back to coercing the whole message (covers "make it one-way" where
    # the keyword and value overlap).
    for candidate in (value_phrase, text):
        if candidate is None:
            continue
        value = _coerce_answer(field, candidate)
        if value is not None:
            return field, value
    return None


def _correction_ack(field: str, value) -> str:
    labels = {
        "traveller_names": "traveller name(s)",
        "num_travellers": "number of travellers",
        "budget": "budget",
        "transport_pref": "transport preference",
        "return_trip": "trip type",
        "travel_date": "travel date",
        "source": "origin",
        "destination": "destination",
    }
    label = labels.get(field, field.replace("_", " "))
    if isinstance(value, list):
        disp = ", ".join(value)
    elif field == "return_trip":
        disp = "round trip" if value else "one-way"
    elif field == "budget":
        disp = f"INR {value:g}" if isinstance(value, (int, float)) else str(value)
    else:
        disp = str(value)
    return f"Got it — updated {label} to {disp}."


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

    # --- Deterministic correction resolution (highest priority) -------------
    # This runs before the pending-question answer so that a message like
    # "change that to Vijay" (while we were asking for the budget) updates the
    # traveller name instead of being misread as a budget answer.
    ack_field = None
    correction = _resolve_correction(text, state)
    if correction:
        field, value = correction
        update[field] = value
        ack_field = field

    # "his name is Sharan not Ramu" style name corrections.
    if "traveller_names" not in update:
        corrected_names = _detect_name_correction(text)
        if corrected_names:
            update["traveller_names"] = corrected_names
            ack_field = ack_field or "traveller_names"

    # Deterministic transport pick-up, e.g. "help me find a train".
    if "transport_pref" not in update:
        tp = re.search(r"\b(flight|train|bus)\b", text, re.I)
        action = re.search(r"\b(find|book|search|want|need|prefer|by|take)\b", text, re.I)
        if tp and action:
            update["transport_pref"] = tp.group(1).lower()

    # Direct answer to the pending question — only if the message wasn't a
    # correction that already answered (or deliberately dodged) it.
    if pending_field and pending_field not in update and _looks_like_direct_answer(text):
        coerced = _coerce_answer(pending_field, text)
        if coerced is not None:
            update[pending_field] = coerced

    if update.get("num_travellers") is not None:
        try:
            if int(update["num_travellers"]) <= 1:
                update["traveller_names"] = None
        except (TypeError, ValueError):
            pass

    # After a completed booking we keep trip details in memory so the user can
    # tweak-and-rebook. But if they give a brand-new route, start a fresh trip.
    if state.get("booking_complete"):
        new_src = update.get("source")
        new_dst = update.get("destination")
        changed_route = (new_src and new_src != state.get("source")) or (
            new_dst and new_dst != state.get("destination")
        )
        if changed_route:
            for f in (
                "travel_date", "return_date", "return_trip", "num_travellers",
                "budget", "transport_pref", "hotel_required", "flexible_dates",
                "seat_pref", "traveller_names",
            ):
                update.setdefault(f, None)
        update["booking_complete"] = False

    # Remember which field the user just answered, so the *next* turn can
    # resolve a vague correction ("change that to ...") back to it.
    if pending_field and update.get(pending_field) is not None:
        update["last_answered_field"] = pending_field

    if ack_field is not None:
        update["correction_ack"] = _correction_ack(ack_field, update[ack_field])

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

    ack = state.get("correction_ack")
    message = f"{ack}\n\n{question}" if ack else question

    return {
        "messages": [AIMessage(content=message)],
        "last_asked_field": field,
        "correction_ack": None,
    }


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

    ack = state.get("correction_ack")
    ack_prefix = f"{ack}\n\n" if ack else ""

    summary = (
        f"{ack_prefix}"
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
        "correction_ack": None,
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
    # Keep the trip details in memory so the user can tweak-and-rebook
    # ("change the budget to 3000 and find a train"). Only the per-transaction
    # fields are cleared here; extract_info_node starts a fresh trip if the user
    # later gives a brand-new route.
    return {
        "messages": [AIMessage(content=message)],
        "booking_complete": True,
        "search_results": None,
        "comparison": None,
        "recommendation": None,
        "approved": None,
        "last_asked_field": None,
        "correction_ack": None,
    }


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
