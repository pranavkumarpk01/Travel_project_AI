from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class TravelState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    user_name: Optional[str]
    user_email: Optional[str]

    intent: Optional[str]

    source: Optional[str]
    destination: Optional[str]
    travel_date: Optional[str]
    return_date: Optional[str]
    return_trip: Optional[bool]
    num_travellers: Optional[int]
    budget: Optional[float]
    transport_pref: Optional[str]
    hotel_required: Optional[bool]
    flexible_dates: Optional[bool]
    seat_pref: Optional[str]
    traveller_names: Optional[list]

    missing_fields: list
    last_asked_field: Optional[str]
    # The field the user most recently *answered*. Lets us resolve vague
    # corrections like "change that to Vijay" to the right slot.
    last_answered_field: Optional[str]
    # Set once a booking is confirmed. Trip details are kept in memory so the
    # user can tweak-and-rebook, but a brand-new route starts a fresh trip.
    booking_complete: Optional[bool]
    # A short "Got it — updated X to Y" note surfaced after a correction.
    correction_ack: Optional[str]

    search_results: Optional[dict]
    comparison: Optional[dict]
    recommendation: Optional[dict]

    approved: Optional[bool]

    booking: Optional[dict]
    previous_bookings: Optional[list]


# traveller_names is deliberately not listed here — it's only required once
# num_travellers > 1, so check_missing_info_node adds it conditionally.
REQUIRED_FIELDS = [
    "source",
    "destination",
    "travel_date",
    "num_travellers",
    "budget",
    "transport_pref",
    "return_trip",
]

FIELD_QUESTIONS = {
    "source": "Which city will you be travelling from?",
    "destination": "Which city are you travelling to?",
    "travel_date": "What date would you like to travel (YYYY-MM-DD)?",
    "num_travellers": "How many travellers should I book for?",
    "budget": "What's your approximate budget for this trip (in INR)?",
    "transport_pref": "Any preferred mode of transport — flight, train, bus, or no preference?",
    "return_trip": "Is this a round trip, or one-way?",
    "traveller_names": "Could you share the name(s) of the other traveller(s), separated by commas?",
    "hotel_required": "Do you need a hotel booked as well?",
    "flexible_dates": "Are your travel dates flexible?",
    "seat_pref": "Any seat preference (window/aisle) or class preference?",
}
