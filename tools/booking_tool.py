from typing import List, Optional

from langchain_core.tools import tool

from database.models import generate_booking_id


@tool
def create_booking_tool(
    user_name: str,
    email: str,
    source: str,
    destination: str,
    travel_date: str,
    transport_type: str,
    amount: float,
    provider: str = "N/A",
    traveller_names: Optional[List[str]] = None,
) -> dict:
    """Simulate a booking transaction and return a booking record."""
    return {
        "booking_id": generate_booking_id(),
        "user_name": user_name,
        "email": email,
        "source": source,
        "destination": destination,
        "travel_date": travel_date,
        "transport_type": transport_type,
        "provider": provider,
        "amount": amount,
        "status": "CONFIRMED",
        "traveller_names": traveller_names or [],
    }
