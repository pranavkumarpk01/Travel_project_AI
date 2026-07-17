from langchain_core.tools import tool

from database.db import SessionLocal, init_db
from database.models import Booking


@tool
def save_booking_tool(booking: dict) -> dict:
    """Persist a booking record to the database and return it with its row id."""
    init_db()
    session = SessionLocal()
    try:
        row = Booking(
            booking_id=booking["booking_id"],
            user_name=booking["user_name"],
            email=booking["email"],
            source=booking["source"],
            destination=booking["destination"],
            travel_date=booking["travel_date"],
            transport_type=booking["transport_type"],
            provider=booking.get("provider", "N/A"),
            traveller_names=",".join(booking.get("traveller_names") or []),
            amount=booking["amount"],
            status=booking.get("status", "CONFIRMED"),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.to_dict()
    finally:
        session.close()


@tool
def fetch_previous_bookings_tool(email: str) -> list:
    """Fetch all previous bookings for a given passenger email, most recent first."""
    init_db()
    session = SessionLocal()
    try:
        rows = (
            session.query(Booking)
            .filter(Booking.email == email)
            .order_by(Booking.created_at.desc())
            .all()
        )
        return [row.to_dict() for row in rows]
    finally:
        session.close()
