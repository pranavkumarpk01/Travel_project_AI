import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def generate_booking_id() -> str:
    return f"TRV-{uuid.uuid4().hex[:8].upper()}"


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    booking_id = Column(String, unique=True, index=True, default=generate_booking_id)

    user_name = Column(String, nullable=False)
    email = Column(String, nullable=False, index=True)

    source = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    travel_date = Column(String, nullable=False)

    transport_type = Column(String, nullable=False)
    provider = Column(String, nullable=True)
    traveller_names = Column(String, nullable=True)  # comma-separated
    amount = Column(Float, nullable=False)
    status = Column(String, default="CONFIRMED")

    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "booking_id": self.booking_id,
            "user_name": self.user_name,
            "email": self.email,
            "source": self.source,
            "destination": self.destination,
            "travel_date": self.travel_date,
            "transport_type": self.transport_type,
            "provider": self.provider,
            "traveller_names": [n for n in (self.traveller_names or "").split(",") if n],
            "amount": self.amount,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
