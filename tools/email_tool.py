import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv
from langchain_core.tools import tool


def _smtp_config() -> dict:
    load_dotenv(override=True)
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME", "").strip(),
        "password": os.getenv("SMTP_PASSWORD", "").strip(),
        "from_name": os.getenv("SMTP_FROM_NAME", "AI Travel Planner"),
    }


def _build_email_body(booking: dict) -> str:
    passengers = [booking["user_name"]] + list(booking.get("traveller_names") or [])
    passenger_line = ", ".join(passengers)
    return f"""Hi {booking['user_name']},

Your booking is confirmed! Tickets have been booked for: {passenger_line}.

  Booking ID:   {booking['booking_id']}
  Passenger(s): {passenger_line}
  Route:        {booking['source']} -> {booking['destination']}
  Travel Date:  {booking['travel_date']}
  Transport:    {booking['transport_type'].title()}
  Provider:     {booking.get('provider', 'N/A')}
  Fare:         Rs. {booking['amount']}
  Status:       {booking['status']}

Thanks for booking with AI Travel Planner!
"""


@tool
def send_booking_email_tool(booking: dict) -> dict:
    """Send a booking confirmation email to the passenger."""
    subject = f"Booking Confirmed - {booking['booking_id']}"
    body = _build_email_body(booking)
    cfg = _smtp_config()

    if not cfg["username"] or not cfg["password"]:
        missing = []
        if not cfg["username"]:
            missing.append("SMTP_USERNAME")
        if not cfg["password"]:
            missing.append("SMTP_PASSWORD")
        print("=" * 60)
        print(f"[email_tool] Missing {', '.join(missing)} in .env — printing email instead:")
        print(f"To: {booking['email']}\nSubject: {subject}\n\n{body}")
        print("=" * 60)
        return {
            "sent": True,
            "mode": "console",
            "detail": f"Printed to console (missing {', '.join(missing)} in .env)",
        }

    try:
        msg = MIMEMultipart()
        msg["From"] = f"{cfg['from_name']} <{cfg['username']}>"
        msg["To"] = booking["email"]
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
            server.starttls()
            server.login(cfg["username"], cfg["password"])
            server.sendmail(cfg["username"], booking["email"], msg.as_string())

        return {"sent": True, "mode": "smtp", "detail": f"Email sent to {booking['email']}"}
    except Exception as exc:
        print("=" * 60)
        print(f"[email_tool] SMTP send failed ({exc}) — printing email instead:")
        print(f"To: {booking['email']}\nSubject: {subject}\n\n{body}")
        print("=" * 60)
        return {
            "sent": True,
            "mode": "console",
            "detail": f"SMTP send failed ({exc}) — printed to console instead. Check SMTP_HOST/network connectivity.",
        }
