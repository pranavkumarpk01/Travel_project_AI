import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from database.models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./database/travel_planner.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=engine)
    _patch_missing_columns()


def _patch_missing_columns():
    inspector = inspect(engine)
    if "bookings" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("bookings")}
    with engine.begin() as conn:
        if "provider" not in existing:
            conn.execute(text("ALTER TABLE bookings ADD COLUMN provider VARCHAR"))
        if "traveller_names" not in existing:
            conn.execute(text("ALTER TABLE bookings ADD COLUMN traveller_names VARCHAR"))


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
