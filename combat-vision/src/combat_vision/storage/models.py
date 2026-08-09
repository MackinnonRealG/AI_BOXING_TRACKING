"""SQLAlchemy schema: fighters, sessions, rounds, and raw events.

Every emitted event is persisted with its full payload (JSON), so metrics can
always be *recomputed* from storage — the analytics layer never needs the
original video.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all Combat Vision tables."""


def _utcnow() -> datetime:
    """Timezone-aware UTC now (SQLite stores naive; we normalize on write)."""
    return datetime.now(UTC)


class Fighter(Base):
    """A person whose metrics are tracked across sessions."""

    __tablename__ = "fighters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    sessions: Mapped[list[SessionFighter]] = relationship(back_populates="fighter")


class Session(Base):
    """One training/sparring session (one live run or one reviewed video)."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport: Mapped[str] = mapped_column(String(40))
    mode: Mapped[str] = mapped_column(String(10))  # "live" | "review"
    source: Mapped[str] = mapped_column(String(255))  # camera id / video path
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated: Mapped[int] = mapped_column(Integer, default=0)  # bool

    rounds: Mapped[list[Round]] = relationship(back_populates="session")
    fighters: Mapped[list[SessionFighter]] = relationship(back_populates="session")
    events: Mapped[list[EventRecord]] = relationship(back_populates="session")


class SessionFighter(Base):
    """Joins a fighter to a session under a per-session label (A/B)."""

    __tablename__ = "session_fighters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"))
    label: Mapped[str] = mapped_column(String(4))  # "A" | "B"

    session: Mapped[Session] = relationship(back_populates="fighters")
    fighter: Mapped[Fighter] = relationship(back_populates="sessions")


class Round(Base):
    """A round within a session.

    ``outcome`` is user-labelled for the pattern-recognition module and
    holds the *winning fighter's label* (``"A"``/``"B"``) or ``"draw"``.
    """

    __tablename__ = "rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    number: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(10), nullable=True)

    session: Mapped[Session] = relationship(back_populates="rounds")


class EventRecord(Base):
    """One emitted pipeline event, stored raw for later recomputation."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    fighter_label: Mapped[str] = mapped_column(String(4), index=True)
    timestamp_s: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict] = mapped_column(JSON)

    session: Mapped[Session] = relationship(back_populates="events")
