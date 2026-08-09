"""Repository: the only place the rest of the app touches the database."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from combat_vision.events.types import Event
from combat_vision.storage.models import Base, EventRecord, Fighter, Round, Session


class SessionRepository:
    """Persists sessions and events; serves the analytics layer read-side."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url)
        # Dev convenience — production schema management is alembic's job.
        Base.metadata.create_all(self._engine)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)

    def create_session(self, sport: str, mode: str, source: str, calibrated: bool) -> int:
        """Insert a new session row and return its id."""
        with self._sessions.begin() as db:
            record = Session(sport=sport, mode=mode, source=source, calibrated=int(calibrated))
            db.add(record)
            db.flush()
            return record.id

    def finish_session(self, session_id: int, duration_s: float) -> None:
        """Stamp the session's final duration."""
        with self._sessions.begin() as db:
            db.get_one(Session, session_id).duration_s = duration_s

    def save_events(self, session_id: int, events: Sequence[Event]) -> None:
        """Persist emitted events with their full dataclass payload as JSON."""
        with self._sessions.begin() as db:
            for event in events:
                payload = dataclasses.asdict(event)
                db.add(
                    EventRecord(
                        session_id=session_id,
                        event_type=type(event).__name__,
                        fighter_label=event.fighter_id,
                        timestamp_s=event.timestamp_s,
                        payload=payload,
                    )
                )

    def get_or_create_fighter(self, name: str) -> int:
        """Look up a fighter by name, creating them on first sight."""
        with self._sessions.begin() as db:
            existing = db.scalar(select(Fighter).where(Fighter.name == name))
            if existing is not None:
                return existing.id
            fighter = Fighter(name=name)
            db.add(fighter)
            db.flush()
            return fighter.id

    def list_sessions(self) -> list[Session]:
        """All sessions, oldest first — the analytics layer's entry point."""
        with self._sessions() as db:
            return list(db.scalars(select(Session).order_by(Session.started_at)))

    def events_for_session(
        self, session_id: int, event_type: str | None = None
    ) -> list[EventRecord]:
        """Raw event records for one session, optionally filtered by type."""
        with self._sessions() as db:
            query = select(EventRecord).where(EventRecord.session_id == session_id)
            if event_type is not None:
                query = query.where(EventRecord.event_type == event_type)
            return list(db.scalars(query.order_by(EventRecord.timestamp_s)))

    def create_round(
        self,
        session_id: int,
        number: int,
        start_s: float,
        end_s: float | None = None,
        outcome: str | None = None,
    ) -> int:
        """Insert a round; ``outcome`` is the winning fighter label or 'draw'."""
        with self._sessions.begin() as db:
            rnd = Round(
                session_id=session_id,
                number=number,
                start_s=start_s,
                end_s=end_s,
                outcome=outcome,
            )
            db.add(rnd)
            db.flush()
            return rnd.id

    def label_round(self, round_id: int, outcome: str) -> None:
        """User-label a round's outcome (winning fighter label or 'draw')."""
        with self._sessions.begin() as db:
            db.get_one(Round, round_id).outcome = outcome

    def rounds_for_session(self, session_id: int) -> list[Round]:
        """All rounds of one session in order."""
        with self._sessions() as db:
            return list(
                db.scalars(
                    select(Round).where(Round.session_id == session_id).order_by(Round.number)
                )
            )

    def orm_session(self) -> OrmSession:
        """Escape hatch for analytics queries that need raw ORM access."""
        return self._sessions()
