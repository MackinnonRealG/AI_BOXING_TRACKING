"""Repository: the only place the rest of the app touches the database."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from combat_vision.events.types import Event
from combat_vision.storage.models import (
    Base,
    EventRecord,
    Fighter,
    Round,
    Session,
    SessionFighter,
)


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

    def events_for_sessions(
        self, session_ids: Sequence[int], event_type: str | None = None
    ) -> dict[int, list[EventRecord]]:
        """Raw event records for multiple sessions, grouped by session id.

        One query covering every session, instead of calling
        :meth:`events_for_session` once per session in a loop.
        """
        if not session_ids:
            return {}
        with self._sessions() as db:
            query = select(EventRecord).where(EventRecord.session_id.in_(session_ids))
            if event_type is not None:
                query = query.where(EventRecord.event_type == event_type)
            query = query.order_by(EventRecord.session_id, EventRecord.timestamp_s)
            records = list(db.scalars(query))
        grouped: dict[int, list[EventRecord]] = {sid: [] for sid in session_ids}
        for record in records:
            grouped[record.session_id].append(record)
        return grouped

    def link_fighter(self, session_id: int, fighter_id: int, label: str) -> None:
        """Bind a named fighter to a session under their per-session label."""
        with self._sessions.begin() as db:
            db.add(SessionFighter(session_id=session_id, fighter_id=fighter_id, label=label))

    def label_for_fighter(self, session_id: int, fighter_id: int) -> str | None:
        """The per-session label ``fighter_id`` was assigned, or None if absent.

        Per-session labels ("A"/"B") are reassigned independently each
        session, so cross-session aggregation must go through this join
        instead of matching on the raw label.
        """
        with self._sessions() as db:
            return db.scalar(
                select(SessionFighter.label).where(
                    SessionFighter.session_id == session_id,
                    SessionFighter.fighter_id == fighter_id,
                )
            )

    def labels_for_fighter(self, fighter_id: int) -> dict[int, str]:
        """session_id -> label for every session this physical fighter appears in.

        One query covering every session, instead of calling
        :meth:`label_for_fighter` once per session in a loop.
        """
        with self._sessions() as db:
            rows = db.execute(
                select(SessionFighter.session_id, SessionFighter.label).where(
                    SessionFighter.fighter_id == fighter_id
                )
            ).all()
        return {session_id: label for session_id, label in rows}

    def fighter_name(self, fighter_id: int) -> str | None:
        """The fighter's display name, or None if no such fighter exists."""
        with self._sessions() as db:
            return db.scalar(select(Fighter.name).where(Fighter.id == fighter_id))

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

    def rounds_for_sessions(self, session_ids: Sequence[int]) -> dict[int, list[Round]]:
        """All rounds of multiple sessions, grouped by session id and in order.

        One query covering every session, instead of calling
        :meth:`rounds_for_session` once per session in a loop.
        """
        if not session_ids:
            return {}
        with self._sessions() as db:
            query = (
                select(Round)
                .where(Round.session_id.in_(session_ids))
                .order_by(Round.session_id, Round.number)
            )
            records = list(db.scalars(query))
        grouped: dict[int, list[Round]] = {sid: [] for sid in session_ids}
        for record in records:
            grouped[record.session_id].append(record)
        return grouped

    def orm_session(self) -> OrmSession:
        """Escape hatch for analytics queries that need raw ORM access."""
        return self._sessions()
