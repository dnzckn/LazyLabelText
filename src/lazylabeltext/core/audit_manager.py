"""Audit trail management."""

from __future__ import annotations

import logging

from lazylabeltext.core.database import Database
from lazylabeltext.core.models import AuditEvent

logger = logging.getLogger("lazylabeltext")


class AuditManager:
    """Logs audit events for all state-changing operations."""

    # Event type constants
    DOCUMENT_LOADED = "document_loaded"
    DOCUMENT_PARSE_FAILED = "document_parse_failed"
    RUBRIC_CREATED = "rubric_created"
    RUBRIC_UPDATED = "rubric_updated"
    CHUNKING_RUN_STARTED = "chunking_run_started"
    CHUNKING_RUN_COMPLETED = "chunking_run_completed"
    CHUNK_MERGED = "chunk_merged"
    CHUNK_SPLIT = "chunk_split"
    LABEL_CREATED = "label_created"
    LABEL_REVIEWED = "label_reviewed"
    PROPAGATION_STARTED = "propagation_started"
    PROPAGATION_COMPLETED = "propagation_completed"
    EXPORT_COMPLETED = "export_completed"

    def __init__(self, database: Database) -> None:
        self.db = database

    def log_event(
        self,
        event_type: str,
        actor: str = "system",
        payload: dict | None = None,
        related_chunk_ids: list[int] | None = None,
    ) -> int:
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            actor=actor,
            payload=payload or {},
            related_chunk_ids=related_chunk_ids or [],
        )
        event_id = self.db.insert_audit_event(event)
        logger.debug("Audit: %s by %s", event_type, actor)
        return event_id

    def get_events(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        return self.db.get_audit_events(event_type=event_type, limit=limit)
