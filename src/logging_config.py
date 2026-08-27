"""Structured application logging with incident context."""

from __future__ import annotations

import logging
import os


class IncidentContextFormatter(logging.Formatter):
    """Guarantee context fields exist for application and dependency logs."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "incident_id"):
            record.incident_id = "system"
        if not hasattr(record, "stage"):
            record.stage = "application"
        return super().format(record)


def configure_logging(level: str | None = None) -> None:
    selected_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler()
    handler.setFormatter(
        IncidentContextFormatter(
            "%(asctime)s level=%(levelname)s logger=%(name)s "
            "incident_id=%(incident_id)s stage=%(stage)s message=%(message)s"
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(selected_level)
