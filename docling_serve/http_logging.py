import logging
import re
from typing import Optional

import requests


class HttpJsonLogHandler(logging.Handler):
    """
    A custom logging handler that sends log records as JSON to an HTTP endpoint.
    """

    def __init__(self, url: str, token: Optional[str] = None, verify: bool = True):
        super().__init__()
        self.url = url
        self.token = token
        self.verify = verify

    def emit(self, record: logging.LogRecord):
        """
        Formats the log record and sends it to the specified URL.
        """
        log_entry = {
            "timestamp": record.asctime
            if hasattr(record, "asctime")
            else logging.Formatter().formatTime(record),
            "level": record.levelname,
            "message": self.format(record),
            "logger_name": record.name,
            "module": record.module,
            "func_name": record.funcName,
            "line_no": record.lineno,
        }

        try:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            requests.post(
                self.url, json=log_entry, headers=headers, verify=self.verify, timeout=5
            )
        except requests.RequestException:
            # We can't log the error using the logger this handler is attached to,
            # as it would cause an infinite loop.
            # For now, we'll fail silently. In a production system, you might
            # use a different method to handle this (e.g., print to stderr).
            pass


class PageProgressLogHandler(logging.Handler):
    """
    A custom logging handler that tracks page-by-page progress by watching
    for specific log messages and sends structured updates to an HTTP endpoint.
    """

    def __init__(
        self, url: str, source_id: str, total_pages: int, token: Optional[str] = None
    ):
        super().__init__()
        self.url = url
        self.source_id = source_id
        self.total_pages = total_pages
        self.token = token
        self.processed_pages = 0
        # This regex matches the specific log message from the VLM model.
        self.progress_regex = re.compile(r"Generated \d+ tokens")

    def emit(self, record: logging.LogRecord):
        """
        Checks if the log message indicates page progress, and if so,
        increments the counter and sends a progress update.
        """
        if self.progress_regex.search(record.getMessage()):
            self.processed_pages += 1

            progress_data = {
                "source_id": self.source_id,
                "status": "PROCESSING",
                "message": f"Processed page {self.processed_pages} of {self.total_pages}",
                "processed_pages": self.processed_pages,
                "total_pages": self.total_pages,
            }

            try:
                headers = {"Content-Type": "application/json"}
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"

                requests.post(self.url, json=progress_data, headers=headers, timeout=5)
            except requests.RequestException:
                pass  # Fail silently
