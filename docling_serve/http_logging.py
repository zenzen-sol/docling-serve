import logging
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
            "timestamp": self.formatTime(record, self.datefmt),
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
