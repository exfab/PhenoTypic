"""Serialization and compare-and-set fencing for output binding requests."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Iterator

__all__ = ["BindingCoordinator", "BindingSupersededError"]


class BindingSupersededError(RuntimeError):
    """Raised when a newer bind/Refresh request supersedes this request."""


class BindingCoordinator:
    """Issue monotonic tickets and serialize selection through publication."""

    def __init__(self) -> None:
        self._request_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._latest_request = 0

    def issue_request(self) -> int:
        """Return a monotonic ticket and make it the latest request."""
        with self._request_lock:
            self._latest_request += 1
            return self._latest_request

    @property
    def latest_request(self) -> int:
        """Return the newest issued request ticket."""
        with self._request_lock:
            return self._latest_request

    def require_latest(self, ticket: int) -> None:
        """Raise when ``ticket`` lost the compare-and-set race."""
        with self._request_lock:
            latest = self._latest_request
        if ticket != latest:
            raise BindingSupersededError(
                f"Binding request {ticket} was superseded by request {latest}."
            )

    def commit_if_latest(
        self,
        ticket: int,
        commit: Callable[[], None],
    ) -> None:
        """Run ``commit`` while ticket issuance is locked to this CAS."""
        with self._request_lock:
            if ticket != self._latest_request:
                raise BindingSupersededError(
                    f"Binding request {ticket} was superseded by request "
                    f"{self._latest_request}."
                )
            commit()

    @contextmanager
    def serialized(self) -> Iterator[None]:
        """Hold the binder-wide lock through selection, build, and publish."""
        with self._publish_lock:
            yield
