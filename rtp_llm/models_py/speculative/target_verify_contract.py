"""Platform-neutral target-verify round contract.

This module deliberately owns only request/row bookkeeping.  Cache geometry
is selected by the caller through an explicit mode; no generation count or
environment value is interpreted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Callable, Optional, Sequence

from rtp_llm.models.dsv4_contracts import (
    Dsv4StateRingMode,
    validate_dsv4_speculative_target_query_len,
)


class TargetVerifyRole(Enum):
    NORMAL = "normal"
    TARGET_VERIFY = "target_verify"


class TargetVerifyMode(Enum):
    DISABLED = "disabled"
    MTP = "mtp"


@dataclass(frozen=True)
class VerifyRowMapping:
    """Request and token index for each dense verify row.

    ``-1`` denotes a graph-padding row.  Rows are request-major and remain
    dense even when a request accepts no proposed token.
    """

    request_ids: tuple[int, ...]
    token_indices: tuple[int, ...]
    row_offsets: tuple[int, ...]
    accepted_lengths: tuple[int, ...]
    next_prefix_lengths: tuple[int, ...]


def _validate_batch(batch_size: int, query_len: int) -> tuple[int, int]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if isinstance(query_len, bool) or not isinstance(query_len, Integral):
        raise TypeError(f"query_len must be an integer, got {query_len!r}")
    return int(batch_size), int(query_len)


def map_verify_rows(
    batch_size: int,
    query_len: int,
    accepted_lengths: Sequence[int],
    base_prefix_lengths: Sequence[int],
    *,
    padded_batch_size: Optional[int] = None,
) -> VerifyRowMapping:
    """Map a request-major dense verify result and derive next prefixes.

    Acceptance is per request and is allowed to be zero (all rejected).  The
    dense output has ``padded_batch_size * query_len`` rows; padded requests
    map to ``(-1, -1)`` and never contribute to the next prefix.
    """

    batch_size, query_len = _validate_batch(batch_size, query_len)
    if query_len < 1:
        raise ValueError("query_len must be positive")
    validate_dsv4_speculative_target_query_len(query_len)
    if len(accepted_lengths) != batch_size or len(base_prefix_lengths) != batch_size:
        raise ValueError("accepted_lengths and base_prefix_lengths must match batch_size")
    if padded_batch_size is None:
        padded_batch_size = batch_size
    if (
        isinstance(padded_batch_size, bool)
        or not isinstance(padded_batch_size, Integral)
        or padded_batch_size < batch_size
    ):
        raise ValueError("padded_batch_size must be an integer >= batch_size")

    accepted = []
    for length in accepted_lengths:
        if isinstance(length, bool) or not isinstance(length, Integral) or not 0 <= length <= query_len:
            raise ValueError(
                f"accepted length must be an integer in 0..{query_len}, got {length!r}"
            )
        accepted.append(int(length))
    prefixes = []
    for prefix in base_prefix_lengths:
        if isinstance(prefix, bool) or not isinstance(prefix, Integral) or prefix < 0:
            raise ValueError(f"base prefix must be a non-negative integer, got {prefix!r}")
        prefixes.append(int(prefix))

    request_ids: list[int] = []
    token_indices: list[int] = []
    row_offsets = tuple(request_id * query_len for request_id in range(batch_size))
    for request_id in range(padded_batch_size):
        for token_index in range(query_len):
            if request_id < batch_size:
                request_ids.append(request_id)
                token_indices.append(token_index)
            else:
                request_ids.append(-1)
                token_indices.append(-1)
    return VerifyRowMapping(
        tuple(request_ids),
        tuple(token_indices),
        row_offsets,
        tuple(accepted),
        tuple(prefix + length for prefix, length in zip(prefixes, accepted)),
    )


@dataclass(frozen=True)
class TargetVerifyContract:
    """Explicit target-verify role/mode and its cache-mode handoff."""

    role: TargetVerifyRole
    mode: TargetVerifyMode
    query_len: int
    state_ring_mode: Dsv4StateRingMode

    def __post_init__(self) -> None:
        if not isinstance(self.role, TargetVerifyRole):
            raise TypeError("role must be a TargetVerifyRole")
        if not isinstance(self.mode, TargetVerifyMode):
            raise TypeError("mode must be a TargetVerifyMode")
        if not isinstance(self.state_ring_mode, Dsv4StateRingMode):
            raise TypeError("state_ring_mode must be a Dsv4StateRingMode")
        if self.role is TargetVerifyRole.NORMAL:
            if self.mode is not TargetVerifyMode.DISABLED or self.state_ring_mode is not Dsv4StateRingMode.NORMAL:
                raise ValueError("NORMAL requires DISABLED mode and NORMAL state ring")
            if isinstance(self.query_len, bool) or not isinstance(self.query_len, Integral) or self.query_len < 0:
                raise ValueError("normal query_len must be a non-negative integer")
        else:
            if (
                self.mode is not TargetVerifyMode.MTP
                or self.state_ring_mode
                is not Dsv4StateRingMode.SPECULATIVE_TARGET_VERIFY
            ):
                raise ValueError("TARGET_VERIFY requires MTP mode and speculative state ring")
            validate_dsv4_speculative_target_query_len(self.query_len)


class NumericalFailureTransaction:
    """Small CPU-testable transaction shell for a future runtime owner."""

    def __init__(self) -> None:
        self.state = "idle"

    def run(
        self,
        prepare: Callable[[], None],
        verify: Callable[[], bool],
        commit: Callable[[], None],
        rollback: Callable[[], None],
    ) -> bool:
        if self.state != "idle":
            raise RuntimeError(f"transaction cannot run from state {self.state!r}")
        rollback_called = False

        def rollback_once() -> None:
            nonlocal rollback_called
            if rollback_called:
                return
            rollback_called = True
            rollback()

        try:
            prepare()
            self.state = "prepared"
            if not verify():
                rollback_once()
                self.state = "rolled_back"
                return False
            commit()
            self.state = "committed"
            return True
        except Exception:
            try:
                rollback_once()
            except Exception:
                self.state = "rollback_failed"
                raise
            self.state = "rolled_back"
            raise
