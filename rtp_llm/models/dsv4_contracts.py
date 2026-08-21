"""Pure-stdlib DeepSeek-V4 speculative cache contracts.

Kept dependency-free so schedulers and CPU contract tests can validate the
target query boundary without importing tensor/native runtime modules.
"""

from enum import Enum
from numbers import Integral

DSV4_SPECULATIVE_C4_MAX_QUERY_LEN = 9


class Dsv4StateRingMode(Enum):
    NORMAL = "normal"
    SPECULATIVE_TARGET_VERIFY = "speculative_target_verify"


def validate_dsv4_speculative_target_query_len(query_len: int) -> int:
    if isinstance(query_len, bool) or not isinstance(query_len, Integral):
        raise TypeError(
            "DeepSeek-V4 speculative target query_len must be an integer: "
            f"value={query_len!r}, type={type(query_len).__name__}"
        )
    query_len = int(query_len)
    if not 1 <= query_len <= DSV4_SPECULATIVE_C4_MAX_QUERY_LEN:
        raise ValueError(
            "DeepSeek-V4 speculative target query_len is outside the safe "
            "C4 state-ring range: "
            f"query_len={query_len}, supported=1.."
            f"{DSV4_SPECULATIVE_C4_MAX_QUERY_LEN}"
        )
    return query_len
