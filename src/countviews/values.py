"""Parse released values while retaining each publisher's nonnumeric tokens."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Count:
    """A displayed nonnegative integer or an uninterpreted publisher token."""

    value: Optional[int]
    token: Optional[str]


def parse_count(text: str) -> Count:
    """Parse integer count text; never replace suppression or missingness by zero.

    Commas are accepted only in conventional three-digit groups. Decimals,
    percentages and negative values remain tokens, requiring explicit handling.
    """
    import re
    value = text.strip()
    if re.fullmatch(r"[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+", value):
        return Count(int(value.replace(",", "")), None)
    return Count(None, value)
