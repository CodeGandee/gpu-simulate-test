"""
User-facing errors for `vidur-cli`.

This module defines a single exception type that carries:
- a readable message
- an optional hint to fix the issue
- optional structured context for debugging
- an exit code for the CLI
"""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any


@dataclass(frozen=True)
class UserFacingError(RuntimeError):
    """An error intended to be shown directly to the user."""

    message: str
    hint: str | None = None
    context: dict[str, Any] | None = None
    exit_code: int = 2

    def format_stderr(self) -> str:
        """Format this exception for printing to stderr."""
        lines: list[str] = [self.message]
        if self.hint:
            lines.append(f"Hint: {self.hint}")
        if self.context:
            lines.append(f"Context: {self.context}")
        return "\n".join(lines)


def format_exception_for_cli(exc: BaseException) -> tuple[str, int]:
    """Format an exception for stderr printing and return (text, exit_code)."""
    if isinstance(exc, UserFacingError):
        return exc.format_stderr(), int(exc.exit_code)
    return f"Unexpected error: {exc}\n{traceback.format_exc()}".rstrip(), 1
