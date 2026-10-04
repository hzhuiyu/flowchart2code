"""
Utilities for injecting self-planning instructions into prompts.
"""

from typing import Optional


def insert_before(prompt: str, marker: str, insertion: str) -> str:
    """Insert `insertion` before the first occurrence of `marker`."""
    if not prompt:
        return insertion
    if marker in prompt:
        return prompt.replace(marker, f"{insertion}\n{marker}", 1)
    return f"{prompt}\n\n{insertion}"


def insert_after(prompt: str, marker: str, insertion: str) -> str:
    """Insert `insertion` after the first occurrence of `marker`."""
    if not prompt:
        return insertion
    if marker in prompt:
        return prompt.replace(marker, f"{marker}\n{insertion}", 1)
    return f"{prompt}\n\n{insertion}"
