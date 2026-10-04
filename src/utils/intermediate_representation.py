"""Intermediate representation related utility functions."""

from typing import Optional


SUPPORTED_INTERMEDIATE_REPRESENTATIONS = {"ilr", "text"}


def normalize_intermediate_representation_type(value: Optional[str]) -> str:
    """Normalize intermediate representation type."""
    normalized = (value or "ilr").strip().lower().replace("_", "-")

    aliases = {
        "ilr": "ilr",
        "text": "text",
        "text-ir": "text",
        "textual": "text",
        "textual-ir": "text",
    }

    result = aliases.get(normalized)
    if result not in SUPPORTED_INTERMEDIATE_REPRESENTATIONS:
        supported = ", ".join(sorted(SUPPORTED_INTERMEDIATE_REPRESENTATIONS))
        raise ValueError(f"Unsupported intermediate representation type: {value}, available values: {supported}")

    return result


def get_intermediate_representation_suffix(representation_type: Optional[str]) -> str:
    """Get the suffix used for output directory distinction."""
    normalized = normalize_intermediate_representation_type(representation_type)
    return "" if normalized == "ilr" else "-text-ir"


def get_intermediate_representation_file_name(representation_type: Optional[str]) -> str:
    """Get the intermediate representation save file name."""
    normalized = normalize_intermediate_representation_type(representation_type)
    return "ilr.jsonl" if normalized == "ilr" else "text_ir.jsonl"


def get_intermediate_representation_subdir(representation_type: Optional[str]) -> str:
    """Get the intermediate representation save subdirectory."""
    normalized = normalize_intermediate_representation_type(representation_type)
    return "" if normalized == "ilr" else "text"


def get_intermediate_representation_display_name(representation_type: Optional[str]) -> str:
    """Get the display name."""
    normalized = normalize_intermediate_representation_type(representation_type)
    return "ILR" if normalized == "ilr" else "Text IR"
