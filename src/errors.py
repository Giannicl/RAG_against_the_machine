"""Error types raised across the pipeline and reported by the CLI."""


class InputError(Exception):
    """Raised when input the user controls is missing, empty or malformed."""
