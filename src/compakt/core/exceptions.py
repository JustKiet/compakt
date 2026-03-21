class CompaktError(Exception):
    """Base exception for Compakt domain errors."""


class UnsupportedDocumentStrategyError(CompaktError):
    """Raised when no strategy can process the parsed document."""


class InvalidRetrievalLevelError(CompaktError):
    """Raised when retrieval level is outside supported bounds."""


class EmptyDocumentError(CompaktError):
    """Raised when the input document cannot be parsed into useful content."""
