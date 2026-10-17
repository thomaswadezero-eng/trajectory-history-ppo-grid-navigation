class ProtocolError(RuntimeError):
    """Raised when a frozen protocol or configuration is inconsistent."""


class IntegrityError(RuntimeError):
    """Raised when an input, action mask, checkpoint, or output fails audit."""

