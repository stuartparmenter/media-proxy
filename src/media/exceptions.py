# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Media-layer exceptions for clean abstraction from library-specific errors.

These exceptions provide a consistent interface for error handling across different
media processing libraries (PyAV, PIL, aiohttp) without exposing library-specific
exception types to upper layers.

Design Pattern:
    All diagnostic information (URLs, error codes, frames decoded, etc.) should be
    logged immediately before raising exceptions. The exception attributes provide
    structured data for retry logic and error categorization, not for logging.
"""


class MediaSourceError(Exception):
    """Base exception for media processing errors.

    Attributes:
        source_url: URL of the media source that caused the error
        error_code: Numeric error code (e.g., errno, HTTP status)
        retryable: Whether the operation should be retried
    """

    def __init__(
        self,
        message: str,
        source_url: str | None = None,
        error_code: int | None = None,
        retryable: bool = False,
    ):
        """Initialize media source error.

        Args:
            message: Human-readable error description
            source_url: URL of the media source
            error_code: Numeric error code (errno, HTTP status, etc.)
            retryable: Whether this error is transient and retryable
        """
        super().__init__(message)
        self.source_url = source_url
        self.error_code = error_code
        self.retryable = retryable


class MediaNetworkError(MediaSourceError):
    """Network/HTTP/I/O errors (typically retryable).

    Raised for:
    - HTTP errors (4xx/5xx)
    - Network connection failures
    - I/O errors during streaming
    - Timeout errors

    Network errors are generally retryable, but 4xx errors may not be.
    """

    def __init__(
        self,
        message: str,
        source_url: str | None = None,
        error_code: int | None = None,
        retryable: bool = True,
    ):
        """Initialize network error.

        Args:
            message: Human-readable error description
            source_url: URL of the media source
            error_code: HTTP status code or errno
            retryable: Whether to retry (default True for network errors)
        """
        super().__init__(message, source_url, error_code, retryable)


class MediaFormatError(MediaSourceError):
    """Unsupported format or corrupt data (not retryable).

    Raised for:
    - Unrecognized media format
    - Invalid/corrupt media data
    - Unsupported codec
    - Format parsing errors

    Format errors are permanent and will not be fixed by retrying.
    """

    def __init__(self, message: str, source_url: str | None = None):
        """Initialize format error.

        Args:
            message: Human-readable error description
            source_url: URL of the media source
        """
        super().__init__(message, source_url, retryable=False)


class MediaDecodeError(MediaSourceError):
    """Decode failures during media processing.

    Raised for:
    - Decode errors during frame iteration
    - Filter graph errors
    - Image processing errors

    Retryability depends on the specific error cause.
    """

    def __init__(
        self,
        message: str,
        source_url: str | None = None,
        error_code: int | None = None,
        retryable: bool = False,
    ):
        """Initialize decode error.

        Args:
            message: Human-readable error description
            source_url: URL of the media source
            error_code: Numeric error code if available
            retryable: Whether this specific decode error is retryable
        """
        super().__init__(message, source_url, error_code, retryable)


class MediaNotFoundError(MediaSourceError):
    """Media file or resource not found (not retryable).

    Raised for:
    - Local file does not exist
    - HTTP 404 Not Found
    - Missing resources

    Not found errors are permanent and will not be fixed by retrying.
    """

    def __init__(self, message: str, source_url: str | None = None):
        """Initialize not found error.

        Args:
            message: Human-readable error description
            source_url: URL or path of the missing media source
        """
        super().__init__(message, source_url, retryable=False)
