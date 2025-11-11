class SentimentAnalysisError(Exception):
    """Raised when sentiment analysis fails."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}
