"""Load local environment configuration from .env when present."""

from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables without overriding values already in the shell."""
    load_dotenv()
