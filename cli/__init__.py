"""Command-Line Interface package for the Loan Approval system."""

def main():
    """Lazy entry point to avoid circular import warnings."""
    from .main import main as _main
    return _main()

__all__ = ["main"]
