"""Local, evidence-scoped semantic conformance for robot policy adapters."""

__version__ = "0.1.0"

from .checker import check

__all__ = ["check", "main"]


def main() -> int:
    from .cli import main as cli_main

    return cli_main()
