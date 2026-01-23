"""Command-line interface for the robotics subnet.

This module serves as the CLI entry point and imports the app from the cli package.
"""

from kinitro.cli import app

__all__ = ["app"]


if __name__ == "__main__":
    app()
