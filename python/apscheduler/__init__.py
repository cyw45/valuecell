"""Small APScheduler-compatible surface used by the local ValueCell runtime.

The production dependency is still declared in pyproject.toml. This module keeps
the quant-only deployment operational when the server cannot reach PyPI/GHCR
during a rebuild.
"""

__version__ = "3.11.3-local"
