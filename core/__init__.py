"""Framework-free assessment pipeline.

Deliberately contains no Django imports: detection, severity, cost and report
generation can all be unit-tested without a database, a request or a settings
module, and could be lifted into a different application unchanged.

    from core.pipeline import assess
"""

from .damage_config import CLASS_NAMES, SEVERITY_ORDER  # noqa: F401

__all__ = ["CLASS_NAMES", "SEVERITY_ORDER"]
