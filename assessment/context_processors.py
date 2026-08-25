"""Template context available on every page."""

from __future__ import annotations

from .services import detector_status


def detector_banner(request) -> dict:
    """Expose whether a trained model is installed.

    A context processor rather than a per-view context entry, because the
    warning has to appear on every page. If it were added view by view, the one
    page somebody forgot would be the page that showed invented detections
    without a warning.
    """
    return {"detector": detector_status()}
