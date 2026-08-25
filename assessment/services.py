"""The only module that knows about both Django and `core`.

Views call `run_assessment`; nothing else in the app imports `core.pipeline`.
Keeping the bridge in one file is what lets `core/` stay free of Django - you can
open a plain Python prompt and price a photograph without a database, a settings
module or a request, which is how the cost and severity logic gets tested.

What happens when an inspection is submitted
-------------------------------------------
1. The rows already exist: one Inspection, and one VehicleImage per photograph,
   with status "Queued".
2. `core.pipeline.assess_walkaround` detects and measures each photograph, then
   prices **once** across all of them.
3. Everything it produced is written to the database inside one transaction -
   detections against the photograph that produced them, cost lines against the
   inspection.
4. The PDF is generated *from the saved rows*, not from the pipeline's return
   value, so the document and the database cannot disagree.

Why the pricing call sits outside the per-photograph loop
--------------------------------------------------------
Because a bumper photographed from the front and again from the front-left is one
bumper to buy. Pricing each view and adding up would charge for two. The merge
happens in `core.cost.estimate_combined`; this module's job is only to make sure
every view reaches it in one call. See core/cost.py for what the merge can and
cannot tell apart.

Why this runs inside the web request
------------------------------------
Inference on one image is roughly a second or two on CPU, so a five-angle
walk-around is several seconds - slow for a web request, but not so slow that a
task queue pays for itself. Celery plus a broker would mean two more processes to
start before a demo, a second failure mode, and a results page that has to poll.
The upgrade path is real if it is ever needed: this function is already the unit
of work a task would wrap, because it takes a primary key's worth of state and
returns nothing. If the angle count grows, that is the change to make.
"""

from __future__ import annotations

import logging
import tempfile
import time
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.db import transaction
from django.utils import timezone

from core import report as report_module
from core.damage_config import ANGLES, COST_SOURCE, PANELS, SEGMENTS
from core.detector import get_detector
from core.pipeline import WalkaroundAssessment, assess_walkaround

from .models import CostLine, DamageDetection, Inspection, VehicleImage

logger = logging.getLogger(__name__)


class AssessmentFailed(RuntimeError):
    """Raised after the failure has been recorded on the Inspection row."""


def _money(value) -> Decimal:
    """float -> Decimal for storage in a money column.

    Via `str`, deliberately. `Decimal(0.1)` captures the float's binary error
    (0.1000000000000000055511151231257827) and carries it into the database;
    `Decimal("0.1")` is exactly a tenth. core.cost already rounds to two
    decimals, so this only has to preserve what it produced.
    """
    return Decimal(str(round(float(value), 2)))


# ---------------------------------------------------------------------------
# Running an assessment
# ---------------------------------------------------------------------------

def run_assessment(inspection: Inspection) -> Inspection:
    """Assess every photograph on the inspection and persist the result.

    Safe to call again on the same row: previous detections and cost lines are
    deleted first, so the admin's "re-run" action replaces results rather than
    accumulating a second copy of them. That matters once real weights land and
    the records assessed by the stub detector need redoing.
    """
    started = time.perf_counter()

    views = list(inspection.images.all())
    if not views:
        # Reachable only by creating an inspection without photographs - the form
        # requires at least one. Failing loudly here beats writing a zero-rupee
        # quotation that looks like a successful assessment of an undamaged car.
        message = "No photographs are attached to this inspection."
        Inspection.objects.filter(pk=inspection.pk).update(
            status=Inspection.Status.FAILED, error_message=message
        )
        inspection.refresh_from_db()
        raise AssessmentFailed(message)

    # FileSystemStorage exposes an on-disk path, which ultralytics needs since it
    # opens the file itself. On a remote storage backend this attribute does not
    # exist and each file would have to be copied to a temp path first - the one
    # place this project assumes local media.
    shots = [(Path(view.image.path), view.panel, view.angle) for view in views]

    detector = get_detector(settings.YOLO_WEIGHTS_PATH)

    try:
        # Annotated images are written to a temp directory and then handed to the
        # ImageField, rather than being written under MEDIA_ROOT directly. That
        # way upload_to, unique-name collision handling and the storage backend
        # all still apply to them - the same as for the uploads.
        with tempfile.TemporaryDirectory() as tmp:
            result = assess_walkaround(
                shots,
                segment=inspection.segment,
                annotated_dir=tmp,
                detector=detector,
            )
            _persist(inspection, views, result, started)

    except Exception as exc:
        logger.exception("assessment failed for inspection %s", inspection.pk)
        Inspection.objects.filter(pk=inspection.pk).update(
            status=Inspection.Status.FAILED, error_message=str(exc)[:2000]
        )
        inspection.refresh_from_db()
        raise AssessmentFailed(str(exc)) from exc

    # Outside the transaction on purpose: a PDF that fails to render must not
    # roll back a perfectly good assessment. The page and the API work without
    # it, and the admin can regenerate it.
    try:
        build_pdf_report(inspection)
    except Exception:
        logger.exception("PDF generation failed for inspection %s", inspection.pk)

    return inspection


@transaction.atomic
def _persist(inspection: Inspection,
             views: list[VehicleImage],
             result: WalkaroundAssessment,
             started: float) -> None:
    """Write the assessment to the database. All of it, or none of it.

    The transaction is what stops a half-written record existing: an inspection
    holding detections but no totals would render a results page with damage
    listed and a bill of zero, which reads as "this repair is free" rather than
    as "something went wrong".

    `views` and `result.views` are the same photographs in the same order -
    `assess_walkaround` preserves the order it was given - which is how each
    detection gets attributed to the photograph it came from.
    """
    quote = result.quotation

    if len(views) != len(result.views):
        # Cannot happen unless assess_walkaround changes its contract. Checked
        # anyway, because the failure mode of a silent mismatch is detections
        # filed against the wrong photograph, which no page would reveal.
        raise RuntimeError(
            f"assessed {len(result.views)} views but {len(views)} were submitted"
        )

    # Re-running replaces previous output rather than adding to it.
    DamageDetection.objects.filter(image__inspection=inspection).delete()
    inspection.cost_lines.all().delete()

    for view, assessed in zip(views, result.views):
        DamageDetection.objects.bulk_create([
            DamageDetection(
                image=view,
                class_name=det.class_name,
                severity=det.severity,
                confidence=det.confidence,
                area_px=det.area_px,
                area_fraction=det.area_fraction,
                reason=det.reason[:200],
                x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3],
            )
            for det in assessed.detections
        ])

        if assessed.annotated_path and Path(assessed.annotated_path).exists():
            annotated = Path(assessed.annotated_path)
            # An earlier overlay for this photograph is deleted first. Without
            # this, storage gives the new file a suffixed name and every re-run
            # leaves another orphan under media/ that nothing links to - and
            # "re-run once the real weights land" is the documented path for
            # every record built against the stub detector, so the leak would be
            # one file per photograph per record.
            if view.annotated_image:
                view.annotated_image.delete(save=False)
            with annotated.open("rb") as handle:
                # save=False: the field is set here and written by the single
                # view.save() below, rather than issuing a second UPDATE.
                view.annotated_image.save(annotated.name, File(handle), save=False)

        view.image_width = assessed.image_width
        view.image_height = assessed.image_height
        view.reference_area_px = assessed.reference_area
        view.processing_ms = assessed.elapsed_ms
        view.save()

    CostLine.objects.bulk_create([
        CostLine(
            inspection=inspection,
            class_name=line.class_name,
            severity=line.severity,
            component=line.component,
            action=line.action[:120],
            instances=line.instances,
            part_cost=_money(line.part_cost),
            labour_cost=_money(line.labour_cost),
            paint_cost=_money(line.paint_cost),
            labour_hours=_money(line.labour_hours),
        )
        for line in quote.lines
    ])

    inspection.status = Inspection.Status.COMPLETE
    inspection.error_message = ""
    inspection.overall_severity = quote.overall_severity
    inspection.parts_total = _money(quote.parts_total)
    inspection.labour_total = _money(quote.labour_total)
    inspection.paint_total = _money(quote.paint_total)
    inspection.subtotal = _money(quote.subtotal)
    inspection.gst_amount = _money(quote.gst)
    inspection.total = _money(quote.total)
    inspection.currency = quote.currency
    inspection.notes = list(quote.notes)
    inspection.model_name = result.model_name
    inspection.is_stub = result.is_stub
    inspection.processing_ms = int((time.perf_counter() - started) * 1000)
    inspection.save()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _report_data(inspection: Inspection) -> report_module.ReportData:
    """Build the PDF's input from the stored record.

    Reading from the database rather than from the pipeline's return value is
    what guarantees the PDF matches the results page, and is also what makes
    "regenerate the report" possible months later without re-running a model.
    """
    local = timezone.localtime(inspection.created_at or timezone.now())
    segment = SEGMENTS.get(inspection.segment)

    views = list(inspection.images.all())

    report_views = []
    detections = []
    for view in views:
        angle = ANGLES.get(view.angle)
        panel = PANELS.get(view.panel)
        angle_label = angle.label if angle else view.angle
        panel_label = panel.label if panel else view.panel

        report_views.append(report_module.ReportView(
            angle_label=angle_label,
            panel_label=panel_label,
            damage_count=len(view.detections.all()),
            annotated_image_path=(
                Path(view.annotated_image.path) if view.annotated_image else None
            ),
            original_image_path=Path(view.image.path) if view.image else None,
        ))

        # Detections are listed with the photograph they came from, because an
        # area percentage is relative to that photograph's reference area and
        # means nothing without knowing which shot it was measured in.
        for det in view.detections.all():
            detections.append(report_module.ReportDetection(
                label=det.get_class_name_display(),
                severity_label=det.get_severity_display(),
                confidence=det.confidence,
                area_percent=det.area_percent,
                reason=det.reason,
                angle_label=angle_label,
            ))

    return report_module.ReportData(
        docket_number=inspection.docket_number,
        created_at=local.strftime("%d %B %Y, %H:%M"),
        owner_name=inspection.owner_name,
        owner_phone=inspection.owner_phone,
        owner_email=inspection.owner_email,
        registration_number=inspection.registration_number,
        vehicle_make=inspection.vehicle_make,
        vehicle_model=inspection.vehicle_model,
        vehicle_year=inspection.vehicle_year,
        segment_label=segment.label if segment else inspection.segment,
        overall_severity_label=inspection.get_overall_severity_display() or "",
        views=report_views,
        detections=detections,
        cost_lines=[
            report_module.ReportCostLine(
                label=line.get_class_name_display(),
                severity_label=line.get_severity_display(),
                component_label=line.get_component_display(),
                action=line.action,
                instances=line.instances,
                part_cost=float(line.part_cost),
                labour_cost=float(line.labour_cost),
                paint_cost=float(line.paint_cost),
                labour_hours=float(line.labour_hours),
            )
            for line in inspection.cost_lines.all()
        ],
        parts_total=float(inspection.parts_total),
        labour_total=float(inspection.labour_total),
        paint_total=float(inspection.paint_total),
        subtotal=float(inspection.subtotal),
        gst_amount=float(inspection.gst_amount),
        total=float(inspection.total),
        notes=list(inspection.notes or []),
        model_name=inspection.model_name,
        is_stub=inspection.is_stub,
        cost_source=COST_SOURCE,
        currency=inspection.currency,
    )


def build_pdf_report(inspection: Inspection) -> bool:
    """Render the PDF and attach it to the inspection. True if it was written."""
    data = _report_data(inspection)

    with tempfile.TemporaryDirectory() as tmp:
        filename = f"{inspection.docket_number}.pdf"
        pdf_path = report_module.build_report(data, Path(tmp) / filename)
        with pdf_path.open("rb") as handle:
            # An earlier PDF for this inspection is deleted first, otherwise
            # storage appends a suffix and media/ fills up with orphaned
            # revisions nothing links to.
            if inspection.report_pdf:
                inspection.report_pdf.delete(save=False)
            inspection.report_pdf.save(filename, File(handle), save=True)

    return True


# ---------------------------------------------------------------------------
# Status, for the banner every page shows
# ---------------------------------------------------------------------------

def detector_status() -> dict:
    """Whether a trained model is installed, for the site-wide banner.

    Surfaced in the UI rather than left in the logs: the stub detector returns
    fixed, invented detections, and a demo that shows those without saying so
    would be presenting fabricated results as real ones.
    """
    weights = Path(settings.YOLO_WEIGHTS_PATH)
    return {
        "weights_path": str(weights),
        "weights_present": weights.exists(),
        "using_stub": not weights.exists(),
    }
