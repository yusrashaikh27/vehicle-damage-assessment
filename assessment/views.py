"""Views: upload, result, history, PDF download.

These are deliberately thin. Each one validates input, calls
`services.run_assessment`, and renders - there is no detection, severity or
pricing logic in this file. If you are looking for a number the site displayed,
it came from `core/`, and the view only carried it.
"""

from __future__ import annotations

import logging

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Q, Sum
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render

from core.damage_config import COST_SOURCE, GST_RATE

from .forms import InspectionForm, VehicleImageFormSet, walkaround_initial
from .models import Inspection
from .services import AssessmentFailed, build_pdf_report, run_assessment

logger = logging.getLogger(__name__)

PAGE_SIZE = 12

# Every Inspection field the history page reads. Listed explicitly and used with
# .only() so a page of twelve records does not drag along notes, error messages
# and six money columns it never shows. The trap with .only() is that touching an
# unlisted field issues a *fresh query per row* - silently, so the page still
# works and just gets slower - which is why this list lives next to the view and
# is named rather than inlined.
HISTORY_FIELDS = (
    "id", "created_at", "owner_name", "registration_number",
    "vehicle_make", "vehicle_model", "vehicle_year",
    "overall_severity", "total", "status", "is_stub",
)


def upload(request):
    """Stage 1 of the report's methodology, and the whole pipeline behind it.

    Two forms are validated together: the owner and vehicle details, and the
    walk-around photographs. Both must pass before anything is written, because a
    saved Inspection with no photographs is a record of nothing - and the
    photographs cannot be saved first, since they need the inspection's id.
    """
    if request.method == "POST":
        form = InspectionForm(request.POST)
        # instance=None makes an unsaved Inspection for the formset to hang the
        # slots off. The real one is attached below, once it has a primary key.
        formset = VehicleImageFormSet(
            request.POST, request.FILES,
            instance=None, initial=walkaround_initial(),
        )

        # Both validated before either is saved: `and` would short-circuit and
        # the user would fix the vehicle details only to discover the photograph
        # problems on the next attempt.
        details_ok = form.is_valid()
        photos_ok = formset.is_valid()

        if details_ok and photos_ok:
            inspection: Inspection = form.save(commit=False)
            if request.user.is_authenticated:
                inspection.submitted_by = request.user
            inspection.status = Inspection.Status.PENDING
            # Saved before assessing, so the photographs are on disk and the row
            # exists. If inference then fails, there is a record of the attempt
            # carrying the error - a failure that leaves no trace is
            # unreportable.
            inspection.save()

            formset.instance = inspection
            formset.save()

            try:
                run_assessment(inspection)
            except AssessmentFailed as exc:
                messages.error(
                    request,
                    "The assessment could not be completed. The record was kept "
                    f"so it can be retried: {exc}",
                )
            return redirect("assessment:result", pk=inspection.pk)
    else:
        form = InspectionForm()
        formset = VehicleImageFormSet(instance=None, initial=walkaround_initial())

    return render(request, "assessment/upload.html", {
        "form": form,
        "formset": formset,
        "gst_percent": GST_RATE * 100,
    })


def result(request, pk: int):
    """One completed inspection: every photograph, what was found, the quotation."""
    inspection = get_object_or_404(
        Inspection.objects.prefetch_related("images__detections", "cost_lines"),
        pk=pk,
    )

    # One entry per photograph, each carrying its own detections and the severity
    # geometry for them. Grouped by photograph rather than presented as one flat
    # table because an area percentage is measured against a single image's
    # reference area - a row that did not say which shot it came from would be a
    # number without a denominator.
    views = []
    for view in inspection.images.all():
        detections = list(view.detections.all())
        views.append({
            "view": view,
            "detections": detections,
            "rows": _severity_scale(detections),
        })

    cost_lines = list(inspection.cost_lines.all())

    return render(request, "assessment/result.html", {
        "inspection": inspection,
        "views": views,
        "cost_lines": cost_lines,
        "gst_percent": GST_RATE * 100,
        "cost_source": COST_SOURCE,
        # True when more than one photograph contributed, which is when the
        # merge-before-pricing explanation is worth showing.
        "multi_view": len(views) > 1,
    })


def history(request):
    """Past inspections, newest first, with a small amount of aggregate detail."""
    queryset = Inspection.objects.all()

    query = (request.GET.get("q") or "").strip()
    if query:
        queryset = queryset.filter(
            Q(registration_number__icontains=query)
            | Q(owner_name__icontains=query)
            | Q(vehicle_make__icontains=query)
            | Q(vehicle_model__icontains=query)
        )

    # One query for all four figures instead of four. `filter=` inside Count is
    # a conditional aggregate - the severity breakdown without a second trip.
    summary = Inspection.objects.aggregate(
        total_count=Count("id"),
        value=Sum("total"),
        severe=Count("id", filter=Q(overall_severity="severe")),
        stubbed=Count("id", filter=Q(is_stub=True)),
    )

    paginator = Paginator(
        # images__detections rather than images: the table shows a damage count
        # per record, and prefetching only the photographs would mean one query
        # per photograph to count what is on it - worse than the N+1 it fixed.
        queryset.prefetch_related("images__detections").only(*HISTORY_FIELDS),
        PAGE_SIZE,
    )
    page = paginator.get_page(request.GET.get("page"))

    return render(request, "assessment/history.html", {
        "page": page,
        "query": query,
        "summary": summary,
    })


def download_report(request, pk: int):
    """Serve the PDF, generating it on demand if it is missing.

    Generating on demand covers the case where a record was created before the
    PDF module existed, or where rendering failed once. It means the download
    link never dead-ends.
    """
    inspection = get_object_or_404(
        Inspection.objects.prefetch_related("images__detections", "cost_lines"),
        pk=pk,
    )

    if not inspection.report_pdf:
        try:
            build_pdf_report(inspection)
        except Exception as exc:  # noqa: BLE001
            logger.exception("on-demand PDF failed for %s", pk)
            raise Http404("The report could not be generated.") from exc

    return FileResponse(
        inspection.report_pdf.open("rb"),
        as_attachment=True,
        filename=f"{inspection.docket_number}.pdf",
        content_type="application/pdf",
    )


# ---------------------------------------------------------------------------
# Presentation helper
# ---------------------------------------------------------------------------

def _severity_scale(detections) -> list[dict]:
    """Per-detection data for the area meter on the results page.

    The meter shows the measured area against the two thresholds that banded it,
    so a reader can see *why* a verdict was reached instead of being asked to
    accept it. That is the honest way to present a rule-based severity: show the
    rule and the measurement side by side.

    The bar is drawn on a scale of three times the moderate threshold, so the
    "severe" region is always visible on the right rather than the marker
    pinning at 100% for anything large.

    Every geometry figure is finished here, in percent, ready to drop into a
    style attribute. The template does no arithmetic: Django's `add` filter
    falls back to string concatenation when given floats, so a subtraction like
    `moderate_pct|add:"-0.5"` would quietly produce a nonsense width rather than
    an error. Layout maths belongs in Python where it can be read and tested.

    One entry is returned per detection, always. `damage_config._validate()`
    guarantees a rule exists for every class in CLASS_NAMES, so the fallback
    below should be unreachable - but a table that quietly renders four rows for
    five detections is a worse outcome than one that says a threshold is
    missing, so the row is kept and labelled.
    """
    from core.damage_config import SEVERITY_RULES

    scale = []
    for det in detections:
        rule = SEVERITY_RULES.get(det.class_name)
        if rule is None:
            scale.append({
                "detection": det,
                "minor_width": 0, "moderate_left": 0, "moderate_width": 0,
                "severe_left": 0, "severe_width": 0, "value_pct": 0,
                "minor_max_pct": None, "moderate_max_pct": None,
                "floor": None,
                "note": "No severity threshold is on record for this class.",
            })
            continue
        span = max(rule.moderate_max * 3, det.area_fraction, 1e-6)
        minor_end = min(rule.minor_max / span * 100, 100)
        moderate_end = min(rule.moderate_max / span * 100, 100)
        scale.append({
            "detection": det,
            # Zone geometry: three abutting bands, each an absolute left + width.
            "minor_width": minor_end,
            "moderate_left": minor_end,
            "moderate_width": max(moderate_end - minor_end, 0),
            "severe_left": moderate_end,
            "severe_width": max(100 - moderate_end, 0),
            # Where the measurement actually landed.
            "value_pct": min(det.area_fraction / span * 100, 100),
            # The thresholds as readable percentages, for the axis labels. These
            # are the two numbers in the rule that decided the verdict.
            "minor_max_pct": rule.minor_max * 100,
            "moderate_max_pct": rule.moderate_max * 100,
            "floor": rule.floor,
            "note": rule.note,
        })
    return scale
