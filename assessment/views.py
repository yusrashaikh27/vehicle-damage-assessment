"""Views: landing, upload, result, history, PDF download.

These are deliberately thin. Each one validates input, calls
`services.run_assessment`, and renders - there is no detection, severity or
pricing logic in this file. If you are looking for a number the site displayed,
it came from `core/`, and the view only carried it.

The chart helpers at the bottom are the one apparent exception, and they are not
one: they turn figures that already exist into pixel coordinates. No figure is
computed here, only positioned. That arithmetic lives in Python rather than in a
template for the same reason `_severity_scale` does - Django's `add` filter
falls back to string concatenation on floats, so layout maths in a template
fails silently rather than loudly.
"""

from __future__ import annotations

import logging

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Q, Sum
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render

from core import evaluation
from core.damage_config import COST_SOURCE, GST_RATE

from .forms import InspectionForm, VehicleImageFormSet, walkaround_initial
from .models import DamageDetection, Inspection
from .services import AssessmentFailed, build_pdf_report, run_assessment

logger = logging.getLogger(__name__)

PAGE_SIZE = 12

# Every Inspection field the history page reads. Listed explicitly and used with
# .only() so a page of twelve records does not drag along notes, error messages
# and six money columns it never shows. The trap with .only() is that touching an
# unlisted field issues a *fresh query per row* - silently, so the page still
# works and just gets slower - which is why this list lives next to the view and
# is named rather than inlined.
#
# range_low / range_high are listed because the history table shows each
# estimate as a range (Inspection.range_display reads both, and falls back to
# `total` for records saved before the range existed).
HISTORY_FIELDS = (
    "id", "created_at", "owner_name", "registration_number",
    "vehicle_make", "vehicle_model", "vehicle_year",
    "overall_severity", "total", "range_low", "range_high",
    "status", "is_stub",
)


def landing(request):
    """The front page: what the system does, and how well it actually does it.

    This page exists to be the honest one. An examiner's first question about a
    project like this is "how accurate is it", and the defensible answer is a
    measured figure with its caveats attached rather than a percentage on a
    slide. So the measured test-split results, the boundary-quality limitation
    that 84% of instances fall under, and the reason accuracy does not track
    training volume are all on the front page, above anything promotional.

    `Inspection.objects.count()` is cheap and this page is not hot, so the live
    figures are read on every request rather than cached - a stale count on the
    front page of a demo is worse than the query.
    """
    return render(request, "assessment/landing.html", {
        "ev": evaluation,
        "class_results": evaluation.CLASS_RESULTS,
        "accuracy_chart": _accuracy_chart(),
        "unreliable_share": evaluation.unreliable_area_share() * 100,
        "gst_percent": GST_RATE * 100,
        "cost_source": COST_SOURCE,
        "inspection_count": Inspection.objects.count(),
    })


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
    """One completed inspection: every photograph, what was found, the quotation.

    The estimate is shown as a range. The template reads it from the inspection:
    `inspection.range_display` for the headline figure, `inspection.parts_range`,
    `labour_range`, `paint_range`, `subtotal_range` and `gst_range` for the
    breakdown, and `line.subtotal_range` for each cost line.
    """
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

    # One query for all the figures instead of one each. `filter=` inside Count
    # is a conditional aggregate - the severity breakdown without a second trip.
    #
    # `value` (the sum of single totals) is kept so an existing template does not
    # break. `low_value` and `high_value` are the same sum taken over the stored
    # range ends: the portfolio is described as a range too, not as one number.
    summary = Inspection.objects.aggregate(
        total_count=Count("id"),
        value=Sum("total"),
        low_value=Sum("range_low"),
        high_value=Sum("range_high"),
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
        # Charts describe the whole table, not the current page or the current
        # search, so they are built from the unfiltered queryset. A distribution
        # that silently changed when you typed in the search box would be
        # describing something other than what its title claims.
        "severity_mix": _severity_mix(),
        "class_mix": _class_mix(),
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
# Presentation helpers
# ---------------------------------------------------------------------------

# Dumbbell chart geometry, in SVG user units. The SVG carries a viewBox and no
# fixed width, so these are a coordinate system rather than pixels - the browser
# scales the whole thing to whatever the card is wide.
_CHART_W = 620
_PLOT_X0 = 152          # left edge of the plot; everything before it is labels
_PLOT_X1 = 596
_ROW_H = 30
_ROW_TOP = 18           # centre of the first row
_AXIS_GAP = 16          # from the last row's centre down to the axis rule


def _accuracy_chart() -> dict:
    """Per-class accuracy as a dumbbell: mAP50-95 and mAP50, joined.

    A dumbbell rather than a bar chart because the *gap* is the finding. A bar
    of mAP50 alone would say crack is the weakest class and stop there; the
    distance between the two dots is how much of the score the model loses when
    the overlap requirement tightens, which is the same thing as how badly it
    draws the outline. Since the pipeline consumes the outline - area becomes
    severity becomes cost - that gap is the error that reaches the rupee figure,
    and it should be the most visible thing on the chart.

    One hue in two shades, not two hues: these are two measurements of one
    quantity at different strictness, not two independent series.

    Only the mAP50 end is labelled. A value beside all fourteen dots would be
    unreadable; the axis carries the rest and the table view below the chart
    carries every figure exactly.
    """
    rows = []
    for i, c in enumerate(evaluation.CLASS_RESULTS):
        y = _ROW_TOP + i * _ROW_H
        rows.append({
            "result": c,
            "y": y,
            "x_strict": _x(c.map50_95),
            "x_loose": _x(c.map50),
            # Label sits just past the outer dot, except where that would run
            # off the plot - then it goes inside, to the left of the dot.
            "label_x": _x(c.map50) + 11,
            "label_anchor": "start",
            "approximate": c.name in evaluation.APPROXIMATE,
        })

    # A label at x=1.0 would overflow the viewBox, so tire_flat's flips inward.
    for row in rows:
        if row["label_x"] > _PLOT_X1 - 26:
            row["label_x"] = _x(row["result"].map50) - 11
            row["label_anchor"] = "end"

    axis_y = _ROW_TOP + (len(rows) - 1) * _ROW_H + _AXIS_GAP

    return {
        "rows": rows,
        "width": _CHART_W,
        "height": axis_y + 22,
        "plot_x0": _PLOT_X0,
        "plot_x1": _PLOT_X1,
        "axis_y": axis_y,
        "grid_top": 6,
        "grid_bottom": axis_y,
        # Two coordinates the template would otherwise have to derive with the
        # `add` filter. They are here for the reason given in the module
        # docstring: `add` concatenates rather than sums when either side is a
        # float, so it produces "152-12" instead of 140 and the SVG silently
        # drops the attribute. Computing them here cannot fail that way.
        "label_col_x": _PLOT_X0 - 12,   # right edge of the class-name column
        "tick_label_y": axis_y + 15,    # baseline of the axis numerals
        # Ticks at clean quarters. Gridlines are hairline and solid, never
        # dashed, and sit behind the marks.
        "ticks": [
            {"value": v, "x": _x(v), "label": f"{v:.2f}".lstrip("0") or "0"}
            for v in (0.0, 0.25, 0.5, 0.75, 1.0)
        ],
    }


def _x(value: float) -> float:
    """Map a 0-1 metric onto the plot's horizontal extent."""
    return _PLOT_X0 + value * (_PLOT_X1 - _PLOT_X0)


def _severity_mix() -> dict:
    """How the stored inspections divide across the three severity grades.

    Part-to-whole, so a single stacked bar. The three grades are a *status*
    scale, not a set of independent categories, which is why they keep the
    frozen green / amber / red rather than taking chart colours - and why every
    segment is labelled, since a status colour must never be the only thing
    carrying the meaning.
    """
    order = [
        ("minor", "Minor", "minor"),
        ("moderate", "Moderate", "moderate"),
        ("severe", "Severe", "severe"),
    ]

    counts = dict(
        Inspection.objects.exclude(overall_severity="")
        .values_list("overall_severity")
        .annotate(n=Count("id"))
    )
    # Assessed but nothing found is a real outcome and belongs in the
    # denominator: leaving it out would make "50% severe" out of two records
    # where one was clean, which overstates the severity of the set.
    none_count = Inspection.objects.filter(
        overall_severity="", status=Inspection.Status.COMPLETE,
    ).count()

    total = sum(counts.values()) + none_count
    segments = []
    for key, label, tone in order:
        n = counts.get(key, 0)
        if n:
            segments.append({
                "label": label, "tone": tone, "count": n,
                "percent": n / total * 100,
            })
    if none_count:
        segments.append({
            "label": "No damage found", "tone": "none", "count": none_count,
            "percent": none_count / total * 100,
        })

    return {"segments": segments, "total": total}


def _class_mix() -> dict:
    """Which damage types the system has actually been finding, by count.

    Magnitude across categories, so plain bars in one hue - the classes are not
    competing series and colouring them seven different ways would imply a
    distinction the data does not carry.

    Bars are drawn as a share of the largest count, not of the total: the
    question this answers is "which of these is most common", and scaling to the
    total would flatten every bar into illegibility as the class count grows.
    """
    rows = list(
        DamageDetection.objects.values("class_name")
        .annotate(n=Count("id"))
        .order_by("-n")
    )
    if not rows:
        return {"rows": [], "total": 0}

    labels = dict(DamageDetection._meta.get_field("class_name").choices)
    biggest = rows[0]["n"]
    total = sum(r["n"] for r in rows)

    return {
        "rows": [
            {
                "label": labels.get(r["class_name"], r["class_name"]),
                "count": r["n"],
                "percent": r["n"] / biggest * 100,
                "share": r["n"] / total * 100,
            }
            for r in rows
        ],
        "total": total,
    }


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