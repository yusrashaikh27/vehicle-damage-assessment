"""Django admin: the report's Admin Panel layer.

The report lists five administrator capabilities. Four come almost free from
registering these models, and it is worth knowing exactly which:

    "Manage registered users"            -> django.contrib.auth's own admin
    "Update and maintain damage records" -> InspectionAdmin
    "Modify repair cost information"     -> the editable CostLine inline
    "View reports and analytics"         -> the changelist: filters, search,
                                            date drill-down, totals column

The fifth, "configure system settings", is deliberately *not* an editable table.
The rate card lives in `core/damage_config.py`, in version control, because a
quotation has to stay reproducible: if an administrator could edit part prices in
the database, a report saved in August could not be re-derived in September and
nobody could say which rates produced it. Changing a rate is a code change with
a commit against it, which is the honest answer for anything that outputs money.

What an administrator *can* adjust is an individual quotation's lines, and when
they do, `save_related` below recomputes the inspection's totals from those
lines. A stored total that disagrees with the lines above it would be worse than
no total at all.

Why detections are not an inline on Inspection any more
------------------------------------------------------
A detection now belongs to a photograph, not to an inspection, because its area
fraction was measured against one frame. Django's admin does not do nested
inlines, so the hierarchy is expressed as two levels of page: an inspection lists
its photographs, and each photograph lists what was found in it. That is a
truthful representation of the data rather than a flattened one - a single table
of detections under an inspection would put percentages with different
denominators in one column.
"""

from __future__ import annotations

from decimal import Decimal

from django.contrib import admin, messages
from django.db.models import QuerySet
from django.urls import reverse
from django.utils.html import format_html

from core.cost import money
from core.damage_config import COST_SOURCE, GST_RATE

from .models import CostLine, DamageDetection, Inspection, VehicleImage

admin.site.site_header = "Vehicle Damage Assessment - administration"
admin.site.site_title = "VDAC admin"
admin.site.index_title = "Damage records, quotations and users"

# GST_RATE is a float in the config (it is a rate, not an amount). Converting
# via str() keeps the Decimal exact: Decimal(0.18) would carry the float's
# binary error into a money calculation, Decimal("0.18") does not.
GST_DECIMAL = Decimal(str(GST_RATE))

SEVERITY_COLOURS = {"minor": "#2e7d5b", "moderate": "#b26b00", "severe": "#b3261e"}


def _thumbnail(image, width: int = 96) -> str:
    """A small preview, or a dash. Shared so the two changelists agree."""
    if not image:
        return "-"
    return format_html(
        '<img src="{}" style="width:{}px;height:auto;border:1px solid #c9cdd3" alt="">',
        image.url,
        width,
    )


# ---------------------------------------------------------------------------
# Inlines
# ---------------------------------------------------------------------------

class VehicleImageInline(admin.TabularInline):
    """The walk-around, on the inspection page.

    Angle, part and file stay editable: adding a missed angle to an existing
    record is a legitimate correction, and the "Re-run assessment" action then
    assesses it. Everything measured *from* the file is read-only, for the same
    reason the detections are.
    """

    model = VehicleImage
    extra = 0
    fields = ("angle", "panel", "image", "preview", "found", "frame", "open_link")
    readonly_fields = ("preview", "found", "frame", "open_link")

    @admin.display(description="Annotated")
    def preview(self, obj: VehicleImage) -> str:
        return _thumbnail(obj.display_image, width=120)

    @admin.display(description="Regions")
    def found(self, obj: VehicleImage) -> str:
        if not obj.pk:
            return "-"
        return str(obj.damage_count)

    @admin.display(description="Frame")
    def frame(self, obj: VehicleImage) -> str:
        if not obj.image_width:
            return "not measured"
        return format_html(
            "{}&times;{} px<br><small>ref. area {} px</small>",
            obj.image_width,
            obj.image_height,
            f"{obj.reference_area_px:,.0f}",
        )

    @admin.display(description="")
    def open_link(self, obj: VehicleImage) -> str:
        """Link through to the photograph's own page, where its detections are.

        This is what stands in for a nested inline.
        """
        if not obj.pk:
            return "-"
        url = reverse("admin:assessment_vehicleimage_change", args=[obj.pk])
        return format_html('<a href="{}">Detections &rarr;</a>', url)


class DamageDetectionInline(admin.TabularInline):
    """Model output, on the photograph page. Read-only on purpose - see below."""

    model = DamageDetection
    extra = 0
    can_delete = False
    fields = ("class_name", "severity", "confidence", "area_percent_display", "reason")
    readonly_fields = fields

    # These rows are what the neural network measured. Letting an administrator
    # edit them would turn a record of what the system found into a record of
    # what somebody typed, and the evaluation numbers in the report depend on
    # the difference.
    def has_add_permission(self, request, obj=None) -> bool:
        return False

    @admin.display(description="Area")
    def area_percent_display(self, obj: DamageDetection) -> str:
        return f"{obj.area_percent:.2f}%"


class CostLineInline(admin.TabularInline):
    """Editable: the report's "modify repair cost information"."""

    model = CostLine
    extra = 0
    fields = (
        "class_name", "severity", "component", "action", "instances",
        "part_cost", "labour_cost", "paint_cost", "labour_hours",
    )


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

@admin.register(Inspection)
class InspectionAdmin(admin.ModelAdmin):
    list_display = (
        "docket_number", "created_at", "owner_name", "registration_number",
        "vehicle_display", "view_count_display", "severity_badge",
        "damage_count", "total_display", "source_display",
    )
    list_filter = ("status", "overall_severity", "is_stub", "segment")
    search_fields = (
        "registration_number", "owner_name", "owner_phone",
        "vehicle_make", "vehicle_model",
    )
    date_hierarchy = "created_at"
    inlines = [VehicleImageInline, CostLineInline]
    actions = ["rerun_assessment", "regenerate_pdf"]

    readonly_fields = (
        "created_at", "docket_number", "model_name", "is_stub",
        "processing_ms", "notes", "error_message", "rate_source",
    )

    fieldsets = (
        ("Record", {
            "fields": ("docket_number", "created_at", "status", "submitted_by"),
        }),
        ("Owner", {
            "fields": ("owner_name", "owner_phone", "owner_email"),
        }),
        ("Vehicle", {
            "fields": (
                "registration_number", "vehicle_make", "vehicle_model",
                "vehicle_year", "segment",
            ),
        }),
        ("Assessment", {
            # Frame sizes and reference areas are per photograph, so they are on
            # the photographs below rather than duplicated here.
            "fields": (
                "overall_severity", "model_name", "is_stub", "processing_ms",
            ),
        }),
        ("Quotation", {
            "fields": (
                "parts_total", "labour_total", "paint_total", "subtotal",
                "gst_amount", "total", "currency", "notes", "rate_source",
            ),
        }),
        ("Report", {"fields": ("report_pdf", "error_message")}),
    )

    def get_queryset(self, request) -> QuerySet[Inspection]:
        # view_count and damage_count both walk the photographs, and damage_count
        # walks their detections. Without prefetching, a 50-row changelist issues
        # 51 queries for the first column and one per photograph for the second.
        # Classic N+1, cheap to avoid.
        return super().get_queryset(request).prefetch_related("images__detections")

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------

    @admin.display(description="Vehicle")
    def vehicle_display(self, obj: Inspection) -> str:
        return obj.vehicle_description

    @admin.display(description="Views")
    def view_count_display(self, obj: Inspection) -> int:
        return obj.view_count

    @admin.display(description="Severity", ordering="overall_severity")
    def severity_badge(self, obj: Inspection) -> str:
        if not obj.overall_severity:
            return "-"
        return format_html(
            '<b style="color:{}">{}</b>',
            SEVERITY_COLOURS.get(obj.overall_severity, "#333"),
            obj.get_overall_severity_display(),
        )

    @admin.display(description="Total", ordering="total")
    def total_display(self, obj: Inspection) -> str:
        return money(obj.total)

    @admin.display(description="Source")
    def source_display(self, obj: Inspection) -> str:
        """Flag placeholder results in the list, not just on the detail page.

        An administrator scanning the changelist must be able to see at a glance
        which records came from the stub detector.
        """
        if obj.is_stub:
            return format_html('<span style="color:#b3261e">stub</span>')
        return format_html('<span style="color:#2e7d5b">model</span>')

    @admin.display(description="Rate card provenance")
    def rate_source(self, obj: Inspection) -> str:
        return COST_SOURCE

    # ------------------------------------------------------------------
    # Keeping totals honest after a manual edit
    # ------------------------------------------------------------------

    def save_related(self, request, form, formsets, change) -> None:
        """Recompute the inspection's totals from its cost lines.

        Runs after the inlines are saved, which is the only point at which the
        edited lines are visible in the database. If an administrator changes a
        part price, the subtotal, GST and total follow it.

        Note what this deliberately does *not* do: adding a photograph through
        the inline does not re-price anything, because pricing needs the detector
        to run over the new frame first. "Re-run assessment" is that step, and
        leaving it explicit means a save never silently changes a quotation the
        administrator was not looking at.
        """
        super().save_related(request, form, formsets, change)

        inspection: Inspection = form.instance
        lines = list(inspection.cost_lines.all())
        if not lines:
            return

        parts = sum((line.part_cost for line in lines), Decimal("0.00"))
        labour = sum((line.labour_cost for line in lines), Decimal("0.00"))
        paint = sum((line.paint_cost for line in lines), Decimal("0.00"))
        subtotal = parts + labour + paint
        gst = (subtotal * GST_DECIMAL).quantize(Decimal("0.01"))

        Inspection.objects.filter(pk=inspection.pk).update(
            parts_total=parts, labour_total=labour, paint_total=paint,
            subtotal=subtotal, gst_amount=gst, total=subtotal + gst,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @admin.action(description="Re-run assessment on the original photographs")
    def rerun_assessment(self, request, queryset: QuerySet[Inspection]) -> None:
        """Re-assess selected records with whatever weights are installed now.

        The reason this exists: everything built before training finished was
        assessed by the stub detector. Once best.pt lands, this converts those
        placeholder records into real ones without re-uploading the photographs.
        It is also how a newly added angle gets assessed.
        """
        from .services import run_assessment  # local import: avoids a cycle

        ok = failed = 0
        for inspection in queryset:
            try:
                run_assessment(inspection)
                ok += 1
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failed += 1
                self.message_user(
                    request, f"{inspection.docket_number}: {exc}", messages.ERROR
                )
        if ok:
            self.message_user(request, f"Re-assessed {ok} inspection(s).",
                              messages.SUCCESS)
        if failed:
            self.message_user(request, f"{failed} failed.", messages.WARNING)

    @admin.action(description="Regenerate the PDF report")
    def regenerate_pdf(self, request, queryset: QuerySet[Inspection]) -> None:
        from .services import build_pdf_report

        count = 0
        for inspection in queryset:
            if build_pdf_report(inspection):
                count += 1
        self.message_user(request, f"Regenerated {count} report(s).",
                          messages.SUCCESS)


# ---------------------------------------------------------------------------
# Photographs
# ---------------------------------------------------------------------------

@admin.register(VehicleImage)
class VehicleImageAdmin(admin.ModelAdmin):
    """One photograph and what was found in it.

    Registered in its own right rather than only as an inline, because this is
    where the detections live - and because "show me every close-up that found
    nothing" is a useful question during evaluation and is one filter here.
    """

    list_display = ("preview", "inspection", "angle", "panel",
                    "damage_count_display", "frame_display")
    list_filter = ("angle", "panel")
    search_fields = (
        "inspection__registration_number",
        "inspection__owner_name",
    )
    list_select_related = ("inspection",)
    inlines = [DamageDetectionInline]

    readonly_fields = ("preview_large", "image_width", "image_height",
                       "reference_area_px", "processing_ms")
    fields = ("inspection", "angle", "panel", "image", "annotated_image",
              "preview_large", "image_width", "image_height",
              "reference_area_px", "processing_ms")

    def get_queryset(self, request) -> QuerySet[VehicleImage]:
        return super().get_queryset(request).prefetch_related("detections")

    def has_add_permission(self, request) -> bool:
        # A photograph is added through its inspection, where the angle and part
        # have an inspection to belong to. Adding one here would invite a row
        # with no parent chosen.
        return False

    @admin.display(description="Annotated")
    def preview(self, obj: VehicleImage) -> str:
        return _thumbnail(obj.display_image, width=110)

    @admin.display(description="Annotated image")
    def preview_large(self, obj: VehicleImage) -> str:
        if not obj.annotated_image:
            return "Not generated."
        return _thumbnail(obj.annotated_image, width=520)

    @admin.display(description="Regions")
    def damage_count_display(self, obj: VehicleImage) -> int:
        return obj.damage_count

    @admin.display(description="Frame")
    def frame_display(self, obj: VehicleImage) -> str:
        if not obj.image_width:
            return "-"
        return f"{obj.image_width}x{obj.image_height}"


@admin.register(DamageDetection)
class DamageDetectionAdmin(admin.ModelAdmin):
    """Registered separately so detections can be filtered across inspections.

    Useful during evaluation: "show me every glass_shatter the model called
    severe" is one click here, and would otherwise be a shell query.
    """

    list_display = ("docket", "image", "class_name", "severity", "confidence",
                    "area_display")
    list_filter = ("class_name", "severity", "image__angle")
    search_fields = (
        "image__inspection__registration_number",
        "image__inspection__owner_name",
    )
    # Two joins deep, because the docket column reads through the photograph to
    # the inspection. Without the second, every row would fetch its own.
    list_select_related = ("image", "image__inspection")

    def has_add_permission(self, request) -> bool:
        return False

    @admin.display(description="Docket", ordering="image__inspection__created_at")
    def docket(self, obj: DamageDetection) -> str:
        return obj.image.inspection.docket_number

    @admin.display(description="Area", ordering="area_fraction")
    def area_display(self, obj: DamageDetection) -> str:
        return f"{obj.area_percent:.2f}%"


@admin.register(CostLine)
class CostLineAdmin(admin.ModelAdmin):
    list_display = ("inspection", "class_name", "severity", "component",
                    "action", "subtotal_display")
    list_filter = ("class_name", "severity", "component")
    search_fields = ("inspection__registration_number", "inspection__owner_name")
    list_select_related = ("inspection",)

    @admin.display(description="Line total")
    def subtotal_display(self, obj: CostLine) -> str:
        return money(obj.subtotal)
