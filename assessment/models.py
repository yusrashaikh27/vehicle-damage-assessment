"""Database models: one inspection, the photographs taken of it, what was found
in each, and the single priced quotation that results.

Four tables, covering what the report's Data Storage Layer says it keeps:
"vehicle damage records, repair cost information, and report metadata".

Three decisions worth being able to explain
-------------------------------------------

**Money is Decimal, measurements are float.** Every rupee field is a
`DecimalField`. Money must not be stored as a binary float, because values like
0.10 have no exact binary representation, so a column of floats does not
reliably add up to its own total - and a quotation whose lines do not sum to its
total is worse than no quotation. Confidence, pixel areas and area fractions are
*measurements*: they are approximate by nature, never summed into a legal
figure, and float is the right type for them.

**The results are stored, not recomputed.** The pipeline runs once, at upload,
and every number it produced is written to these tables. Reloading a results
page re-reads the database; it does not re-run the model. That is deliberate:
inference takes seconds, and more importantly a saved report must not silently
change when the weights file is replaced. An inspection is a record of what the
system concluded on a given day with a given model - which is why `model_name`
and `is_stub` are stored alongside the numbers.

**Owner details live on the inspection, not in a Customer table.** Normalising
them out would be the textbook answer, and it is the wrong one here. An
inspection is a document issued on a date: if the owner later changes their phone
number, the docket raised last month must still show the number recorded at the
time, exactly as a paper job card would. That is the same argument as storing the
totals rather than recomputing them. The cost of the choice is honest and small -
a repeat customer's details are typed again - and nothing in this system needs to
report across a customer's history.

Why photographs are their own table
-----------------------------------
A damage assessment is a walk-around: front, front-left, left, and so on. Each
photograph has its own angle, its own panel, its own dimensions and its own
detections, so it is a row, not a column. Detections therefore hang off the
photograph that produced them - `DamageDetection.image` - while `CostLine` hangs
off the *inspection*, because the quotation is priced once across every view. See
`core.cost.estimate_combined` for why that has to be so: a bumper photographed
from two angles is still one bumper to buy.
"""

from __future__ import annotations

from decimal import Decimal

from django.conf import settings
from django.db import models
from django.urls import reverse

from core.damage_config import (
    ANGLE_CHOICES,
    ANGLES,
    CLASS_CHOICES,
    CURRENCY,
    DEFAULT_ANGLE,
    DEFAULT_PANEL,
    DEFAULT_SEGMENT,
    PANEL_CHOICES,
    SEGMENT_CHOICES,
    SEVERITY_CHOICES,
)

# All money columns share these limits: up to 99,999,999.99. A written-off
# luxury car does not reach eight figures of panel work, so this is ample.
MONEY = {"max_digits": 10, "decimal_places": 2, "default": Decimal("0.00")}


class Inspection(models.Model):
    """One vehicle, photographed from several angles, assessed and priced."""

    class Status(models.TextChoices):
        PENDING = "pending", "Queued"
        COMPLETE = "complete", "Complete"
        FAILED = "failed", "Failed"

    # --- who and when ---------------------------------------------------
    # SET_NULL rather than CASCADE: deleting a staff account must not delete
    # the inspection records they happened to create. The record outlives the
    # user, which is how any real claims system behaves.
    submitted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="inspections",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    # --- the owner, as recorded at the time -------------------------------
    # See the module docstring for why these are columns here and not a foreign
    # key to a Customer table.
    owner_name = models.CharField("owner's name", max_length=120)
    owner_phone = models.CharField("contact number", max_length=20)
    owner_email = models.EmailField("email", blank=True)

    # --- vehicle details, supplied by the user at upload ------------------
    # The report's stage 1: "the user uploads an image of the damaged vehicle
    # along with relevant vehicle details such as make, model, year, and
    # registration number".
    vehicle_make = models.CharField(max_length=60)
    vehicle_model = models.CharField(max_length=60)
    vehicle_year = models.PositiveIntegerField()
    registration_number = models.CharField(max_length=20)

    # Segment drives part prices and the labour rate for the whole vehicle. The
    # panel is per photograph and lives on VehicleImage, because different angles
    # show different components. Neither can be predicted from the dataset - no
    # class in it names a component - so both are asked for. See core/cost.py.
    segment = models.CharField(
        max_length=20, choices=SEGMENT_CHOICES, default=DEFAULT_SEGMENT
    )

    # --- files ------------------------------------------------------------
    # The photographs are rows in VehicleImage. Only the generated report is a
    # file on the inspection itself.
    report_pdf = models.FileField(upload_to="reports/%Y/%m/", blank=True, null=True)

    # --- outcome ----------------------------------------------------------
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.PENDING
    )
    error_message = models.TextField(blank=True)

    overall_severity = models.CharField(
        max_length=10, choices=SEVERITY_CHOICES, blank=True
    )

    parts_total = models.DecimalField(**MONEY)
    labour_total = models.DecimalField(**MONEY)
    paint_total = models.DecimalField(**MONEY)
    subtotal = models.DecimalField(**MONEY)
    gst_amount = models.DecimalField(**MONEY)
    total = models.DecimalField(**MONEY)
    currency = models.CharField(max_length=8, default=CURRENCY)

    # Caveats the cost model produced ("bumper repaired rather than replaced,
    # so only 35% of the part price is charged", "the same damage type appears in
    # more than one photograph"). A list of strings. Stored rather than
    # regenerated so the saved report and the PDF always agree.
    notes = models.JSONField(default=list, blank=True)

    # --- provenance -------------------------------------------------------
    # Which model produced this, and was it the stub? Without these two
    # columns you cannot later tell a real assessment from a placeholder one,
    # and mistaking one for the other in a demo would be indefensible.
    model_name = models.CharField(max_length=120, blank=True)
    is_stub = models.BooleanField(default=False)

    # Wall-clock time for the whole inspection: every photograph, plus pricing.
    # Per-photograph timings are on VehicleImage.
    processing_ms = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at"], name="insp_created_desc_idx"),
            models.Index(fields=["registration_number"], name="insp_regno_idx"),
            # The history page searches on the owner's name, so it gets an index
            # too. Registration is the usual lookup; name is the fallback for
            # when somebody has the caller on the phone and not the plate.
            models.Index(fields=["owner_name"], name="insp_owner_idx"),
        ]
        verbose_name = "inspection"
        verbose_name_plural = "inspections"

    def __str__(self) -> str:
        return f"{self.docket_number} · {self.registration_number}"

    def get_absolute_url(self) -> str:
        return reverse("assessment:result", args=[self.pk])

    # ------------------------------------------------------------------
    # Derived display helpers. Computed rather than stored, because a stored
    # copy of something derivable is one more thing that can fall out of sync.
    #
    # Each of these walks `self.images.all()` rather than issuing its own query,
    # so that a caller who has done prefetch_related("images__detections") pays
    # for one query and not one per inspection. That is why they use len() and
    # generator expressions instead of .count() and .aggregate().
    # ------------------------------------------------------------------

    @property
    def docket_number(self) -> str:
        """Human-facing reference, e.g. VDA-2026-000042.

        Derived from the primary key and the creation year, so it needs no
        column of its own and cannot collide.
        """
        year = self.created_at.year if self.created_at else 0
        return f"VDA-{year}-{self.pk or 0:06d}"

    @property
    def vehicle_description(self) -> str:
        return f"{self.vehicle_make} {self.vehicle_model} ({self.vehicle_year})"

    @property
    def view_count(self) -> int:
        return len(self.images.all())

    @property
    def damage_count(self) -> int:
        """Damage instances found across every photograph.

        This counts detections, not distinct physical damage: the same dent
        photographed from two angles is counted twice here. The *quotation* does
        not double-count it - see core.cost.estimate_combined - but this figure
        is a tally of what the model output, and inflating or deflating it to
        look tidier would misrepresent the model.
        """
        return sum(len(view.detections.all()) for view in self.images.all())

    @property
    def primary_image(self):
        """The photograph to show as the record's thumbnail.

        The close-up is preferred because it is the shot where the damage fills
        the frame; otherwise the first angle uploaded.
        """
        views = list(self.images.all())
        if not views:
            return None
        for view in views:
            if view.angle == DEFAULT_ANGLE:
                return view
        return views[0]

    @property
    def panels_assessed(self) -> list[str]:
        """Distinct component labels the quotation covers, for a one-line summary."""
        seen: list[str] = []
        for line in self.cost_lines.all():
            label = line.get_component_display()
            if label not in seen:
                seen.append(label)
        return seen


class VehicleImage(models.Model):
    """One photograph of the vehicle, from one angle, showing one panel."""

    inspection = models.ForeignKey(
        Inspection, on_delete=models.CASCADE, related_name="images"
    )

    angle = models.CharField(max_length=20, choices=ANGLE_CHOICES,
                             default=DEFAULT_ANGLE)
    # Which component this shot contains. Defaulted from the angle at upload
    # (a front shot is usually the front bumper) but always editable, because
    # the panel sets the part price and a wrong one costs real money.
    panel = models.CharField(max_length=20, choices=PANEL_CHOICES,
                             default=DEFAULT_PANEL)

    image = models.ImageField(upload_to="uploads/%Y/%m/")
    annotated_image = models.ImageField(
        upload_to="annotated/%Y/%m/", blank=True, null=True
    )

    # Per-photograph measurements. Area fractions are relative to *this* image's
    # reference area, so a stored severity verdict is only interpretable next to
    # the reference area that produced it.
    image_width = models.PositiveIntegerField(default=0)
    image_height = models.PositiveIntegerField(default=0)
    reference_area_px = models.FloatField(default=0.0)
    processing_ms = models.PositiveIntegerField(null=True, blank=True)

    # Position in the walk-around, so photographs display in a sensible order
    # rather than by upload time. Set in save() from the order of ANGLES.
    sort_order = models.PositiveSmallIntegerField(default=0, editable=False)

    class Meta:
        # sort_order puts the walk-around in its conventional order; pk breaks
        # ties, which happens when two photographs share an angle.
        ordering = ["sort_order", "pk"]
        verbose_name = "photograph"
        verbose_name_plural = "photographs"
        # Deliberately NOT unique on (inspection, angle). Two close-ups of one
        # scratch, or two shots of a long dent, are a reasonable thing for an
        # owner to upload, and the cost model merges duplicate views safely
        # rather than charging twice - so forbidding them would remove a useful
        # option for no benefit.

    def save(self, *args, **kwargs):
        angle_keys = list(ANGLES)
        self.sort_order = (angle_keys.index(self.angle)
                           if self.angle in angle_keys else len(angle_keys))
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.get_angle_display()} · {self.get_panel_display()}"

    @property
    def display_image(self):
        """The annotated overlay if there is one, else the original upload."""
        return self.annotated_image if self.annotated_image else self.image

    @property
    def angle_hint(self) -> str:
        angle = ANGLES.get(self.angle)
        return angle.hint if angle else ""

    @property
    def damage_count(self) -> int:
        return len(self.detections.all())


class DamageDetection(models.Model):
    """One damage instance the model found, with its severity verdict.

    Stored per instance rather than per damage type, because the annotated
    image shows individual polygons and the results page has to be able to
    explain each one. Pricing works on the aggregated groups instead - see
    CostLine and core/pipeline.py's aggregate().

    Hangs off the photograph, not the inspection: the bounding box and the area
    fraction are both in the coordinate space of one particular image, so a
    detection is meaningless without knowing which.
    """

    image = models.ForeignKey(
        VehicleImage, on_delete=models.CASCADE, related_name="detections"
    )

    class_name = models.CharField(max_length=30, choices=CLASS_CHOICES)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    confidence = models.FloatField()

    area_px = models.FloatField()
    area_fraction = models.FloatField()
    # The sentence severity.classify() produced, e.g. "covers 3.12% of the
    # reference area". Kept verbatim so the page can show *why* a verdict was
    # reached instead of asserting it.
    reason = models.CharField(max_length=200, blank=True)

    # Bounding box in pixels of the original upload. Four columns rather than a
    # JSON blob so they can be queried and averaged if the evaluation needs it.
    x1 = models.FloatField(default=0.0)
    y1 = models.FloatField(default=0.0)
    x2 = models.FloatField(default=0.0)
    y2 = models.FloatField(default=0.0)

    class Meta:
        # Largest damage first, matching the order the pipeline returns and the
        # order the results table and PDF display.
        ordering = ["-area_fraction", "class_name"]
        verbose_name = "damage detection"

    def __str__(self) -> str:
        return f"{self.get_class_name_display()} ({self.get_severity_display()})"

    @property
    def area_percent(self) -> float:
        return self.area_fraction * 100


class CostLine(models.Model):
    """One line of the quotation: a damage type, its repair action, its price.

    Attached to the inspection rather than to a photograph, because the estimate
    is priced once across every view. A line that said "front bumper, replace"
    could not honestly be assigned to the front shot or the front-left shot when
    both showed the damage that justified it.
    """

    inspection = models.ForeignKey(
        Inspection, on_delete=models.CASCADE, related_name="cost_lines"
    )

    class_name = models.CharField(max_length=30, choices=CLASS_CHOICES)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    # The component actually priced. Usually the panel the user picked for the
    # photograph, but overridden for classes that name their own component - a
    # broken lamp is a lamp whatever panel was photographed. See
    # cost.resolve_component().
    component = models.CharField(max_length=20, choices=PANEL_CHOICES)
    action = models.CharField(max_length=120)
    instances = models.PositiveIntegerField(default=1)

    part_cost = models.DecimalField(**MONEY)
    labour_cost = models.DecimalField(**MONEY)
    paint_cost = models.DecimalField(**MONEY)
    labour_hours = models.DecimalField(max_digits=5, decimal_places=2,
                                      default=Decimal("0.00"))

    class Meta:
        # Insertion order. core.cost pricing emits the most expensive line
        # first - the line that justified buying the part - and that ordering
        # carries meaning, so it is preserved rather than re-sorted here.
        ordering = ["pk"]
        verbose_name = "cost line"

    def __str__(self) -> str:
        return f"{self.get_class_name_display()} on {self.get_component_display()}"

    @property
    def subtotal(self) -> Decimal:
        return self.part_cost + self.labour_cost + self.paint_cost
