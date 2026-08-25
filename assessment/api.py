"""REST API - the report's Backend API Layer.

    GET  /api/inspections/            list
    GET  /api/inspections/<pk>/       one inspection, its photographs and lines
    POST /api/assess/                 multipart upload, returns the assessment

Reading is open so the API can be demonstrated from a browser; creating requires a
login, because it runs the model and writes files (`REST_FRAMEWORK` in settings).

One decision worth pointing at: **the POST endpoint validates with the same
`InspectionForm` and the same `validate_photograph` the web page uses**, rather
than a parallel DRF serializer. Writing the rules twice is how the API ends up
accepting a 40 MB TIFF that the web form rejects. Validation lives in one file;
this endpoint only translates the result into JSON.

How photographs are addressed
-----------------------------
The web page posts a Django formset, which needs management fields an API client
should not have to know about. So the API takes a simpler contract: **one file per
angle, named by the angle key.**

    curl -u user:pass http://127.0.0.1:8000/api/assess/ \\
         -F owner_name="Yusra Fathima" -F owner_phone="+91 98765 43210" \\
         -F registration_number="KA 14 EX 4821" \\
         -F vehicle_make=Maruti -F vehicle_model=Swift -F vehicle_year=2019 \\
         -F segment=hatchback \\
         -F closeup=@dent.jpg -F front_left=@corner.jpg

Naming the field after the angle rather than sending parallel `image[]` and
`angle[]` lists means the two can never get out of step, and the request is
readable on its own. `panel_<angle>` overrides the part that angle usually shows;
left out, it defaults from the angle. A bare `image` field is accepted as a
convenience and treated as the close-up.

Why there is no flat `detections` list on an inspection
------------------------------------------------------
Detections are nested inside the photograph they were found in, because
`area_fraction` is measured against one frame. A flat list would put fractions
with different denominators side by side and invite a consumer to compare or sum
them, which is not valid. `damage_count` is provided for the cases that only need
the number.
"""

from __future__ import annotations

from rest_framework import generics, serializers, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from django.core.exceptions import ValidationError

from core.damage_config import ANGLES, COST_SOURCE, DEFAULT_ANGLE, PANELS

from .forms import (
    MAX_TOTAL_UPLOAD_BYTES,
    InspectionForm,
    validate_photograph,
)
from .models import CostLine, DamageDetection, Inspection, VehicleImage
from .services import AssessmentFailed, run_assessment


def _absolute(file_field, request) -> str | None:
    """Absolute URL, so a client is not left guessing the host.

    Module-level rather than a serializer method, because both the inspection and
    its photographs need it and a second copy would be a second thing to keep
    right.
    """
    if not file_field:
        return None
    url = file_field.url
    return request.build_absolute_uri(url) if request else url


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

class DamageDetectionSerializer(serializers.ModelSerializer):
    damage = serializers.CharField(source="get_class_name_display",
                                   read_only=True)
    severity_label = serializers.CharField(source="get_severity_display",
                                           read_only=True)
    area_percent = serializers.FloatField(read_only=True)

    class Meta:
        model = DamageDetection
        fields = ("class_name", "damage", "severity", "severity_label",
                  "confidence", "area_px", "area_fraction", "area_percent",
                  "reason", "x1", "y1", "x2", "y2")


class VehicleImageSerializer(serializers.ModelSerializer):
    """One photograph, with the frame it was measured against and what was in it.

    `image_width`, `image_height` and `reference_area_px` are on this object and
    not on the inspection for a reason worth stating: they are the denominator of
    every `area_fraction` below them. Keeping them together means a consumer can
    re-derive any percentage in this payload from the pixels.
    """

    angle_label = serializers.CharField(source="get_angle_display",
                                        read_only=True)
    panel_label = serializers.CharField(source="get_panel_display",
                                        read_only=True)
    image_url = serializers.SerializerMethodField()
    annotated_image_url = serializers.SerializerMethodField()
    damage_count = serializers.IntegerField(read_only=True)
    detections = DamageDetectionSerializer(many=True, read_only=True)

    class Meta:
        model = VehicleImage
        fields = ("id", "angle", "angle_label", "panel", "panel_label",
                  "image_url", "annotated_image_url",
                  "image_width", "image_height", "reference_area_px",
                  "processing_ms", "damage_count", "detections")

    def get_image_url(self, obj: VehicleImage) -> str | None:
        return _absolute(obj.image, self.context.get("request"))

    def get_annotated_image_url(self, obj: VehicleImage) -> str | None:
        return _absolute(obj.annotated_image, self.context.get("request"))


class CostLineSerializer(serializers.ModelSerializer):
    damage = serializers.CharField(source="get_class_name_display",
                                   read_only=True)
    component_label = serializers.CharField(source="get_component_display",
                                            read_only=True)
    subtotal = serializers.DecimalField(max_digits=10, decimal_places=2,
                                        read_only=True)

    class Meta:
        model = CostLine
        fields = ("class_name", "damage", "severity", "component",
                  "component_label", "action", "instances", "part_cost",
                  "labour_cost", "paint_cost", "labour_hours", "subtotal")


class InspectionSerializer(serializers.ModelSerializer):
    docket_number = serializers.CharField(read_only=True)
    images = VehicleImageSerializer(many=True, read_only=True)
    cost_lines = CostLineSerializer(many=True, read_only=True)
    severity_label = serializers.CharField(source="get_overall_severity_display",
                                           read_only=True)
    segment_label = serializers.CharField(source="get_segment_display",
                                          read_only=True)
    view_count = serializers.IntegerField(read_only=True)
    damage_count = serializers.IntegerField(read_only=True)
    report_pdf_url = serializers.SerializerMethodField()
    # Provenance travels with the numbers. A consumer of this API has to be able
    # to tell a real assessment from a placeholder one, and to know which rate
    # card produced the money - otherwise the figures are unattributable.
    cost_source = serializers.SerializerMethodField()

    class Meta:
        model = Inspection
        fields = (
            "id", "docket_number", "created_at", "status",
            "owner_name", "owner_phone", "owner_email",
            "registration_number", "vehicle_make", "vehicle_model",
            "vehicle_year", "segment", "segment_label",
            "overall_severity", "severity_label",
            "parts_total", "labour_total", "paint_total", "subtotal",
            "gst_amount", "total", "currency", "notes",
            "model_name", "is_stub", "processing_ms",
            "view_count", "damage_count",
            "report_pdf_url", "cost_source",
            "images", "cost_lines", "error_message",
        )

    def get_report_pdf_url(self, obj: Inspection) -> str | None:
        return _absolute(obj.report_pdf, self.context.get("request"))

    def get_cost_source(self, obj: Inspection) -> str:
        return COST_SOURCE


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# Photographs, their detections, and the cost lines, in one round trip. Without
# the prefetch a page of 20 inspections issues a query per inspection for its
# photographs and another per photograph for its detections.
_WITH_RELATED = Inspection.objects.prefetch_related(
    "images__detections", "cost_lines"
)


class InspectionListAPI(generics.ListAPIView):
    serializer_class = InspectionSerializer
    queryset = _WITH_RELATED


class InspectionDetailAPI(generics.RetrieveAPIView):
    serializer_class = InspectionSerializer
    queryset = _WITH_RELATED


class AssessAPI(APIView):
    """Upload photographs plus owner and vehicle details; get the assessment back.

    Synchronous: the response carries the result rather than a job id, because
    the work takes a second or two per photograph. If it ever needs to be queued,
    this is the method that would return 202 and a status URL.
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        form = InspectionForm(request.data)
        shots, shot_errors = self._collect_shots(request)

        # Both checked before anything is written, so a caller with two problems
        # is told about both rather than discovering the second on a retry.
        errors: dict = {}
        if not form.is_valid():
            # form.errors is already {field: [messages]}, which is the shape DRF
            # uses for validation errors, so it passes straight through.
            errors.update(form.errors)
        errors.update(shot_errors)
        if not shots and "images" not in errors:
            errors["images"] = [
                "Attach at least one photograph, in a field named after its "
                f"angle. Valid angles: {', '.join(ANGLES)}."
            ]
        if errors:
            return Response(errors, status=status.HTTP_400_BAD_REQUEST)

        inspection: Inspection = form.save(commit=False)
        if request.user.is_authenticated:
            inspection.submitted_by = request.user
        inspection.status = Inspection.Status.PENDING
        inspection.save()

        for angle, panel, upload in shots:
            VehicleImage.objects.create(
                inspection=inspection, angle=angle, panel=panel, image=upload
            )

        try:
            run_assessment(inspection)
        except AssessmentFailed as exc:
            # 422, not 500: the request was well-formed and the record was
            # created - the model could not assess these particular images. The
            # id is returned so the caller can retry or inspect the record.
            return Response(
                {"detail": str(exc), "id": inspection.pk,
                 "docket_number": inspection.docket_number},
                status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        inspection.refresh_from_db()
        serializer = InspectionSerializer(inspection,
                                         context={"request": request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    # ------------------------------------------------------------------
    # Request parsing
    # ------------------------------------------------------------------

    def _collect_shots(self, request) -> tuple[list[tuple], dict]:
        """Pull the angle-named files out of the request.

        Returns `[(angle, panel, upload)]` in walk-around order, plus a dict of
        errors keyed by the field that caused them - so a bad file is reported
        against `front_left`, the name the caller actually sent.

        Iterating over ANGLES rather than over the request keys is deliberate:
        an unrecognised field is ignored rather than guessed at, and the order of
        the result is the walk-around order regardless of how the client
        assembled the request.
        """
        shots: list[tuple] = []
        errors: dict = {}

        candidates = [(key, angle.default_panel) for key, angle in ANGLES.items()]

        for key, default_panel in candidates:
            upload = request.FILES.get(key)
            # A bare `image` stands in for the default angle, so the simplest
            # possible call - details plus one photograph - works.
            if upload is None and key == DEFAULT_ANGLE:
                upload = request.FILES.get("image")
            if upload is None:
                continue

            try:
                validate_photograph(upload)
            except ValidationError as exc:
                # ValidationError carries `.messages`, which is already the list
                # shape DRF renders for a field.
                errors[key] = exc.messages
                continue

            panel = request.data.get(f"panel_{key}") or default_panel
            if panel not in PANELS:
                errors[f"panel_{key}"] = [
                    f"{panel!r} is not a known part. Valid parts: "
                    f"{', '.join(PANELS)}."
                ]
                continue

            shots.append((key, panel, upload))

        # The per-file ceiling is inside validate_photograph; the total is only
        # knowable here, and Django has no total-request limit for uploads.
        total = sum(upload.size for _, _, upload in shots)
        if total > MAX_TOTAL_UPLOAD_BYTES:
            errors["images"] = [
                f"Those {len(shots)} photographs come to "
                f"{total / 1024 / 1024:.0f} MB in total. The limit is "
                f"{MAX_TOTAL_UPLOAD_BYTES // 1024 // 1024} MB."
            ]

        return shots, errors
