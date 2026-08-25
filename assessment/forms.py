"""The upload form: who the owner is, what the car is, and the walk-around.

Three forms working together:

* `InspectionForm`     - owner and vehicle details. One row in Inspection.
* `VehicleImageForm`   - one photograph, its angle and the panel it shows.
* `VehicleImageFormSet` - one slot per walk-around angle, on the front page.

Why every angle is offered but none is compulsory
------------------------------------------------
A damage assessor photographs a car from a fixed set of positions, so the page
offers all of them: front, the four 3/4 corners, both sides, rear, roof, and a
close-up. Requiring all ten would be wrong. The roof shot only matters if the roof
is damaged, and a car with a scuffed rear bumper does not need a photograph of its
bonnet. Demanding them would either block a real inspection or - worse - teach the
user to upload filler images, which the model would then dutifully find nothing
in.

So each slot is optional and the formset enforces one rule: at least one
photograph. More angles genuinely improve the estimate, because area is measured
against what is visible in the frame, and the page says so rather than pretending
the rule is arbitrary.

What is validated, and why each check exists
--------------------------------------------
Django's `ImageField` already runs the file through Pillow, so a text file
renamed to `.jpg` is rejected before this code sees it. What it does *not* check
is anything domain-specific, and those are the checks added here:

1. **A size ceiling per photograph.** Without one, a 60 MB raw photograph is
   decoded in memory inside a web request.
2. **A ceiling on the whole submission.** Ten photographs at the per-file limit
   would be a 100 MB POST. Django has no total-request limit for file uploads, so
   this is the only place it can be enforced.
3. **A format allowlist.** Pillow opens dozens of formats; the pipeline and the
   PDF are only tested against JPEG, PNG and WebP.
4. **A minimum resolution.** This one is not cosmetic. Severity is derived from
   the *area* of a mask, so on a 100x60 thumbnail a real dent is a handful of
   pixels and the quotation built from it is meaningless. Refusing the image is
   more honest than pricing a guess.
"""

from __future__ import annotations

import datetime as dt
import re

from django import forms
from django.core.exceptions import ValidationError
from django.forms import BaseInlineFormSet, inlineformset_factory

from core.damage_config import ANGLES

from .models import Inspection, VehicleImage

MAX_UPLOAD_BYTES = 10 * 1024 * 1024          # 10 MB per photograph
MAX_TOTAL_UPLOAD_BYTES = 40 * 1024 * 1024    # 40 MB for the whole walk-around
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_EDGE_PX = 320                            # shortest side

# Registration plates vary far more than tutorials assume - old-format state
# plates, BH series, trade plates, imports - so this only enforces the shape all
# of them share rather than one country's current pattern. Rejecting a valid
# plate would block a real inspection to enforce a rule nobody asked for.
PLATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-]{2,19}$")

# Indian mobile and landline numbers, optionally with +91 and separators. Kept
# deliberately loose for the same reason as the plate pattern: this is a contact
# number for a workshop to ring, and rejecting a valid one to enforce a format
# helps nobody.
PHONE_RE = re.compile(r"^\+?[\d][\d \-]{7,17}$")


def _style(field: forms.Field) -> None:
    """Attach the stylesheet's classes to a widget.

    Done in Python rather than in the template so that every form in the project
    picks up the same classes automatically, including the formset's extra slots -
    which the template never names individually.
    """
    css = "field-input"
    if isinstance(field.widget, forms.Select):
        css += " field-select"
    elif isinstance(field.widget, forms.ClearableFileInput):
        css += " field-file"
    existing = field.widget.attrs.get("class", "")
    field.widget.attrs["class"] = f"{existing} {css}".strip()


def validate_photograph(image):
    """The domain checks on one uploaded photograph. Raises ValidationError.

    A module-level function rather than a method, so the API and any management
    command apply exactly the same rules as the web form. A second copy of these
    checks that drifted from the first would be worse than no checks.
    """
    if image.size > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"That file is {image.size / 1024 / 1024:.1f} MB. The limit is "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB per photograph - most phones "
            f"can share a smaller copy."
        )

    pillow_image = _decode(image)

    fmt = (pillow_image.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ValidationError(
            f"{fmt or 'That format'} is not supported. Use JPEG, PNG or WebP."
        )

    width, height = pillow_image.size
    if min(width, height) < MIN_EDGE_PX:
        raise ValidationError(
            f"That image is {width}x{height}. Severity is measured from how much "
            f"area the damage covers, and below {MIN_EDGE_PX}px there are too few "
            f"pixels for that measurement to mean anything."
        )

    # Pillow's decode leaves the file pointer at the end. Anything that reads the
    # upload afterwards - saving it to media/ - would write zero bytes. Rewinding
    # is the fix, and forgetting it produces an unhelpful empty-file bug much
    # later.
    image.seek(0)
    return image


def _decode(upload):
    """The uploaded file as a Pillow image, however it arrived.

    Django's `forms.ImageField` decodes the upload and caches the result on
    `upload.image`, so the web form's file already carries one. A file taken
    straight out of `request.FILES` - which is what the API does - does not, and
    reading the attribute that is not there would have made every API upload fail
    with "that file could not be read as an image". So: use the cached decode if
    it exists, otherwise do it here.
    """
    cached = getattr(upload, "image", None)
    if cached is not None:
        return cached

    from PIL import Image, UnidentifiedImageError

    upload.seek(0)
    try:
        decoded = Image.open(upload)
        # `open` is lazy - it reads the header and stops. verify() forces the
        # rest, which is what catches a file that is a valid JPEG for 200 bytes
        # and then garbage.
        decoded.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValidationError(
            "That file could not be read as an image."
        ) from exc

    # verify() leaves the image unusable for further reads, so reopen to get the
    # size and format from a working object. Documented Pillow behaviour, and the
    # reason this is two opens rather than one.
    upload.seek(0)
    return Image.open(upload)


# ---------------------------------------------------------------------------
# Owner and vehicle
# ---------------------------------------------------------------------------

class InspectionForm(forms.ModelForm):
    """Stage 1 of the report's methodology: the details the user supplies."""

    class Meta:
        model = Inspection
        fields = [
            "owner_name", "owner_phone", "owner_email",
            "registration_number", "vehicle_make", "vehicle_model",
            "vehicle_year", "segment",
        ]
        labels = {
            "owner_name": "Full name",
            "owner_phone": "Phone number",
            "owner_email": "Email (optional)",
            "registration_number": "Registration number",
            "vehicle_make": "Make",
            "vehicle_model": "Model",
            "vehicle_year": "Year",
            "segment": "Vehicle segment",
        }
        help_texts = {
            "owner_email": "Only used to send you a copy of the estimate.",
            "segment": "Sets the part prices and the hourly labour rate.",
        }
        widgets = {
            "owner_name": forms.TextInput(
                attrs={"placeholder": "Yusra Fathima", "autocomplete": "name"}
            ),
            "owner_phone": forms.TextInput(
                attrs={"placeholder": "+91 98765 43210", "autocomplete": "tel",
                       "inputmode": "tel"}
            ),
            "owner_email": forms.EmailInput(
                attrs={"placeholder": "you@example.com", "autocomplete": "email"}
            ),
            "registration_number": forms.TextInput(
                attrs={"placeholder": "KA 14 EX 4821",
                       "autocapitalize": "characters"}
            ),
            "vehicle_make": forms.TextInput(attrs={"placeholder": "Maruti Suzuki"}),
            "vehicle_model": forms.TextInput(attrs={"placeholder": "Swift"}),
            "vehicle_year": forms.NumberInput(attrs={"placeholder": "2019"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Year bounds live on the widget as well as in the validator, so the
        # browser can stop an obvious typo before a round trip. The server-side
        # check below is the one that actually enforces it - a browser
        # constraint is a convenience, never a validation.
        next_year = dt.date.today().year + 1
        self.fields["vehicle_year"].widget.attrs.update(
            {"min": 1980, "max": next_year, "step": 1}
        )
        for field in self.fields.values():
            _style(field)

    def clean_owner_name(self) -> str:
        name = " ".join((self.cleaned_data["owner_name"] or "").split())
        if len(name) < 2:
            raise ValidationError("Enter the vehicle owner's name.")
        return name

    def clean_owner_phone(self) -> str:
        raw = (self.cleaned_data["owner_phone"] or "").strip()
        phone = re.sub(r"\s+", " ", raw)
        if not PHONE_RE.match(phone):
            raise ValidationError(
                "Enter a contact number with at least 8 digits - for example "
                "+91 98765 43210."
            )
        return phone

    def clean_registration_number(self) -> str:
        raw = (self.cleaned_data["registration_number"] or "").upper()
        # Collapse runs of whitespace so "KA14  EX4821" and "KA 14 EX 4821"
        # become the same stored value, which is what makes search work.
        plate = re.sub(r"\s+", " ", raw).strip()
        if not PLATE_RE.match(plate):
            raise ValidationError(
                "Use letters, digits, spaces or hyphens - for example KA 14 EX 4821."
            )
        return plate

    def clean_vehicle_year(self) -> int:
        year = self.cleaned_data["vehicle_year"]
        next_year = dt.date.today().year + 1
        if not (1980 <= year <= next_year):
            raise ValidationError(f"Enter a year between 1980 and {next_year}.")
        return year

    def clean_vehicle_make(self) -> str:
        return self.cleaned_data["vehicle_make"].strip()

    def clean_vehicle_model(self) -> str:
        return self.cleaned_data["vehicle_model"].strip()


# ---------------------------------------------------------------------------
# One photograph
# ---------------------------------------------------------------------------

class VehicleImageForm(forms.ModelForm):
    """One angle of the walk-around: the photograph and the panel it shows."""

    class Meta:
        model = VehicleImage
        fields = ["angle", "panel", "image"]
        widgets = {
            # The angle is decided by which slot on the page this is, so it is
            # carried in a hidden field rather than asked twice. It still goes
            # through the ChoiceField, so a tampered value is rejected.
            "angle": forms.HiddenInput(),
            "image": forms.ClearableFileInput(
                attrs={"accept": "image/jpeg,image/png,image/webp"}
            ),
        }
        labels = {"panel": "Part shown"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Every slot is optional; the formset requires one photograph overall.
        # Marking the field itself optional is what lets an untouched slot pass
        # without an error message next to it.
        self.fields["image"].required = False
        self.fields["panel"].required = False
        for field in self.fields.values():
            _style(field)

    @property
    def angle_key(self) -> str:
        """Which angle this slot is for, whether bound or not.

        Reads the submitted value first so an invalid re-render keeps its
        headings, and falls back to the initial value on a fresh form.
        """
        if self.is_bound:
            value = self.data.get(self.add_prefix("angle"))
            if value:
                return value
        return self.initial.get("angle") or self.instance.angle or ""

    @property
    def angle_label(self) -> str:
        angle = ANGLES.get(self.angle_key)
        return angle.label if angle else self.angle_key

    @property
    def angle_hint(self) -> str:
        angle = ANGLES.get(self.angle_key)
        return angle.hint if angle else ""

    def clean_image(self):
        image = self.cleaned_data.get("image")
        if not image:
            return image
        return validate_photograph(image)

    def clean(self):
        cleaned = super().clean()
        image = cleaned.get("image")
        panel = cleaned.get("panel")

        # A panel chosen with no photograph is almost certainly a forgotten file
        # rather than a deliberate blank, so say so instead of dropping the slot.
        if panel and not image and not self.errors.get("image"):
            self.add_error(
                "image",
                "Choose a photograph for this angle, or reset the part to leave "
                "the angle out.",
            )

        # A photograph with no panel would be priced against the "other" rate
        # card. Falling back to the angle's usual component is more accurate than
        # that, and the form pre-selects it anyway - this covers a cleared field.
        if image and not panel:
            angle = ANGLES.get(self.angle_key)
            cleaned["panel"] = angle.default_panel if angle else "other"

        return cleaned


class BaseVehicleImageFormSet(BaseInlineFormSet):
    """The walk-around. Every angle offered, at least one photograph required."""

    def clean(self):
        super().clean()
        # Field-level problems are already reported next to the field that has
        # them; adding a formset-wide error on top would be noise.
        if any(self.errors):
            return

        photographs = [
            form.cleaned_data.get("image")
            for form in self.forms
            if form.cleaned_data.get("image")
        ]

        if not photographs:
            raise ValidationError(
                "Add at least one photograph of the damage. The close-up slot is "
                "the most useful single shot if you only have one."
            )

        total = sum(photo.size for photo in photographs)
        if total > MAX_TOTAL_UPLOAD_BYTES:
            raise ValidationError(
                f"Those {len(photographs)} photographs come to "
                f"{total / 1024 / 1024:.0f} MB in total. The limit is "
                f"{MAX_TOTAL_UPLOAD_BYTES // 1024 // 1024} MB - upload the angles "
                f"that show the damage, or share smaller copies."
            )

    def save_new(self, form, commit=True):
        # Slots the user left empty reach here only if something upstream
        # changes; guarding is cheaper than an inspection with a blank row.
        if not form.cleaned_data.get("image"):
            return None
        return super().save_new(form, commit=commit)


# One slot per angle, in walk-around order. `extra` is the number of angles
# because the front page is a blank walk-around: nothing exists yet to edit.
VehicleImageFormSet = inlineformset_factory(
    Inspection,
    VehicleImage,
    form=VehicleImageForm,
    formset=BaseVehicleImageFormSet,
    extra=len(ANGLES),
    max_num=len(ANGLES),
    validate_max=True,
    can_delete=False,
)


def walkaround_initial() -> list[dict]:
    """Initial data that pins each slot to one angle and its usual panel.

    Passed as the formset's `initial`, which Django applies to the extra forms in
    order - so slot 0 is the front, slot 1 the front-left corner, and so on, and
    the panel dropdown arrives pre-selected with the component that angle usually
    shows.
    """
    return [
        {"angle": key, "panel": angle.default_panel}
        for key, angle in ANGLES.items()
    ]
