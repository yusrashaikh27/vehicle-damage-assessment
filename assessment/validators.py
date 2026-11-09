import re

from django.core.exceptions import ValidationError

# Valid RTO state / UT codes
STATE_CODES = (
    "AN|AP|AR|AS|BR|CH|CG|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ|NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB"
)

STANDARD_RE = re.compile(rf"^({STATE_CODES})\d{{1,2}}[A-Z]{{0,3}}\d{{4}}$")
BHARAT_RE = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")


def normalise_registration(value: str) -> str:
    """Uppercase and strip spaces / hyphens: 'ka 14-ex 4821' -> 'KA14EX4821'."""
    return re.sub(r"[\s\-]", "", (value or "").upper())


def validate_registration_number(value: str) -> str:
    cleaned = normalise_registration(value)
    if STANDARD_RE.match(cleaned) or BHARAT_RE.match(cleaned):
        return cleaned
    raise ValidationError(
        "Enter a valid Indian registration number, for example KA 14 EX 4821 "
        "or 22 BH 1234 AA.",
        code="invalid_registration",
    )