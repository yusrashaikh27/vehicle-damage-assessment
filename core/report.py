"""PDF inspection report - objective 4 of the report.

    "To generate and store a user-friendly report containing the annotated
     image, detected damage, severity level, and estimated repair cost."

All four of those appear on the page, in that order.

Two things about this module
---------------------------

**It takes plain data, not a Django model.** `build_report` is handed a
`ReportData` built by `assessment/services.py`, which reads it back out of the
database *after* the assessment has been saved. So the PDF is generated from the
stored record rather than from the model's live output, which means the document
and the database can never disagree - and regenerating an old report produces the
same figures rather than whatever the current weights would say.

**Money is printed as "Rs.", not the rupee sign.** ₹ is U+20B9, added to Unicode
in 2010. PDF's built-in Helvetica is a Type 1 font from the 1980s and has no
glyph for it, so the character silently renders as a black box - the kind of
defect you only notice in the printed copy. Embedding a TrueType font would fix
it and add a font file the demo has to find at runtime; "Rs." is unambiguous, is
what Indian workshop invoices print anyway, and cannot break. The web pages use ₹
freely, because a browser has fonts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as PdfImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .damage_config import CURRENCY, GST_RATE

# Palette shared with the web pages: workshop primer grey and ink, with the
# three severity colours carrying real meaning rather than decoration.
INK = colors.HexColor("#15181B")
MUTED = colors.HexColor("#5C646D")
RULE = colors.HexColor("#C9CDD3")
WASH = colors.HexColor("#EDEEF0")
SIGNAL = colors.HexColor("#FFC300")
SEVERITY_COLOURS = {
    "Minor": colors.HexColor("#2E7D5B"),
    "Moderate": colors.HexColor("#B26B00"),
    "Severe": colors.HexColor("#B3261E"),
}

PAGE_MARGIN = 15 * mm
CONTENT_WIDTH = A4[0] - 2 * PAGE_MARGIN


# ---------------------------------------------------------------------------
# Input structures - deliberately plain, so this module needs no database
# ---------------------------------------------------------------------------

@dataclass
class ReportDetection:
    label: str
    severity_label: str
    confidence: float
    area_percent: float
    reason: str
    # Which photograph this was measured in. An area percentage is relative to
    # one image's reference area, so a detection listed without its photograph is
    # a number with no denominator.
    angle_label: str = ""


@dataclass
class ReportView:
    """One photograph of the vehicle, as it appears in the report."""
    angle_label: str
    panel_label: str
    damage_count: int = 0
    annotated_image_path: Path | None = None
    original_image_path: Path | None = None


@dataclass
class ReportCostLine:
    label: str
    severity_label: str
    component_label: str
    action: str
    instances: int
    part_cost: float
    labour_cost: float
    paint_cost: float
    labour_hours: float

    @property
    def subtotal(self) -> float:
        return self.part_cost + self.labour_cost + self.paint_cost


@dataclass
class ReportData:
    docket_number: str
    created_at: str
    owner_name: str
    owner_phone: str
    registration_number: str
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    segment_label: str
    overall_severity_label: str
    owner_email: str = ""
    views: list[ReportView] = field(default_factory=list)
    detections: list[ReportDetection] = field(default_factory=list)
    cost_lines: list[ReportCostLine] = field(default_factory=list)
    parts_total: float = 0.0
    labour_total: float = 0.0
    paint_total: float = 0.0
    subtotal: float = 0.0
    gst_amount: float = 0.0
    total: float = 0.0
    notes: list[str] = field(default_factory=list)
    model_name: str = ""
    is_stub: bool = False
    cost_source: str = ""
    currency: str = CURRENCY

    @property
    def panels_inspected(self) -> str:
        """Distinct components covered, for the summary block.

        Distinct rather than one-per-photograph: three shots of the same door
        should read "Front door", not "Front door, Front door, Front door".
        """
        seen: list[str] = []
        for view in self.views:
            if view.panel_label not in seen:
                seen.append(view.panel_label)
        return ", ".join(seen) if seen else "-"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def rupees(amount: float) -> str:
    """Indian digit grouping with an ASCII currency prefix.

    Indian grouping breaks after three digits from the right and every two
    thereafter: 1234567 prints as 12,34,567, not 1,234,567. Python's `:,`
    format cannot do that, hence the manual split.
    """
    whole = f"{int(round(float(amount))):d}"
    negative = whole.startswith("-")
    if negative:
        whole = whole[1:]
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts: list[str] = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{'-' if negative else ''}Rs. {whole}"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=15, leading=18, textColor=INK, spaceAfter=1,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Courier",
            fontSize=8.5, leading=11, textColor=MUTED,
        ),
        "eyebrow": ParagraphStyle(
            "eyebrow", parent=base["Normal"], fontName="Courier-Bold",
            fontSize=7.5, leading=10, textColor=MUTED, spaceAfter=3,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=INK,
            spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.5, leading=11.5, textColor=INK,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.8, leading=10, textColor=INK,
        ),
        # Header cells need their own style rather than relying on the table's
        # TEXTCOLOR: a Paragraph carries its own colour, and that colour wins
        # over the enclosing TableStyle. Setting white here and black in the
        # TableStyle is what makes an inverted header row legible - without it
        # the text is ink-on-ink and the header reads as a solid black bar.
        "cellhead": ParagraphStyle(
            "cellhead", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.8, leading=10, textColor=colors.white,
        ),
        "cellnum": ParagraphStyle(
            "cellnum", parent=base["Normal"], fontName="Courier",
            fontSize=7.0, leading=9.5, textColor=INK, alignment=TA_RIGHT,
        ),
        "cellnumhead": ParagraphStyle(
            "cellnumhead", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.8, leading=10, textColor=colors.white, alignment=TA_RIGHT,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.2, leading=9.5, textColor=MUTED,
        ),
        "warn": ParagraphStyle(
            "warn", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.5, leading=11.5, textColor=colors.HexColor("#8A1C15"),
        ),
    }


def _scaled_image(path: Path, max_width: float,
                  max_height: float) -> PdfImage | None:
    """Fit an image inside a box without distorting it.

    reportlab will happily stretch an image to whatever width and height it is
    given, so the aspect ratio has to be preserved here. Returns None when the
    file is missing or unreadable rather than raising - a report without its
    photograph is still a useful report, and losing the whole PDF over one
    missing file would be the wrong trade.
    """
    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            width, height = im.size
    except Exception:
        return None
    if not width or not height:
        return None

    scale = min(max_width / width, max_height / height)
    return PdfImage(str(path), width=width * scale, height=height * scale)


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

def _page_furniture(canvas, doc) -> None:
    """Drawn on every page: the signal rule at the top, a footer, page number."""
    canvas.saveState()

    # A 3pt hazard-yellow bar across the head of the page. This is the one
    # decorative element, and it ties the PDF to the web pages so a printed
    # report is recognisably from the same system.
    canvas.setFillColor(SIGNAL)
    canvas.rect(0, A4[1] - 4 * mm, A4[0], 4 * mm, stroke=0, fill=1)

    canvas.setFont("Courier", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        PAGE_MARGIN, 10 * mm,
        "Computer-generated estimate. Indicative only; not a binding quotation.",
    )
    canvas.drawRightString(A4[0] - PAGE_MARGIN, 10 * mm, f"Page {doc.page}")

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(PAGE_MARGIN, 13 * mm, A4[0] - PAGE_MARGIN, 13 * mm)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _header(data: ReportData, st: dict) -> list:
    left = [
        Paragraph("Vehicle damage assessment", st["title"]),
        Paragraph("Automated inspection and repair cost estimate", st["subtitle"]),
    ]
    right = [
        Paragraph(f"<b>{data.docket_number}</b>", st["subtitle"]),
        Paragraph(data.created_at, st["subtitle"]),
    ]
    table = Table([[left, right]], colWidths=[CONTENT_WIDTH * 0.62,
                                             CONTENT_WIDTH * 0.38])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, INK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return [table, Spacer(1, 8)]


def _stub_banner(st: dict) -> list:
    """Loud, unmissable warning that these numbers are placeholders.

    Only reachable when no trained weights were installed. A demo that shows
    fabricated detections without saying so is the single worst outcome for this
    project, so the warning is a filled red-edged block, not a footnote.
    """
    text = Paragraph(
        "PLACEHOLDER RESULTS - no trained model was installed when this "
        "inspection ran. The detections and costs below were produced by the "
        "stub detector for testing and describe nothing about this vehicle.",
        st["warn"],
    )
    table = Table([[text]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBE9E7")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#B3261E")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return [table, Spacer(1, 9)]


def _vehicle_block(data: ReportData, st: dict) -> list:
    contact = data.owner_phone
    if data.owner_email:
        contact = f"{data.owner_phone}, {data.owner_email}"

    rows = [
        ["Owner", data.owner_name, "Contact", contact],
        ["Registration", data.registration_number,
         "Segment", data.segment_label],
        ["Make and model", f"{data.vehicle_make} {data.vehicle_model}",
         "Panels inspected", data.panels_inspected],
        ["Year", str(data.vehicle_year),
         "Overall severity", data.overall_severity_label or "-"],
    ]
    severity_row = len(rows) - 1

    table = Table(rows, colWidths=[CONTENT_WIDTH * 0.17, CONTENT_WIDTH * 0.33,
                                   CONTENT_WIDTH * 0.19, CONTENT_WIDTH * 0.31])
    style = [
        ("FONTNAME", (0, 0), (0, -1), "Courier"),
        ("FONTNAME", (2, 0), (2, -1), "Courier"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("BACKGROUND", (0, 0), (-1, -1), WASH),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        # Contact details are typed by hand and can be long, so this one cell is
        # allowed to shrink rather than pushing the table off the page.
        ("FONTSIZE", (3, 0), (3, 0), 7.4),
    ]
    # Colour the severity value to match the web page and the severity legend.
    # Indexed off the row count rather than hard-coded, so adding a row above it
    # cannot silently paint the wrong cell.
    colour = SEVERITY_COLOURS.get(data.overall_severity_label)
    if colour is not None:
        style.append(("TEXTCOLOR", (3, severity_row), (3, severity_row), colour))
    table.setStyle(TableStyle(style))
    return [table, Spacer(1, 10)]


# Two photographs to a row. A walk-around can be ten shots; printing each at
# full width would make a ten-page report out of an estimate that fits on two.
VIEW_COLUMNS = 2
VIEW_IMAGE_HEIGHT = 46 * mm
VIEW_GUTTER = 5 * mm


def _views_block(data: ReportData, st: dict) -> list:
    """A contact sheet of every photograph, captioned with angle and panel."""
    if not data.views:
        return []

    cell_width = (CONTENT_WIDTH - VIEW_GUTTER * (VIEW_COLUMNS - 1)) / VIEW_COLUMNS

    cells: list = []
    for view in data.views:
        image = None
        if view.annotated_image_path:
            image = _scaled_image(Path(view.annotated_image_path),
                                  cell_width, VIEW_IMAGE_HEIGHT)
        if image is None and view.original_image_path:
            image = _scaled_image(Path(view.original_image_path),
                                  cell_width, VIEW_IMAGE_HEIGHT)

        found = ("no damage detected" if view.damage_count == 0
                 else f"{view.damage_count} damage instance"
                      f"{'s' if view.damage_count != 1 else ''}")
        caption = (f"<b>{view.angle_label}</b> &bull; {view.panel_label}<br/>"
                   f"{found}")

        block: list = []
        if image is not None:
            block.append(image)
        else:
            # Say so rather than leaving a gap: a missing photograph in an
            # inspection report is information, not an empty space.
            block.append(Paragraph("Photograph unavailable.", st["small"]))
        block.append(Spacer(1, 2))
        block.append(Paragraph(caption, st["small"]))
        cells.append(block)

    # Pad the last row so every row has the same number of columns, which
    # reportlab requires.
    while len(cells) % VIEW_COLUMNS:
        cells.append("")

    rows = [cells[i:i + VIEW_COLUMNS]
            for i in range(0, len(cells), VIEW_COLUMNS)]

    table = Table(rows, colWidths=[cell_width] * VIEW_COLUMNS)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), VIEW_GUTTER),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    annotated = any(v.annotated_image_path for v in data.views)
    caption = ("Detected damage outlined by the segmentation model."
               if annotated else "Photographs as uploaded.")

    return [
        Paragraph("PHOTOGRAPHS ASSESSED", st["eyebrow"]),
        table,
        Paragraph(caption, st["small"]),
        Spacer(1, 10),
    ]


def _detections_table(data: ReportData, st: dict) -> list:
    if not data.detections:
        return [
            Paragraph("Detected damage", st["h2"]),
            Paragraph("No damage was detected in any of the photographs.",
                      st["body"]),
            Spacer(1, 8),
        ]

    header = ["Photograph", "Damage", "Severity", "Confidence", "Area",
              "Basis for severity"]
    numeric = {3, 4}
    rows: list[list] = [[
        Paragraph(text, st["cellnumhead"] if i in numeric else st["cellhead"])
        for i, text in enumerate(header)
    ]]
    for det in data.detections:
        rows.append([
            Paragraph(det.angle_label or "-", st["cell"]),
            Paragraph(det.label, st["cell"]),
            Paragraph(det.severity_label, st["cell"]),
            Paragraph(f"{det.confidence * 100:.0f}%", st["cellnum"]),
            Paragraph(f"{det.area_percent:.2f}%", st["cellnum"]),
            Paragraph(det.reason, st["small"]),
        ])

    widths = [CONTENT_WIDTH * w for w in (0.15, 0.16, 0.11, 0.10, 0.08, 0.40)]
    table = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    for i, det in enumerate(data.detections, start=1):
        colour = SEVERITY_COLOURS.get(det.severity_label)
        if colour is not None:
            style.append(("TEXTCOLOR", (2, i), (2, i), colour))
    table.setStyle(TableStyle(style))

    note = (
        "Severity is computed from the share of the reference area each mask "
        "covers, banded by documented per-class thresholds. It is not predicted "
        "by the model - the training data carries no severity labels. Area is "
        "measured within the photograph named on each row, so the same damage "
        "seen in two shots may show two different percentages."
    )
    if len(data.views) > 1:
        note += (
            " This table lists every detection; the estimate below merges damage "
            "seen in more than one photograph so that a panel is charged once."
        )

    return [Paragraph("Detected damage", st["h2"]), table, Spacer(1, 4),
            Paragraph(note, st["small"]),
            Spacer(1, 10)]


def _quotation_table(data: ReportData, st: dict) -> list:
    if not data.cost_lines:
        return []

    header = ["Damage", "Component", "Repair action", "Qty", "Hrs",
              "Parts", "Labour", "Paint", "Line total"]
    # Numeric headers right-aligned so they sit over the figures they label.
    numeric = {3, 4, 5, 6, 7, 8}
    rows: list[list] = [[
        Paragraph(text, st["cellnumhead"] if i in numeric else st["cellhead"])
        for i, text in enumerate(header)
    ]]
    for line in data.cost_lines:
        rows.append([
            Paragraph(f"{line.label}<br/><font color='#5C646D'>"
                      f"{line.severity_label}</font>", st["cell"]),
            Paragraph(line.component_label, st["cell"]),
            Paragraph(line.action, st["cell"]),
            Paragraph(str(line.instances), st["cellnum"]),
            Paragraph(f"{line.labour_hours:.2f}", st["cellnum"]),
            Paragraph(rupees(line.part_cost), st["cellnum"]),
            Paragraph(rupees(line.labour_cost), st["cellnum"]),
            Paragraph(rupees(line.paint_cost), st["cellnum"]),
            Paragraph(f"<b>{rupees(line.subtotal)}</b>", st["cellnum"]),
        ])

    # Widths tuned so no money cell wraps: "Rs. 10,100" is ten Courier
    # characters, and a wrapped currency figure ("Rs. 2,5" / "00") is unreadable
    # on an invoice. Descriptive columns absorb the space instead, because prose
    # wraps gracefully and numbers do not.
    widths = [CONTENT_WIDTH * w for w in
              (0.145, 0.115, 0.185, 0.045, 0.055, 0.105, 0.105, 0.09, 0.155)]
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [Paragraph("Repair cost estimate", st["h2"]), table, Spacer(1, 8)]


def _totals_block(data: ReportData, st: dict) -> list:
    """The totals, right-aligned, with the payable figure in a filled box."""
    gst_label = f"GST at {GST_RATE * 100:.0f}%"
    rows = [
        ["Parts", rupees(data.parts_total)],
        ["Labour", rupees(data.labour_total)],
        ["Paint material", rupees(data.paint_total)],
        ["Subtotal", rupees(data.subtotal)],
        [gst_label, rupees(data.gst_amount)],
        ["Estimated total", rupees(data.total)],
    ]
    inner = Table(rows, colWidths=[38 * mm, 32 * mm])
    inner.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
        ("FONTNAME", (1, 0), (1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEABOVE", (0, 3), (-1, 3), 0.5, RULE),
        # The payable line: heavier type on the signal colour, so the eye lands
        # on it first. This is the number the reader opened the report for.
        ("FONTNAME", (0, 5), (-1, 5), "Helvetica-Bold"),
        ("FONTSIZE", (0, 5), (-1, 5), 10.5),
        ("BACKGROUND", (0, 5), (-1, 5), SIGNAL),
        ("LINEABOVE", (0, 5), (-1, 5), 1.2, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 5), (-1, 5), 6),
        ("BOTTOMPADDING", (0, 5), (-1, 5), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))

    wrapper = Table([[ "", inner ]],
                    colWidths=[CONTENT_WIDTH - 70 * mm, 70 * mm])
    wrapper.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return [wrapper, Spacer(1, 10)]


def _notes_block(data: ReportData, st: dict) -> list:
    if not data.notes:
        return []
    items = "".join(f"<br/>&bull;&nbsp; {note}" for note in data.notes)
    return [
        Paragraph("How this estimate was reached", st["h2"]),
        Paragraph(items.lstrip("<br/>"), st["small"]),
        Spacer(1, 8),
    ]


def _provenance_block(data: ReportData, st: dict) -> list:
    lines = [
        f"<b>Detection model:</b> {data.model_name or 'unknown'}",
        f"<b>Rate card:</b> {data.cost_source}",
        "<b>Method:</b> damage located and outlined by a YOLO11 segmentation "
        "model; severity derived from mask area against documented per-class "
        "thresholds; cost assembled from a fixed rate card of part prices, "
        "labour hours and paint material.",
        "<b>Limitations:</b> the model was trained on photographs of cars only. "
        "Area is measured relative to the frame, so how close the camera was "
        "affects the severity band. Prices are indicative and exclude "
        "consumables, towing and any concealed structural damage found on strip.",
    ]
    body = "<br/><br/>".join(lines)
    table = Table([[Paragraph(body, st["small"])]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WASH),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return [Paragraph("Basis and limitations", st["h2"]), table]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_report(data: ReportData, out_path: Path | str) -> Path:
    """Write the inspection report and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st = _styles()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN, bottomMargin=20 * mm,
        title=f"Damage assessment {data.docket_number}",
        author="Vehicle Damage Assessment and Cost Estimator",
        subject=f"{data.vehicle_make} {data.vehicle_model} "
                f"({data.registration_number})",
    )

    story: list = []
    story += _header(data, st)
    if data.is_stub:
        story += _stub_banner(st)
    story += _vehicle_block(data, st)
    story += _views_block(data, st)
    story += _detections_table(data, st)
    story += _quotation_table(data, st)
    story += _totals_block(data, st)
    story += _notes_block(data, st)
    # KeepTogether stops the basis-and-limitations box being split across a page
    # break, which would leave a stray heading at the foot of page one.
    story.append(KeepTogether(_provenance_block(data, st)))

    doc.build(story, onFirstPage=_page_furniture, onLaterPages=_page_furniture)
    return out_path
