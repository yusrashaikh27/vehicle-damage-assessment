#!/usr/bin/env python3
# Builds a journal-club deck from the BMC Oral Health 2024;24:786 paper,
# mirroring the chapter structure and plain academic styling of "ROHIT PPT.pptx".
import copy
import re
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

WORK = "/sessions/optimistic-admiring-goodall/work"
OUT = "/sessions/optimistic-admiring-goodall/mnt/outputs"
FIG = os.path.join(WORK, "fig")

SERIF = "Times New Roman"
NAVY = RGBColor(0x1E, 0x27, 0x61)
BLACK = RGBColor(0x11, 0x11, 0x11)
GREY = RGBColor(0x60, 0x60, 0x60)

SW, SH = 13.333333333333334, 7.5

prs = Presentation()
# exact EMU dimensions of ROHIT PPT.pptx (16:9, 13.333" x 7.5")
prs.slide_width = Emu(12192000)
prs.slide_height = Emu(6858000)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def new_slide():
    return prs.slides.add_slide(BLANK)


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return tb, tf


def style_run(run, size, bold=False, italic=False, color=BLACK,
              underline=False, font=SERIF):
    f = run.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.underline = underline
    f.color.rgb = color
    # make the font apply to east-asian / complex-script slots too
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:cs", "a:ea"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def set_bullet(para, char="•", indent_in=0.28, hang_in=0.28,
               font="Arial"):
    """Attach a real OOXML bullet to a paragraph."""
    pPr = para._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Inches(indent_in)).emu if False else int(indent_in * 914400)))
    pPr.set("indent", str(-int(hang_in * 914400)))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buFont"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    buFont = pPr.makeelement(qn("a:buFont"), {"typeface": font})
    buChar = pPr.makeelement(qn("a:buChar"), {"char": char})
    pPr.append(buFont)
    pPr.append(buChar)


def no_bullet(para):
    pPr = para._p.get_or_add_pPr()
    for tag in ("a:buChar", "a:buAutoNum", "a:buFont"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    pPr.append(pPr.makeelement(qn("a:buNone"), {}))


def space_after(para, pts):
    para.space_after = Pt(pts)


def title(slide, text, size=28, y=0.34, h=1.0):
    tb, tf = textbox(slide, 0.55, y, SW - 1.1, h)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    style_run(r, size, bold=True, color=NAVY, underline=True)
    return tb


def bullets(slide, items, x=0.85, y=1.55, w=None, h=None, size=16,
            gap=10, bullet="•", line=1.0):
    """items: list of str, or (str, dict) with keys bold/italic/size/bullet/indent."""
    if w is None:
        w = SW - 2 * x
    if h is None:
        h = SH - y - 0.55
    tb, tf = textbox(slide, x, y, w, h)
    first = True
    for item in items:
        opts = {}
        if isinstance(item, tuple):
            item, opts = item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = opts.get("line", line)
        space_after(p, opts.get("gap", gap))
        bc = opts.get("bullet", bullet)
        ind = opts.get("indent", 0.0)
        if bc is None:
            no_bullet(p)
            if ind:
                p._p.get_or_add_pPr().set("marL", str(int(ind * 914400)))
        else:
            set_bullet(p, bc, indent_in=ind + 0.30, hang_in=0.30)
        # inline "Label: rest" bolding
        txt = item
        r = p.add_run()
        r.text = txt
        style_run(r, opts.get("size", size), bold=opts.get("bold", False),
                  italic=opts.get("italic", False),
                  color=opts.get("color", BLACK))
    return tb, tf


def label_value_pair(slide, rows, y=1.6, size=15, lx=1.15, lw=3.5,
                     vx=4.85, vw=7.6, gap=9):
    """Two aligned text boxes so label/value rows line up (as in the reference deck)."""
    tbl, tfl = textbox(slide, lx, y, lw, SH - y - 0.5)
    tbv, tfv = textbox(slide, vx, y, vw, SH - y - 0.5)
    for i, (lab, val) in enumerate(rows):
        pl = tfl.paragraphs[0] if i == 0 else tfl.add_paragraph()
        pv = tfv.paragraphs[0] if i == 0 else tfv.add_paragraph()
        for p in (pl, pv):
            p.line_spacing = 1.0
            space_after(p, gap)
        set_bullet(pl, "▸", indent_in=0.26, hang_in=0.26)
        no_bullet(pv)
        rl = pl.add_run()
        rl.text = lab
        style_run(rl, size, bold=True)
        rv = pv.add_run()
        rv.text = val
        style_run(rv, size)
    return tbl, tbv


def two_col(slide, head_l, items_l, head_r, items_r, y=1.5, size=14):
    xs = [0.75, 7.0]
    ws = [5.85, 5.85]
    for x, w, head, items in ((xs[0], ws[0], head_l, items_l),
                              (xs[1], ws[1], head_r, items_r)):
        tb, tf = textbox(slide, x, y, w, SH - y - 0.55)
        p = tf.paragraphs[0]
        no_bullet(p)
        space_after(p, 10)
        r = p.add_run()
        r.text = head
        style_run(r, size + 2, bold=True, color=NAVY, underline=True)
        for it in items:
            p = tf.add_paragraph()
            p.line_spacing = 1.0
            space_after(p, 8)
            set_bullet(p, "•", indent_in=0.28, hang_in=0.28)
            r = p.add_run()
            r.text = it
            style_run(r, size)


def picture(slide, path, y=1.45, max_h=None, max_w=None):
    from PIL import Image
    iw, ih = Image.open(path).size
    max_h = max_h or (SH - y - 0.55)
    max_w = max_w or (SW - 1.6)
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    slide.shapes.add_picture(path, Inches((SW - w) / 2), Inches(y),
                            Inches(w), Inches(h))
    return h


def caption(slide, text, y=None, size=11):
    y = y if y is not None else SH - 0.52
    tb, tf = textbox(slide, 0.75, y, SW - 1.5, 0.35)
    p = tf.paragraphs[0]
    no_bullet(p)
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    style_run(r, size, italic=True, color=GREY)


def footer_number(slide, n):
    tb, tf = textbox(slide, SW - 1.15, SH - 0.48, 0.6, 0.3)
    p = tf.paragraphs[0]
    no_bullet(p)
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = str(n)
    style_run(r, 11, color=GREY)


# ---------------------------------------------------------------- references
def parse_references():
    raw = open(os.path.join(WORK, "raw.txt"), encoding="utf-8").read()
    raw = raw[raw.index("1. Merle CL"):]
    # drop the Table 3 block and page furniture that interleaves the ref list
    raw = re.sub(r"Table 3 Clinical periodontal.*?PPD: periodontal probing depth",
                 "", raw, flags=re.S)
    keep = []
    for ln in raw.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("Özkal Emino"):
            continue
        if re.match(r"^Page \d+ of \d+$", s):
            continue
        if s.startswith("Publisher") or s.startswith("Springer Nature remains") \
           or s.startswith("published maps"):
            continue
        keep.append(s)
    joined = ""
    for s in keep:
        if joined.endswith("-"):
            joined = joined[:-1] + s
        else:
            joined = (joined + " " + s) if joined else s
    parts = re.split(r"(?:(?<=\s)|^)(\d{1,2})\.\s", joined)
    refs = {}
    for i in range(1, len(parts) - 1, 2):
        num = int(parts[i])
        body = parts[i + 1]
        if num in refs:
            continue
        body = body.split("http")[0]
        body = re.sub(r"s’(?=[a-z])", "s’ ", body)
        body = re.sub(r"\s+", " ", body).strip().rstrip(",;")
        if not body.endswith("."):
            body += "."
        refs[num] = body
    return [refs[i] for i in range(1, 55) if i in refs], refs


REFS, REFMAP = parse_references()
missing = [i for i in range(1, 55) if i not in REFMAP]
assert not missing, f"missing refs: {missing}"

# ================================================================ SLIDES
n = 0

# ---- 1 title -------------------------------------------------------------
s = new_slide()
tb, tf = textbox(s, 0.9, 1.55, SW - 1.8, 2.7)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
p.line_spacing = 1.15
r = p.add_run()
r.text = ("USING PATIENTS’ ORAL PHOTOGRAPHS FOR ORAL HYGIENE "
          "MOTIVATION PROMOTES PERIODONTAL HEALTH – A PROSPECTIVE "
          "RANDOMIZED CONTROLLED TRIAL")
style_run(r, 30, bold=True, color=NAVY)

tb, tf = textbox(s, 0.9, 4.45, SW - 1.8, 0.8)
p = tf.paragraphs[0]
no_bullet(p)
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = ("Ozkal Eminoglu D, Sahin BN, Bicer D, Dal G, "
          "Huseynli I, Dadashov A, Dogan DB, Kasali K.")
style_run(r, 15, italic=True, color=BLACK)
p = tf.add_paragraph()
no_bullet(p)
p.alignment = PP_ALIGN.CENTER
space_after(p, 0)
r = p.add_run()
r.text = "BMC Oral Health (2024) 24:786"
style_run(r, 15, bold=True, color=GREY)

tb, tf = textbox(s, SW - 4.6, 5.9, 3.9, 1.1)
for i, line in enumerate(["Submitted by:", "YUSRA"]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    no_bullet(p)
    p.alignment = PP_ALIGN.RIGHT
    space_after(p, 2)
    r = p.add_run()
    r.text = line
    style_run(r, 14, bold=(i == 1))

# ---- 2 contents ----------------------------------------------------------
n += 1
s = new_slide()
title(s, "CONTENTS")
bullets(s, [
    "Journal details",
    "Author details",
    "Abstract",
    "Introduction",
    "Aim and null hypothesis",
    "Materials and methods",
    "Results",
    "Discussion",
    "Study limitations and perspectives",
    "CONSORT 2010 flow chart",
    "Conclusion",
    "References",
], x=3.9, y=1.75, w=6.5, size=18, gap=11, bullet="➤")
footer_number(s, n)

# ---- 3 journal details ---------------------------------------------------
n += 1
s = new_slide()
title(s, "JOURNAL DETAILS")
label_value_pair(s, [
    ("Journal", "BMC Oral Health"),
    ("Journal title abbreviation", "BMC Oral Health"),
    ("Country", "United Kingdom (London)"),
    ("Publisher", "BioMed Central – Springer Nature"),
    ("Publication frequency", "Continuous publication; one volume per year"),
    ("Year of publication started", "2001"),
    ("Editorial model", "In-house BMC series editors with an external Editorial Board"),
    ("Language", "English"),
    ("Peer reviewed journal", "Yes"),
    ("Access", "Open access (CC BY 4.0)"),
    ("Impact factor", "2.6 (JCR 2023)"),
    ("Article", "2024; volume 24, article 786"),
    ("DOI", "10.1186/s12903-024-04553-7"),
    ("Received / Accepted / Published", "26 Feb 2024 / 01 Jul 2024 / 13 Jul 2024"),
], y=1.6, size=14, lx=0.95, lw=3.7, vx=5.0, vw=7.4, gap=7)
footer_number(s, n)

# ---- 4 author details ----------------------------------------------------
n += 1
s = new_slide()
title(s, "AUTHOR DETAILS")
tb, tf = textbox(s, 1.0, 1.85, SW - 2.0, 4.6)
p = tf.paragraphs[0]
no_bullet(p)
p.line_spacing = 1.25
space_after(p, 26)
r = p.add_run()
r.text = ("Didem Ozkal Eminoglu¹*; Beyza Nur Sahin¹; "
          "Dilek Bicer¹; Gulsum Dal¹; "
          "Ismayil Huseynli¹; Anvar Dadashov¹; "
          "Didar Betul Dogan¹; Kamber Kasali²")
style_run(r, 17)
for txt in [
    "¹Faculty of Dentistry, Department of Periodontology, Ataturk University, Erzurum, Turkey",
    "²Faculty of Medicine, Department of Biostatistics, Ataturk University, Erzurum, Turkey",
    "*Corresponding author – Didem Ozkal Eminoglu (ddm_ozkal@hotmail.com)",
]:
    p = tf.add_paragraph()
    no_bullet(p)
    p.line_spacing = 1.1
    space_after(p, 14)
    r = p.add_run()
    r.text = txt
    style_run(r, 15)
p = tf.add_paragraph()
no_bullet(p)
space_after(p, 0)
r = p.add_run()
r.text = ("The study was conducted at the Department of Periodontology Clinic, "
          "Faculty of Dentistry, Ataturk University, Erzurum, Turkey, "
          "between December 2023 and March 2024.")
style_run(r, 14, italic=True, color=GREY)
footer_number(s, n)

# ---- 5 abstract ----------------------------------------------------------
n += 1
s = new_slide()
title(s, "ABSTRACT")
bullets(s, [
    ("Objectives – The aim of this prospective, randomized, controlled, "
     "single-center, examiner-blinded clinical trial was to evaluate the "
     "effectiveness of a personalized and visual oral health education program "
     "in addition to conventional oral hygiene education.", {}),
    ("Materials and methods – Fifty-six non-smoking, right-handed participants "
     "(mean age 30.34 ± 11.46 years) without clinical signs of periodontitis were "
     "randomly allocated to two groups: the intervention group (n = 28) received "
     "personalized, visually aided oral health education combined with conventional "
     "oral hygiene education, and the control group (n = 28) received conventional "
     "oral hygiene education only. Periodontal parameters (PI, GI, BOP and PPD) "
     "were assessed in all participants at baseline, one month and three months.", {}),
    ("Results – A significant reduction (p < 0.001) in PI, GI and BOP was observed "
     "at both follow-up visits compared with baseline in both groups. No difference "
     "was found in either the inter-group (p > 0.05) or the intra-group (p = 1) "
     "comparison of PPD. At the third-month follow-up, PI (p = 0.012), GI "
     "(p = 0.032) and BOP (p = 0.024) scores were significantly lower in the "
     "intervention group than in the control group.", {}),
], x=0.85, y=1.55, size=15, gap=13, bullet="•")
footer_number(s, n)

# ---- 6 abstract contd ----------------------------------------------------
n += 1
s = new_slide()
title(s, "ABSTRACT")
bullets(s, [
    ("Conclusions – Clinical outcomes of periodontal health were significantly "
     "better with the personalized and visual oral health education program applied "
     "in this study than with the conventional oral hygiene education program.", {}),
    ("Clinical relevance – Numerous studies have reported additional interventions "
     "alongside oral hygiene education programs; however, the authors found no "
     "published studies investigating the role of patients’ own intra-oral "
     "photographs in oral care. The results of this study suggest that a visually "
     "aided education program for oral hygiene motivation may help improve oral "
     "health.", {}),
    ("Clinical trial registration – Registration number NCT06316505; "
     "registered on 18 March 2024.", {}),
    ("Keywords – Oral hygiene, Periodontal health, Periodontal indices, "
     "Motivation, Modified Bass technique.", {"italic": True}),
], x=0.85, y=1.6, size=15, gap=15)
footer_number(s, n)

# ---- 7 introduction (5 points) -------------------------------------------
n += 1
s = new_slide()
title(s, "INTRODUCTION")
bullets(s, [
    "Oral health \u2013 the state of the oral cavity and related tissues that lets a "
    "person eat, speak and communicate without active disease or discomfort \u2013 is "
    "linked to general health and quality of life. Poor oral health causes pain, "
    "malnutrition and absenteeism from school or work and can lead to infectious and "
    "degenerative disease, yet knowledge of how to promote oral care effectively is "
    "scarce; the WHO is actively promoting global awareness of oral health.",
    "Dental caries and periodontal disease are the most common, and preventable, "
    "diseases worldwide, and the tooth loss they cause is a common clinical indicator "
    "of poor oral health. Gingivitis affects 48.5\u201388% and periodontitis 5\u201315% of "
    "the world population \u2013 49\u201372% and 2\u201346.4% respectively in Turkey.",
    "Dental plaque is the major cause of periodontal disease, and the mouth\u2019s natural "
    "cleansing mechanisms cannot eliminate supragingival plaque. Controlling "
    "gingivitis through daily personal oral hygiene \u2013 tooth brushing supported by "
    "interdental and chemical plaque control \u2013 is therefore the most critical "
    "determinant in preventing and managing periodontal disease.",
    "A lifelong hygiene regimen depends on the patient\u2019s motivation and manual "
    "dexterity, so the dentist must teach an efficient routine and motivate adherence "
    "to it. Motivation peaks in the first days of training and declines over time, so "
    "individual-specific recall intervals are needed \u2013 three months is the most "
    "preferred time frame.",
    "Motivation strategies include chair-side instruction, message reminders, "
    "behavior modification and the Hawthorne effect, delivered through verbal and "
    "written instructions, photographs, videos, models, text messages, AI-based "
    "mobile health applications and WhatsApp tooth-selfie groups. Visual aids improve "
    "comprehension and motivation and produce more enduring change; although they "
    "demand chair time, they save time in the long run.",
], size=15, gap=13)
footer_number(s, n)

# ---- 10 aim --------------------------------------------------------------
n += 1
s = new_slide()
title(s, "AIM AND NULL HYPOTHESIS")
tb, tf = textbox(s, 1.5, 1.75, SW - 3.0, 5.1)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
no_bullet(p)
p.alignment = PP_ALIGN.CENTER
p.line_spacing = 1.25
space_after(p, 30)
r = p.add_run()
r.text = "Aim"
style_run(r, 19, bold=True, color=NAVY, underline=True)
p = tf.add_paragraph()
no_bullet(p)
p.alignment = PP_ALIGN.CENTER
p.line_spacing = 1.25
space_after(p, 30)
r = p.add_run()
r.text = ("This randomized controlled trial aimed to determine the effectiveness of "
          "a personalized and visual oral health education program, in addition to "
          "conventional education, on periodontal parameters.")
style_run(r, 18, italic=True)
p = tf.add_paragraph()
no_bullet(p)
p.alignment = PP_ALIGN.CENTER
space_after(p, 14)
r = p.add_run()
r.text = "Null hypothesis"
style_run(r, 19, bold=True, color=NAVY, underline=True)
p = tf.add_paragraph()
no_bullet(p)
p.alignment = PP_ALIGN.CENTER
p.line_spacing = 1.25
r = p.add_run()
r.text = ("Using oral photographs of patients in addition to conventional oral "
          "health education has no effect on oral hygiene.")
style_run(r, 18, italic=True)
footer_number(s, n)

# ---- 11 materials and methods -------------------------------------------
n += 1
s = new_slide()
title(s, "MATERIALS AND METHODS")
bullets(s, [
    "Study design: a randomized, controlled, examiner-blinded study with two "
    "parallel arms (prospective, single-centered).",
    "Before the study, each participant signed an informed consent form.",
    "The study was conducted in accordance with the guidelines of the World Medical "
    "Association Declaration of Helsinki of 1975, as revised in 2013.",
    "Ethical approval was obtained from the Ethics Committee of the Faculty of "
    "Dentistry at Ataturk University (meeting date 23.11.2023; meeting number 11; "
    "decision no. 58).",
    "The study design was based on the CONSORT (Consolidated Standards of Reporting "
    "Trials) 2010 statement.",
    "Clinical trial registration number “NCT06316505”; date of registration "
    "18.03.2024.",
], size=16, gap=13)
footer_number(s, n)

# ---- 12 setting and population ------------------------------------------
n += 1
s = new_slide()
title(s, "SETTING AND STUDY POPULATION")
bullets(s, [
    "Setting: Ataturk University Faculty of Dentistry, Department of "
    "Periodontology Clinic, Erzurum, Turkey.",
    "Study period: December 2023 to March 2024, comprising a baseline data "
    "collection and two follow-up assessments.",
    "Sample size: 56 participants (28 per group); mean age 30.34 ± 11.46 years; "
    "all non-smokers and right-handed, without clinical signs of periodontitis.",
    "Subjects were recruited among patients attending periodontal health care.",
    "Assessment points: baseline (T0), first follow-up at one month (T1) and second "
    "follow-up at three months (T2).",
    "For standardization, only right-handed patients at toothbrushing were selected, "
    "because right-handers clean the left jaw better than the right jaw and vice "
    "versa.",
], size=16, gap=13)
footer_number(s, n)

# ---- 13 inclusion / exclusion -------------------------------------------
n += 1
s = new_slide()
title(s, "INCLUSION AND EXCLUSION CRITERIA")
two_col(s,
        "Inclusion criteria", [
            "Age between 18 and 60 years",
            "A minimum of 12 teeth",
            "Availability for follow-up assessments",
            "No clinical signs of periodontitis",
            "Non-smoker and right-handed at toothbrushing",
        ],
        "Exclusion criteria", [
            "Systemic diseases or conditions that could impact periodontal health",
            "Neurological or psychiatric disorders",
            "Physical or mental disabilities",
            "Smoking habits or medication use that could impact periodontal health",
            "History of periodontal disease treatment",
            "Use of antibacterial mouth rinses in the preceding six months",
        ],
        y=1.6, size=15)
footer_number(s, n)

# ---- 14 group allocation -------------------------------------------------
n += 1
s = new_slide()
title(s, "GROUP ALLOCATION")
bullets(s, [
    ("Control group (CG) – n = 28", {"bullet": None, "bold": True, "size": 18,
                                          "gap": 8}),
    ("Participants received conventional oral hygiene education (COHE) only.",
     {"size": 16, "gap": 26, "indent": 0.15}),
    ("Intervention group (IG) – n = 28", {"bullet": None, "bold": True,
                                               "size": 18, "gap": 8}),
    ("In addition to COHE, participants received visual motivation – they were "
     "shown photographs (F1, F2, F3 and F4) of the changes in their own oral health "
     "status taken at each session.", {"size": 16, "gap": 26, "indent": 0.15}),
    ("Both groups were educated face to face, one by one, at every session "
     "(T0, T1 and T2).", {"bullet": None, "size": 15, "italic": True,
                          "color": GREY}),
], x=1.1, y=1.75, size=16)
footer_number(s, n)

# ---- 15 COHE protocol ----------------------------------------------------
n += 1
s = new_slide()
title(s, "CONVENTIONAL ORAL HYGIENE EDUCATION (COHE)")
bullets(s, [
    "COHE was provided by a single educator (DBD) for each participant one by one "
    "and face to face.",
    "Tooth cleaning techniques – the modified Bass technique and interdental "
    "brushing – were demonstrated in each session (T0, T1 and T2).",
    "Patients were requested to brush their teeth in front of a mirror after the "
    "demonstration of brushing procedures.",
    "Each participant was provided with an identical toothbrush and toothpaste to "
    "minimize potential effects on oral hygiene.",
    "All participants were recommended to brush twice daily for two minutes and "
    "were encouraged to adhere to oral self-care recommendations.",
    "The procedure lasted approximately 20 minutes and was performed in a quiet "
    "environment.",
], size=16, gap=12)
footer_number(s, n)

# ---- 16 visual motivation protocol --------------------------------------
n += 1
s = new_slide()
title(s, "VISUAL MOTIVATION PROTOCOL (INTERVENTION GROUP)")
bullets(s, [
    ("Photography – intra-oral images were taken by one operator (AD) at each "
     "session using a digital camera with a ring flash and a 100-mm macro lens; "
     "five photos per participant (frontal, right and left lateral, maxillary and "
     "mandibular views).", {}),
    ("Session 1 – baseline (T0): a photograph was taken before (F1) and after "
     "(F2) full-mouth scaling and root planing (fmSRP), and the patient was "
     "presented with the difference.", {}),
    ("Session 2 – one month after baseline (T1): a photograph (F3) was taken and "
     "the patient was presented with the difference between F1 and F3.", {}),
    ("Session 3 – three months after baseline (T2): a photograph (F4) was taken "
     "and the patient was presented with the difference between F1, F3 and F4.", {}),
], size=16, gap=14)
footer_number(s, n)

# ---- 17 oral assessment / calibration -----------------------------------
n += 1
s = new_slide()
title(s, "ORAL ASSESSMENT AND EXAMINER CALIBRATION")
bullets(s, [
    "Parameters measured: plaque index (PI), gingival index (GI), bleeding on "
    "probing (BOP) and periodontal probing depth (PPD), recorded at T0, T1 and T2 "
    "using a Williams periodontal probe (Hu-Friedy, Chicago, IL, USA).",
    "Oral assessments were recorded blindly by four dentists (IH, BNS, DB and "
    "GD), trained and calibrated by an experienced calibrated examiner (DOE).",
    "For intra-investigator calibration, measurements were repeated at baseline and "
    "one week later in a number of patients equal to 10% of the participants; these "
    "patients were not included in the study.",
    "Intra-rater correlations: PI ICC = 1.000 (p < 0.001); BOP ICC = 1.000 "
    "(p < 0.001); PPD ICC = 0.828 (p = 0.038); GI ICC = 0.995 (p < 0.001).",
], size=16, gap=15)
footer_number(s, n)

# ---- 17 sample size, randomization and blinding --------------------------
n += 1
s = new_slide()
title(s, "SAMPLE SIZE, RANDOMIZATION AND BLINDING", size=25)
bullets(s, [
    ("Sample size estimation", {"bullet": None, "bold": True, "size": 17, "gap": 7}),
    ("G-Power\u00ae version 3.1 (University of D\u00fcsseldorf, Germany) was used for "
     "sample size estimation, based on a previous study; oral hygiene status was "
     "calculated using the Silness\u2013L\u00f6e Plaque Index.",
     {"size": 15, "gap": 6, "indent": 0.15}),
    ("Assumed change in plaque index in the motivation group: baseline 1.14 \u00b1 0.51 "
     "and third month 0.85 \u00b1 0.51, giving an expected mean difference of 0.29 and an "
     "effect size of 0.560.", {"size": 15, "gap": 6, "indent": 0.15}),
    ("Twenty-eight volunteers per group were required for a type I error of 5% and a "
     "power of 80%.", {"size": 15, "gap": 18, "indent": 0.15}),
    ("Randomization and blinding", {"bullet": None, "bold": True, "size": 17, "gap": 7}),
    ("Participants were divided into two groups using the www.randomizer.org website "
     "\u2013 a research randomizer whose pseudo-random number generator is initialized "
     "by the computer\u2019s clock.", {"size": 15, "gap": 6, "indent": 0.15}),
    ("Examiner-blinded, two parallel arm controlled trial: the oral hygiene "
     "instructor, the clinical examiners and the statistical analyzer were all "
     "blinded to the groups of the patients.",
     {"size": 15, "gap": 0, "indent": 0.15}),
], x=1.1, y=1.7, size=15)
footer_number(s, n)

# ---- 20 statistical analysis --------------------------------------------
n += 1
s = new_slide()
title(s, "STATISTICAL ANALYSIS")
bullets(s, [
    "Analyses were performed (by KK) with IBM SPSS 20. Data are presented as "
    "mean ± standard deviation, median, minimum, maximum, percentage and number.",
    "Normal distribution was assessed with the Shapiro–Wilk and "
    "Kolmogorov–Smirnov tests, Q-Q plots, skewness and kurtosis.",
    "Two independent groups: independent samples t-test if normality was met, "
    "otherwise the Mann–Whitney U test.",
    "More than two dependent variables: Repeated Measures ANOVA if normal, "
    "otherwise the Friedman test. Post-hoc: Tukey’s test (homogeneous variances) "
    "or Tamhane’s T2; Friedman 2-way ANOVA by ranks after the Friedman test.",
    "Categorical variables: Pearson chi-square, chi-square Yates or Fisher’s "
    "Exact test according to the expected value; Fisher–Freeman–Halton test "
    "for comparisons larger than 2 × 2.",
    "Correlation: Pearson test if normally distributed, otherwise the Spearman test. "
    "Statistical significance was set at p < 0.05.",
], size=15, gap=12)
footer_number(s, n)

# ---- 21 results: participant flow ---------------------------------------
n += 1
s = new_slide()
title(s, "RESULTS")
bullets(s, [
    "247 patients were assessed for eligibility; 191 were excluded (39 declined to "
    "participate and 152 did not meet the inclusion criteria).",
    "56 participants were enrolled and randomly separated into two groups "
    "(n = 28 each).",
    "Over the follow-up period (first and third months) no participant dropped out.",
    "Comparison of demographic characteristics at baseline revealed no significant "
    "differences between the two groups: age (p = 0.325), gender (p = 0.057) and "
    "educational level (p = 0.756).",
    "No correlation was found between age and oral hygiene parameters measured in "
    "all sessions (p > 0.05), and no significant difference was observed between "
    "gender and the oral hygiene parameters (p > 0.05).",
], size=16, gap=14)
footer_number(s, n)

# ---- 22 Table 1 ----------------------------------------------------------
n += 1
s = new_slide()
title(s, "RESULTS – DEMOGRAPHIC CHARACTERISTICS", size=26)
h1 = picture(s, os.path.join(FIG, "table1.png"), y=1.85, max_h=4.0, max_w=10.9)
caption(s, "Table 1 – Demographic characteristics of participants "
           "(IG: intervention group; CG: control group).", y=1.85 + h1 + 0.22)
footer_number(s, n)

# ---- 23 Table 3 ----------------------------------------------------------
n += 1
s = new_slide()
title(s, "RESULTS – CLINICAL PERIODONTAL PARAMETERS", size=26)
picture(s, os.path.join(FIG, "table3.png"), y=1.6, max_h=5.15, max_w=11.9)
caption(s, "Table 3 – PI, GI, BOP and PPD at baseline (T0), 1st month (T1) "
           "and 3rd month (T2).")
footer_number(s, n)

# ---- 24 results contd ----------------------------------------------------
n += 1
s = new_slide()
title(s, "RESULTS \u2013 PERIODONTAL PARAMETER COMPARISONS", size=26)
bullets(s, [
    "At baseline (T0) no significant difference was seen between groups for "
    "PI (p = 0.491), GI (p = 0.557), BOP (p = 0.466) or PPD (p = 0.529).",
    "Intra-group: a statistically significant difference was observed between T0 and "
    "T1 and between T0 and T2 in both groups for PI, GI and BOP (p < 0.001).",
    "No statistically significant difference was observed for PPD scores between "
    "T0, T1 and T2 in either group (p = 1.0).",
    "Inter-group at three months: the intervention group displayed significantly "
    "greater improvement than the control group in PI (p = 0.012), GI (p = 0.032) "
    "and BOP (p = 0.024).",
    "PPD showed no significant difference between the two groups three months after "
    "T0 (p = 0.503).",
], size=16, gap=14)
footer_number(s, n)

# ---- 25 chart ------------------------------------------------------------
n += 1
s = new_slide()
title(s, "RESULTS – GRAPHICAL SUMMARY", size=26)

cd = CategoryChartData()
cd.categories = ["PI T0", "PI T1", "PI T2", "GI T0", "GI T1", "GI T2"]
cd.add_series("Intervention group", (1.76, 0.87, 0.57, 1.33, 0.80, 0.43))
cd.add_series("Control group", (1.79, 1.08, 0.92, 1.29, 0.86, 0.74))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8),
                        Inches(1.75), Inches(7.4), Inches(4.9), cd)
ch = gf.chart
ch.has_title = True
ch.chart_title.text_frame.text = "Mean plaque and gingival index scores"
for r in ch.chart_title.text_frame.paragraphs[0].runs:
    style_run(r, 14, bold=True, color=NAVY)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.font.size = Pt(11)
ch.font.name = SERIF
pl = ch.plots[0]
pl.has_data_labels = True
pl.data_labels.number_format = "0.00"
pl.data_labels.number_format_is_linked = False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(9)
pl.data_labels.font.name = SERIF
ch.series[0].format.fill.solid()
ch.series[0].format.fill.fore_color.rgb = NAVY
ch.series[1].format.fill.solid()
ch.series[1].format.fill.fore_color.rgb = RGBColor(0xA9, 0xB4, 0xD6)

bullets(s, [
    ("Bleeding on probing (%)", {"bullet": None, "bold": True, "size": 16,
                                 "gap": 10}),
    ("IG: 54.57% → 18.41% → 4.36%", {"size": 15, "gap": 6}),
    ("CG: 45.36% → 21.77% → 11.96%", {"size": 15, "gap": 22}),
    ("Probing depth (mm)", {"bullet": None, "bold": True, "size": 16, "gap": 10}),
    ("IG: 2.13 → 2.13 → 2.15", {"size": 15, "gap": 6}),
    ("CG: 2.00 → 2.00 → 2.00", {"size": 15, "gap": 22}),
    ("Both groups improved from baseline, but the visually motivated group was "
     "significantly better at three months for PI, GI and BOP.",
     {"bullet": None, "size": 14, "italic": True, "color": GREY}),
], x=8.6, y=2.0, w=4.0, h=4.6, size=15)
footer_number(s, n)

# ---- 24 discussion 1 ----------------------------------------------------
n += 1
s = new_slide()
title(s, "DISCUSSION")
bullets(s, [
    "This study aimed to examine the impact of a personalized visual oral hygiene "
    "motivation program, in addition to regular oral hygiene instruction, on "
    "clinical periodontal parameters.",
    "Adopting effective brushing techniques is more crucial than choosing the right "
    "brushing instrument to manage dental plaque. Of the techniques that emerged "
    "over the last half-century, only the Bass technique has proven effective in "
    "cleansing the gingival sulcus \u2013 hence the modified Bass technique was used "
    "here.",
    "Participants were recommended to brush twice daily for two minutes. An average "
    "of 42% of dental plaque can be eliminated with two minutes of manual brushing, "
    "and two minutes yields superior results compared with one minute.",
    "Extending brushing time to three minutes or more does not seem to enhance "
    "plaque removal and is likely to discourage most participants.",
    "Despite consistent and frequent professional reminders during dental "
    "appointments, modifying patients\u2019 oral hygiene behavior tends to be "
    "challenging. Chair-side instruction may still be the gold standard for oral "
    "hygiene education.",
    "Oral health education based on a conventional method, with or without visual "
    "materials, effectively improved oral hygiene \u2013 PI, GI and BOP were "
    "significantly improved from T0. This may be due to periodic training, and the "
    "Hawthorne effect should also be considered.",
], size=15, gap=11)
footer_number(s, n)

# ---- 25 discussion 2 ----------------------------------------------------
n += 1
s = new_slide()
title(s, "DISCUSSION")
bullets(s, [
    "The initial reduction in plaque in both groups can be explained by the fmSRP "
    "itself; professional plaque removal may affect the first-month PI scores.",
    "The significant decrease in BOP and GI at both follow-ups may be related to "
    "reduced plaque accumulation. BOP is the most important sign of gingivitis, so "
    "its reduction is a sensitive and reliable indicator of a positive change in "
    "oral care.",
    "The present results confirm that personalized education has a more positive "
    "effect on PI, GI and BOP after three months than standardized oral hygiene "
    "instructions \u2013 in line with studies proving the advantage of individualized "
    "instruction methods.",
    "A systematic review reported that an oral hygiene motivation program in "
    "patients undergoing fixed orthodontic treatment leads to reduced microbial "
    "plaque accumulation and improved gingival health.",
    "The findings support the recommendation to employ a direct counseling method: "
    "oral hygiene education should attend to the individual strengths and weaknesses "
    "of patients and implement a motivational program accordingly. A qualitative "
    "study showed that a personalized dialogue and a good caregiver\u2013patient "
    "relationship affect lifestyle habits positively.",
    "Scaling, root planing and oral hygiene instructions did not affect the "
    "stability of shallow PPD \u2013 mean PPD at the initial session was under 3 mm and "
    "the population had no clinical signs of periodontitis, so no pocket-depth "
    "reduction was possible.",
], size=15, gap=11)
footer_number(s, n)

# ---- 29 limitations ------------------------------------------------------
n += 1
s = new_slide()
title(s, "STUDY LIMITATIONS AND PERSPECTIVES")
bullets(s, [
    "The study was conducted on a relatively small sample size.",
    "Participants were observed for a short time frame of three months, so the "
    "findings may not provide information about potential long-term outcomes.",
    "Different operators were involved in the oral assessment.",
    "The Hawthorne effect is a limitation of studies involving participants’ "
    "motivation.",
    "Future studies should include a larger sample size involving patients with "
    "periodontitis, an extended observation period, and other brushing techniques "
    "and different toothbrushes.",
    "Future studies could also involve elderly participants, since age is a crucial "
    "factor involving cognitive abilities.",
], size=16, gap=13)
footer_number(s, n)

# ---- 27-28 CONSORT (split in two for legibility) -------------------------
for img, sub, cap, iy, mh in (
    ("consort_top.png", "ENROLMENT AND ALLOCATION", None, 1.45, 5.5),
    ("consort_bottom.png", "FOLLOW-UP AND ANALYSIS",
     "Fig. 1 (part 2) – no drop-outs at T1 (1 month) or T2 (3 months); "
     "28 participants per group analyzed.", 1.75, 4.55),
):
    n += 1
    s = new_slide()
    title(s, f"CONSORT 2010 FLOW CHART – {sub}", size=24)
    ih = picture(s, os.path.join(FIG, img), y=iy, max_h=mh, max_w=11.9)
    if cap:
        caption(s, cap, y=iy + ih + 0.22)
    footer_number(s, n)

# ---- 29 conclusion -------------------------------------------------------
n += 1
s = new_slide()
title(s, "CONCLUSION")
tb, tf = textbox(s, 1.4, 1.8, SW - 2.8, 5.0)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
for i, (txt, opts) in enumerate([
    ("The two groups displayed a significant reduction in PI, GI and BOP scores.",
     {"gap": 26}),
    ("The beneficial effect of the client-centered, directive therapeutic, visually "
     "aided oral health education approach significantly improved clinical outcomes "
     "compared to conventional oral hygiene education alone.", {"gap": 0}),
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    no_bullet(p)
    p.alignment = PP_ALIGN.CENTER
    p.line_spacing = 1.3
    space_after(p, opts["gap"])
    r = p.add_run()
    r.text = txt
    style_run(r, 19, italic=True)
footer_number(s, n)

# ---- 30 references (condensed, single slide) -----------------------------
CONDENSE_FIX = {
    7:  "Akp\u0131nar A, et al. Cumhur Dent J. 2001.",
    9:  "Jakubovics NS, et al. Periodontol 2000. 2021.",
    22: "Wynn RM, et al. J Healthc Eng. 2018.",
    30: "Uguz HN, et al. J Clin Pediatr Dent. 2023.",
    38: "Ainamo J, Bay I. Int Dent J. 1975.",
    40: "Urbaniak GC, Plous S. Research Randomizer v4.0 [software]. 2013.",
}


def condense(num, ref):
    """'Surname I, et al. Journal. Year.' - short form of a full reference."""
    if num in CONDENSE_FIX:
        return CONDENSE_FIX[num]
    m = re.match(r"^(.*?)\.\s", ref)
    authors = m.group(1) if m else ref[:40]
    names = [a.strip() for a in authors.split(",") if a.strip()]
    if len(names) == 1:
        who = names[0] + "."
    elif len(names) == 2:
        who = names[0] + ", " + names[1] + "."
    else:
        who = names[0] + ", et al."
    yrs = list(re.finditer(r"(?:19|20)\d{2}", ref))
    year = yrs[-1].group(0) if yrs else ""
    before = ref[:yrs[-1].start()] if yrs else ref
    chunk = [c for c in re.split(r"\.\s", before) if c.strip()]
    jour = chunk[-1].strip(" .;:,()") if chunk else ""
    return f"{who} {jour}. {year}.".replace("..", ".")


n += 1
s = new_slide()
title(s, "REFERENCES", size=26)
COLS, ROWS = 2, 27
short = [condense(i + 1, r) for i, r in enumerate(REFS)]
for c in range(COLS):
    tb, tf = textbox(s, 0.7 + c * 6.2, 1.42, 5.95, SH - 2.0)
    col = short[c * ROWS:(c + 1) * ROWS]
    for j, ref in enumerate(col):
        para = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        no_bullet(para)
        para.line_spacing = 1.0
        space_after(para, 1.5)
        pPr = para._p.get_or_add_pPr()
        pPr.set("marL", str(int(0.30 * 914400)))
        pPr.set("indent", str(-int(0.30 * 914400)))
        r = para.add_run()
        r.text = f"{c * ROWS + j + 1}.  {ref}"
        style_run(r, 10.5)
caption(s, "Short form (first author, journal, year); the full reference list "
           "appears in the published article.", y=SH - 0.5)
footer_number(s, n)

# ---- last: thank you -----------------------------------------------------
n += 1
s = new_slide()
tb, tf = textbox(s, 1.0, 3.0, SW - 2.0, 1.5)
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
no_bullet(p)
p.alignment = PP_ALIGN.CENTER
r = p.add_run()
r.text = "THANK YOU"
style_run(r, 44, bold=True, color=NAVY)

dest = os.path.join(OUT, "Oral_Photographs_Oral_Hygiene_Motivation_Journal_Club.pptx")
prs.save(dest)
print("saved:", dest)
print("slides:", len(prs.slides.__iter__.__self__._sldIdLst))
print("refs parsed:", len(REFS))
