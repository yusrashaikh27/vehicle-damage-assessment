"""Static checks on the Django templates.

Django is not installed in this sandbox, so the templates cannot be rendered
here. These are the three failures that would otherwise surface as a 500 in
front of an examiner:

1. unbalanced block tags ({% if %} without {% endif %}, and so on);
2. a {% url %} name that does not exist in assessment/urls.py;
3. a filter that is neither a Django built-in nor registered in our own
   templatetags (the classic being a custom filter used without {% load %}).

Plus an HTML element-balance check on the markup with the template tags stripped.
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

# The project root is found by walking up from this file, never hardcoded and
# never assumed to be the working directory - so `python tests/check_templates.py` works
# from the project root, from inside tests/, or from anywhere else.
def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "assessment" / "models.py").exists():
            return candidate
    raise SystemExit(
        "Cannot find the project root: no parent of this file contains "
        "assessment/models.py. Run this from inside the project."
    )


ROOT = _project_root()
TEMPLATES = sorted(ROOT.glob("templates/**/*.html"))

TAG_RE = re.compile(r"\{%\s*(\w+)(.*?)%\}", re.S)
VAR_RE = re.compile(r"\{\{(.*?)\}\}", re.S)

PAIRED = {
    "if": "endif", "for": "endfor", "block": "endblock", "with": "endwith",
    "comment": "endcomment", "spaceless": "endspaceless", "blocktrans":
    "endblocktrans", "autoescape": "endautoescape", "filter": "endfilter",
    "verbatim": "endverbatim", "ifchanged": "endifchanged",
}
MIDDLE = {"else", "elif", "empty"}

# Django built-in filters (django.template.defaultfilters) plus ours.
BUILTIN_FILTERS = {
    "add", "addslashes", "capfirst", "center", "cut", "date", "default",
    "default_if_none", "dictsort", "dictsortreversed", "divisibleby", "escape",
    "escapejs", "escapeseq", "filesizeformat", "first", "floatformat",
    "force_escape", "get_digit", "iriencode", "join", "json_script", "last",
    "length", "length_is", "linebreaks", "linebreaksbr", "linenumbers", "ljust",
    "lower", "make_list", "phone2numeric", "pluralize", "pprint", "random",
    "rjust", "safe", "safeseq", "slice", "slugify", "stringformat", "striptags",
    "time", "timesince", "timeuntil", "title", "truncatechars",
    "truncatechars_html", "truncatewords", "truncatewords_html", "unordered_list",
    "upper", "urlencode", "urlize", "urlizetrunc", "wordcount", "wordwrap", "yesno",
}
OUR_FILTERS = {"rupees"}


class Balance(HTMLParser):
    """Element-balance check. Only reports genuinely unclosed elements."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}
    # Elements whose end tag HTML lets you omit.
    OPTIONAL_END = {"li", "p", "td", "th", "tr", "thead", "tbody", "tfoot",
                    "option", "dt", "dd"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                for name, line in self.stack[i + 1:]:
                    if name not in self.OPTIONAL_END:
                        self.problems.append(
                            f"<{name}> opened line {line} not closed before </{tag}>")
                del self.stack[i:]
                return
        self.problems.append(f"stray </{tag}> at line {self.getpos()[0]}")


def url_names() -> set[str]:
    text = (ROOT / "assessment" / "urls.py").read_text()
    return set(re.findall(r"name=[\"'](\w+)[\"']", text))


def check(path: Path, known_urls: set[str]) -> list[str]:
    src = path.read_text()
    errs: list[str] = []

    # --- 1. block tag balance -------------------------------------------
    stack: list[tuple[str, int]] = []
    for m in TAG_RE.finditer(src):
        name = m.group(1)
        line = src.count("\n", 0, m.start()) + 1
        if name in PAIRED:
            stack.append((name, line))
        elif name.startswith("end"):
            want = name
            if not stack:
                errs.append(f"line {line}: {{% {name} %}} with nothing open")
                continue
            opened, oline = stack[-1]
            if PAIRED.get(opened) != want:
                errs.append(
                    f"line {line}: {{% {name} %}} closes {{% {opened} %}} "
                    f"opened at line {oline}")
            stack.pop()
        elif name in MIDDLE and not stack:
            errs.append(f"line {line}: {{% {name} %}} outside a block")
    for opened, oline in stack:
        errs.append(f"line {oline}: {{% {opened} %}} never closed")

    # --- 2. url names ----------------------------------------------------
    for m in TAG_RE.finditer(src):
        if m.group(1) != "url":
            continue
        line = src.count("\n", 0, m.start()) + 1
        target = re.search(r"[\"']([\w:]+)[\"']", m.group(2))
        if not target:
            continue
        ref = target.group(1)
        if ":" in ref:
            namespace, name = ref.split(":", 1)
            if namespace == "assessment" and name not in known_urls:
                errs.append(f"line {line}: url name 'assessment:{name}' not in urls.py")

    # --- 3. filters ------------------------------------------------------
    loaded = set()
    for m in TAG_RE.finditer(src):
        if m.group(1) == "load":
            loaded.update(m.group(2).split())

    for m in VAR_RE.finditer(src):
        line = src.count("\n", 0, m.start()) + 1
        # Strip quoted filter arguments before splitting on |, so a literal
        # pipe inside a string cannot be read as a filter.
        expr = re.sub(r"\"[^\"]*\"|'[^']*'", "''", m.group(1))
        for part in expr.split("|")[1:]:
            fname = part.split(":")[0].strip()
            if not fname:
                continue
            if fname in OUR_FILTERS:
                if "money" not in loaded:
                    errs.append(f"line {line}: filter '{fname}' used without "
                                f"{{% load money %}}")
            elif fname not in BUILTIN_FILTERS:
                errs.append(f"line {line}: unknown filter '{fname}'")

    # --- 4. html balance --------------------------------------------------
    # Comment bodies go first: they are prose and may contain stray angle
    # brackets. Stripping tags before comments would leave the body behind.
    body = re.sub(r"\{%\s*comment.*?endcomment\s*%\}", "", src, flags=re.S)
    stripped = VAR_RE.sub("x", TAG_RE.sub("", body))
    parser = Balance()
    parser.feed(stripped)
    for name, line in parser.stack:
        if name not in Balance.OPTIONAL_END:
            parser.problems.append(f"<{name}> opened line {line} never closed")
    errs.extend(f"html: {p}" for p in parser.problems)

    return errs


def main() -> int:
    known = url_names()
    print(f"url names in assessment/urls.py: {sorted(known)}\n")
    total = 0
    for path in TEMPLATES:
        errs = check(path, known)
        total += len(errs)
        label = path.relative_to(ROOT)
        if errs:
            print(f"FAIL  {label}")
            for e in errs:
                print(f"      {e}")
        else:
            print(f"ok    {label}")
    print(f"\n{len(TEMPLATES)} templates, {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
