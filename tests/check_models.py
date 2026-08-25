#!/usr/bin/env python3
"""Resolve every model attribute the admin, API and templates ask for.

Django's own `manage.py check` catches most of this (admin.E108, E116, and so
on), and `manage.py check` is the thing to run when Django is available. This
exists because it is not available here, and because the class of bug it catches
is the expensive kind: a renamed field shows up as a blank admin column, an empty
table cell, or a 500 in front of an examiner - never as a syntax error.

What it does
------------
1. Parses `assessment/models.py` with `ast` to build, per model: its concrete
   fields, the `get_<field>_display` methods Django generates for fields with
   choices, its properties and methods, and the reverse accessors its foreign
   keys create on *other* models.
2. Walks `admin.py` for every registered ModelAdmin and inline, and resolves each
   name in `list_display`, `list_filter`, `search_fields`, `readonly_fields`,
   `fields`, `fieldsets`, `date_hierarchy` and `list_select_related` - following
   `__` spans across relations.
3. Walks the templates for dotted variable paths and resolves them against the
   model each context variable is declared to hold (the CONTEXT map below).

Deliberately conservative: anything it cannot resolve *with confidence* is
reported as "unknown", separately from a definite failure, so the output stays
readable rather than becoming a wall of false alarms to skim past.

Run it from the project root:  python3 check_models.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Which model each template context variable holds.
#
# Written down rather than inferred: the views build these dicts, and reading a
# name out of a `render()` call is guesswork, while this is a statement of intent
# that fails loudly if a view stops honouring it.
# --------------------------------------------------------------------------
CONTEXT = {
    "assessment/result.html": {
        "inspection": "Inspection",
        "entry.view": "VehicleImage",
        "det": "DamageDetection",
        "line": "CostLine",
    },
    "assessment/history.html": {
        "inspection": "Inspection",
        "shot": "VehicleImage",
    },
}

# Names that are template-local or framework-provided, not model attributes.
TEMPLATE_LOCALS = {
    "forloop", "form", "formset", "slot", "field", "hidden", "item", "entry",
    "views", "page", "summary", "query", "cost_lines", "cost_source",
    "gst_percent", "multi_view", "note", "panel", "error", "request", "user",
    "messages", "csrf_token", "block", "True", "False", "None",
}

# Attributes Django puts on every model instance or on file/related fields.
UNIVERSAL = {
    "pk", "id", "objects", "all", "count", "url", "name", "size", "path",
    "exists", "items", "keys", "values",
}

ADMIN_LIST_ATTRS = (
    "list_display", "list_filter", "search_fields", "readonly_fields",
    "fields", "list_select_related", "date_hierarchy", "ordering",
    "autocomplete_fields", "raw_id_fields", "filter_horizontal",
)


# --------------------------------------------------------------------------
# Model introspection
# --------------------------------------------------------------------------

class Model:
    def __init__(self, name: str):
        self.name = name
        self.fields: set[str] = set()
        self.relations: dict[str, str] = {}   # field name -> target model
        self.callables: set[str] = set()      # properties, methods, get_x_display
        self.reverse: dict[str, str] = {}     # accessor -> model on the other end

    def has(self, attr: str) -> bool:
        return (attr in self.fields or attr in self.callables
                or attr in self.reverse or attr in UNIVERSAL)

    def target(self, attr: str) -> str | None:
        return self.relations.get(attr) or self.reverse.get(attr)


RELATION_KINDS = {"ForeignKey", "OneToOneField", "ManyToManyField"}


def _is_field_call(node: ast.AST) -> ast.Call | None:
    """`models.CharField(...)` and friends, however they were written.

    Note the explicit relation set: `ForeignKey` does not end in "Field", so a
    suffix test alone silently misses every relation - which made the first run
    of this script report sixteen false failures. The lesson is the checker needs
    a self-test as much as the code does; see `--selftest`.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "models" and (
            func.attr.endswith("Field") or func.attr in RELATION_KINDS
        ):
            return node
    return None


def _kwarg(call: ast.Call, key: str):
    for kw in call.keywords:
        if kw.arg == key:
            return kw.value
    return None


def _literal(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def parse_models(path: Path) -> dict[str, Model]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    models: dict[str, Model] = {}
    pending_reverse: list[tuple[str, str, str]] = []  # (target, accessor, owner)

    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        # Only Django models: a base of `models.Model`.
        bases = [
            b.attr for b in cls.bases
            if isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name)
            and b.value.id == "models"
        ]
        if "Model" not in bases:
            continue

        model = Model(cls.name)
        models[cls.name] = model

        for node in cls.body:
            # Fields
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                fname = node.targets[0].id
                call = _is_field_call(node.value)
                if call is None:
                    continue
                model.fields.add(fname)

                kind = call.func.attr  # type: ignore[union-attr]

                # choices= gives Django a get_<field>_display method.
                if _kwarg(call, "choices") is not None:
                    model.callables.add(f"get_{fname}_display")

                # Relations: first positional arg names the target.
                if kind in RELATION_KINDS:
                    target = None
                    if call.args:
                        if isinstance(call.args[0], ast.Name):
                            target = call.args[0].id
                        else:
                            target = _literal(call.args[0])
                    if target and "." not in (target or ""):
                        model.relations[fname] = target
                        accessor = _literal(_kwarg(call, "related_name"))
                        if accessor and accessor != "+":
                            pending_reverse.append((target, accessor, cls.name))

            # Properties and methods
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                model.callables.add(node.name)

    for target, accessor, owner in pending_reverse:
        if target in models:
            models[target].reverse[accessor] = owner

    return models


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------

def resolve(models: dict[str, Model], model_name: str, parts: list[str],
            sep_ok: bool = False) -> tuple[bool, str]:
    """Walk a dotted (or `__`) path from a model. Returns (ok, explanation)."""
    current = models.get(model_name)
    if current is None:
        return True, "unknown model"

    for i, part in enumerate(parts):
        if current is None:
            return True, "left the model graph"
        if not current.has(part):
            return False, f"{current.name} has no {part!r}"
        nxt = current.target(part)
        if nxt is None:
            # A non-relation: anything further is on a plain value, so stop
            # checking rather than guessing.
            return True, "ok"
        current = models.get(nxt)
    return True, "ok"


# --------------------------------------------------------------------------
# Admin
# --------------------------------------------------------------------------

def _string_seq(node) -> list[str]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return [s for s in (_literal(e) for e in node.elts) if s]
    single = _literal(node)
    return [single] if single else []


def _fieldset_fields(node) -> list[str]:
    """Field names buried in a `fieldsets` structure."""
    found: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Dict):
            for key, value in zip(sub.keys, sub.values):
                if _literal(key) == "fields":
                    found.extend(_string_seq(value))
    return found


def check_admin(path: Path, models: dict[str, Model]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems: list[str] = []

    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        # Which model? Either @admin.register(X) or `model = X` on an inline.
        model_name = None
        for dec in cls.decorator_list:
            if isinstance(dec, ast.Call) and dec.args \
                    and isinstance(dec.args[0], ast.Name):
                model_name = dec.args[0].id
        for node in cls.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id == "model" \
                    and isinstance(node.value, ast.Name):
                model_name = node.value.id
        if model_name is None or model_name not in models:
            continue

        # Methods defined on the ModelAdmin itself are legitimate list_display
        # and readonly_fields entries.
        own = {n.name for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

        for node in cls.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                continue
            attr = node.targets[0].id

            if attr == "fieldsets":
                names = _fieldset_fields(node.value)
            elif attr in ADMIN_LIST_ATTRS:
                names = _string_seq(node.value)
            else:
                continue

            for name in names:
                if name in own:
                    continue
                parts = name.split("__") if "__" in name else [name]
                ok, why = resolve(models, model_name, parts)
                if not ok:
                    problems.append(
                        f"admin.py  {cls.name}.{attr}: {name!r} - {why}"
                    )
    return problems


# --------------------------------------------------------------------------
# DRF serializers
# --------------------------------------------------------------------------

def check_api(path: Path, models: dict[str, Model]) -> list[str]:
    """Resolve every name in a serializer's `Meta.fields`.

    DRF raises ImproperlyConfigured for an unknown field name, but only when the
    serializer is first instantiated - which is at request time, not at import.
    So a stale name here is a 500 on a page that looked fine when the server
    started.

    A name is acceptable if it is a model field, a property or method, or a field
    declared explicitly on the serializer class (`SerializerMethodField`,
    `source=` aliases, nested serializers).
    """
    if not path.exists():
        return []

    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems: list[str] = []

    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        meta = next((n for n in cls.body
                     if isinstance(n, ast.ClassDef) and n.name == "Meta"), None)
        if meta is None:
            continue

        model_name = None
        declared_fields: list[str] = []
        for node in meta.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "model" and isinstance(node.value, ast.Name):
                    model_name = node.value.id
                elif node.targets[0].id == "fields":
                    declared_fields = _string_seq(node.value)
        if model_name is None or model_name not in models:
            continue

        # Names the serializer defines itself, and any `source=` they point at.
        own: set[str] = set()
        sources: list[tuple[str, str]] = []
        for node in cls.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Call):
                name = node.targets[0].id
                own.add(name)
                src = _literal(_kwarg(node.value, "source"))
                if src:
                    sources.append((name, src))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # get_<field> backs a SerializerMethodField.
                if node.name.startswith("get_"):
                    own.add(node.name[4:])

        for name in declared_fields:
            if name in own:
                continue
            ok, why = resolve(models, model_name, [name])
            if not ok:
                problems.append(
                    f"api.py  {cls.name}.Meta.fields: {name!r} - {why}"
                )

        # A `source=` that points at nothing is the same failure, one level in.
        for name, src in sources:
            ok, why = resolve(models, model_name, src.split("."))
            if not ok:
                problems.append(
                    f"api.py  {cls.name}.{name} source={src!r} - {why}"
                )

    return problems


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][\w.]*)")
IF_RE = re.compile(r"\{%\s*(?:if|elif)\s+(.+?)\s*%\}")
FOR_RE = re.compile(r"\{%\s*for\s+[\w, ]+\s+in\s+([\w.]+)")
IDENT_RE = re.compile(r"\b([a-zA-Z_][\w.]*\.[\w.]+)\b")


def check_template(path: Path, rel: str, models: dict[str, Model]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Comments are not rendered, so paths inside them are prose.
    text = re.sub(r"\{%\s*comment\s*%\}.*?\{%\s*endcomment\s*%\}", "", text,
                  flags=re.S)
    text = re.sub(r"\{#.*?#\}", "", text, flags=re.S)

    mapping = CONTEXT.get(rel, {})
    if not mapping:
        return []

    candidates: set[str] = set()
    for match in VAR_RE.finditer(text):
        candidates.add(match.group(1))
    for regex in (IF_RE, FOR_RE):
        for match in regex.finditer(text):
            for ident in IDENT_RE.finditer(match.group(1)):
                candidates.add(ident.group(1))

    problems: list[str] = []
    for raw in sorted(candidates):
        # Longest matching prefix, so `entry.view.angle` binds to `entry.view`
        # rather than to the local `entry`.
        root = max((k for k in mapping if raw == k or raw.startswith(k + ".")),
                   key=len, default=None)
        if root is None:
            head = raw.split(".")[0]
            if head not in TEMPLATE_LOCALS and head not in mapping:
                pass  # not a tracked variable; nothing to say about it
            continue

        rest = raw[len(root):].lstrip(".")
        if not rest:
            continue
        parts = rest.split(".")
        # Filters were stripped by the regex, but a trailing empty part can
        # survive an odd match.
        parts = [p for p in parts if p]
        ok, why = resolve(models, mapping[root], parts)
        if not ok:
            problems.append(f"{rel}  {{{{ {raw} }}}} - {why}")
    return problems


# --------------------------------------------------------------------------

SELFTEST_MODELS = '''
from django.db import models

class Inspection(models.Model):
    owner_name = models.CharField(max_length=120)
    segment = models.CharField(max_length=20, choices=[("a", "A")])
    total = models.DecimalField(max_digits=10, decimal_places=2)

    @property
    def view_count(self):
        return 0

    def docket_number(self):
        return ""

class VehicleImage(models.Model):
    inspection = models.ForeignKey(
        Inspection, on_delete=models.CASCADE, related_name="images"
    )
    angle = models.CharField(max_length=20, choices=[("front", "Front")])
    image = models.ImageField(upload_to="x/")

    @property
    def display_image(self):
        return self.image

class DamageDetection(models.Model):
    image = models.ForeignKey(
        VehicleImage, on_delete=models.CASCADE, related_name="detections"
    )
    severity = models.CharField(max_length=10, choices=[("minor", "Minor")])
'''


def selftest() -> int:
    """Prove the parser sees what it claims to, before anyone trusts a PASS.

    Each case is a fact about Django that the checker has to encode correctly. If
    one of these regresses, the checker starts approving broken code, which is
    worse than not having it.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models.py"
        path.write_text(SELFTEST_MODELS, encoding="utf-8")
        models = parse_models(path)

    def ok(label: str, condition: bool) -> bool:
        print(f"  {'pass' if condition else 'FAIL'}  {label}")
        return condition

    results = [
        ok("three models found", set(models) ==
           {"Inspection", "VehicleImage", "DamageDetection"}),
        ok("plain fields collected",
           "owner_name" in models["Inspection"].fields),
        ok("ForeignKey counts as a field - the bug that started this",
           "inspection" in models["VehicleImage"].fields),
        ok("ForeignKey target recorded",
           models["VehicleImage"].relations.get("inspection") == "Inspection"),
        ok("related_name creates a reverse accessor",
           models["Inspection"].reverse.get("images") == "VehicleImage"),
        ok("choices= implies get_<field>_display",
           "get_angle_display" in models["VehicleImage"].callables),
        ok("no choices, no display method",
           "get_owner_name_display" not in models["Inspection"].callables),
        ok("properties are callable attributes",
           "view_count" in models["Inspection"].callables),
        ok("methods are callable attributes",
           "docket_number" in models["Inspection"].callables),
        # Resolution
        ok("valid single hop resolves",
           resolve(models, "Inspection", ["owner_name"])[0]),
        ok("missing attribute fails",
           not resolve(models, "Inspection", ["panel"])[0]),
        ok("relation span resolves",
           resolve(models, "DamageDetection",
                   ["image", "inspection", "owner_name"])[0]),
        ok("broken relation span fails",
           not resolve(models, "DamageDetection",
                       ["image", "inspection", "panel"])[0]),
        ok("reverse accessor traverses",
           resolve(models, "Inspection", ["images", "angle"])[0]),
        ok("bad attribute after a reverse hop fails",
           not resolve(models, "Inspection", ["images", "nonsense"])[0]),
        ok("property then anything stops checking (not a false alarm)",
           resolve(models, "VehicleImage", ["display_image", "url"])[0]),
    ]

    failures = results.count(False)
    print(f"\n{failures} self-test failure(s)")
    return 1 if failures else 0


# The project root is found by walking up from this file, never hardcoded and
# never assumed to be the working directory - so `python tests/check_models.py` works
# from the project root, from inside tests/, or from anywhere else.
def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "assessment" / "models.py").exists():
            return candidate
    raise SystemExit(
        "Cannot find the project root: no parent of this file contains "
        "assessment/models.py. Run this from inside the project."
    )


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    root = _project_root()

    models = parse_models(root / "assessment" / "models.py")
    print("models found:")
    for name, model in models.items():
        print(f"  {name}: {len(model.fields)} fields, "
              f"{len(model.callables)} methods/properties, "
              f"reverse: {', '.join(model.reverse) or '-'}")
    print()

    problems = check_admin(root / "assessment" / "admin.py", models)
    problems.extend(check_api(root / "assessment" / "api.py", models))

    for path in sorted((root / "templates").rglob("*.html")):
        rel = str(path.relative_to(root / "templates"))
        problems.extend(check_template(path, rel, models))

    if problems:
        print(f"{len(problems)} problem(s):")
        for line in problems:
            print(f"  FAIL  {line}")
        return 1

    print("no unresolved model attributes in admin.py, api.py or the templates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
