# Running the site

Everything below assumes you are in the project folder:

    cd "/Users/yusra/ALL PROJECTS OF YUSRA/vehicle damage assessment"

## First time only

    source venv/bin/activate
    python manage.py migrate
    python manage.py createsuperuser

`migrate` alone is enough. `assessment/migrations/0001_initial.py` already exists
and is committed, so there is nothing to generate — it describes the four-model
shape (Inspection, VehicleImage, DamageDetection, CostLine) and `migrate` executes
it against SQLite.

You only need `makemigrations assessment` **after changing a model field**. It
writes a new migration file describing the change; `migrate` then applies it. The
two steps are separate on purpose: the written file is committed, which is what
lets the same schema be rebuilt on any machine without anyone re-typing it.

`createsuperuser` is the login for `/admin/`. Any username and password will do;
the password prompt stays blank as you type, which is normal. If `db.sqlite3`
already has a superuser from an earlier session, you can skip this.

## Every time after that

    source venv/bin/activate
    python manage.py runserver

Then open:

**http://127.0.0.1:8000/**

Leave the terminal running — that *is* the web server. `Ctrl+C` stops it. Every
request shows up as a log line in that window, which is the first place to look
when a page misbehaves.

## The four pages

| URL | What it is |
| --- | --- |
| http://127.0.0.1:8000/ | The front page: owner details, car details, and one upload slot per angle. |
| http://127.0.0.1:8000/inspections/ | Every inspection assessed so far, searchable by plate, owner, make or model. |
| http://127.0.0.1:8000/admin/ | Django admin — the report's Admin Panel layer. |
| http://127.0.0.1:8000/api/inspections/ | The REST API, browsable in the browser. |

A result page lives at `/inspections/<id>/` and its PDF at
`/inspections/<id>/report.pdf`; both are linked from the pages above, so you
should never need to type them.

## Use `127.0.0.1`, not `localhost`

Both are in `ALLOWED_HOSTS`, so both work. Prefer `127.0.0.1` anyway: on some
macOS setups `localhost` resolves to IPv6 `::1` first and the dev server binds
IPv4 by default, which shows up as a connection refused that looks like the
server failed to start when it is running perfectly.

## Before the demo

    python tests/run_all.py

Five checks, about a second, no GPU or database needed. See `tests/README.md`
for what each one proves. Running this first means a broken template or a
renamed field surfaces in your terminal rather than on a projector.

## The banner about the stub detector

Until trained weights exist at `weights/best.pt`, every page shows a banner
saying the detections are placeholders. That is deliberate and should be left
alone: the stub detector returns fixed, invented regions, and a demo that
displayed those without saying so would be presenting fabricated results as real
measurements. Once training finishes, drop `best.pt` into `weights/` and restart
the server — the banner disappears on its own, and existing records can be
re-assessed from the admin's re-run action.

Weights can also live elsewhere:

    YOLO_WEIGHTS_PATH=/path/to/best.pt python manage.py runserver

## If something goes wrong

**`command not found: python`** — the venv is not activated. `source
venv/bin/activate` first; your prompt should then start with `(venv)`.

**`ModuleNotFoundError: No module named 'django'`** — same cause. If it persists
with the venv active, the venv is broken; rebuild it:

    python3.12 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

Python 3.12 specifically, not 3.14 — torch and ultralytics do not publish
reliable wheels for 3.14, and building them from source is not a thing to
discover the week before a submission.

**`That port is already in use`** — a previous `runserver` is still alive. Either
`python manage.py runserver 8001` and use port 8001, or find and stop the old
one:

    lsof -ti:8000 | xargs kill

**Uploads rejected as too small** — the minimum is 320px on the shortest side.
That is not fussiness: severity is derived from the *area* a damage mask covers,
and on a thumbnail a real dent is a handful of pixels, so the quotation built
from it would be meaningless. Refusing the image is more honest than pricing a
guess.

**A page 500s after you change a model** — you almost certainly changed a field
without migrating. `makemigrations assessment` then `migrate`. If the schema has
drifted badly and there is nothing in the database worth keeping, deleting
`db.sqlite3` and re-running `migrate` is a legitimate reset during development.
