"""Django settings.

Two choices here that you should be ready to defend, because both differ from
what the report literally says:

**SQLite, not MySQL.** The report's Data Storage Layer names MySQL. Under the
Django ORM the model code, queries and migrations are byte-for-byte identical
either way - swapping to MySQL is this DATABASES block and a `pip install
mysqlclient`, nothing more. SQLite ships inside Python, so a demo machine needs
no database server running, no credentials and no network. For a single-user
demo that is strictly fewer things that can fail in front of an examiner. The
honest framing: this is a deployment choice, not an architectural one.

**No Redis cache.** The report says Redis "can be used" for caching. It is left
out. Nothing here is read often enough to cache: each inspection is computed
once and then read from SQLite. The expensive step is model inference, and its
result is already persisted in the database, which *is* the cache.

Secrets and DEBUG come from the environment with development defaults, so the
same file works on your machine and on a host, and there is no real secret key
committed to git.
"""

from __future__ import annotations

import os
from pathlib import Path

# BASE_DIR is the project root - the folder holding manage.py. Everything else
# is derived from it, so the project can be moved or cloned anywhere.
BASE_DIR = Path(__file__).resolve().parent.parent


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean from the environment. "1", "true", "yes", "on" are true."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

# Development default is insecure ON PURPOSE and is obviously named so, so that
# an unset SECRET_KEY in production is a visible mistake rather than a silent
# one. Set DJANGO_SECRET_KEY before deploying anywhere real.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "dev-only-insecure-key-change-before-deploying",
)

DEBUG = env_flag("DJANGO_DEBUG", default=True)

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    if h.strip()
]

# Django 4+ checks the Origin header of unsafe requests against this list. The
# dev server is reached over http, so both schemes are listed explicitly.
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third party
    "rest_framework",
    # Local
    "assessment",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "vdac.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Project-level templates dir, plus each app's own templates/ folder.
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                # Puts the "no trained weights installed" warning on every page.
                "assessment.context_processors.detector_banner",
            ],
        },
    },
]

WSGI_APPLICATION = "vdac.wsgi.application"
ASGI_APPLICATION = "vdac.asgi.application"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {
            # Write-ahead logging lets a reader and a writer work at the same
            # time instead of the reader getting "database is locked". Model
            # inference holds a request open for a second or two, so this is
            # worth having even on a demo machine.
            "init_command": "PRAGMA journal_mode=WAL;",
        },
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ---------------------------------------------------------------------------
# Passwords, i18n
# ---------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-in"
# Timestamps are stored in UTC and rendered in this zone. Kolkata, because the
# cost tables are in rupees and the demo is in India.
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True


# ---------------------------------------------------------------------------
# Static and media
# ---------------------------------------------------------------------------
# static/ = files we wrote (CSS). media/ = files users uploaded plus everything
# the pipeline generated from them. Keeping them apart matters: media is
# untrusted input and must never be executed or collected into staticfiles.

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}

# A phone photograph is routinely 4-8 MB, well over Django's 2.5 MB default for
# buffering an upload in memory. Raising the threshold to 10 MB keeps ordinary
# uploads off the temp disk; the hard limit users actually hit is
# forms.MAX_UPLOAD_BYTES, which produces a readable error message.
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 15 * 1024 * 1024


# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------
# The report's Backend API Layer. Read access is open so the API can be
# demonstrated from a browser; creating an inspection requires a login, because
# it runs the model and writes files.

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticatedOrReadOnly",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# core.detector logs at WARNING when it falls back to the stub detector. That
# message must reach the console, because a demo silently showing fabricated
# detections is the worst possible failure here.

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "{levelname} {name}: {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
    },
    "root": {"handlers": ["console"], "level": "WARNING"},
    "loggers": {
        "core": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "assessment": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}


# ---------------------------------------------------------------------------
# Project-specific
# ---------------------------------------------------------------------------
# Where core.detector looks for the trained model. Missing = stub detector, and
# every page says so. Copy best.pt out of Colab to this path.
YOLO_WEIGHTS_PATH = Path(
    os.environ.get("YOLO_WEIGHTS_PATH", BASE_DIR / "weights" / "best.pt")
)
