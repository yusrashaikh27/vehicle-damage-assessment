"""ASGI entry point. Present for completeness; the project is synchronous."""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vdac.settings")

application = get_asgi_application()
