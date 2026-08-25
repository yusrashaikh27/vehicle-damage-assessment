#!/usr/bin/env python
"""Django's command-line entry point.

Run everything through this file from the project root:

    ./venv/bin/python manage.py migrate
    ./venv/bin/python manage.py runserver
"""

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vdac.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:  # pragma: no cover - only hit on a broken env
        raise ImportError(
            "Django could not be imported. Are you using the project venv?\n"
            "  ./venv/bin/python manage.py ..."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
