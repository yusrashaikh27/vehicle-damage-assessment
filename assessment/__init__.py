"""The Django application: web pages, API, database models.

Everything in here is Django-specific glue. The assessment itself - detection,
severity, cost, PDF - lives in the framework-free `core` package, and the only
module allowed to bridge the two is `services.py`.
"""
