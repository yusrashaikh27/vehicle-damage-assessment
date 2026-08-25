"""Root URL configuration.

    /            -> the assessment app (upload, results, history)
    /api/        -> DRF endpoints, the report's Backend API Layer
    /admin/      -> Django admin, the report's Admin Panel layer
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("assessment.api_urls")),
    path("", include("assessment.urls")),
]

# Uploaded and generated images live under MEDIA_ROOT, which Django's dev server
# does not serve on its own. This helper wires it up for development only - in
# production the web server in front of Django serves /media/ directly, which is
# why it is guarded by DEBUG rather than added unconditionally.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
