"""Page URLs for the assessment app."""

from django.urls import path

from . import views

app_name = "assessment"

urlpatterns = [
    path("", views.upload, name="upload"),
    path("inspections/", views.history, name="history"),
    path("inspections/<int:pk>/", views.result, name="result"),
    path("inspections/<int:pk>/report.pdf", views.download_report,
         name="download_report"),
]
