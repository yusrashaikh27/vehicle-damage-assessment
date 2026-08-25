"""API routes, kept in their own module so /api/ and the page URLs stay separate."""

from django.urls import path

from . import api

app_name = "api"

urlpatterns = [
    path("assess/", api.AssessAPI.as_view(), name="assess"),
    path("inspections/", api.InspectionListAPI.as_view(), name="inspection-list"),
    path("inspections/<int:pk>/", api.InspectionDetailAPI.as_view(),
         name="inspection-detail"),
]
