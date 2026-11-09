"""Page URLs for the assessment app."""

from django.urls import path

from . import views

app_name = "assessment"

urlpatterns = [
    # The landing page takes "/" and the form moves to "/inspect/". The order
    # here is not significant - none of these patterns can shadow another - but
    # the reason for the arrangement is: the first thing an examiner sees should
    # be the measured accuracy and its caveats, not a form. Every link to the
    # form is written {% url 'assessment:upload' %}, so the name is what the
    # templates depend on and the path is free to move.
    path("", views.landing, name="landing"),
    path("inspect/", views.upload, name="upload"),
    path("inspections/", views.history, name="history"),
    path("inspections/<int:pk>/", views.result, name="result"),
    path("inspections/<int:pk>/report.pdf", views.download_report,
         name="download_report"),
]
