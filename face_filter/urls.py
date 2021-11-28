from django.urls import path
from face_filter.view.views import LiveStream


urlpatterns = [
    path("live_stream", LiveStream.as_view(), name="live_stream"),
]
