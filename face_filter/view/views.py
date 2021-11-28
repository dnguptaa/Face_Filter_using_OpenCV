from . import v1
from rest_framework import generics


class LiveStream(generics.GenericAPIView):
    def dispatch(self, request, *args, **kwargs):

        view = v1.LiveStream.as_view()
        return view(request, *args, **kwargs)
