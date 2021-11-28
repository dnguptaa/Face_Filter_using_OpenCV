import cv2
import dlib
import imutils
from django.shortcuts import render
from rest_framework.views import APIView
from django.utils.translation import gettext as _
from face_filter.helpers import eye_filter, full_face_filter, nose_filter, filter_helper


class LiveStream(APIView):
    """
    API for user to change filter in live video
    """

    def get(self, request):
        return render(request, "index.html")

    def post(self, request):
        eyes_image = None
        nose_image = None
        face_image = None

        filter_type = request.POST.get("filter_type")

        if filter_type == "face":
            face_image = cv2.imread("data/dog2.png")
        if filter_type in ("eye", "eye_and_nose"):
            eyes_image = cv2.imread("data/google.png")
        if filter_type in ("nose", "eye_and_nose"):
            nose_image = cv2.imread("data/pig_nose.png")
        # for webcam
        cap = cv2.VideoCapture(0)

        # set width height brightness for the webcam window
        cap.set(3, 1920)  # id 3 is width
        cap.set(4, 1080)  # id 4 is height
        cap.set(10, 100)  # id 10 is brightness

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("data/data")

        while cap.isOpened():
            try:
                success, image = cap.read()
                if not success:
                    return render(
                        request, "index.html", {"errors": "Failure in reading video"}
                    )

                # apply filter
                image = imutils.resize(image, width=500)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale image
                rects = detector(gray, 1)

                # loop over the face detections
                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    if face_image is not None:
                        image = full_face_filter(face_image, image, shape)
                    if nose_image is not None:
                        image = nose_filter(nose_image, image, shape)
                    if eyes_image is not None:
                        image = eye_filter(eyes_image, image, shape)
                    if filter_type not in ("eye", "nose", "eye_and_nose", "face"):
                        image = filter_helper(image, filter_type)

                cv2.imshow("Video", image)
                if cv2.waitKey(10) == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return render(request, "index.html")
            except Exception as e:
                return render(request, "index.html", {"errors": e})
