import cv2
import numpy as np
from collections import Counter
from math import hypot


def filter_helper(image, type):
    if type == "edge_detection":
        img_canny = cv2.Canny(image, 100, 100)
        return img_canny
    elif type == "blur":
        img_blur = cv2.GaussianBlur(image, (15, 15), 0)
        return img_blur
    else:
        return apply_effects(image, type)


def resize_frame(img, scale=30):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    resize_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resize_image


def remove_background(img_path):
    bg_color = BackgroundColorDetector(img_path)
    bgcolor = bg_color.detect()
    f_img = cv2.imread(img_path)
    f_img = cv2.cvtColor(f_img, cv2.COLOR_BGR2BGRA)
    f_img[np.all(f_img == bgcolor + [255], axis=2)] = [0, 0, 0, 0]
    filter_img = cv2.cvtColor(f_img, cv2.COLOR_BGRA2BGR)
    return filter_img


def apply_effects(img, effect):
    if effect == "lens":
        filter_image_path = "data/filter_glass.jpg"

    elif effect == "cat":
        filter_image_path = "data/snowcat.jpg"

    elif effect == "dog":
        filter_image_path = "data/dog1.jpg"

    path = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")

    filter_img = remove_background(filter_image_path)

    original_filter_img_h, original_filter_img_w, filter_img_channels = filter_img.shape
    img_h, img_w, img_channels = img.shape

    # convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filter_img_gray = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)

    ret, original_mask = cv2.threshold(filter_img_gray, 10, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)

    # find faces in image using classifier
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        # filter_img size in relation to face by scaling
        filter_img_width = int(1 * face_w)
        filter_img_height = int(
            filter_img_width * original_filter_img_h / original_filter_img_w
        )

        # setting location of coordinates of filter_img
        filter_img_x1 = face_x2 - int(face_w / 2) - int(filter_img_width / 2)
        filter_img_x2 = filter_img_x1 + filter_img_width
        filter_img_y1 = face_y1
        filter_img_y2 = filter_img_y1 + filter_img_height

        # check to see if out of frame
        if filter_img_x1 < 0:
            filter_img_x1 = 0
        if filter_img_y1 < 0:
            filter_img_y1 = 0
        if filter_img_x2 > img_w:
            filter_img_x2 = img_w
        if filter_img_y2 > img_h:
            filter_img_y2 = img_h

        # Account for any out of frame changes
        filter_img_width = filter_img_x2 - filter_img_x1
        filter_img_height = filter_img_y2 - filter_img_y1

        # resize filter_img to fit on face
        filter_img = cv2.resize(
            filter_img,
            (filter_img_width, filter_img_height),
            interpolation=cv2.INTER_AREA,
        )
        mask = cv2.resize(
            original_mask,
            (filter_img_width, filter_img_height),
            interpolation=cv2.INTER_AREA,
        )
        mask_inv = cv2.resize(
            original_mask_inv,
            (filter_img_width, filter_img_height),
            interpolation=cv2.INTER_AREA,
        )

        # take ROI for filter_img from background that is equal to size of filter_img image
        roi = img[filter_img_y1:filter_img_y2, filter_img_x1:filter_img_x2]

        # original image in background (bg) where filter_img is not present
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
        roi_fg = cv2.bitwise_and(filter_img, filter_img, mask=mask_inv)
        dst = cv2.add(roi_bg, roi_fg)

        # put back in original image
        img[filter_img_y1:filter_img_y2, filter_img_x1:filter_img_x2] = dst
        return img


class BackgroundColorDetector:
    """
    This is to detect the background color of a image
    """

    def __init__(self, imageLoc):
        self.img = cv2.imread(imageLoc, 1)
        self.manual_count = {}
        self.w, self.h, self.channels = self.img.shape
        self.total_pixels = self.w * self.h

    def count(self):
        for y in range(0, self.h):
            for x in range(0, self.w):
                RGB = (self.img[x, y, 2], self.img[x, y, 1], self.img[x, y, 0])
                if RGB in self.manual_count:
                    self.manual_count[RGB] += 1
                else:
                    self.manual_count[RGB] = 1

    def average_colour(self):
        red = 0
        green = 0
        blue = 0
        sample = 10
        for top in range(0, sample):
            red += self.number_counter[top][0][0]
            green += self.number_counter[top][0][1]
            blue += self.number_counter[top][0][2]

        average_red = red / sample
        average_green = green / sample
        average_blue = blue / sample
        return [average_red, average_green, average_blue]

    def twenty_most_common(self):
        self.count()
        self.number_counter = Counter(self.manual_count).most_common(20)

    def detect(self):
        self.twenty_most_common()
        self.percentage_of_first = float(self.number_counter[0][1]) / self.total_pixels
        if self.percentage_of_first > 0.5:
            return list(self.number_counter[0][0])
        else:
            return self.average_colour()


def eye_filter(filter, image, shape):
    # eyes filter
    eye_left = (shape.part(37).x, shape.part(37).y)
    eye_right = (shape.part(46).x, shape.part(46).y)
    eye_mid = (shape.part(28).x, shape.part(28).y)

    eye_width = int(hypot(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]) * 2)
    eye_height = int(eye_width * 0.7)

    top_left = (int(eye_mid[0] - eye_width / 2), int(eye_mid[1] - eye_height / 2))

    eye = cv2.resize(filter, (eye_width, eye_height))

    eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, eye_mask = cv2.threshold(eye_gray, 25, 255, cv2.THRESH_BINARY_INV)

    eye_area = image[
        top_left[1] : top_left[1] + eye_height, top_left[0] : top_left[0] + eye_width
    ]

    no_eye_area = cv2.bitwise_and(eye_area, eye_area, mask=eye_mask)

    image[
        top_left[1] : top_left[1] + eye_height, top_left[0] : top_left[0] + eye_width
    ] = cv2.add(no_eye_area, eye)

    return image


def nose_filter(filter, image, shape):
    center_nose = (shape.part(30).x, shape.part(30).y)
    left_nose = shape.part(32).x, shape.part(32).y
    right_nose = shape.part(36).x, shape.part(36).y

    nose_width = int(hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]))
    nose_height = int(nose_width * 0.77)
    # New nose position
    top_left = (
        int(center_nose[0] - nose_width / 2),
        int(center_nose[1] - nose_height / 2),
    )

    nose_pig = cv2.resize(filter, (nose_width, nose_height))
    nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
    _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

    nose_area = image[
        top_left[1] : top_left[1] + nose_height, top_left[0] : top_left[0] + nose_width
    ]

    nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
    final_nose = cv2.add(nose_area_no_nose, nose_pig)
    image[
        top_left[1] : top_left[1] + nose_height, top_left[0] : top_left[0] + nose_width
    ] = final_nose
    return image


def full_face_filter(filter, image, shape):
    left = (shape.part(2).x, shape.part(2).y)
    right = (shape.part(16).x, shape.part(16).y)
    top = (shape.part(28).x, shape.part(28).y)
    bottom = (shape.part(34).x, shape.part(34).y)

    height = int(hypot(bottom[0] - top[0], bottom[1] - top[1]))
    face_height = int(3 * height)
    face_width = int(hypot(left[0] - right[0], left[1] - right[1]))

    cv2.rectangle(
        image, (right[0], bottom[1] - face_height), (left[0], bottom[1]), (0, 255, 0), 2
    )

    new_filter = cv2.resize(filter, (face_width, face_height))
    center = (shape.part(30).x, shape.part(30).y)
    top_left = int(center[0] - face_width / 2), int(bottom[1] - face_height)
    cv2.circle(image, top_left, 3, (255, 0, 0), -1)
    filter_gray = cv2.cvtColor(new_filter, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(filter_gray, 25, 255, cv2.THRESH_BINARY_INV)

    area = image[
        top_left[1] : top_left[1] + face_height, top_left[0] : top_left[0] + face_width
    ]

    no_filter_area = cv2.bitwise_and(area, area, mask=mask)
    final_img = cv2.add(no_filter_area, new_filter)
    image[
        top_left[1] : top_left[1] + face_height, top_left[0] : top_left[0] + face_width
    ] = final_img
    return image
