from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import cv2
import dlib
import numpy as np
import pkg_resources

from face_trigger.utils.common import clamp_rectangle, shape_to_np

FACE_LANDMARK_INDICES = {
    "reye_right": 0,
    "reye_left": 1,
    "leye_right": 3,
    "leye_left": 2,
    "nose": 4
}


class FaceDetector():

    """
    HOG-based frontal face detector class.
    """

    def __init__(self, face_area_threshold=0.25):
        """
        Initialise a 'FaceDetector' object

        :param face_area_threshold: minimum area the face must cover w.r.t the frame
        :type face_area_threshold: float [0,1]
        """

        self.detector = dlib.get_frontal_face_detector()
        self.face_area_threshold = face_area_threshold
        self.logger = logging.getLogger(__name__)

    def detect(self, gray_frame):
        """
        Detect faces in the frame.
        If multiple faces are detected, only the largest face is taken into consideration.

        :param gray_frame: grayscale image that might include a face
        :type gray_frame: numpy.ndarray
        :returns: bounding box coordinates signifying the location of the face
        :rtype: dlib.rectangle
        """

        bounding_box = None

        faces = self.detector(gray_frame, 0)

        if len(faces) > 0:

            areas = [0 for face in faces]
            (x_max, y_max) = gray_frame.shape
            frame_area = x_max*y_max

            for idx, face in enumerate(faces):
                x1, y1, x2, y2 = clamp_rectangle(x1=face.left(), y1=face.top(
                ), x2=face.right(), y2=face.bottom(), x2_max=x_max-1, y2_max=y_max-1)
                areas[idx] = (x2-x1)*(y2-y1)

            largest_face_idx = np.argmax(np.array(areas))

            if areas[largest_face_idx]/frame_area > self.face_area_threshold:
                bounding_box = faces[largest_face_idx]

        return bounding_box

    def detect_unbounded(self, gray_frame):
        """
        Detect all faces in the frame

        :param gray_frame: grayscale image that might faces
        :type gray_frame: numpy.ndarray
        :returns: bounding box coordinates signifying the location of the faces
        :rtype: list of dlib.rectangle
        """

        faces = self.detector(gray_frame, 0)

        return faces


class FaceAlign():

    """
    Align a face by performing affine transformations.
    Inspired from 'Face Alignment with OpenCV and Python'
    https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    """

    def __init__(self, left_eye_offset=(0.35, 0.35), final_width=256, final_height=None):
        """
        Instantiate a 'FaceAlign' object

        :param left_eye_offset: the amount by which to anchor the left eye center
        :type left_eye_offset: 2-tuple
        :param final_width: the width in pixels of the aligned image
        :type final_width: int
        :param final_height: the height in pixels of the aligned image
        :type final_height: int
        """

        self.left_eye_offset = left_eye_offset
        self.final_width = final_width

        if final_height is None:
            self.final_height = final_width
        else:
            self.final_height = final_height

        self.logger = logging.getLogger(__name__)

    def align(self, img, landmarks):
        """
        Align the given image according to given face landmarks -
        1. Eyes are aligned on a horizontal axis
        2. Face is scaled to keep eye centers on the same offset location w.r.t the face
        3. Face is centered on the center of mass of the eyes.

        :param img: rgb or gray-scale image/frame
        :type img: numpy.ndarray
        :landmarks: the x,y-coordinates of the facial landmarks
        :type landmarks: dlib shape
        :returns: aligned image of final dimensions
        :rtype: numpy.ndarray
        """

        landmarks = shape_to_np(landmarks)

        # eye centers
        leye = tuple(np.add(landmarks[FACE_LANDMARK_INDICES["leye_right"]],
                            landmarks[FACE_LANDMARK_INDICES["leye_left"]])/2)

        reye = tuple(np.add(landmarks[FACE_LANDMARK_INDICES["reye_right"]],
                            landmarks[FACE_LANDMARK_INDICES["reye_left"]])/2)

        # angle between eye centers
        dY = reye[1] - leye[1]
        dX = reye[0] - leye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # right eye coordinates as a function of the left eye
        right_eye_offset_x = 1.0 - self.left_eye_offset[0]

        # scale
        current_dist = np.sqrt(dY**2+dX**2)
        desired_dist = (right_eye_offset_x -
                        self.left_eye_offset[0])*self.final_width
        scale = desired_dist/current_dist

        # median of eye centers
        eye_median = ((reye[0]+leye[0])//2, (reye[1] + leye[1])//2)

        # transformation matrix
        transform_matrix = cv2.getRotationMatrix2D(eye_median, angle, scale)

        # update the matrix's transaltion compopnents
        tX = self.final_width * 0.5
        tY = self.final_height * self.left_eye_offset[1]
        transform_matrix[0, 2] += (tX - eye_median[0])
        transform_matrix[1, 2] += (tY - eye_median[1])

        # align the face
        aligned_face = cv2.warpAffine(
            img, transform_matrix, (self.final_width, self.final_height), flags=cv2.INTER_CUBIC)

        return aligned_face


class LandmarkDetector:

    """
        A landmark detector that annotates face bounding boxes with 5 landmarks
    """

    def __init__(self, predictor_path=None):
        """
        Instantiates the 'LandmarkDetector' object

        :param predictor_path: path to trained face predictor model
        :type predictor_path: str
        """

        if predictor_path is None:
            predictor_path = pkg_resources.resource_filename(
                "face_trigger", "pre_trained/shape_predictor_5_face_landmarks.dat")

        self.predictor = dlib.shape_predictor(predictor_path)
        self.logger = logging.getLogger(__name__)

    def predict(self, bounding_box, grayImg):
        """
        Provides an array of tuples for facial landmarks, predicted within a bounding box

        :param bounding_box: bounding box coordinates in dlib format
        :type bounding_box: dlib.rectangle
        :param grayImg: grayscale image
        :type grayImg: numpy.ndarray
        :returns: 5-tuple
        :rtype: dlib.full_object_detection
        """

        shape = self.predictor(grayImg, bounding_box)

        return shape
