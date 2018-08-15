from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import cv2
import tqdm
import numpy as np

"""
LBPH-based Face recognition module
"""


class FaceRecognizer():

    """
    Face recognition class ikmplementing the LBPH algorithm
    """

    def __init__(self):
        """
        Instantiate a FaceRecognizer object
        """

        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.logger = logging.getLogger(__name__)

    def train(self, images, labels):
        """
        Train the recognizer on the training set

        :param images: the images to train on
        :type images: numpy.ndarray shape: (num_images, image_height, image_width)
        :param labels: the labels/subjects the corresponding faces belong to
        :type labels: numpy.ndarray shape: (num_images,)
        """

        self.model.train(images, labels)

    def predict(self, images):
        """
        Predicts the labels of the given images

        :param images: the images to test on
        :type images: numpy.ndarray shape: (num_images, image_height, image_width)
        :returns: the predicted labels
        :rtype: array
        """

        predictions = []
        for i in tqdm.trange(0, len(images)):
            prediction = self.model.predict(images[i])
            predictions.append(prediction)
            i += 1

        return predictions

    def evaluate(self, predictions, ground_truths):

        assert(len(predictions) == len(ground_truths))

        true_positive = np.count_nonzero(
            np.equal(ground_truths, np.array(predictions)[:, 0]))

        precision_perc = true_positive/len(predictions)

        self.logger.info(
            "Precision@1: {0}/{1}={2:.3%}".format(true_positive, len(predictions), precision_perc))

    def save(self, name):

        self.model.write(name)

    def load(self, name):

        self.model.read(name)
