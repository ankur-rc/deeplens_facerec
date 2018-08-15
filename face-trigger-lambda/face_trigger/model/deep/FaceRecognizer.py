from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import tqdm
import numpy as np
import dlib
import cv2
import sklearn
import time
import pkg_resources

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from face_trigger.process.post_process import FaceDetector

"""
Deep Learning based face recognition module.
"""


class FaceRecognizer():

    """
    Trains a face recognition model based on deep metric learning.
    https://arxiv.org/abs/1503.03832
    """

    def __init__(self, dnn_model_path=None, classifier_model_path=None, label_map_path=None):
        """
        Instantiate a FaceRecognizer object

        :param dnn_model_path: path to the trainined dnn featyure extractor
        :type dnn_model_path: str
        :param classifier_model_path: path to the trained sklearn classifier
        :type classifier_model_path: str
        :param label_map_path: path to the pickled label map
        :type label_map_path: str
        """
        self.logger = logging.getLogger(__name__)

        if dnn_model_path is None:
            self.logger.debug("No DNN model path specified, using default.")
            dnn_model_path = pkg_resources.resource_filename(
                "face_trigger", "pre_trained/dlib_face_recognition_resnet_model_v1.dat")

        self.dnn_model = dlib.face_recognition_model_v1(dnn_model_path)

        if classifier_model_path is not None:
            self.classifier = self.load(classifier_model_path)
        else:
            raise Exception("No classifier model path given!")

        if label_map_path is not None:
            self.label_map = self.load_label_mapping(
                label_map_path=label_map_path)
        else:
            raise Exception("No label mapping provided!")

    def train(self):
        """
        Train the recognizer on the training set. Not required when working with pre-trained models.
        TODO: Implement the training mechanism
        """
        raise NotImplementedError()

    def embed(self, images=None, landmarks=None):
        """
        Generates embeddings for the given images. THe images should be a result of the face detector phase, 
        i.e these images should contain a face detected by the face detector.

        :param images: the images to get embeddings of
        :type images: list of numpy.nadarray: (num_images, image_height, image_width)
        :param landmarks: the facial landmarks of the images
        :type landmarks: list, shape: (num_images, 5, 2)
        :returns: the face embeddings
        :rtype: list

        **Note:** The images contain the entire frame, and not just the cropped out face. Alignmnet is taken care of when we generate the embeddings.

        """

        assert len(images) == len(landmarks)

        embeddings = []

        # convert from gray to rgb
        # images = np.array(images)
        # images = images.reshape(images.shape + (1,))
        # images = np.repeat(images, 3, axis=3)

        images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in images]

        start_time = time.time()
        # self.logger.debug("Start timestamp: {}".format(start_time))
        embeddings = [self.dnn_model.compute_face_descriptor(
            image, landmarks[i]) for i, image in enumerate(images)]

        end_time = time.time()  # batch:100 s: ~1.5 sec; p: n/a
        # self.logger.debug("End time: {}. Runtime: {}".format(
        # end_time, (end_time-start_time)))

        return embeddings

    def save(self, classifier_model_path):
        """
        Save the trained classifier.
        Call only after fitting the embeddings, otherwise will throw an exception.

        :param classifier_model_path: path along with name specifiying where to save the model. Extension should be .pkl for brevity.
        :type classifier_model_path: string
        """
        joblib.dump(self.classifier, classifier_model_path)

    def load(self, classifier_model_path):
        """
        Load the saved classifier model.

        :param classifier_model_path: path to the trained classifier model
        :type classifier_model_path: string
        """

        if not os.path.exists(classifier_model_path):
            raise Exception("Path to trained classifier model does not exist!")

        classifier_model_path = os.path.realpath(classifier_model_path)

        return joblib.load(classifier_model_path)

    def infer(self, embeddings, threshold=0.20, unknown_index=-1):
        """
        Infer and return a predicted face identity.

        :param embeddings: 128D face embeddings
        :type embeddings: list
        :param threshold: probability threshold to accept a prediction result
        :type threshold: float
        :param unknown_index: a integer id that denotes an unknown class
        :type unknown_index: int
        :returns: an identity
        :rtype: int
        """

        unknown = unknown_index

        # get prediction probabilities across all classes for each sample
        predictions = self.classifier.predict_proba(np.array(embeddings))

        # get the index of the highest predicted class
        prediction_indices = np.argmax(predictions, axis=1)
        # get the probability of the highest predicted class
        prediction_probabilities = np.max(predictions, axis=1)

        self.logger.debug(
            "Predicted indices before thresholding: {}".format(prediction_indices))

        # get the boolean mask for all indices that have a probability less than the threshold value
        thresholded_probabilities = prediction_probabilities < threshold
        # extract the indices from the boolean mask
        thresholded_indices = np.nonzero(thresholded_probabilities)
        # set the indices below the threshold to belong to an unknown class
        prediction_indices[thresholded_indices] = unknown

        self.logger.debug(
            "Predicted indices after thresholding: {}".format(prediction_indices))

        # get the index that occured the most in the batch that was evaluated
        predicted_identity = np.max(prediction_indices)

        if predicted_identity != unknown:
            predicted_identity = self.label_map[predicted_identity]

        return predicted_identity

    def load_label_mapping(self, label_map_path=None):
        """
        Loads the mapping between the real labels and the ones used by sklearn during training.

        :param label_map_path: path to the pickled label map
        :type label_map_path: str
        :returns: a dictionary mapping from encoded label to real label
        :rtype: dict
        """

        if not os.path.exists(label_map_path):
            raise Exception("Path to label map does not exist!")

        label_map_path = os.path.realpath(label_map_path)

        return joblib.load(label_map_path)
