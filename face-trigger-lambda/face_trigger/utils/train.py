from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import random

import numpy as np
import cv2
import uuid
import pprint
import tqdm


from face_trigger.process.post_process import FaceDetector, LandmarkDetector, FaceAlign
from face_trigger.model.deep.FaceRecognizer import FaceRecognizer


def generate_embeddings_for_dataset(dataset_path=None):
    """
    Generates embeddings by sequentially reading images from the dataset.

    :param dataset_path: path to the dataset
    :type dataseet_path: str
    :returns: X, y where X is a list of numpy array (128-d vectors) and y is a numpy array representing the labels

    """

    assert os.path.exists(dataset_path)

    X = []
    y = []

    pbar = tqdm.tqdm(total=None)

    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    landmark_detector = LandmarkDetector()

    for root, dirs, files in os.walk(dataset_path):

        if root == dataset_path:
            pbar.total = len(dirs)

        for image in files:

            img_path = os.path.join(root, image)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # detect faces
            faces = face_detector.detect_unbounded(img)

            if len(faces) == 1:

                # get face (only interested if there's one and only one)
                face_bb = faces[0]

                # get the landmarks
                landmarks = landmark_detector.predict(face_bb, img)

                if landmarks is not None:
                    embedding = face_recognizer.embed([img], [landmarks])
                    X.append(embedding[0])
                    y.append(os.path.basename(root))

        if root != dataset_path:
            pbar.update()

    pbar.close()

    return X, y


def generate_embeddings_for_split_and_fold(dataset_path=None, split_path=None, fold=None, num_train=None):
    """
    Generate embeddings of dataset for a particular split and fold.

    :param dataset_path: path to the dataset
    :type dataset_path: str
    :param split_path: path to the directory holding train-test split info
    :type split_path: str
    :param fold: which fold to generate embeddings on 
    :type fold: int
    :param num_train: folder name, indacating the number of training samples per subject
    :type num_train: int
    :returns: X_train, y_train, X_test, y_test

    """

    assert os.path.exists(dataset_path)
    assert os.path.exists(split_path)

    logger = logging.getLogger(__name__)

    if fold is None:
        raise Exception("Fold identifier is uspecified!")

    if num_train is None:
        raise Exception("Training sample identifier unspecified!")

    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    landmark_detector = LandmarkDetector()

    suffixes = ["train", "test"]
    X = {}
    y = {}

    for suffix in suffixes:
        logger.info("Generating embeddings for {}...".format(suffix))
        csv_file = os.path.join(
            split_path, str(num_train), str(fold), "{}.csv".format(suffix))

        X[suffix] = []
        y[suffix] = []

        with open(csv_file, "r") as csv_file:
            for line in csv_file:
                parts = line.split("\n")
                parts = parts[0].split(",")
                label = parts[0]
                imgstrs = parts[1:]

                images_path = os.path.join(dataset_path, label)
                for image_string in imgstrs:
                    img = cv2.imread(os.path.join(
                        images_path, image_string.strip()), cv2.IMREAD_GRAYSCALE)

                    # detect faces
                    faces = face_detector.detect_unbounded(img)

                    if len(faces) == 1:

                        # get face (only interested if there's one and only one)
                        face_bb = faces[0]

                        # get the landmarks
                        landmarks = landmark_detector.predict(face_bb, img)

                        if landmarks is not None:
                            embedding = face_recognizer.embed(
                                [img], [landmarks])
                            X[suffix].append(embedding[0])
                            y[suffix].append(label)

    y['train'] = np.array(y['train'])
    y['test'] = np.array(y['test'])

    return X['train'], y['train'], X['test'], y['test']
