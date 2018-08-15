from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import os
import random
import shutil

import numpy as np
import cv2
import uuid
import pprint
import tqdm

from face_trigger.process.post_process import FaceDetector, LandmarkDetector, FaceAlign


class Dataset():

    """
    A class for generating manual train-test splits and loading data from generated splits.

    Example:
    --------------------------------------------------------
    >>> dataset = Dataset(dataset_path="/media/ankurrc/new_volume/softura/facerec/datasets/norm_standard_att")
    >>> folds = 3
    >>> training_samples = [2, 5, 8]
    >>> dataset.split(num_train_list=training_samples, folds=folds, 
                    split_path="/media/ankurrc/new_volume/softura/facerec/split_path")

    """

    def __init__(self, dataset_path=None, split_path=None):
        """
        Instantiate a Dataset object

        :param dataset_path: path to the dataset
        :type dataset_path: str
        """

        self.logger = logging.getLogger(__name__)

        if not os.path.exists(os.path.realpath(dataset_path)):
            raise Exception("Invalid dataset path!")

        self.dataset_path = os.path.realpath(dataset_path)

    def split(self, split_path=None, num_train_list=None, folds=1):
        """
        Generates a train-test split based on the number of training samples

        :param num_train_list: number of training samples per fold
        :type num_train_list: list of inttegers
        :param fold: total number of folds
        :type fold: int
        :param split_path: path to store the dataset train-test split information. Will rename the old directory, if it exists.
        :type split_path: str
        """

        assert split_path is not None

        split_path = os.path.realpath(split_path)

        if os.path.isfile(split_path):
            raise Exception(
                "A file with the same name as 'split_path' already exists. Please make sure split_path location is a valid one!")

        if os.path.isdir(split_path):
            os.rename(split_path, os.path.join(os.path.split(
                split_path)[0], os.path.split(
                split_path)[1] + "_" + str(uuid.uuid4().get_hex())))

        os.makedirs(split_path)

        if not isinstance(num_train_list, list):
            raise Exception("num_train_list should be a list!")

        train_file_suffix = "train.csv"
        test_file_suffix = "test.csv"

        for training_sample in num_train_list:
            self.logger.info(
                "Generating for {0:d} training samples per subject.".format(training_sample))

            for i in range(1, folds+1):

                self.logger.info("Generating: Fold {0:d}".format(i))

                fold_path = os.path.join(
                    split_path, str(training_sample), str(i))

                self.logger.info("Creating directory: {0}".format(fold_path))
                os.makedirs(fold_path)
                self.logger.info("done.")

                train_file = os.path.join(fold_path, train_file_suffix)
                test_file = os.path.join(fold_path, test_file_suffix)

                self.logger.info(train_file)

                # bar = pbar.tqdm(total=int(9e9))
                subjects = 0
                rejected_dirs = []

                with open(train_file, "a+") as train, \
                        open(test_file, "a+") as test:

                    for root, dirs, files in os.walk(self.dataset_path):

                        # here root denotes the folder indicating the user id
                        if root != self.dataset_path:
                            # bar.set_description(
                            #     "Dir-->" + os.path.basename(root))
                            random.Random().shuffle(files)

                            # bar.update()

                            if len(files) - 1 < training_sample:
                                rejected_dirs.append(os.path.basename(root))
                                continue

                            subjects += 1

                            train_split = files[0:training_sample]
                            test_split = files[training_sample:]

                            # create csv entry as user id, file1, file2...
                            train_sample = os.path.basename(root) + ", " + \
                                reduce(lambda l, s: l + ", " +
                                       s, train_split) + "\n"

                            test_sample = os.path.basename(root) + ", " + \
                                reduce(lambda l, s: l + ", " +
                                       s, test_split) + "\n"

                            train.write(train_sample)
                            test.write(test_sample)

                        # elif root == self.dataset_path:
                            # bar.total = len(dirs)

                            # bar.close()

                i += 1

            if len(rejected_dirs) > 0:
                self.logger.info(
                    "The following directories were rejected: {}".format(rejected_dirs))
            self.logger.info(
                "We have {0:d} subjects in our dataset.".format(subjects))

    def load_data(self, split_path=None, is_train=None, num_train=None, fold=None):
        """
        Gets the test or train data

        :param is_train: a flag to indicate whether we need training data or test data
        :type is_train: bool
        :param fold: the fold for which data needs to be loaded
        :type fold: int
        :param num_train: the subdirectory indicating number of training samples per subject
        :type num_train: int
        :param split_path: path to store the dataset train-test split information. Will rename the old directory, if it exists.
        :type split_path: str
        :returns: a tuple of a vector of faces and it's corresponding labels
        :rtype: (numpy.ndarray, numpy.ndarray)
        """

        if not os.path.exists(split_path):
            raise Exception(
                "Dataset test-train split has not been generated, yet! Please run split() before loading data.")

        if fold is None:
            raise Exception("Fold identifier is uspecified!")

        if num_train is None:
            raise Exception("Training sample identifier unspecified!")

        if is_train is None:
            raise Exception("Train/Test flag is unspecified!")

        suffix = None
        if is_train:
            suffix = "train.csv"
        else:
            suffix = "test.csv"

        csv_file = os.path.join(
            split_path, str(num_train), str(fold), suffix)
        X = []
        y = []

        with open(csv_file, "r") as file:
            for line in file:
                parts = line.split("\n")
                parts = parts[0].split(",")
                label = parts[0]
                imgs = parts[1:]

                # print label, "imgs:", train_imgs

                images_path = os.path.join(self.dataset_path, label)
                for image in imgs:
                    img = cv2.imread(os.path.join(
                        images_path, image.strip()), cv2.IMREAD_GRAYSCALE)
                    # print img.shape
                    X.append(img)
                    y.append(int(label))

        y = np.array(y)

        return X, y


def dataset_filter(dataset_path=None, output_path=None):
    """
    Filter the dataset by:
    1. A face is detected in the image. If no face or more than one face is detected, the image is rejected.
    2. For each detected face, 5-landmarks are detected. If landmarks are not detected, image is rejected.
    3. A new dataset is created by not including the rejected images.

    :param dataset_path: path to the original dataset
    :type dataset_path: str
    :param output_path: path to the filtered dataset
    :type output_path: str
    :returns: dictionary of rejected images
    :rtype: dict

    Example:
    ------------------------------------------------------------------
    >>> dataset_path = "/media/ankurrc/new_volume/softura/facerec/datasets/standard_att_copy"
    >>> output_path = "/media/ankurrc/new_volume/softura/facerec/att_norm"

    >>> dataset_filter(
        dataset_path=dataset_path, output_path=output_path)
    """

    logger = logging.getLogger(__name__)

    face_detector = FaceDetector()
    landmark_predictor = LandmarkDetector()

    rejected_faces = {}

    bar = tqdm.tqdm(total=None)

    if not os.path.exists(dataset_path):
        raise Exception("Invalid dataset path!")

    # setup output directory
    if os.path.isdir(output_path):
        os.rename(output_path, os.path.join(os.path.split(
            output_path)[0], os.path.split(
            output_path)[1] + "_" + str(uuid.uuid4().get_hex())))
    os.makedirs(output_path)

    for root, dirs, files in os.walk(dataset_path):

        if root == dataset_path:
            bar.total = len(dirs)

        for direc in dirs:
            # create output directory for this presonality
            output_direc_path = os.path.join(output_path, direc)
            os.mkdir(output_direc_path)

        for img in files:

            img_path = os.path.join(root, img)

            # read the image
            rgbImg = cv2.imread(img_path)

            grayImg = None
            if rgbImg is None:
                break
            elif rgbImg.shape[2] == 3:
                grayImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
            else:
                grayImg = rgbImg

            # detect faces
            faces = face_detector.detect_unbounded(grayImg)

            if len(faces) == 1:

                # get face (only interested if there's one and only one)
                face_bb = faces[0]

                # get the landmarks
                landmarks = landmark_predictor.predict(face_bb, grayImg)

                # reject this image in case no landmarks found
                if landmarks is None:

                    root = os.path.basename(root)
                    if root in rejected_faces:
                        rejected_faces[root].append(img)
                    else:
                        rejected_faces[root] = [img]
                else:
                    # write to output directory
                    save_path = os.path.join(
                        output_path, os.path.basename(root), img)

                    cv2.imwrite(save_path, rgbImg)

            else:
                root = os.path.basename(root)
                if root in rejected_faces:
                    rejected_faces[root].append(img)
                else:
                    rejected_faces[root] = [img]

        if root != dataset_path:
            bar.update()

    bar.close()

    logger.info("Filtered dataset created at {}".format(output_path))

    print("Rejected directories:")
    pprint.pprint(rejected_faces)

    return rejected_faces
