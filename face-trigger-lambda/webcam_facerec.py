# *****************************************************
#                                                     *
# Author: Ankur Roy Chowdhury (Softura)               *
#                                                     *
# *****************************************************

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread, Event
import os
import awscam

import cv2
import time
import numpy as np
from collections import deque
import logging

from gtts import gTTS
from pygame import mixer
from tempfile import TemporaryFile
import pyttsx

import face_trigger

from face_trigger.model.deep.FaceRecognizer import FaceRecognizer
from face_trigger.process.post_process import FaceDetector, LandmarkDetector, FaceAlign
from face_trigger.utils.common import RepeatedTimer

from configurator import setup_config, setup_logging

fps_counter = None  # repeated timer object
frame_count = 0  # frames ingested
fps = 0  # computed fps
sequence = 0  # sequence indicating consecutive face detections
landmarks = []  # list to hold the face landmarks across the batch
faces = []  # list to hold face bounding boxes across the batch
# queue holding information of the last fps counts; used to generate avg, fps
fps_queue = deque(maxlen=100)
counter_delay = 5.0  # fps counter update frequency in secs


class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """

    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p': (1920, 1080), '720p': (
            1280, 720), '480p': (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode(
            '.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()


def speak(unknown=False, person_name=None):
    """
    Speaks out the text fed to it.
    :param bool unknown: flag to indicate whether the person was unidentified
    :param str person_name: the name of the person. If unknown is set to TRue, no need to set this parameter.
    """

    person_unknown_msg = "Person cannot be identified"
    person_id_msg_prefix = "Person identified as "
    msg = ""
    try:
        tts = None
        if unknown is True:
	    msg = person_unknown_msg
        else:
            msg = person_id_msg_prefix + \
                str(unicode(person_name, encoding='utf-8'))
            
	tts = gTTS(msg)

        mixer.init()
        sf = TemporaryFile()
        tts.write_to_fp(sf)
        sf.seek(0)
        mixer.music.load(sf)
	if not mixer.music.get_busy():
        	mixer.music.play()

        time.sleep(4)
	#engine = pyttsx.init()
	#engine.say(msg)
	#engine.runAndWait()

    except Exception as ex:
        logging.exception("Exception: {}".format(ex))
	
	try:
		engine = pyttsx.init()
		engine.say(msg)
		engine.runAndWait()
	
	except Exception as ex:
        	logging.exception("Exception: {}".format(ex))

def fps_count():
    """
    Outputs the frames per second
    """
    global frame_count
    global fps
    global fps_queue
    global counter_delay

    fps = frame_count/counter_delay
    fps_queue.append(fps)

    frame_count = 0


def cleanup():
    """
    Runs cleanup services
    """
    global fps_counter
    global fps_queue

    fps_counter.stop()

    logging.debug("Avg. FPS: {:0.3f}".format(np.mean(np.array(fps_queue))))


def start_over():
    """
    Resets the following variables, if there is a discontinuity in detecting a face among consecutive frames:
    1. faces - all detected faces 
    2. landmarks - facial landmarks of the detected faces
    3. sequence - the counter for the sequence
    """

    global sequence
    global landmarks
    global faces

    sequence = 0
    faces = []
    landmarks = []


def infinite_infer_run():
    """ Entry point of the lambda function"""

    setup_logging()
    config = setup_config()

    try:

        """
        Main loop of the program
        """

        logger = logging.getLogger(__name__)

        global frame_count
        global fps_counter
        global fps
        global sequence
        global landmarks
        global faces
        global counter_delay

        # setup the configuration
        face_area_threshold = config.get("face_area_threshold", 0.03)
        cam_height = config.get(
            "cam_height", 480)
        cam_width = config.get("cam_width", 858)
        resolution = config.get("resolution", "480p")
        batch_size = config.get("batch_size", 1)
        face_recognition_confidence_threshold = config.get(
            "face_recognition_confidence_threshold", 0.25)
        frame_skip_factor = config.get("frame_skip_factor", 5)
        unknown_class = config.get("unknown_class", -1)

        svm_model_path = config.get(
            "svm_model_path", "classifier.pkl")
        label_mapping_path = config.get(
            "label_mapping_path", "label_mapping.pkl")

        print(face_area_threshold)

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay(resolution)
        local_display.start()

        # init the fps counter object
        fps_counter = RepeatedTimer(interval=5.0, function=fps_count)

        # reference to face detector
        face_detector = FaceDetector(face_area_threshold=face_area_threshold)
        # reference to landmark detector
        landmark_detector = LandmarkDetector(
            predictor_path=None)
        # reference to face recognizer
        face_recognizer = FaceRecognizer(
            dnn_model_path=None, classifier_model_path=svm_model_path, label_map_path=label_mapping_path)

        # initialise the sequence count
        sequence = 0

        # start the fps counter
        fps_counter.start()

        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame = cv2.resize(frame, (cam_width, cam_height))

            frame = cv2.flip(frame, 1)

            # increment frame count; for fps calculation
            frame_count += 1

            # only process every 'frame_skip_factor' frame
            if not frame_count % frame_skip_factor == 0:
                        # get frame rate
                fps_text = "FPS:" + str(fps)
                cv2.putText(frame, fps_text, (1, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                # Set the next frame in the local display stream.
                local_display.set_frame_data(frame)
                continue

            # convert to grayscale
            grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # equalize the histogram
            # grayImg = cv2.equalizeHist(grayImg)

            # detect the largest face
            face = face_detector.detect(grayImg)

            # if a face was detected
            if face is not None:

                # increment sequence count
                sequence += 1

                # draw a rectangle around the detected face
                cv2.rectangle(frame, (face.left(), face.top()),
                              (face.right(), face.bottom()), (255, 0, 255))

                # get the landmarks
                landmark = landmark_detector.predict(face, grayImg)

                # accumulate face and landmarks till we get a batch of batch_size
                if landmark is not None:
                    faces.append(grayImg)
                    landmarks.append(landmark)

                # recognize the face in the batch
                if len(faces) == batch_size:
                    start_time = time.time()
                    logger.debug("Start timestamp: {}".format(start_time))

                    face_embeddings = face_recognizer.embed(
                        images=faces, landmarks=landmarks)

                    predicted_identity = face_recognizer.infer(
                        face_embeddings, threshold=face_recognition_confidence_threshold, unknown_index=unknown_class)

                    end_time = time.time()  # batch:100 s: ~1.5 sec; p:
                    logger.debug("End time: {}. Runtime: {}".format(
                        end_time, (end_time-start_time)))

                    if predicted_identity == unknown_class:
                        speak(unknown=True)
                    else:
                        speak(unknown=False, person_name=predicted_identity)

                    logger.info("Predicted identity: {}".format(
                        predicted_identity))

                    # start a new face recognition activity
                    start_over()

            else:
                # start a new face recognition activity, because no face face was detected in the frame
                start_over()

            # get frame rate
            fps_text = "FPS:" + str(fps)
            cv2.putText(frame, fps_text, (1, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)

    except Exception as ex:
        logging.exception("Exception: {}".format(ex))

    finally:
        cleanup()


infinite_infer_run()
