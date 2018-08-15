import threading
import time
import numpy as np


def clamp_rectangle(x1=None, y1=None, x2=None, y2=None, x1_min=0, y1_min=0, x2_max=None, y2_max=None):
    """
    Clamps the coordinates of a rectangle to within specified limits

    :param x1: the leftmost x-coordinate
    :type x1: int
    :param y1: the topmost y-coordinate
    :type y1: int
    :param x2: the rightmost x-coordinate
    :type x2: int
    :param y2: the bottommost y-coordinate
    :type y2: int
    :param x1_min: the leftmost possible x-coordinate
    :type x1_min: int
    :param y1: the topmost possible y-coordinate
    :type y1: int
    :param x2: the rightmost possible x-coordinate
    :type x2: int
    :param y2: the bottommost possible y-coordinate
    :type y2: int
    :returns: clamped coordinates (x1_clamped, y1_clamped, x2_clamped, y2_clamped)
    :rtype: 4-tuple
    """

    return (max(x1_min, x1), max(y1_min, y1), min(x2_max, x2), min(y2_max, y2))


def shape_to_np(shape, dtype="int"):
    """
    Convert a dlib.shape to a numpy array

    :param shape: a dlib.shape object
    :type shape: dlib.shape 
    :returns: converted numpy array 
    :rtype: numpy.array
    """

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


class RepeatedTimer(object):
    """
    Creates a timer object that can call a specified function repeatedly after a specified delayed. Does not drift over time. 
    """

    def __init__(self, interval, function, *args, **kwargs):
        """
        Instantiates the object

        :param interval: the delay (in seconds) between each function call
        :type interval: float 
        :param function: the refernce to the function to repeat
        :type function: function reference
        :param *args: named argument for the function call
        :param **kwargs: un-named arguments for the function call
        """
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()

    def _run(self):
        """
        Used by the timer thread. Reponsible for calling the 'function'
        """
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        """
        Starts the timer thread and schedules the next call
        """
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(
                self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        """
        Stops the timer thread
        """
        self._timer.cancel()
        self.is_running = False
