from settings.opt import DETECTORS
import cv2
import logging

log = logging.getLogger(__name__)


class DetectorManager():
    def create(type="None"):
        if type is "None":
            log.error("No detector selected!")
            return Detector
        elif type is DETECTORS[0]:
            log.info(type)
            return DetectorFast()
        elif type is DETECTORS[1]:
            log.info(type)
            return DetectorSIFT()
        elif type is DETECTORS[2]:
            log.info(type)
            return DetectorORB()


class Detector():
    def __init__(self, type="None"):
        self.type = type
        self.feature_detector = None

    def get_type(self):
        return self.type

    # def compute(self, gray, points):
    #     return self.feature_detector.compute(gray, points)

    def __call__(self, *args, **kwargs):
        return self


class DetectorFast(Detector):
    def __init__(self, type="fast"):
        super().__init__(type=type)
        self.feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                               nonmaxSuppression=True)

    def detect(self, img, *args):
        return self.feature_detector.detect(img, *args), None


class DetectorSIFT(Detector):
    def __init__(self, type="sift"):
        super().__init__(type=type)
        self.feature_detector = cv2.xfeatures2d.SIFT_create()

    def detect(self, img, *args):
        return self.feature_detector.detectAndCompute(img, *args)


class DetectorORB(Detector):
    def __init__(self, type="orb"):
        super().__init__(type=type)
        self.feature_detector = cv2.ORB_create(5000)

    def detect(self, img, *args):
        return self.feature_detector.detectAndCompute(img, *args)
