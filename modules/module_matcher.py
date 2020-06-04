from settings.opt import MATCHERS
import cv2
import logging
import numpy as np

log = logging.getLogger(__name__)



class MatcherManager():
    def create(type="None"):
        if type is "None":
            log.error("No detector selected!")
            return Matcher
        elif type is MATCHERS[0]:
            log.info(type)
            return MatcherBF()
        elif type is MATCHERS[1]:
            log.info(type)
            return MatcherFLANN()
        elif type is MATCHERS[2]:
            log.info(type)
            return MatcherLK()


class Matcher():
    def __init__(self, type="None"):
        self.type = type
        self.feature_matcher = None

    def get_type(self):
        return type

    def match(self, *args, **kwargs):
        pass

    def require_descriptor(self):
        return False

    def __call__(self):
        return self


class MatcherBF(Matcher):
    def __init__(self, type="bf"):
        super().__init__(type=type)
        self.bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        self.matches = None

    def match(self, des1, des2):
        self.matches = self.bf.match(des1, des2)

    def knnMatch(self, des1, des2):
        self.matches = self.bf.knnMatch(des1, des2, k=2)

    def ratiotest(self):
        good = []
        for m, n in self.matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        self.matches = good

    def getpoints(self, prev, cur):
        good_prev = []
        good_cur = []
        for m in self.matches:
            good_prev.append([prev[m.queryIdx].pt])
            good_cur.append([cur[m.trainIdx].pt])
        return np.array(good_prev), np.array(good_cur)

    def require_descriptor(self):
        return True


class MatcherFLANN(MatcherBF):
    def __init__(self, type="flann"):
        super().__init__(type=type)
        self.bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        self.matches = None


class MatcherLK(Matcher):
    def __init__(self, type="lk"):
        super().__init__(type=type)
        self.lk_params = dict(winSize=(21, 21),
                              criteria=(cv2.TERM_CRITERIA_EPS |
                                        cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    def match(self, prev_image, image, points):
        return cv2.calcOpticalFlowPyrLK(prev_image,
                                        image, points,
                                        None, **self.lk_params)
