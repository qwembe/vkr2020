from app import SampleApp
import logging
import tracemalloc

import cv2


sift = cv2.xfeatures2d.SIFT_create()
fast = cv2.FastFeatureDetector()
path_groundtruth = ""
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


if __name__ == "__main__":
    tracemalloc.start()
    my_app = SampleApp()
    my_app.run()
    tracemalloc.stop()






