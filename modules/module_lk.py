import logging
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tools.trajectory import *

fh = logging.FileHandler('spam.log')
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setLevel(logging.DEBUG)
log.addHandler(fh)
epsilon = 0.001
SCALE = 1.0
SCALE_MUL = 1


def calc_euclid_dist(p1, p2):
    a = math.pow((p1[0] - p2[0]), 2.0) + math.pow((p1[1] - p2[1]), 2.0)
    return math.sqrt(a)


def udjust_map(X=600, Y=600):
    traj = np.zeros((X, Y, 3), dtype=np.uint8)
    return traj


def run(dataset=None):
    if dataset is None:
        return 1

    feature_detector = cv2.FastFeatureDetector_create(threshold=25,
                                                      nonmaxSuppression=True)
    lk_params = dict(winSize=(21, 21),
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)

    # create graph.
    rotation_error_list = []
    frame_index_list = []
    traj = trajectory()

    prev_image = None

    valid_ground_truth = False
    if dataset.ground_truth is not None:
        valid_ground_truth = True

    if dataset.camera_matrix is not None:
        camera_matrix = dataset.camera_matrix()
    else:
        camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                  [0.0, 718.8560, 185.2157],
                                  [0.0, 0.0, 1.0]])

    for index in range(dataset.image_count):
        # load image
        image = cv2.imread(dataset.image_path_left(index))
        if image is None:
            log.error("Corrupted dataset!")
            return 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check for slow movement

        # main process
        keypoint = feature_detector.detect(image, None)
        if prev_image is None:
            prev_image = image
            prev_keypoint = keypoint
            continue

        points = np.array(list(map(lambda x: [x.pt], prev_keypoint)),
                          dtype=np.float32)

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_image,
                                               image, points,
                                               None, **lk_params)
        E, mask = cv2.findEssentialMat(p1, points, camera_matrix,
                                       cv2.RANSAC, 0.999, 1.0, None)

        points, R, t, mask = cv2.recoverPose(E, p1, points, camera_matrix)

        scale = SCALE
        # calc scale from ground truth if exists.
        if valid_ground_truth:
            ground_truth = dataset.ground_truth[index]
            ground_truth_pos = [ground_truth[0, 3], ground_truth[2, 3]]
            previous_ground_truth = dataset.ground_truth[index - 1]
            previous_ground_truth_pos = [
                previous_ground_truth[0, 3],
                previous_ground_truth[2, 3]]

            scale = calc_euclid_dist(ground_truth_pos * SCALE_MUL,
                                     previous_ground_truth_pos * SCALE_MUL)
            if index == 1:
                log.debug(str(current_pos[:]))
                current_pos[:,0] = ground_truth[:, 3]
            if scale < epsilon:
                log.debug("Scale " + str(scale))
                continue

        current_pos += current_rot.dot(t) * scale
        current_rot = R.dot(current_rot)
        # log.debug("SCALE_ " + str(scale))

        # get ground truth if exist.
        if valid_ground_truth:
            ground_truth = dataset.ground_truth[index]
            true_x, true_y = int(ground_truth[0, 3] * SCALE_MUL), int(ground_truth[2, 3] * SCALE_MUL)
            # log.debug("GT_ " + str([true_x, true_y]))
            traj.add_gt([true_x, true_y])

        # calc rotation error with ground truth.
        # if valid_ground_truth:
        #     ground_truth = dataset.ground_truth[index]
        #     ground_truth_rotation = ground_truth[0: 3, 0: 3]
        #     r_vec, _ = cv2.Rodrigues(current_rot.dot(ground_truth_rotation.T))
        #     rotation_error = np.linalg.norm(r_vec)
        #     frame_index_list.append(index)
        #     rotation_error_list.append(rotation_error)

        draw_x, draw_y = int(current_pos[0][0] * SCALE_MUL), int(current_pos[2][0] * SCALE_MUL)
        log.debug("VO_ " + str([current_pos]))
        traj.add_vo([draw_x, draw_y], index)

        img = cv2.drawKeypoints(image, keypoint, None)

        cv2.imshow('image', img)
        traj.plot()
        cv2.waitKey(1)

        prev_image = image
        prev_keypoint = keypoint
    # rotation_error_axes.bar(frame_index_list, rotation_error_list)
    # error_figure.savefig("error.png")
