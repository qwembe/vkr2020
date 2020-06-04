import logging
import numpy as np
import cv2
import math
import time
import tracemalloc

from tools.trajectory import trajectory

log = logging.getLogger(__name__)
epsilon = 0.09
SCALE = 1.0
SCALE_MUL = 1


def ATE(p1, p2):
    return abs((p1[0] - p2[0]) + (p1[1] - p2[1]))


def calc_euclid_dist(p1, p2):
    a = math.pow((p1[0] - p2[0]), 2.0) + math.pow((p1[1] - p2[1]), 2.0)
    return math.sqrt(a)


def run(dataset=None, detector=None, matcher=None, other=None):
    if dataset is None:
        log.error("No dataset found!")
        return 1
    if detector is None:
        log.error("No detector found!")
        return 1
    if matcher is None:
        log.error("No matcher found!")
        return 1
    enable_gt, enable_knn, disable_vis = other

    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)
    final_stat = None
    all_stat = []
    time_start = 0
    time_acc = 0
    final_stat = 0.0

    # create graph.
    traj = trajectory()

    prev_image = None
    prev_des = None

    valid_ground_truth = False
    if dataset.ground_truth is not None:
        valid_ground_truth = True and enable_gt

    if dataset.camera_matrix is not None:
        camera_matrix = dataset.camera_matrix()
    else:
        camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                  [0.0, 718.8560, 185.2157],
                                  [0.0, 0.0, 1.0]])

    for index in range(dataset.image_count):
        time_start = time.time()

        # load image
        image = cv2.imread(dataset.image_path_left(index))
        if image is None:
            log.error("Corrupted dataset!")
            return 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # TODO: Check for slow movement

        # main process
        keypoint, des = detector.detect(image, None)
        if prev_image is None:
            prev_image = image
            prev_keypoint = keypoint
            prev_des = des
            continue

        points = np.array(list(map(lambda x: [x.pt], prev_keypoint)),
                          dtype=np.float32)

        # Matcher in progress
        if des is None or not matcher.require_descriptor():
            p1, st, err = matcher.match(prev_image, image, points)
        else:
            if enable_knn:
                matcher.knnMatch(prev_des, des)
                matcher.ratiotest()
            else:
                matcher.match(prev_des, des)
            points, p1 = matcher.getpoints(prev_keypoint, keypoint)

        # Essential matrix
        E, mask = cv2.findEssentialMat(p1, points, camera_matrix,
                                       cv2.RANSAC, 0.999, 1.0, None)

        # Recover pose!
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
                # log.debug(str(current_pos[:]))
                current_pos[:, 0] = ground_truth[:, 3]
            if scale < epsilon:
                log.debug("Scale " + str(scale))
                continue

        # Concatinate pos
        current_pos += current_rot.dot(t) * scale
        current_rot = R.dot(current_rot)
        # log.debug("SCALE_ " + str(scale))
        log.info("-----------------------")
        draw_x, draw_y = int(current_pos[0][0] * SCALE_MUL), int(current_pos[2][0] * SCALE_MUL)
        # get ground truth if exist.
        if valid_ground_truth:
            ground_truth = dataset.ground_truth[index]
            true_x, true_y = int(ground_truth[0, 3] * SCALE_MUL), int(ground_truth[2, 3] * SCALE_MUL)
            # log.debug("GT_ " + str([true_x, true_y]))
            traj.add_gt([true_x, true_y])
            all_stat.append(float(final_stat))
            final_stat = ATE([current_pos[0][0], current_pos[2][0]], [ground_truth[0, 3], ground_truth[2, 3]])
            log.debug("final ATE -  " + str(final_stat))

        # Draw
        traj.add_vo([draw_x, draw_y], index)
        if not disable_vis:

            img = cv2.drawKeypoints(image, keypoint, None)

            cv2.imshow('image', img)
            traj.plot()
        else:
            if index > 100:
                break
            log.info("index " + str(index))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current, peak = tracemalloc.get_traced_memory()
        log.info("Current memory usage is " + str(current / 10 ** 6) + "MB; Peak was " + str(peak / 10 ** 6) + "MB")
        time_acc += -time_start + time.time()
        log.info("FPS = " + str(index / time_acc))
        prev_image = image
        prev_keypoint = keypoint
        prev_des = des
    cv2.destroyAllWindows()
    traj.imwrite("./map.jpg")
    average_ate = sum(all_stat) / len(all_stat)
    fps = index / time_acc
    memory, peak = tracemalloc.get_traced_memory()
    return [average_ate, final_stat, fps, str(peak / 10 ** 6) + "MB"]