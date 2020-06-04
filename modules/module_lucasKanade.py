import cv2
import numpy as np
import logging

from tools.visual_odometry import PinholeCamera, VisualOdometry
log = logging.getLogger(__name__)

TRAJ_WINDOW_SIZE_X = 600
TRAJ_WINDOW_SIZE_Y = 600

# def run(path, pic_src, groundtruth):
def run(dataset):
    # cam = PinholeCamera(640.0, 480.0, 718.8560, 718.8560, 607.1928, 185.2157)
    # init = False
    cam = PinholeCamera(640.0, 480.0, 1, 1, 1, 1)
    vo = VisualOdometry(cam, groundtruth)

    traj = np.zeros((TRAJ_WINDOW_SIZE_X, TRAJ_WINDOW_SIZE_Y, 3), dtype=np.uint8)

    with open(pic_src) as src:
        img_annotation = src.readlines()
        log.debug(len(img_annotation))
        for img_id in range(len(img_annotation)):
            if img_annotation[img_id].startswith("#"):
                continue
            path_img = path + "/" + img_annotation[img_id].split()[1]
            img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
            log.debug(img.shape)

            # if not init:
            #     size = img.shape
            #     cam = PinholeCamera(size[1], size[0], 1, 1, int(size[1]/2), int(size[0]/2))
            #     vo = VisualOdometry(cam, groundtruth)
            #     init = True

            vo.update(img, img_id)
            if img_id % 3 != 0:
                continue

            cur_t = vo.cur_t
            if (img_id > 3):
                x, y, z = cur_t[0], cur_t[1], cur_t[2]
            else:
                x, y, z = 0., 0., 0.

            # plot

            draw_x, draw_y = int(x + TRAJ_WINDOW_SIZE_X / 2), int(z + TRAJ_WINDOW_SIZE_Y / 2)
            true_x, true_y = int(vo.trueX) + 290, int(vo.trueZ) + 90

            cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)  # VO
            cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)  # GT
            cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
            text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
            cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            cv2.imshow('Road facing camera', img)
            cv2.imshow('Trajectory', traj)
            cv2.waitKey(1)

        cv2.imwrite('map.png', traj)
