import numpy as np
import cv2
import logging

log = logging.getLogger("Trajectory")


class trajectory:
    def __init__(self, X=20, Y=20):
        self.X = X
        self.Y = Y
        self.brush_X = X // 2
        self.brush_Y = Y // 2
        self.trajectory = np.zeros((X, Y, 3), dtype=np.uint8)
        self.vo_traj = np.array([])
        self.gt_traj = np.array([])
        self.adjust_size = 15

    def add_vo(self, p, index):
        self.vo_traj = np.append(self.vo_traj, p)
        self.udjust_map(p)
        x, y = p
        cv2.circle(self.trajectory,
                   (x + self.brush_X, y + self.brush_Y),
                   1, (index * 255 / 4540, 255 - index * 255 / 4540, 0), 1)  # VO

    def add_gt(self, p):
        self.gt_traj = np.append(self.gt_traj, p)
        self.udjust_map(p)
        x, y = p
        cv2.circle(self.trajectory,
                   (x + self.brush_X, y + self.brush_Y),
                   1, (0, 0, 255), 2)  # GT

    def udjust_map(self, p):
        px, py = p
        px += self.brush_X
        py += self.brush_Y

        if px > self.X:
            self.X += self.adjust_size
            adj = np.zeros((self.Y, self.adjust_size, 3))
            self.trajectory = np.hstack((self.trajectory, adj))

        if py > self.Y:
            self.Y += self.adjust_size
            adj = np.zeros((self.adjust_size, self.X, 3))
            self.trajectory = np.vstack((self.trajectory, adj))

        if py < 0:
            self.Y += self.adjust_size
            adj = np.zeros((self.adjust_size, self.X, 3))
            self.trajectory = np.vstack((adj, self.trajectory))
            self.brush_Y += self.adjust_size

        if px < 0:
            self.X += self.adjust_size
            adj = np.zeros((self.Y, self.adjust_size, 3))
            self.trajectory = np.hstack((adj, self.trajectory))
            self.brush_X += self.adjust_size

    def plot(self):
        cv2.imshow('traj', self.trajectory)

    def imwrite(self, name):
        log.info("Image saved at " + name)
        cv2.imwrite(name, self.trajectory)
