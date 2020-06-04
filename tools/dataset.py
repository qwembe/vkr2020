#!/usr/bin/env python
# coding: utf-8

__author__ = "Masahiko Toyoshi"
__copyright__ = "Copyright 2007, Masahiko Toyoshi."
__license__ = "GPL"
__version__ = "1.0.0"

import glob
import os
import yaml

import numpy as np
import logging

log = logging.getLogger(__name__)
FACTOR_SCALE = 100.0

class CameraParameters():
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @property
    def camera_matrix(self):
        matrix = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fx, self.cy],
                           [0.0, 0.0, 1.0]])
        return matrix

    def __call__(self):
        return self.camera_matrix


class Dataset():
    def __init__(self):
        log.info("Loading dataset...")
        self.type = None
        pass

    # bug os path join and format string
    def image_path_left(self, index):
        return os.path.join(self.path, self.image_format_left.format(index))

    def count_image(self):
        extension = os.path.splitext(self.image_format_left)[-1]
        wildcard = os.path.join(self.path, '*' + extension)
        self.image_count = len(glob.glob(wildcard))

    def load_ground_truth_pose(self, gt_path):
        ground_truth = None
        if not os.path.exists(gt_path):
            print("ground truth path is not found.")
            return None

        ground_truth = []

        with open(gt_path) as gt_file:
            gt_lines = gt_file.readlines()
            while gt_lines[0].startswith("#"):
                gt_lines = gt_lines[1:]

            for gt_line in gt_lines:
                pose = self.convert_text_to_ground_truth(gt_line)
                ground_truth.append(pose)
        return ground_truth

    def convert_text_to_ground_truth(self, gt_line):
        pass


class KittiDataset(Dataset):
    def __init__(self, path):
        log.info("Loading kitti dataset...")
        self.type = "kitti"
        self.image_format_left = '{:06d}.png'
        self.path = os.path.join(path, 'image_0')
        self.calibfile = os.path.join(path, 'calib.txt')
        sequence_count = os.path.dirname(self.path).split('/')[-1]

        gt_path = os.path.join(self.path, '..', '..', '..', '..', '..',
                               'data_odometry_poses', 'dataset', 'poses', sequence_count + '.txt')

        self.count_image()
        self.ground_truth = self.load_ground_truth_pose(gt_path)
        self.camera_matrix = self.load_camera_parameters(self.calibfile)

    def convert_text_to_ground_truth(self, gt_line):
        matrix = np.array(gt_line.split()).reshape((3, 4)).astype(np.float32)
        return matrix

    def load_camera_parameters(self, calibfile):
        if not os.path.exists(calibfile):
            print("camera parameter file path is not found.")
            return None

        with open(calibfile, 'r') as f:
            line = f.readline()
            part = line.split()
            param = CameraParameters(float(part[1]), float(part[6]),
                                     float(part[3]), float(part[7]))
            # log.debug("KITTI CAM - " + str(param()))
            return param


class TumDatset(Dataset):
    def __init__(self, path):
        log.info("Loading tum dataset...")
        self.type = "tum"
        self.image_format_left = ".png"
        image_path = os.path.join(path, "rgb.txt")
        gt_path = os.path.join(path, "groundtruth.txt")

        self.img_annotation = None
        with open(image_path) as src:
            img_annotation = src.readlines()
            while img_annotation[0].startswith("#"):
                img_annotation = img_annotation[1:]
            if len(img_annotation) > 2:
                self.img_annotation = img_annotation

        # sufficse_src_path = os.path.dirname(img_annotation[0].split()[-1])
        # self.calibfile = os.path.abspath(os.path.join(path, '..', '..',
        #                                               'pinhole_example_calib.txt'))
        # self.calibfile = "./settings/"
        self.conf_file = "None"
        if "freiburg1" in path:
            self.conf_file = "TUM1.YAML"
        elif "freiburg2" in path:
            self.conf_file = "TUM2.YAML"
        elif "freiburg3" in path:
            self.conf_file = "TUM3.YAML"

        __proj_path__ = os.path.dirname(os.path.abspath(__file__))
        self.calibfile = os.path.join(__proj_path__, '..', 'settings', self.conf_file)
        log.debug(str(self.calibfile))
        # self.path = os.path.join(path, sufficse_src_path)
        self.path = path
        # self.count_image()
        self.ground_truth = self.load_ground_truth_pose(gt_path)
        self.img_annotation, self.ground_truth = self.sync()
        # log.debug(str(self.img_annotation))
        self.image_count = len(self.img_annotation)
        self.camera_matrix = self.load_camera_parameters(self.calibfile)

    def sync(self):
        log.info("Sync time begin ...")
        slice_res = (s.split() for s in self.img_annotation)
        time__path = list((float(timestamp), path) for timestamp, path in slice_res)
        gt = list((float(*s[0]), s[1]) for s in self.ground_truth)
        matches = self.associate(time__path, gt)

        images = []
        gt_poses = []
        for i_pic, j_gt, _ in matches:
            images.append(time__path[i_pic][1])
            gt_poses.append(self.ground_truth[j_gt][1])
        log.info("Sync time completed!")
        return images, gt_poses

    def image_path_left(self, index):
        return os.path.join(self.path, self.img_annotation[index])

    def convert_text_to_ground_truth(self, gt_line):
        src = gt_line.split()
        matrix = np.array([[0, 0, 0, float(src[1])*FACTOR_SCALE],
                           [0, 0, 0, float(src[3])*FACTOR_SCALE],
                           [0, 0, 0, float(src[2])*FACTOR_SCALE]]).astype(np.float32)
        matrix = np.array([[src[0]], matrix])
        return matrix

    def load_camera_parameters(self, calibfile):
        if not os.path.exists(calibfile):
            print("camera parameter file path is not found.")
            return None

        with open(calibfile, 'r') as f:
            try:
                param = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

            line = f.readline()
            param = CameraParameters(float(param["Camera.fx"]), float(param["Camera.fy"]),
                                     float(param["Camera.cx"]), float(param["Camera.cy"]))
            # log.debug("TUM CAM - " + str(param()))
            return param

    @staticmethod
    def associate(first_list, second_list, offset=0, max_difference=0.02):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Input:
        first_list -- first list of (stamp,data) tuples
        second_list -- second list of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

        """
        potential_matches = [(abs(float(a[0]) - (float(b[0]) + offset)), ia, ib)
                             # a[0] and b[0] extract the first element which is a timestamp
                             for ia, a in enumerate(first_list)  # for counter, value in enumerate(some_list)
                             for ib, b in enumerate(second_list)
                             if abs(float(a[0]) - (float(b[0]) + offset)) < max_difference]
        potential_matches.sort()
        matches = []
        first_flag = [False] * len(first_list)
        second_flag = [False] * len(second_list)
        for diff, ia, ib in potential_matches:
            if first_flag[ia] is False and second_flag[ib] is False:
                # first_list.remove(a)
                first_flag[ia] = True
                # second_list.remove(b)
                second_flag[ib] = True
                matches.append((ia, ib, diff))
        matches.sort()
        return matches


dataset_dict = {'kitti': KittiDataset, 'tum': TumDatset}


def create_dataset(options):
    return dataset_dict[options.dataset](options.path)
