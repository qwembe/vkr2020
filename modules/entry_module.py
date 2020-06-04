import argparse
import logging
import os
from tkinter import filedialog as fd

# import modules.module_lucasKanade as lk
import tools.dataset as dataset
import modules.module_vo as vo
from modules.module_detector import DetectorManager
from modules.module_matcher import MatcherManager

log = logging.getLogger(__name__)


def run(dataset="kitti", detetector="fast", matcher="None", other="None"):
    log.info("Dataset: " + dataset)
    log.info("Detector: " + detetector)
    log.info("Matcher: " + matcher)
    log.info("GT , KNN: " + str(other))

    # Dataset
    try:
        dataset = loadDataset(dataset)
    except OSError as e:
        log.error(e)
        return 1

    # Detector
    m_detector = DetectorManager.create(type=detetector)
    if m_detector().get_type() is "None":
        log.error("No detector find...")
        return 1

    # Matcher
    m_matcher = MatcherManager.create(type=matcher)
    if m_matcher().get_type() is "None":
        log.error("No matcher found...")
        return 1

    # Algo
    stat = vo.run(dataset, m_detector, m_matcher , other)

    return stat


def loadDataset(m_dataset):
    log.info("loading dataset ...")
    path = fd.askdirectory()
    option = parse_argument(path, m_dataset)
    my_dataset = dataset.create_dataset(option)
    if my_dataset.image_count < 3:
        raise OSError("Images don't found")
    return my_dataset


def parse_argument(path, dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='kitti')
    parser.add_argument('--path', required=True)
    return parser.parse_args(args=('--path', path, '--dataset', dataset))
