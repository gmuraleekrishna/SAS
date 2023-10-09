#!/usr/bin/env python3

''' Script to precompute image features using a Caffe ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT
    and VFOV parameters. '''

import numpy as np
import cv2
import json
import math
import base64
import csv
import sys
from PIL import Image

csv.field_size_limit(sys.maxsize)

# Caffe and MatterSim need to be on the Python path
import MatterSim

import pickle
import time

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36  # Number of discretized views from one viewpoint
# BATCH_SIZE = 4  # Some fraction of viewpoint size - batch size 4 equals 11GB memory
GPU_ID = 0
IMAGENET_FEAT = 'img_features/ResNet-152-imagenet.tsv'
OUTFILE = 'img_features/Detectron_feat.pkl'
GRAPHS = 'connectivity/'

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 60

import torch


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def load_viewpointids():
    viewpointIds = []
    with open(GRAPHS + 'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS + scan + '_connectivity.json') as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print('Loaded %d viewpoints' % len(viewpointIds))
    return viewpointIds


def cvmat_to_numpy(cvmat):
    return np.array(cvmat, copy=True).astype(np.float32)


def transform_img(im):
    ''' Prep opencv 3 channel image for the network '''
    im = np.array(im, copy=True)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= np.array([[[103.1, 115.9, 123.2]]])  # BGR pixel mean
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, :, :, :] = im_orig
    blob = blob.transpose((0, 3, 1, 2))
    return blob


def build_tsv():
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    count = 0
    t_render = Timer()
    t_net = Timer()

    # Loop all the viewpoints in the simulator
    viewpointIds = load_viewpointids()
    for scanId, viewpointId in viewpointIds:
        t_render.tic()
        # Loop all discretized views from this location
        img_list = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])

            state = sim.getState()[0]
            assert state.viewIndex == ix

            # append BGR for cv2
            img_list.append(cvmat_to_numpy(state.rgb)[:, :, ::-1])

        t_render.toc()
        t_net.tic()

        for idx, img in enumerate(img_list):
            # with open('./data/v1/features/all_views_rgb/{}_{}_{}.jpg'.format(scanId, viewpointId, idx), 'wb') as f:
            # print(np.asarray(img).max(), np.asarray(img).min())  # approximately 0-255
            img_cv2 = np.asarray(img).astype(np.uint8)
            cv2.imwrite('./data/v1/features/all_views_rgb/{}_{}_{}.jpg'.format(scanId, viewpointId, idx), img_cv2)
        # exit(0)

        count += 1
        t_net.toc()
        if count % 100 == 0:
            print('Processed %d / %d viewpoints, %.1fs avg render time, %.1fs avg net time, projected %.1f hours' % \
                  (count, len(viewpointIds), t_render.average_time, t_net.average_time,
                   (t_render.average_time + t_net.average_time) * len(viewpointIds) / 3600))


def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=TSV_FIELDNAMES)
        for item in reader:
            item['scanId'] = item['scanId']
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['vfov'] = int(item['vfov'])
            item['features'] = np.frombuffer(base64.b64decode(item['features']),
                                             dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            in_data.append(item)
    return in_data


if __name__ == "__main__":
    build_tsv()


