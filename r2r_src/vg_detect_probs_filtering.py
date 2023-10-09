import json
import copy
import pickle
import yaml
import numpy as np
import torch
from tslearn.metrics import dtw_path_from_metric
from numpy import array  # for solving the bug in eval()

from detect_feat_reader_vln_bert_v4 import PanoFeaturesReader_my_v4
import os

'''
Find out all vg classes that relate to noun words instructions
'''


def main():
    # vg detection feat
    feat_reader = PanoFeaturesReader_my_v4(
        path='./genome/matterport-ResNet-101-faster-rcnn-genome.lmdb',
        in_memory=True)

    vg_class_filtered = np.zeros(1600)

    vg_detect_probs_filtered = {}

    split = 'train'  # 'test', 'val_seen', 'val_unseen'
    source = os.path.abspath('./data/OBJ_FGR2R/data/OBJVG_FGR2R_{}.json'.format(split))

    with open(source, 'r') as f_:
        data = json.load(f_)

    total_length = len(data)

    print('Now calculating filtered classes')
    for idx, item in enumerate(data):
        obj_vg_class_list = eval(item['obj_vg_class_list'])

        for i_ins in range(len(obj_vg_class_list)):
            obj_vg_class = obj_vg_class_list[i_ins]
            for obj_cls in obj_vg_class:
                if obj_cls > -1:
                    vg_class_filtered[obj_cls] = 1

        if idx > 0 and idx % 200 == 0:
            print('{}/{} finished.'.format(idx, total_length))

    print('Filter out classes number:', vg_class_filtered.sum())

    # vg_class_path = './KB/data/entities.txt'
    # vg_class_name = []
    # with open(vg_class_path) as f:
    #     lines = f.readlines()
    #     for idx, line_ in enumerate(lines):
    #         line = line_.strip('\n')
    #         vg_class_name.append(line)
    #
    # for i in range(1600):
    #     if vg_class_filtered[i] == 1:
    #         print(vg_class_name[i])

    # use the filtered classes to read detection prob
    # in each viewpoint, the class prob is the maximum over all detections

    for split in ['train', 'val_seen', 'val_unseen']:
        print('Now caching class prob on {} split'.format(split))
        source = os.path.abspath('./datasets/FGR2R/FGR2R_{}.json'.format(split))

        with open(source, 'r') as f_:
            data = json.load(f_)

        total_length = len(data)

        for idx, item in enumerate(data):
            obs = []
            for vp in item['path']:
                obs.append({'scan': item['scan'],
                            'viewpoint': vp})

            for i, ob in enumerate(obs):
                if '{}_{}'.format(ob['scan'], ob['viewpoint']) not in vg_detect_probs_filtered:
                    probs, _, view_box_centroids, view_box_depths, view_box_areas = feat_reader[
                        '{}-{}'.format(ob['scan'], ob['viewpoint'])]
                    probs = np.asarray(probs)

                    probs_indices = probs.argmax(axis=0)
                    probs = probs.max(axis=0)
                    # print(probs.shape)
                    probs = probs[vg_class_filtered == 1]
                    view_box_centroids = view_box_centroids[probs_indices][vg_class_filtered == 1, :]
                    view_box_depths = view_box_depths[probs_indices][vg_class_filtered == 1]
                    view_box_areas = view_box_areas[probs_indices][vg_class_filtered == 1]

                    vg_detect_probs_filtered[f"{ob['scan']}_{ob['viewpoint']}"] = (
                    probs, view_box_centroids, view_box_depths, view_box_areas)

            if idx > 0 and idx % 200 == 0:
                print('{}/{} finished.'.format(idx, total_length))

    with open('data/pkls/vg_detect_probs_filtered.pkl', 'wb') as f:
        pickle.dump(vg_detect_probs_filtered, f)


if __name__ == '__main__':
    main()
