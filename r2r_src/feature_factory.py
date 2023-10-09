import torch
import numpy as np
import pickle
import os
import json
import copy

from detect_feat_reader_vln_bert import PanoFeaturesReader_my
from utils import read_img_features


# used in detect_genome_obj_label
def get_view_labels(label_set, scanID, viewpointID):
    query = '_'.join([scanID, viewpointID])
    result = []
    for i in range(0, 36):
        query1 = query + f'_{i}'
        try:
            labels = label_set[query1]
        except:
            labels = [0]
        result.append(labels)
    return result


class feature_reader():
    def __init__(self, feat_name, root_dir=None):
        super(feature_reader, self).__init__()

        if root_dir is None:
            self.root_dir = '/home/EnvDrop_my/data/v1/features'
        else:
            self.root_dir = root_dir

        print('Loading feature: {}'.format(feat_name))

        self.feat_name = feat_name
        self.feat_data = None
        self.feat_dim = None

        if self.feat_name == 'detect_detectron':
            path = os.path.join(self.root_dir, 'detectron_1_all', 'detect_feat_all.pkl')
            with open(path, 'rb') as pklfile:
                self.feat_data = pickle.load(pklfile)
            self.feat_dim = 256

        # https://dl.dropbox.com/s/67k2vjgyjqel6og/matterport-ResNet-101-faster-rcnn-genome.lmdb.zip
        if self.feat_name == 'detect_genome_vlnbert':
            path = os.path.join(self.root_dir, 'genome', 'matterport-ResNet-101-faster-rcnn-genome.lmdb')
            self.feat_data = PanoFeaturesReader_my(
                path=path,
                in_memory=True,
            )
            self.feat_dim = 2048

        # from Room-and-Object Aware Knowledge Reasoning
        if self.feat_name == 'detect_genome_obj_label':
            path = os.path.join(self.root_dir, 'genome', 'all_labels.json')
            self.feat_data = json.load(open(path))
            self.feat_dim = 1601

        if self.feat_name == 'CLIP-ViT-B-32-views':
            path = os.path.join(self.root_dir, 'CLIP', 'CLIP-ViT-B-32-views.tsv')
            self.feat_data = read_img_features(path)
            self.feat_dim = 512

        if self.feat_name == 'region_place365':
            path = os.path.join(self.root_dir, 'scene_feat', 'ResNet-152-places365.tsv')
            self.feat_data = read_img_features(path)
            self.feat_dim = 2048

        if self.feat_name == 'region_type':
            path_house_pano_info = os.path.join(self.root_dir, 'region_cls', 'house_panos_gt.json')
            with open(path_house_pano_info, 'r') as f:
                self.feat_data = json.load(f)
            self.feat_dim = 1

    def get_feature(self, scene_id, viewpoint_id, view_id=None):
        if self.feat_name == 'detect_genome_vlnbert':
            return self.feat_data['{}-{}'.format(scene_id, viewpoint_id)]
        elif self.feat_name == 'region_type':
            return int(self.feat_data[scene_id][viewpoint_id])
        elif self.feat_name == 'detect_genome_obj_label':
            return get_view_labels(self.feat_data, scene_id, viewpoint_id)
        else:
            return self.feat_data['{}_{}'.format(scene_id, viewpoint_id)]

    def get_feat_dim(self):
        return self.feat_dim


if __name__ == '__main__':
    # , root_dir = '/home/haitian/Projects/data/Matterport3D/v1/features/'
    feat_reader = feature_reader(feat_name='region_type')
    x = feat_reader.get_feature('WYY7iVyf5p8', 'd214e451bca9470a941a260ab287d904')  # should return bathroom
    print(x)





