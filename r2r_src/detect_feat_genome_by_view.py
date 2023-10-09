import pickle

import numpy as np
import json
import torch
from detect_feat_reader_vln_bert_v4 import PanoFeaturesReader_my_v4

GRAPHS = 'connectivity/'


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


def get_genome_class_glove_embed():
    embedding_path = ''
    emb_size = 300

    vg_class_path = './KB/data/entities.txt'
    vg_class = {}  # class_name: class_id

    with open(vg_class_path) as f:
        lines = f.readlines()
        for idx, line_ in enumerate(lines):
            line = line_.strip('\n')
            if line.find(',') > -1:  # multiple names for a class
                all_names = line.split(',')
                for name in all_names:
                    vg_class[name.replace(' ', '_')] = idx
            else:
                vg_class[line.replace(' ', '_')] = idx

    embeddings = {}  # class_name: class_embed
    with open(embedding_path, 'r', encoding='utf-8') as emb_file:
        for line_no, line in enumerate(emb_file):
            if line_no % 100000 == 0:
                print('Loaded line {}.'.format(line_no))
            line = line.split(' ')
            assert len(line) == emb_size + 1
            entity = line[0]
            if entity not in vg_class and entity != '<unk>':
                continue
            embeddings[entity] = torch.Tensor([float(e) for e in line[1:]])
    unknown = '<unk>'
    for entity in vg_class:
        if entity not in embeddings:
            embeddings[entity] = embeddings[unknown]

    vg_class_embed = {}  # class_id: class_embed
    for class_name, class_id in vg_class.items():
        if class_id not in vg_class_embed:
            vg_class_embed[class_id] = embeddings[class_name]

    with open('vg_class_glove_embed.pkl', 'wb') as f:
        pickle.dump(vg_class_embed, f)


def main():
    features_reader = PanoFeaturesReader_my_v4(
        path="/data/haitian/data/Matterport3D/v1/features/genome/matterport-ResNet-101-faster-rcnn-genome.lmdb",
        in_memory=True,
    )

    detect_feat_genome_by_view = dict()

    # Loop all the viewpoints in the simulator
    viewpointIds = load_viewpointids()
    cnt = 0
    for scanId, viewpointId in viewpointIds:
        viewpoint_detects = []
        for viewId in range(36):
            key = '{}-{}'.format(scanId, viewpointId)
            probs, viewids = features_reader[key]

            probs = np.asarray(probs).astype(float)
            viewids = np.asarray(viewids).astype(int)

            view_detects = []

            # for each detection, keep the top 3 prob classes
            if np.any(viewids == viewId):
                itemindex = np.argwhere(viewids == viewId)
                for idx in itemindex:
                    prob = probs[idx[0]]
                    top_3_classes = np.argsort(prob)[-3:]

                    # print(top_3_classes)
                    # print(prob[top_3_classes])

                    detect_data = np.zeros((6, )).astype(float)
                    detect_data[:3] = top_3_classes
                    detect_data[3:] = prob[top_3_classes]

                    view_detects.append(detect_data)

            viewpoint_detects.append(view_detects)

        detect_feat_genome_by_view['{}_{}'.format(scanId, viewpointId)] = viewpoint_detects

        cnt += 1
        if cnt % 100 == 0:
            print('{}/{} finished.'.format(cnt, len(viewpointIds)))

    with open('detect_feat_genome_by_view.pkl', 'wb') as f:
        pickle.dump(detect_feat_genome_by_view, f)


if __name__ == "__main__":
    # main()
    # get_genome_class_glove_embed()

    with open('vg_class_glove_embed.pkl', 'rb') as f:
        vg_class_embed = pickle.load(f)

    print(vg_class_embed[1].shape)

