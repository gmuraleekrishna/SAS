import copy
import json

import numpy as np
from tslearn.metrics import dtw_path_from_metric

from r2r_src.detect_feat_reader_vln_bert_v4 import PanoFeaturesReader

MAX_CLASS_DETECT = 50
MAX_ATTR_DETECT = 10
'''
Precompute alignment of object words
'''


def main():
    split = 'val_seen' #, 'val_unseen'
    source = './data/OBJ_FGR2R/data/OBJVG_FGR2R_{}.json'.format(split)
    action_source = './data/OBJ_FGR2R/data/OBJ_FGR2R_{}.json'.format(split)
    all_action_source = './data/OBJ_FGR2R/data/all_action_{}.json'.format(split)
    target = './data/OBJ_FGR2R/data/OBJVGv3_FGR2R_{}.json'.format(split)

    # vg detection feat
    feat_reader = PanoFeaturesReader(
        path='./genome/matterport-ResNet-101-faster-rcnn-genome.lmdb',
        in_memory=True)

    with open(source, 'r') as f_:
        data = json.load(f_)

    with open(action_source, 'r') as f_:
        action_data = json.load(f_)

    with open(all_action_source, 'r') as f_:
        all_action_data = json.load(f_)
    act_classes = list(all_action_data.keys())
    new_data = copy.deepcopy(data)

    total_length = len(data)

    for idx, (item, action_item) in enumerate(zip(data, action_data)):
        # print('now:', idx)
        obs = []
        for vp in item['path']:
            obs.append({'scan': item['scan'],
                        'viewpoint': vp})

        detection_conf = np.zeros((len(obs), MAX_CLASS_DETECT, 1600), dtype=np.float32)
        attr_conf = np.zeros((len(obs), MAX_ATTR_DETECT, 400), dtype=np.float32)
        areas = np.zeros((len(obs), MAX_CLASS_DETECT), dtype=np.float32)
        depths = np.zeros((len(obs), MAX_CLASS_DETECT), dtype=np.float32)
        for i, ob in enumerate(obs):
            _, probs, viewids, view_box_centroids, view_box_depths, view_box_areas, attr_probs = feat_reader['{}-{}'.format(ob['scan'], ob['viewpoint'])]
            detection_conf[i] = probs[:MAX_CLASS_DETECT, :]
            attr_conf[i] = attr_probs[:MAX_ATTR_DETECT, :]
            areas[i] = view_box_areas[:MAX_CLASS_DETECT]
            depths[i] = view_box_depths[:MAX_CLASS_DETECT]

        # detection_conf = torch.from_numpy(detection_conf).cuda()
        def get_item(li, it):
            try:
                return li.index(it)
            except ValueError:
                return -1
        act_list = eval(action_item['subins_act_list'])
        new_intrs = eval(item['new_instructions'])
        all_act_class_list = []
        for acts, intrs in zip(act_list, new_intrs):
            act_classes_list = []
            for act, intr in zip(acts, intrs):
                sub_act_classes = -1 * np.ones(len(intr), dtype=int)
                split = act.split()
                i = 0
                while i < len(intr):
                    start = get_item(intr, split[0])
                    end = get_item(intr, split[-1])
                    if start == -1 or end == -1:
                        break
                    end = end + 1
                    sub_act_classes[start: end] = act_classes.index(act)
                    i += 1
                act_classes_list.append(sub_act_classes.tolist())
            all_act_class_list.append(act_classes_list)
        obj_vg_class_list = eval(item['obj_vg_class_list'])
        obj_word_vp_list = []  # contains obj_word_vp in all 3 instructions

        for obj_vg_class in obj_vg_class_list:
            obj_word_cls = np.asarray(obj_vg_class)  # sentence annotation of word vg class
            obj_word_pos = np.where(obj_word_cls > 0)[0]  # find the position of obj words in the sentence
            if obj_word_pos.shape[0] == 0:  # no valid object words
                obj_word_vp_list.append([])
                continue

            # calculate the pairwise distance (n_viewpoint * n_obj_word)
            distance = np.zeros((len(obs), obj_word_pos.shape[0]))
            for i_obj_word, word_pos in enumerate(obj_word_pos.tolist()):
                # take max over all detections at each viewpoint
                path_cls_prob = np.max(detection_conf[:, :, obj_word_cls[word_pos]], axis=1)
                distance[:, i_obj_word] = path_cls_prob.reshape(-1)

            # a larger detection probability means a shorter matching distances, so take a negative log
            distance = -np.log(distance)

            # calculate the alignment with DTW
            align_pairs, _ = dtw_path_from_metric(distance, metric="precomputed")  # TODO: check alignment

            align = np.zeros((len(obs), obj_word_pos.shape[0]))
            for i in range(len(align_pairs)):
                align[align_pairs[i][0], align_pairs[i][1]] = 1

            # calculate the class of each detection by the argmax prob
            # detect_cls = torch.argmax(detection_conf, query_dim=2)  # (a, MAX_CLASS_DETECT)

            # for each object word, select a viewpoint with max prob (min dist) in matched viewpoint(s)
            obj_word_vp = np.zeros_like(obj_word_pos)
            for i_obj_word, word_pos in enumerate(obj_word_pos.tolist()):
                matched_vps = np.where(align[:, i_obj_word] == 1)[0]
                obj_word_vp[i_obj_word] = matched_vps[np.argmin(distance[align[:, i_obj_word] == 1, i_obj_word])]

            obj_word_vp_list.append(obj_word_vp.tolist())

        all_act_vps = []
        for views in item['chunk_view']:
            act_vp_views = []
            for i, act in enumerate(views):
                first_act = act[0]-1
                last_act = act[-1]-1
                if i != 0:
                    first_act = act_vp_views[-1][-1]
                if last_act - first_act > 1:
                    actions = list(range(first_act, last_act))
                else:
                    actions = [first_act, last_act]
                act_vp_views.append(actions)
            all_act_vps.append(act_vp_views)

        new_data[idx]['obj_word_vp_list'] = str(obj_word_vp_list)
        new_data[idx]['act_word_class_list'] = str(all_act_class_list)
        new_data[idx]['act_vp_list'] = str(all_act_vps)

        if idx > 0 and idx % 200 == 0:
            print('{}/{} finished.'.format(idx, total_length))

    with open(target, 'w') as file_:
        json.dump(new_data, file_, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
