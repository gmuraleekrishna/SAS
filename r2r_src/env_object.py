''' Batched Room-to-Room navigation environment '''
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.append('buildpy36')
import MatterSim
import csv
import numpy as np
import math
import utils
import json
import random
import networkx as nx

from utils import load_nav_graphs
from param import args

csv.field_size_limit(2100000000)


# to fix the mismatch of instruction and path
# the path in the original env.py is from the shortest path which is calculated by dijkstra algorithm
# it may be different with the path in the instruction annotation json file

class EnvBatch:
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:  # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)  # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))

            # sim.setNavGraphPath('D:/MyFiles/Projects/EnvDrop_my/connectivity')

            # sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            # self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            # state = sim.getState()
            state = sim.getState()[0]

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]  # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched x).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class RoomDesc:
    def __init__(self, tok, room_desc_path='./data/R2R/annotations/room_desc.json'):
        self.vp_embedding = {}
        self.vp_names = {}
        self.words = []
        embedding_path = './data/glove_rooms.txt'
        glove_embeds = {}
        emb_size = 300
        with open(embedding_path, 'r', encoding='utf-8') as emb_file:
            for line_no, line in enumerate(emb_file):
                line = line.split(' ')
                assert len(line) == emb_size + 1, f"Invalid embedding in file {embedding_path} line {line_no}."
                now_word = line[0]
                glove_embeds[now_word] = np.array([float(e) for e in line[1:]])
        with open(room_desc_path, 'r') as rjsn:
            scan_room_desc = json.load(rjsn)
        for k, v in scan_room_desc.items():
            for vp, name in v.items():
                self.vp_embedding[vp] = torch.vstack([torch.from_numpy(glove_embeds[word])
                                                      for word in name.split(' ')
                                                      if word in glove_embeds]).mean(dim=0)

                self.vp_names[vp] = name


class R2RBatch:
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, batch_size=100, seed=10, splits=('train'), tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        self.room_desc = RoomDesc(tokenizer)

        for split in splits:
            for item in load_datasets_FGR2R([split]):
                # Split multiple instructions into separate entries

                if tokenizer:
                    if 'action_instructions' in item:
                        item['short_instr_encodings'] = [tokenizer.encode_sentence(act_instr) for act_instr in
                                                         item['action_instructions']]
                        del item['action_instructions']
                    if len(item['instructions']) > 3:
                        item['action_only_encodings'] = [tokenizer.encode_sentence(item['instructions'][-1])]
                        item['instructions'] = item['instructions'][:-1]
                    item['vp_embeddings'] = [self.room_desc.vp_embedding[vp] for vp in item['path']]
                    item['vp_names'] = [self.room_desc.vp_names[vp] for vp in item['path']]
                for j, instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:  # For fast training
                        continue

                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    sentences = instr.split('.')
                    unique_sentences = list(dict.fromkeys(sentences))
                    instr = '. '.join(unique_sentences)
                    new_item['instructions'] = instr
                    # new_item['instr_encoding'] = item['instr_encodings'][j]
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if name == 'train' and j >= 3:
                        continue

                    if name in ['train', 'aug']:
                        new_item['obj_vg_class_list'] = eval(item['obj_vg_class_list'])[
                            j] if 'obj_vg_class_list' in item else []
                        new_item['obj_word_vp_list'] = eval(item['obj_word_vp_list'])[
                            j] if 'obj_word_vp_list' in item else []
                        new_item['act_vp_list'] = eval(item['act_vp_list'])[j] if 'act_vp_list' in item else []
                        new_item['act_word_class_list'] = eval(item['act_word_class_list'])[
                            j] if 'act_word_class_list' in item else []
                        if 'subins_pos_list' not in new_item:
                            vps = [0, sum(new_item['instr_encoding'] != 0) + 1]
                            vps.extend([i for i, tok in enumerate(new_item['instr_encoding']) if tok ==
                                        self.tok.word_to_index['.']])
                            vps = list(sorted(set(vps)))
                            new_item['subins_pos_list'] = [[f, l - 1] for f, l in zip(vps, vps[1:])]
                        else:
                            try:
                                new_item['subins_pos_list'] = eval(item['subins_pos_list'])[j]
                            except TypeError:
                                new_item['subins_pos_list'] = item['subins_pos_list'][j]

                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        # self.data = self.data[:64]
        random.seed(self.seed)
        random.shuffle(self.data)
        if args.trial:
            self.data = self.data[:batch_size]
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix + batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    # original dijkstra shortest path action
    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId  # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    # actions from the json annotation
    def _annotation_action(self, state, annotation_path):
        now_pos = annotation_path.index(state.location.viewpointId)
        if now_pos + 1 >= len(annotation_path):
            return annotation_path[-1]  # Just stop here
        nextViewpointId = annotation_path[now_pos + 1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                # state = self.sim.getState()
                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId,  # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                 ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                  'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        max_obs = 0
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id': item['instr_id'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'view_ndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'candidate': candidate,
                'navigableLocations': state.navigableLocations,
                'instructions': item['instructions'],
                'teacher': self._annotation_action(state, item['path']),
                # 'teacher': self._shortest_path_action(state, item['path'][-1]),
                'path_id': item['path_id'],
                # 'new_instructions': item['new_instructions'],
                # 'chunk_view': item['chunk_view']
                'obj_vg_class_list': item.get('obj_vg_class_list', None),
                'obj_word_vp_list': item.get('obj_word_vp_list', None),
                'subins_pos_list': item.get('subins_pos_list', None),
                'action_only_encodings': item.get('action_only_encodings', None),
                'short_instr_encodings': item.get('short_instr_encodings', None),
                'vp_encodings': item.get('vp_encodings', None),
                'act_vp_list': item.get('act_vp_list', None),
                'path': item.get('path', []),
                'act_word_class_list': item.get('act_word_class_list', None),
            })
            # print(type(state.navigableLocations))
            # print(state.navigableLocations)
            # print(state.location.viewpointId, [loc.viewpointId for loc in state.navigableLocations], len(state.navigableLocations))
            if obs[-1]['teacher'] not in [can['viewpointId'] for can in obs[-1]['candidate']] and obs[-1]['teacher'] != \
                    obs[-1]['viewpoint']:
                print('Error: json path teacher action is not in candidate list.')
                print(obs[-1]['viewpoint'])
                print(item['path'])
                print([can['viewpointId'] for can in obs[-1]['candidate']])
                print(obs[-1]['teacher'])
                print(self._shortest_path_action(state, item['path'][-1]))
                exit(1)

            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:  # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:  # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:  # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


def load_datasets_FGR2R(splits):
    """

    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    import random
    data = []
    old_state = random.getstate()
    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        new_data = []
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        # if split in ['train', 'val_seen', 'val_unseen', 'test',
        #              'val_unseen_half1', 'val_unseen_half2', 'val_seen_half1', 'val_seen_half2']:       # Add two halves for sanity check
        if "/" not in split:
            if split == 'train':
                with open('data/OBJ_FGR2R/data/OBJVGv3_FGR2R_%s.json' % split) as f:
                    obj_data = json.load(f)
                    new_data.extend(obj_data)
            else:
                with open('data/R2R/annotations/R2R_%s_enc.json' % split) as f:
                    new_data = json.load(f)

        else:
            with open(split) as f:
                new_data = json.load(f)

        # Partition
        if number > 0:
            random.seed(0)  # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)  # Recover the state of the random generator
    return data
