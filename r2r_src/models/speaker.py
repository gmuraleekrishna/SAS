import json
import os
import pickle
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.losses import SequencePenaltyCriterion
from models.model_sf import SoftDotAttention
from models.pano_gcn import pano_att_gcn_v5, pano_att_gcn_v6
from models.utils import LogitsSampling
from models.weight_drop import WeightDrop
from param import args
from torch.autograd import Variable
from tslearn.metrics import dtw_path_from_metric
from numba.core.errors import NumbaWarning
import math
import warnings

warnings.simplefilter('ignore', category=NumbaWarning)

import utils
from models.pano_gcn import PanoAttentionv2

def top_k_logits(logits, k):
    """Masks everything but the k top entries as -infinity (1e10).
    From: https://github.com/huggingface/pytorch-pretrained-BERT"""
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e5, logits)

def build_viewpoint_loc_embedding(viewIndex):
    """
    Position embedding:
    heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    angle_inc = np.pi / 6.
    embedding = np.zeros((36, 128), np.float32)
    for absViewIndex in range(36):
        relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
        rel_heading = (relViewIndex % 12) * angle_inc
        rel_elevation = (relViewIndex // 12 - 1) * angle_inc
        embedding[absViewIndex, 0:32] = np.sin(rel_heading)
        embedding[absViewIndex, 32:64] = np.cos(rel_heading)
        embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
        embedding[absViewIndex, 96:] = np.cos(rel_elevation)
    return embedding


_static_loc_embeddings = [
    build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]

glove_embeddings_dict = {}
with open('r2r_src/glove_spatial.txt') as glv_file:
    for line in glv_file:
        values = line.strip().split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        glove_embeddings_dict[word] = vector

NEAR = np.array([v for k, v in glove_embeddings_dict.items() if 'near' in k]).mean(0)
CLOSE = np.array([v for k, v in glove_embeddings_dict.items() if 'close' in k]).mean(0)
FAR = np.array([v for k, v in glove_embeddings_dict.items() if 'far' in k]).mean(0)

SMALL = np.array([v for k, v in glove_embeddings_dict.items() if 'small' in k]).mean(0)
MEDIUM = np.array([v for k, v in glove_embeddings_dict.items() if 'medium' in k]).mean(0)
LARGE = np.array([v for k, v in glove_embeddings_dict.items() if 'large' in k]).mean(0)

LEFT = np.array([v for k, v in glove_embeddings_dict.items() if 'left' in k]).mean(0)
RIGHT = np.array([v for k, v in glove_embeddings_dict.items() if 'right' in k]).mean(0)
UP = np.array([v for k, v in glove_embeddings_dict.items() if 'up' in k]).mean(0)
DOWN = np.array([v for k, v in glove_embeddings_dict.items() if 'down' in k]).mean(0)
FRONT = np.array([v for k, v in glove_embeddings_dict.items() if 'front' in k]).mean(0)
BEHIND = np.array([v for k, v in glove_embeddings_dict.items() if 'behind' in k]).mean(0)


def get_distance_embeds(dlabels):
    low_upper = 1
    mid_upper = 2.5
    embeds = []
    labels = []
    for dlabel in dlabels:
        if dlabel < low_upper:
            embed = CLOSE
            label = 0
        elif low_upper < dlabel < mid_upper:
            embed = NEAR
            label = 1
        else:
            embed = FAR
            label = 2
        embeds.append(embed)
        labels.append(label)
    return torch.from_numpy(np.array(embeds)), labels


def get_size_embeds(alabels):
    low_upper = 0.1
    mid_upper = 0.3
    embeds = []
    labels = []
    for alabel in alabels:
        if alabel < low_upper:
            embed = SMALL
            label = 0
        elif low_upper < alabel < mid_upper:
            embed = MEDIUM
            label = 1
        else:
            embed = LARGE
            label = 2
        embeds.append(embed)
        labels.append(label)
    return torch.from_numpy(np.array(embeds)), labels


def get_direction_embeds(view_ndex):
    """glove directional embedding"""
    # low elevation view 0-11
    # middle elevation view 12-23
    # high elevation view 24-35

    pano_a = np.zeros((3, 12, 300), dtype=float)
    pano_a[0, :] = UP
    pano_a[2, :] = DOWN
    pano_a[:, (view_ndex - 4) % 12:(view_ndex - 2) % 12] += LEFT
    pano_a[:, (view_ndex - 2) % 12:(view_ndex + 2) % 12] += FRONT
    pano_a[:, (view_ndex + 2) % 12:view_ndex + 4] += RIGHT
    pano_a[:, (view_ndex + 4) % 12:] += BEHIND
    pano_a[:, :(view_ndex - 4) % 12] += BEHIND

    return pano_a


_direction_embeds = [get_direction_embeds(viewIndex) for viewIndex in range(0, 12)]


class Speaker:
    env_actions = {
        'left': ([0], [-1], [0]),  # left
        'right': ([0], [1], [0]),  # right
        'up': ([0], [0], [1]),  # up
        'down': ([0], [0], [-1]),  # down
        'forward': ([1], [0], [0]),  # forward
        '<end>': ([0], [0], [0]),  # <end>
        '<start>': ([0], [0], [0]),  # <start>
        '<ignore>': ([0], [0], [0])  # <ignore>
    }

    def __init__(self, env, listener, tok, args):
        self.block_ngram_repeat = 4
        self.env = env
        self.feature_size = self.env.feature_size
        self.tok = tok
        self.top_k_facts = args.hparams.get('top_k_facts', 6)
        self.top_k_attribs = args.hparams.get('top_k_attribs', 6)
        self.tok.finalize()
        self.listener = listener
        self.args = args

        # Model
        print("VOCAB_SIZE", self.tok.vocab_size())
        if args.encoder == 'pano':
            self.encoder = PanoSpeakerEncoder(self.feature_size + self.args.angle_feat_size, self.args.rnn_dim,
                                              self.args.dropout, bidirectional=self.args.bidir, args=args).cuda()
            self.decoder = PanoSpeakerDecoder(self.tok.vocab_size(), self.args.wemb, self.tok.word_to_index['<PAD>'],
                                              self.args.rnn_dim, self.args.dropout, args).cuda()
        else:
            self.encoder = SpeakerEncoder(self.feature_size + self.args.angle_feat_size, self.args.rnn_dim,
                                          self.args.dropout, bidirectional=self.args.bidir, args=args).cuda()
            self.decoder = SpeakerDecoder(self.tok.vocab_size(), self.args.wemb, self.tok.word_to_index['<PAD>'],
                                          self.args.rnn_dim, self.args.dropout, args).cuda()

        self.encoder_optimizer = self.args.optimizer(self.encoder.parameters(), lr=self.args.lr,
                                                     weight_decay=self.args.weight_decay)
        self.decoder_optimizer = self.args.optimizer(self.decoder.parameters(), lr=self.args.lr,
                                                     weight_decay=self.args.weight_decay)

        # DTW new
        # self.subins_summarizer = nn.LSTM(self.args.rnn_dim, self.args.rnn_dim, batch_first=True).cuda()

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])
        self.seq_penalty_loss = SequencePenaltyCriterion(sequence_ngram_n=5,
                                                         sequence_prefix_length=1,
                                                         sequence_candidate_type='repeat',
                                                         reduce='mean')
        # self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        # self.repetition_penalty_loss = RepetitionPenaltyLogitsProcessor(penalty=1.2)
        if args.sample_top_p:
            self.top_k_sampler = LogitsSampling(method='top_k_p', top_k=0, top_p=.9)
        elif args.sample_top_k:
            self.top_k_sampler = LogitsSampling(method='top_k', top_k=900)
        # Will be used in beam search
        self.nonreduced_softmax_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.tok.word_to_index['<PAD>'],
            size_average=False,
            reduce=False
        )

        # self.baseline_projection = nn.Sequential(
        #     nn.Linear(self.hidden_size, 128),
        #     nn.ReLU(),
        #     nn.Dropout(self.args.dropout),
        #     nn.Linear(128, 1)
        # )

        self.acc_iter = 0

        # detection feature
        with open(f"data/pkls/detect_feat_{self.top_k_facts}_{self.top_k_attribs}_genome_by_view.pkl", 'rb') as f:
            self.detect_feat_genome_by_view = pickle.load(f)

        with open('data/pkls/vg_class_glove_embed.pkl', 'rb') as f:
            self.vg_class_glove_embed = pickle.load(f)

        # self.object_proposals = self.load_obj_proposals('./data/REVERIE/annotations/BBoxes.json')

    @staticmethod
    def get_obj_glove_embeddings(embedding_path='./data/glove_objects.json', selected_strings=[]):
        glove_embeds = {}
        emb_size = 300
        emb_cnt = 0
        with open(embedding_path, 'r') as jsn:
            glove_embeds = json.load(jsn)
        obj_glove_embeds = {}
        unknown = '<unk>'
        for obj_name in selected_strings:
            obj_name_ = re.sub('[^A-Za-z0-9]+', ' ', obj_name)
            r = obj_name_.lower().split()
            mean_embeds = np.mean([embed for word, embed in glove_embeds.items() if word in r], 0)
            if np.isnan(mean_embeds).any():
                obj_glove_embeds[obj_name] = np.zeros(300)
            else:
                obj_glove_embeds[obj_name] = mean_embeds
                emb_cnt += 1

        print('embed found: {}/{}.'.format(emb_cnt, len(obj_glove_embeds)))
        return obj_glove_embeds

    def load_obj_proposals(self, bbox_file):
        obj_proposals = {}
        obj2viewpoint = {}
        with open(os.path.join(bbox_file)) as f:
            data = json.load(f)
        for scanvp, boxes in data.items():
            for objid, objinfo in boxes.items():
                if objinfo['visible_pos']:
                    # visible_pose = [((d - 12) % 12) + 12 for d in objinfo['visible_pos']]
                    visible_pose = objinfo['visible_pos']
                    if obj2viewpoint.__contains__(objinfo['name']):
                        obj2viewpoint[objinfo['name']].append(scanvp)
                    else:
                        obj2viewpoint[objinfo['name']] = [scanvp]

                    if obj_proposals.__contains__(scanvp):
                        obj_proposals[scanvp]['name'].append(objinfo['name'])
                        obj_proposals[scanvp]['visible_pos'].append(visible_pose)
                    else:
                        obj_proposals[scanvp] = {'visible_pos': [visible_pose], 'name': [objinfo['name']]}
        obj_class_list = list(obj2viewpoint.keys())
        embeddings = self.get_obj_glove_embeddings(selected_strings=obj_class_list)
        for svp, objinfos in obj_proposals.items():
            objinfos['embeddings'] = []
            objinfos['class_ids'] = []
            objinfos['vp_embeds'] = np.zeros((36, 300))
            for i, obname in enumerate(objinfos['name']):
                objinfos['embeddings'].append(embeddings[obname])
                objinfos['class_ids'].append(obj_class_list.index(obname))
                objinfos['vp_embeds'][objinfos['visible_pos'][i], :] = embeddings[obname]
        return obj_proposals

    def step_optimizer(self):
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def zero_grad_optimizers(self):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

    def train_tf(self, iters):
        for i in range(iters):
            self.env.reset()

            self.zero_grad_optimizers()
            loss = self.teacher_forcing(train=True, now_iter=self.acc_iter)

            loss.backward()
            self.step_optimizer()

            self.acc_iter += 1

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for idx in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch()  # Get the insts of the result
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.tok.shrink(inst)  # Shrink the words
        return path2inst

    def valid(self, *args, **kwargs):
        """

        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(*args, **kwargs)

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 1 if self.args.fast_train else 3  # Set the iter to 1 if the fast_train (o.w. the problem occurs)
        metrics = np.zeros(3)
        for i in range(N):
            self.env.reset()
            metrics += np.array(self.teacher_forcing(train=False))
        metrics /= N

        return path2inst, *metrics

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        def take_action(i, idx, name):
            if type(name) is int:  # Go to the next view
                self.env.env.sims[idx].makeAction([name], [0], [0])
            else:  # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            # state = self.env.env.sims[idx].getState()
            state = self.env.env.sims[idx].getState()[0]
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:  # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['view_ndex']
                trg_point = select_candidate['pointId']
                src_level = src_point // 12  # The point idx started from 0
                trg_level = trg_point // 12
                while src_level < trg_level:  # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:  # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState()[0].viewIndex != trg_point:  # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().view_ndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _teacher_action(self, obs, ended, tracker=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:  # Next view point
                        a[i] = k
                        break
                else:  # Stop here
                    assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size + self.args.angle_feat_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['feature']  # Image feat
        return torch.from_numpy(candidate_feat).cuda()

    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        """
        :param return_obj_sizes:
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs))  # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        teacher_action_view_ids = []
        detect_feats = []
        detect_labels = []
        distance_feats = []
        distance_labels = []
        area_feats = []
        area_labels = []

        all_structural_embeddings = []
        all_structural_labels = []

        first_feat = np.zeros((len(obs), self.feature_size + self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -self.args.angle_feat_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self.listener._feature_variable(obs))
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1  # Stop Action

            with torch.no_grad():
                detect_feat_batch = []
                detect_labels_batch = []
                distance_feat_batch = []
                distance_labels_batch = []
                area_feat_batch = []
                area_labels_batch = []

                all_structural_embeddings_batch = []
                all_structural_labels_batch = []

                tea_act_batch = torch.zeros(len(obs)).long()
                for i, ob in enumerate(obs):
                    if teacher_action[i] > -1:
                        teacher_action_viewid = ob['candidate'][teacher_action[i]]['pointId']
                    else:
                        # for stop action, use state.view_ndex as the view for extracting detections
                        teacher_action_viewid = ob['view_ndex']

                    tea_act_batch[i] = int(teacher_action_viewid)
                    view_detects = self.detect_feat_genome_by_view['{}_{}'.format(ob['scan'], ob['viewpoint'])]
                    glove_embeds = [self.vg_class_glove_embed[int(detect[0])] for detect in
                                    view_detects[teacher_action_viewid]]

                    all_glove_embeds = [self.vg_class_glove_embed[int(detect[0][0])] if len(detect) > 0 else
                                        torch.zeros(300) for detect in view_detects]

                    detect_label = [int(detect[0]) for detect in view_detects[teacher_action_viewid]]
                    all_detect_label = [int(detect[0][0]) if len(detect) > 0 else -1
                                        for detect in view_detects]
                    # attrib_label = [int(detect[12]) for detect in view_detects]
                    distance_feat = np.array([detect[self.top_k_facts * 2 + 2 * self.top_k_attribs + 2:
                                                     self.top_k_facts * 2 + 2 * self.top_k_attribs + 3][0].mean()
                                              for detect in view_detects[teacher_action_viewid]], dtype=np.float32)
                    all_distance_feat = [detect[0][self.top_k_facts * 2 + 2 * self.top_k_attribs + 2:
                                                   self.top_k_facts * 2 + 2 * self.top_k_attribs + 3].mean()
                                         if len(detect) > 0 else -1 for detect in view_detects]
                    area_feat = np.array([detect[self.top_k_facts * 2 + 2 * self.top_k_attribs + 3:][0].mean()
                                          for detect in view_detects[teacher_action_viewid]], dtype=np.float32)

                    all_area_feat = [detect[0][self.top_k_facts * 2 + 2 * self.top_k_attribs + 3:].mean()
                                     if len(detect) > 0 else -1 for detect in view_detects]

                    if len(glove_embeds) > 0:
                        detect_feat_batch.append(torch.stack(glove_embeds, 0).contiguous().mean(
                            dim=0))
                        emb_dis, lab_dis = get_distance_embeds(distance_feat)
                        distance_feat_batch.append(emb_dis.mean(0))
                        emb_area, lab_area = get_size_embeds(area_feat)
                        area_feat_batch.append(emb_area.mean(0))
                    else:
                        detect_feat_batch.append(torch.zeros(300))
                        distance_feat_batch.append(torch.zeros(300))
                        area_feat_batch.append(torch.zeros(300))
                        lab_dis = [-1]
                        lab_area = [-1]

                    n_labels = self.top_k_facts
                    all_glove_embeds = torch.stack(all_glove_embeds)
                    all_area_feat = torch.from_numpy(np.array(all_area_feat)).unsqueeze(1)
                    all_distance_feat = torch.from_numpy(np.array(all_distance_feat)).unsqueeze(1)
                    area_labels_ = lab_area[:n_labels] + [-1] * (n_labels - len(lab_area[:n_labels]))
                    distance_labels_ = lab_dis[:n_labels] + [-1] * (n_labels - len(lab_dis[:n_labels]))

                    all_emb_areas, lab_areas = get_size_embeds(all_area_feat)
                    all_emb_dists, lab_dists = get_distance_embeds(all_distance_feat)

                    view_angles = torch.from_numpy(_static_loc_embeddings[teacher_action_viewid])
                    all_direction_name_embeds = torch.from_numpy(_direction_embeds[teacher_action_viewid % 12]).reshape(
                        36, -1)

                    # top_3_classes[0:3], prob[3:6], centroid[6:8], box_depth[8], box_area[9]
                    detect_label_ = detect_label[:n_labels] + [-1] * (n_labels - len(detect_label[:n_labels]))
                    detect_labels_batch.append(torch.LongTensor(detect_label_))
                    area_labels_batch.append(torch.LongTensor(area_labels_))
                    distance_labels_batch.append(torch.LongTensor(distance_labels_))

                    all_structural_embeds = torch.cat([all_glove_embeds, all_direction_name_embeds, all_emb_areas,
                                                       all_emb_dists], 1)
                    all_structural_feats = torch.cat([torch.LongTensor(all_detect_label),
                                                      torch.LongTensor(lab_areas),
                                                      torch.LongTensor(lab_dists)])
                    all_structural_labels_batch.append(all_structural_feats)
                    all_structural_embeddings_batch.append(all_structural_embeds)

                teacher_action_view_ids.append(tea_act_batch)
                detect_feats.append(torch.stack(detect_feat_batch, 0).contiguous())
                detect_labels.append(torch.stack(detect_labels_batch, 0).contiguous())
                distance_feats.append(torch.stack(distance_feat_batch, 0).contiguous())
                area_feats.append(torch.stack(area_feat_batch, 0).contiguous())
                area_labels.append(torch.stack(area_labels_batch, 0).contiguous())
                distance_labels.append(torch.stack(distance_labels_batch, 0).contiguous())

                all_structural_labels.append(torch.stack(all_structural_labels_batch, 0).contiguous())
                all_structural_embeddings.append(torch.stack(all_structural_embeddings_batch, 0).contiguous())

            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        detect_feats = torch.stack(detect_feats, 1).squeeze(2).contiguous()  # batch_size, max_len, 300
        detect_labels = torch.stack(detect_labels, 1).contiguous()  # batch_size, max_len, n_labels

        area_labels = torch.stack(area_labels, 1).contiguous()  # batch_size, max_len, n_labels
        distance_labels = torch.stack(distance_labels, 1).contiguous()  # batch_size, max_len, n_labels

        distance_feats = torch.stack(distance_feats, 1).contiguous()
        area_feats = torch.stack(area_feats, 1).contiguous()

        all_structural_labels = torch.stack(all_structural_labels, 1).contiguous()  # batch_size, # 36, # n_labels
        all_structural_embeddings = torch.stack(all_structural_embeddings, 1).contiguous()

        teacher_action_view_ids = torch.stack(teacher_action_view_ids, 1).contiguous()  # batch_size, max_len

        returns = {
            'img_feats': img_feats,
            'can_feats': can_feats,
            'first_feat': first_feat.cuda(),
            'teacher_action_view_ids': teacher_action_view_ids,
            'detect_labels': detect_labels,
            'detect_feats': detect_feats.cuda(),
            'distance_feats': distance_feats.float().cuda(),
            'area_feats': area_feats.float().cuda(),
            'area_labels': area_labels.cuda(),
            'distance_labels': distance_labels.cuda(),
            'all_structural_embeddings': all_structural_embeddings.float().cuda(),
            'all_structural_labels': all_structural_labels.cuda(),
            'length': length
        }
        return returns

    # def gt_words(self, obs):
    #     """
    #     See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
    #     """
    #     seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
    #     return torch.from_numpy(seq_tensor).cuda()

    def gt_words(self, obs):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        full_instr_seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        short_insts = np.array([ob['short_instr_encodings'] for ob in obs])
        act_only_insts = np.array([ob['action_only_encodings'] for ob in obs])
        # vp_encodings = np.array([ob['vp_encodings'] for ob in obs])
        act_insts_lens = [len(insts) if insts is not None else 0 for insts in short_insts]
        # action_instr_seq_list = np.vstack([act_inst for act_inst, lens in zip(act_insts, act_insts_lens)])
        # return (torch.from_numpy(full_instr_seq_tensor).cuda(), torch.from_numpy(action_instr_seq_list).cuda(),
        #         act_insts_lens)
        return torch.from_numpy(full_instr_seq_tensor).cuda()
        # [torch.from_numpy(np.stack(act_inst)).cuda() for act_inst in short_insts], torch.from_numpy(act_only_insts).cuda(), act_insts_lens  # , torch.from_numpy(vp_encodings).cuda()

    def dtw_align_loss(self, obs, ctx, x_dtw, att_pred, path_lengths, multiview=False, teacher_action_view_ids=[]):
        """
        Align sampled representations of sub-instructions to visual feature sequence with DTW.
        x_dtw: contextualized representation of words (i.e. representation to be used in attention)
        """

        subins_pos_list_batch = [ob['subins_pos_list'] for ob in obs]

        loss_dtw_align = 0
        word_cnt = 0
        subins_cnt = 0
        loss_contrast = 0

        # h0 = torch.zeros(1, 1, self.args.rnn_dim).cuda()
        # c0 = torch.zeros(1, 1, self.args.rnn_dim).cuda()

        for i, ob in enumerate(obs):
            acc_len = 0
            summarized_subins = []
            att_pred_subins = []
            for j, subins_pos in enumerate(subins_pos_list_batch[i]):
                st, ed = subins_pos
                ed += 1

                # _, (h1, c1) = self.subins_summarizer(x_dtw[i, st:ed, :].unsqueeze(0), (h0, c0))

                # summarized_subins.append(h1.squeeze(0))

                summarized_subins.append(torch.mean(x_dtw[i, st:ed, :], dim=0).unsqueeze(0))
                acc_len += ed - st

            if len(summarized_subins) == 0:
                continue

            summarized_subins = torch.cat(summarized_subins, dim=0)  # to [n_subins, hidden_size]
            # print(summarized_subins.shape)

            # normalizing to unit norm
            summarized_subins_normed = summarized_subins / torch.norm(summarized_subins, dim=1).unsqueeze(1)

            if multiview:
                ctx_ = ctx[:, torch.arange(ctx.size(1)), teacher_action_view_ids[i], :]
            else:
                ctx_ = ctx
            ctx_normed = ctx_[i, :, :] / torch.norm(ctx_[i, :, :], dim=1).unsqueeze(1)
            ctx_normed = ctx_normed[:path_lengths[i], :]

            # print('aaa', summarized_subins_normed.shape, ctx_normed.shape)
            att_gt = DTW(summarized_subins_normed, ctx_normed, ctx_.shape[1])
            att_gt_expanded = []

            for j, subins_pos in enumerate(subins_pos_list_batch[i]):
                st, ed = subins_pos
                ed += 1
                if multiview:
                    att_pred_st_ed = att_pred[i, st:ed, torch.arange(att_pred.size(2)), teacher_action_view_ids[i]]
                else:
                    att_pred_st_ed = att_pred[i, st:ed, :, teacher_action_view_ids[i]]
                l_ = att_pred_st_ed.shape[0]

                att_pred_subins.append(att_pred_st_ed)

                with torch.no_grad():
                    att_gt_expanded.append(att_gt[j].unsqueeze(0).expand(l_, -1))

            att_pred_subins = torch.cat(att_pred_subins, dim=0)  # to [n_subins, len_path]
            with torch.no_grad():
                att_gt_expanded = torch.cat(att_gt_expanded, dim=0)  # to [b, n_subins, len_path]

            eps = 1e-6

            pos = torch.log((att_pred_subins * att_gt_expanded).sum(dim=1) + eps)
            neg = torch.log(1.0 - (att_pred_subins * (1 - att_gt_expanded)).sum(dim=1) + eps)

            loss_dtw_align += -1 * (pos.sum() + neg.sum())
            # subins_cnt += att_pred_subins.shape[0]
            word_cnt += att_pred_subins.shape[0]

            # summarized_subins_normed [n_subins, hidden_size]
            inner_products = torch.matmul(summarized_subins_normed, ctx_normed.permute(1, 0))
            inner_products = torch.exp(inner_products * (512 ** -0.5))

            # inner_products [n_subins, max_path_len]
            pos_pair = torch.masked_select(inner_products, att_gt[:, :path_lengths[i]].eq(1))
            neg_pair = torch.masked_select(inner_products, att_gt[:, :path_lengths[i]].eq(0))

            if len(pos_pair) > 0 and len(neg_pair) > 0:
                pos_sum = pos_pair.sum()
                neg_sum = neg_pair.sum()
                loss_contrast += (-1 * torch.log((pos_sum + 1e-5) / (pos_sum + neg_sum)))
                subins_cnt += att_gt.shape[0]

        loss_dtw_align /= subins_cnt
        loss_contrast /= subins_cnt

        return loss_dtw_align

    def action_align_loss(self, ob, ctx, x_dtw, att_pred, path_lengths):
        """
        Align sampled representations of sub-instructions to visual feature sequence with DTW.
        x_dtw: contextualized representation of words (i.e. representation to be used in attention)
        """

        subins_pos_list = ob['subins_pos_list']

        loss_dtw_align = 0
        word_cnt = 0
        subins_cnt = 0
        loss_contrast = 0

        summarized_subins = []
        att_pred_subins = []
        summarized_subins = torch.mean(x_dtw, dim=1)

        eps = 1e-6
        # normalizing to unit norm
        summarized_subins_normed = summarized_subins / torch.norm(summarized_subins, dim=0)
        ctx_normed = (ctx / torch.norm(ctx + eps, dim=1)).squeeze(0)

        # print('aaa', summarized_subins_normed.shape, ctx_normed.shape)
        att_gt = DTW(summarized_subins_normed, ctx_normed, ctx.shape[1])
        att_gt_expanded = []

        l_ = att_pred.shape[1]
        with torch.no_grad():
            att_gt_expanded = att_gt.unsqueeze(1).expand(path_lengths, l_, -1)

        pos = torch.log((att_pred * att_gt_expanded).sum(dim=0) + eps)
        neg = torch.log(1.0 - (att_pred * (1 - att_gt_expanded)).sum(dim=0) + eps)

        loss_dtw_align = -1 * (pos.sum() + neg.sum())

        # summarized_subins_normed [n_subins, hidden_size]
        inner_products = torch.matmul(summarized_subins_normed, ctx_normed.permute(1, 0))
        inner_products = torch.exp(inner_products * (512 ** -0.5))

        # inner_products [n_subins, max_path_len]
        pos_pair = torch.masked_select(inner_products, att_gt[:path_lengths].eq(1))
        neg_pair = torch.masked_select(inner_products, att_gt[:path_lengths].eq(0))

        if len(pos_pair) > 0 and len(neg_pair) > 0:
            pos_sum = pos_pair.sum()
            neg_sum = neg_pair.sum()
            loss_contrast = (-1 * torch.log((pos_sum + 1e-5) / (pos_sum + neg_sum)))
            subins_cnt = att_gt.shape[0]

        loss_dtw_align /= subins_cnt
        loss_contrast /= subins_cnt

        return loss_dtw_align, loss_contrast

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def teacher_forcing(self, train=True, features=None, insts=None, reduce=False, now_iter=None):
        if train:
            self.train()
        else:
            self.eval()

        # Get Image Input & Encode
        if features is not None:
            # It is used in calulating the speaker score in beam-search
            assert insts is not None
            (img_feats, can_feats), lengths = features
            ctx = self.encoder(can_feats, img_feats, lengths)
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            short_outs = self.from_shortest_path()  # (distance_feats, area_feats)
            ctx = self.encoder(short_outs['can_feats'], short_outs['img_feats'],
                               short_outs['teacher_action_view_ids'], short_outs['detect_labels'],
                               short_outs['detect_feats'], short_outs['distance_feats'],
                               short_outs['distance_labels'], short_outs['area_feats'],
                               short_outs['area_labels'], short_outs['all_structural_embeddings'],
                               short_outs['all_structural_labels'], short_outs['length'])

        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ctx_mask = utils.length2mask(short_outs['length'])

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)  # Language short_acts_insts, act_only_insts, act_inst_lens

        # vp_names
        short_loss = 0.
        action_align_loss = 0.
        contrast_align_loss = 0.
        # For attention loss
        if train:
            # Decode
            logits, _, _, attn, x_dtw = self.decoder(insts, ctx, ctx_mask, h_t,
                                                     c_t, train=True)
        else:
            # Decode
            logits, _, _ = self.decoder(insts, ctx, ctx_mask, h_t, c_t)
        # Because the softmax_loss only allow query_dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1).contiguous()
        # summary_logits = summary.permute(0, 2, 1).contiguous()
        if args.sample_top_k:
            logits = self.top_k_sampler.sample(logits)

        loss = self.softmax_loss(
            input=logits[:, :, :-1],  # -1 for aligning
            target=insts[:, 1:]  # "1:" to ignore the word <BOS>
        )

        # loss += args.hparams.get('l_sum', 0.) * summary_loss

        if train:
            if now_iter is not None and now_iter >= int(self.args.hparams['warmup_iter']):
                sp_loss = self.seq_penalty_loss(logits)
                sp_loss = args.hparams.get('l_sp', 0.) * sp_loss
                loss_att = self.dtw_align_loss(obs, ctx, x_dtw,
                                               attn, short_outs['length'], self.args.encoder == 'pano',
                                               short_outs['teacher_action_view_ids']
                                               )
                # print('loss check', loss, loss_att)  # 6.9075, 2.3872
                loss_att *= self.args.hparams.get('w_loss_att', 0.)
                loss_act_att = self.args.hparams.get('l_act_ali', 0.) * action_align_loss
                loss_contrast = self.args.hparams.get('l_contrast', 0.) * contrast_align_loss
                loss += loss_att + loss_act_att + sp_loss + loss_contrast

        if reduce:
            return self.nonreduced_softmax_loss(
                input=logits[:, :, :-1],  # -1 for aligning
                target=insts[:, 1:]  # "1:" to ignore the word <BOS>
            )

        if train:
            return loss
        else:
            # Evaluation
            _, predict = logits.max(dim=1)  # BATCH, LENGTH
            gt_mask = (insts != self.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]  # Not pad and equal to gt
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()  # Exclude <BOS>
            sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
            return loss.item(), word_accu, sent_accu

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, query_dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.train()
        else:
            self.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        short_outs = self.from_shortest_path(viewpoints=viewpoints_list)
        # Image Feature (from the shortest path)

        # This code block is only used for the featdrop.
        if featdropmask is not None:
            short_outs['img_feats'][..., :-self.args.angle_feat_size] *= featdropmask
            short_outs['can_feats'][..., :-self.args.angle_feat_size] *= featdropmask

        # Encoder
        ctx = self.encoder(short_outs['can_feats'], short_outs['img_feats'],
                           short_outs['teacher_action_view_ids'], short_outs['detect_labels'],
                           short_outs['detect_feats'], short_outs['distance_feats'],
                           short_outs['distance_labels'], short_outs['area_feats'],
                           short_outs['area_labels'], short_outs['all_structural_embeddings'],
                           short_outs['all_structural_labels'], short_outs['length'],
                           already_dropfeat=(featdropmask is not None))
        ctx_mask = utils.length2mask(short_outs['length'])

        # Decoder
        words = []
        log_probs = []
        hidden_states = []
        entropies = []
        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ended = np.zeros(len(obs), np.bool)
        word = np.ones(len(obs), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
        word = torch.from_numpy(word).view(-1, 1).cuda()
        for i in range(self.args.maxDecode):
            # Decode Step
            logits, h_t, c_t = self.decoder(word, ctx, ctx_mask, h_t, c_t)  # Decode, logits: (b, 1, vocab_size)

            # Select the word
            logits = logits.squeeze()  # logits: (b, vocab_size)
            logits[:, self.tok.word_to_index['<UNK>']] = -float("inf")  # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    hidden_states.append(h_t.squeeze())
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    hidden_states.append(h_t.squeeze().detach())
                    entropies.append(m.entropy().detach())
            else:
                values, word = logits.max(1)

            # Append the word
            cpu_word = word.cpu().numpy()
            cpu_word[ended] = self.tok.word_to_index['<PAD>']
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break

        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(hidden_states, 1), torch.stack(entropies,
                                                                                                             1)
        else:
            return np.stack(words, 1)  # [(b), (b), (b), ...] --> [b, l]

    def save(self, path='', epoch=None):
        """ Snapshot models """
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        if epoch is None:
            epoch_ = ''
        else:
            epoch_ = epoch + 1

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch_,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state, strict=False)
            if self.args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])

        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        # return states['encoder']['epoch'] - 1

    def rl_train(self, reward_func, iters, ml_weight=0., policy_weight=1., baseline_weight=.5, entropy_weight=0.,
                 self_critical=False, ml_env=None):
        """
        :param reward_func: A function takes the [(path, inst)] list as x, returns the reward for each inst
        :param iters:       Train how many iters
        :param ml_weight:   weight for maximum likelihood
        :param policy_weight:   weight for policy loss
        :param baseline_weight: weight for critic loss (baseline loss)
        :param entropy_weight:  weight for the entropy
        :param self_critical: Use the self_critical baseline
        :param ml_env:        Specific env for ml (in case that the train_env is aug_env)
        :return:
        """
        from collections import defaultdict
        log_dict = defaultdict(lambda: 0)
        for itr in (range(iters)):
            joint_loss = 0.
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            # Reset Env
            if args.same_in_batch:
                self.env.reset(tile_one=True)
            else:
                self.env.reset()
            rl_batch = self.env.batch

            # RL training
            insts, log_probs, hiddens, entropies = self.infer_batch(sampling=True, train=True)  # Sample a batch

            # Get the Reward ( and the length, mask)
            path_ids = [ob['path_id'] for ob in self.env._get_obs()]  # Gather the path ids
            path2inst = {path_id: self.tok.shrink(inst) for path_id, inst in zip(path_ids, insts)}
            reward = reward_func(rl_batch, path2inst)  # The reward func will evaluate the instruction
            reward = torch.FloatTensor(reward).cuda()
            length = np.argmax(np.array(insts) == self.tok.word_to_index['<EOS>'], 1) + 1  # Get length (pos of EOS) + 1
            length[length == 1] = insts.shape[1]  # If there is no EOS, change the length to max length.
            mask = 1. - utils.length2mask(length).float()

            # Get the baseline
            if args.normalize_reward:
                baseline = reward.mean()
            else:
                if self_critical:
                    self.env.reset(rl_batch)
                    insts = self.infer_batch(sampling=False, train=False)  # Argmax Decoding
                    pathXinst = {path_id: self.tok.shrink(inst) for path_id, inst in zip(path_ids, insts)}
                    baseline = reward_func(rl_batch, pathXinst)  # The reward func will evaluate the instruction
                    baseline = torch.FloatTensor(baseline).cuda().unsqueeze(1)
                else:
                    baseline_hiddens = hiddens if args.grad_baseline else hiddens.detach()
                    baseline = self.decoder.baseline_projection(baseline_hiddens).squeeze()

            # print("Reward Mean %0.4f, std %0.4f" % (reward.mean().detach().cpu().item(), reward.std().detach().cpu().item()))
            # print("Baseline Mean %0.4f, std %0.4f" % (baseline.mean().detach().cpu().item(), baseline.std().detach().cpu().item()))
            # print("Avg abs(Reward - Baseline): %0.4f" % (torch.abs(reward - baseline).mean().detach().cpu().item()))

            # Calculating the Loss
            reward = reward.unsqueeze(1)  # (batch_size,) --> (batch_size, 1)

            if args.normalize_reward:  # Normalize the reward to mean 0, std 1
                advantage = (reward - baseline) / reward.std() * 0.2
            else:
                advantage = reward - baseline

            policy_loss = (advantage.detach() * (-log_probs) * mask).sum()
            policy_loss /= self.env.batch_size  # Normalized by the batch_size
            baseline_loss = (advantage ** 2 * mask).sum() / self.env.batch_size
            avg_entropy = (entropies * mask).sum() / self.env.batch_size

            # Add the Loss to the joint_loss
            if baseline_weight != 0.:  # To support the pretrain phase
                joint_loss += baseline_loss * baseline_weight

            if policy_weight != 0.:  # To support the finetune phase
                joint_loss += policy_loss * policy_weight

            if entropy_weight != 0.:  # Note that the negative entrop is added to encourage exploration
                joint_loss += - avg_entropy * entropy_weight

            # ML env preparation
            if ml_env is not None:  # Get the env from ml_env
                old_env = self.env
                self.env = ml_env
                self.env.reset()
            else:  # else reset the same env as RL
                self.env.reset(batch=rl_batch)

            # ML Training
            assert ml_weight != 0  # Because I always log the ml_weight. And it should always exists!
            if ml_weight != 0.:
                ml_loss = self.teacher_forcing(train=True, now_iter=itr)
                joint_loss += ml_loss * ml_weight
            else:
                ml_loss = 0.

            if ml_env is not None:
                self.env = old_env

            # print("Reward Mean %0.4f, std %0.4f" % (reward.mean().detach().cpu().item(), reward.std().detach().cpu().item()))
            # print("Baseline Mean %0.4f, std %0.4f" % (baseline.mean().detach().cpu().item(), baseline.std().detach().cpu().item()))
            # print("Avg abs(Reward - Baseline): %0.4f" % (torch.abs(reward - baseline).mean().detach().cpu().item()))

            # Log:
            for name, loss in (('baseline_loss', baseline_loss.detach().item()),
                               ('policy_loss', policy_loss.detach().item()),
                               ('ml_loss', ml_loss.detach().item()),
                               ('baseline', baseline.mean().detach().item()),
                               ('reward', reward.mean().detach().item()),
                               ('baseline_std', baseline.std().detach().item()),
                               ('reward_std', reward.std().detach().item()),
                               ('reward_diff', torch.abs(reward - baseline).mean().detach().item()),
                               ('entropy', avg_entropy.item())):
                log_dict[name] += loss

            # Backward
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
        return log_dict

    def sample(self, obs, sample_max, rl_training=False, ml_training=False, auxiliary=False, pad=False, tf=False):
        # Image Input for the Encoder
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        short_outs = self.from_shortest_path(viewpoints=viewpoints_list)
        # Image Feature (from the shortest path)

        ctx = self.encoder(short_outs['can_feats'], short_outs['img_feats'],
                           short_outs['teacher_action_view_ids'], short_outs['detect_labels'],
                           short_outs['detect_feats'], short_outs['distance_feats'],
                           short_outs['distance_labels'], short_outs['area_feats'],
                           short_outs['area_labels'], short_outs['all_structural_embeddings'],
                           short_outs['all_structural_labels'], short_outs['length'])

        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ctx_mask = utils.length2mask(short_outs['length'])

        # Get Language Input
        insts = self.gt_words(obs)  # Language short_acts_insts, act_only_insts, act_inst_lens

        seq = []
        seq_log_probs = []
        attns = []
        x_dtws = []
        logits = []

        if rl_training:
            baseline = []
        # For attention loss
        last_word = np.ones((len(obs), 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
        last_word = torch.from_numpy(last_word).view(-1, 1).cuda()
        ended = np.zeros(len(obs), np.bool)
        hiddens = []
        for t in range(args.maxDecode):
            if not tf:
                last_word = Variable(last_word)
            else:
                last_word = insts[:, t].unsqueeze(1)
            if ml_training:
                logit, h_t, c_t, attn, x_dtw = self.decoder(last_word, ctx, ctx_mask, h_t, c_t, True)
                attns.append(attn)
                x_dtws.append(x_dtw)
            else:
                logit, h_t, c_t = self.decoder(last_word, ctx, ctx_mask, h_t, c_t)
            logit = logit.squeeze()
            if not rl_training and not ml_training:
                logit[:, self.tok.word_to_index['<UNK>']] = -float("inf")
            logits.append(logit)
            log_probs = F.log_softmax(logit, dim=1)

            if sample_max:
                values, last_word = logit.max(1)
            else:
                probs = F.softmax(logit, -1)
                m = torch.distributions.Categorical(probs)
                last_word = m.sample()
                log_probs = m.log_prob(last_word)

            cpu_word = last_word.cpu().numpy()
            if pad:
                cpu_word[ended] = self.tok.word_to_index['<PAD>']
            seq.append(cpu_word)  # seq[t] the x of t time step
            seq_log_probs.append(log_probs)
            last_word = last_word.view(-1, 1)
            hiddens.append(h_t)
            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.word_to_index['<EOS>'])
            # if ended.all():
            #     break
            # if t < 6:
            #     mask = np.zeros((batch_size, log_probs.size(-1)), 'float32')
            #     mask[:, 0] = -1000
            #     mask = Variable(torch.from_numpy(mask)).cuda()
            #     log_probs = log_probs + mask
            #
            # if sample_max:
            #     sample_log_prob, last_word = torch.max(log_probs, 1)
            #     last_word = last_word.data.view(-1).long()
            # else:
            #     # fetch prev distribution: shape Nx(M+1)
            #     prob_prev = torch.exp(log_probs.data).cpu()
            #     last_word = torch.multinomial(prob_prev, 1).cuda()
            #     # gather the logprobs at sampled positions
            #     sample_log_prob = log_probs.gather(1, Variable(last_word))
            #     # flatten indices for downstream processing
            #     last_word = last_word.view(-1).long()
            #
            # if t == 0:
            #     unfinished = last_word > 0
            # else:
            #     unfinished = unfinished * (last_word > 0)
            # if unfinished.sum() == 0 and t >= 1 and not pad:
            #     break
            # last_word = last_word * unfinished.type_as(last_word)
            # last_word = last_word.unsqueeze(1)
            # seq.append(last_word)  # seq[t] the x of t time step
            # seq_log_probs.append(sample_log_prob.view(-1))



        # concatenate output lists
        seq = torch.from_numpy(np.array(seq).T)  # batch_size * 5, seq_length
        seq_log_probs = torch.cat([_.unsqueeze(1) for _ in seq_log_probs], 1)
        seq = seq.view(-1, 1, seq.size(1))
        seq_log_probs = seq_log_probs.view(-1, 1, seq_log_probs.size(1))

        # Because the softmax_loss only allow query_dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        # logits = logits.permute(0, 2, 1).contiguous()
        # logits[:, self.tok.word_to_index['<UNK>']] = -float("inf")

        if rl_training:
            hiddens = torch.cat([_.unsqueeze(1) for _ in hiddens], 1)
            baseline = self.decoder.baseline_projection(hiddens.detach())
            baseline = baseline.view(-1, 1, baseline.size(1))
            returns = (seq, seq_log_probs, baseline)
        else:
            returns = (seq, seq_log_probs)

        if ml_training:
            x_dtws = torch.stack(x_dtws, 1).squeeze(2)  # batch, seq_len, embed
            attns = torch.stack(attns, 1).squeeze(2)  # batch, seq_len, 7, 36
            logits = torch.stack(logits, 1)
            logits = logits.permute(0, 2, 1)
            tf_loss = self.softmax_loss(
                input=logits[:, :, :-1],  # -1 for aligning
                target=insts[:, 1:]  # "1:" to ignore the word <BOS>
            )
            if auxiliary:
                sp_loss = self.seq_penalty_loss(logits)
                loss_dtw = self.dtw_align_loss(obs, ctx, x_dtws,
                                               attns, short_outs['length'], self.args.encoder == 'pano',
                                               short_outs['teacher_action_view_ids'])
                sp_loss *= args.hparams.get('l_sp', 0.)
                loss_dtw *= args.hparams.get('w_loss_att', 0.)
                tf_loss = loss_dtw + sp_loss
            return returns + (tf_loss,)
        else:
            return returns

    def _sample(self, obs, sample_max, rl_training=False, ml_train=False, auxiliary=False, pad=False, tf=False):
        # Image Input for the Encoder
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        short_outs = self.from_shortest_path(viewpoints=viewpoints_list)
        # Image Feature (from the shortest path)

        ctx = self.encoder(short_outs['can_feats'], short_outs['img_feats'],
                           short_outs['teacher_action_view_ids'], short_outs['detect_labels'],
                           short_outs['detect_feats'], short_outs['distance_feats'],
                           short_outs['distance_labels'], short_outs['area_feats'],
                           short_outs['area_labels'], short_outs['all_structural_embeddings'],
                           short_outs['all_structural_labels'], short_outs['length'])

        # Get Language Input
        insts = self.gt_words(obs)  # Language short_acts_insts, act_only_insts, act_inst_lens

        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ctx_mask = utils.length2mask(short_outs['length'])

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)  # Language short_acts_insts, act_only_insts, act_inst_lens

        if tf:
            # Decode
            logits, _, _, attn, x_dtw = self.decoder(insts, ctx, ctx_mask, h_t,
                                                     c_t, train=True)
        else:
            # Decode
            logits, _, _ = self.decoder(insts, ctx, ctx_mask, h_t, c_t)
        log_probs = F.log_softmax(logits, dim=1)

        if sample_max:
            log_prob, seq = torch.max(log_probs, 1)
        else:
            probs = F.softmax(logits, -1)
            m = torch.distributions.Categorical(probs)
            word = m.sample()
            log_prob = m.log_prob(word)

        if rl_training:
            # cut off the gradient passing using detech()
            baseline = self.decoder.baseline_projection(h_t.detach())

        logits[:, self.tok.word_to_index['<UNK>']] = -float("inf")

        if rl_training:
            baseline = torch.cat([_.unsqueeze(1) for _ in baseline], 1)
            baseline = baseline.view(-1, 1, baseline.size(1))
            returns = (seq, seq_log_probs, baseline)
        else:
            returns = (seq, seq_log_probs)

        if ml_train:
            x_dtws = torch.stack(x_dtws, 1).squeeze(2)  # batch, seq_len, embed
            attns = torch.stack(attns, 1).squeeze(2)  # batch, seq_len, 7, 36
            logits = torch.stack(logits, 1)
            logits = logits.permute(0, 2, 1)
            tf_loss = self.softmax_loss(
                input=logits[:, :, :-1],  # -1 for aligning
                target=insts[:, 1:]  # "1:" to ignore the word <BOS>
            )
            if auxiliary:
                sp_loss = self.seq_penalty_loss(logits)
                loss_dtw = self.dtw_align_loss(obs, ctx, x_dtws,
                                               attns, short_outs['length'], self.args.encoder == 'pano',
                                               short_outs['teacher_action_view_ids'])
                sp_loss *= args.hparams.get('l_sp', 0.)
                loss_dtw *= args.hparams.get('w_loss_att', 0.)
                tf_loss = loss_dtw + sp_loss
            return returns + (tf_loss,)
        else:
            return returns


    def rl_training(self, reward_func, ml_w=0.05, rl_w=1., e_w=0.005):
        loss = 0.

        # RL_training
        # Sampling a batch
        insts, log_probs, entropies, hiddens = self.infer_batch(
            sampling=True, train=True
        )
        mask = torch.from_numpy((insts[:, 1:] != self.tok.word_to_index['<PAD>']).astype(np.float32)).cuda()
        tokens = mask.sum()  # Dominate by tokens just like ML

        # Get reward
        obs = self.env._get_obs()
        path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
        path2inst = {}
        for path_id, inst in zip(path_ids, insts):
            if path_id not in path2inst:
                path2inst[path_id] = self.tok.shrink(inst)
        batch_reward = reward_func(path2inst)
        batch_reward = torch.from_numpy(batch_reward).cuda()

        # Get baseline
        if args.baseline == 'none':
            baseline = 0.
        elif args.baseline == 'self':
            max_insts = self.infer_batch(
                sampling=False, train=False
            )
            path2inst = {}
            for path_id, inst in zip(path_ids, max_insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.tok.shrink(inst)
            max_reward = reward_func(path2inst)
            baseline = torch.from_numpy(max_reward).unsqueeze(1).cuda()
        elif args.baseline == 'linear':
            baseline = self.decoder.baseline_projection(hiddens.detach()).squeeze(2)
            loss += 0.5 * (((baseline - batch_reward.unsqueeze(1)) ** 2) * mask).sum() / tokens
        else:
            assert False

        # Calculate loss
        entropy = (- entropies * mask).sum() / tokens
        rl_loss = ((batch_reward.unsqueeze(1) - baseline.detach()) * (- log_probs) * mask).sum() / tokens

        # ML_training
        ml_loss, word_accu, sent_accu = self.teacher_forcing(train=True)

        loss += ml_w * ml_loss + rl_w * rl_loss + e_w * entropy
        return loss, word_accu, sent_accu


class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional, args):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size
        self.args = args

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        # self.attention_layer = model.SoftDotAttention(self.hidden_size, feature_size)
        # self.attention_layer = SlotKnowledgeAttention(1, self.hidden_size, feature_size, knowledge_dim=900,
        #                                                   iters=3, eps=1e-8,
        #                                                   hidden_dim=768, drop_rate=args.featdropout)
        # elif args.attn == 'pano':

        self.attention_layer = pano_att_gcn_v6(self.hidden_size, feature_size, knowledge_dim=900)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

        self.knowledge_module = Entity_Knowledge(fact_dropout=self.args.hparams['fact_dropout'],
                                                 top_k_facts=self.args.hparams['top_k_facts'])
        self.detection_projection = nn.Linear(300 + 1 + 1, 300)

    def forward(self, action_embeds, img_feats, teacher_action_view_ids, detect_labels, detect_feats, distance_feats,
                distance_labels, area_feats, area_labels, all_structural_embeddings, all_structural_labels, lengths,
                already_dropfeat=False):

        x = action_embeds
        if not already_dropfeat:
            x[..., :-self.args.angle_feat_size] = self.drop3(
                x[..., :-self.args.angle_feat_size])  # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)
        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            img_feats[..., :-self.args.angle_feat_size] = self.drop3(
                img_feats[..., :-self.args.angle_feat_size])  # Dropout the image feature

        knowledge = self.knowledge_module(detect_labels)
        knowledge = torch.cat([knowledge, area_labels, distance_labels], -1)
        detect_feats = torch.cat([detect_feats, area_feats, distance_feats], -1)

        x, _ = self.attention_layer(  # Attend to the feature map
            ctx.contiguous(),  # (batch, length, hidden) --> (batch x length, hidden)
            img_feats,  # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
            detect_feats,  # (batch, length, 900) -> (batch * length, 900)
            knowledge,
            teacher_action_view_ids  # (batch, length) -> (batch * length)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x


class PanoSpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional, args):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size
        self.args = args

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        # self.attention_layer = model.SoftDotAttention(self.hidden_size, feature_size)
        self.attention_layer = PanoAttentionv2(self.hidden_size, feature_size, knowledge_dim=900)  # PanoAttention()

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.structural_module = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1), padding=0, bias=True)
        self.knowledge_module = PanoEntityKnowledge(fact_dropout=self.args.hparams['fact_dropout'],
                                                    top_k_facts=self.args.hparams['top_k_facts'])
        self.detection_projection = nn.Linear(300 + 1 + 1, 300)

    def forward(self, action_embeds, img_feats, teacher_action_view_ids, detect_labels,
                detect_feats, distance_feats, area_feats, distance_labels, area_labels, all_glove_embeddings,
                all_detect_labels, lengths,
                already_dropfeat=False):
        x = action_embeds
        if not already_dropfeat:
            x[..., :-self.args.angle_feat_size] = self.drop3(
                x[..., :-self.args.angle_feat_size])  # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            img_feats[..., :-self.args.angle_feat_size] = self.drop3(
                img_feats[..., :-self.args.angle_feat_size])  # Dropout the image feature

        knowledge = self.knowledge_module(all_detect_labels[:, :, :36])
        structural_labels = all_detect_labels[:, :, 36:].reshape(batch_size, max_length, -1, 2)
        structural_labels = structural_labels.permute(0, 3, 1, 2).float()
        structural_embeds = self.structural_module(structural_labels).permute(0, 2, 1, 3).squeeze(2)
        x, _ = self.attention_layer(  # Attend to the feature map
            ctx.contiguous(),  # (batch, length, hidden) --> (batch x length, hidden)
            img_feats,
            all_glove_embeddings,
            knowledge,
            structural_embeds,
            teacher_action_view_ids,
        )
        x = x.view(batch_size, max_length * 36, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        x = x.view(batch_size, max_length, 36, -1)
        return x


class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio, args):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.args = args

        # self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.lstm = WeightDrop(nn.LSTM(embedding_size, hidden_size, batch_first=False), ['weight_hh_l0'],
                               dropout=self.args.hparams['weight_drop'])

        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)

        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def inst_embed(self, words):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        return embeds
    def _sample_topp(self, probs, sampling_topp):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.
        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.
        Args:
            probs: (bsz x input_beam_size x vocab_size)  IK: here we dont have beam ! so bsz x vocab_size
                the model's log-probabilities over the vocabulary at the current step
        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        # sort the last dimension (vocab dimension) in descending order
        sorted_probs, sorted_indices = probs.sort(descending=True)

        # compute a mask to indicate the words to be included in the top-P set.
        cumsum_probs = sorted_probs.cumsum(dim=1)
        mask = cumsum_probs.lt(sampling_topp)

        # note that mask was computed by 'lt'. One more word needs to be included
        # so that the cumulative probability mass can exceed p.
        cumsum_mask = mask.cumsum(dim=1)
        last_included = cumsum_mask[:, -1:]
        last_included.clamp_(0, mask.size()[1] - 1)
        mask = mask.scatter_(1, last_included, 1)

        # truncate unnecessary dims.
        max_dim = last_included.max()
        truncated_mask = mask[:, :max_dim + 1]
        truncated_probs = sorted_probs[:, :max_dim + 1]
        truncated_indices = sorted_indices[:, :max_dim + 1]

        # trim the words that are not in top-P by setting their probabilities
        # to 0, so that they would not be sampled later.
        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
        return trimed_probs, truncated_indices

    def _topk_deterministic_decode(self, K, logits):
        top1_indexs = torch.topk(logits, K)[1][:, 0].view(-1, 1)
        topk_indexs = torch.topk(logits, K)[1][:, K - 1].view(-1, 1)
        probs = torch.softmax(logits, -1)
        return topk_indexs, torch.gather(probs, index=topk_indexs, dim=1), top1_indexs, torch.gather(probs,
                                                                                                     index=top1_indexs,
                                                                                                     dim=1)

    def _topk_decode(self, logits, topk, topp, return_prob=False):
        """WARNING!!! This can modify the `self.pad` position of `logits`."""
        logits[:, self.pad] = -math.inf
        if topk == 1 and topp == 0:  # greedy
            pred_tok = logits.argmax(dim=1, keepdim=True)

        else:
            if topk > 1:
                logits[:, self.pad] = -1e10  # never select pad
                logits = top_k_logits(logits, topk)
                pred_tok = torch.softmax(logits, -1).multinomial(1)
            else:
                assert topp > 0.0
                filtered_probs, bookkeep_idx = self._sample_topp(torch.softmax(logits, 1), sampling_topp=topp)
                selected = filtered_probs.multinomial(1)
                pred_tok = torch.gather(bookkeep_idx, index=selected, dim=1)
        if return_prob:
            return pred_tok, torch.gather(torch.softmax(logits, -1), index=pred_tok, dim=1)
        return pred_tok

    def get_token_similarity(self, tokens):
        # tokens: B * T
        tok_embeddings = self.decoder.inst_embed(tokens)
        similarity_matrix = F.cosine_similarity(tok_embeddings[..., None, :, :], tok_embeddings[..., :, None, :],
                                                dim=-1)
        ad_sim = torch.diagonal(similarity_matrix, offset=1, dim1=1, dim2=2)
        return ad_sim

    def forward(self, words, ctx, ctx_mask, h0, c0, train=False):
        embeds = self.inst_embed(words)
        embeds = embeds.permute(1, 0, 2)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        # print(x.shape)
        x = x.permute(1, 0, 2)

        x = self.drop(x)

        # Keep x here for DTW align loss
        x_dtw = x

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)  # By using this, it also supports the beam-search

        # Att and Handle with the shape
        x = x.contiguous().view(batchXlength, self.hidden_size)
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        ctx = ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1, self.hidden_size)
        # Expand the ctx_ from  (b, a, r)    --> (b(word)*l(word), a, r)
        ctx_mask = ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, attn = self.attention_layer(
            x,
            ctx,
            mask=ctx_mask
        )  # (x - batchXlength, self.hidden_size), (attn - batchXlength, seqlen)
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        if train:
            attn = attn.reshape(words.size(0), words.size(1), -1)
            return logit, h1, c1, attn, x_dtw

        return logit, h1, c1


class PanoSpeakerDecoder(SpeakerDecoder):

    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio, args):
        super(PanoSpeakerDecoder, self).__init__(vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio, args)
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.args = args

        # self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.lstm = WeightDrop(nn.LSTM(embedding_size, hidden_size, batch_first=False), ['weight_hh_l0'],
                               dropout=self.args.hparams['weight_drop'])

        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.summarizer = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )


    def forward(self, words, ctx, ctx_mask, h0, c0, train=False):
        embeds = self.inst_embed(words)
        embeds = embeds.permute(1, 0, 2)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        # print(x.shape)
        x = x.permute(1, 0, 2)
        x = self.drop(x)

        ctx_mask = ctx_mask.unsqueeze(2).repeat(1, 1, 36)
        # Keep x here for DTW align loss
        x_dtw = x

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)  # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx_ from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, attn = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1, -1).contiguous().view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)
        attn = attn.view(-1, ctx.size(1), ctx.size(2))
        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        if train:
            attn = attn.reshape(words.size(0), words.size(1), -1, 36)
            return logit, h1, c1, attn, x_dtw

        return logit, h1, c1


def cosine_dist(a, b):
    '''
    i.e. inverse cosine similarity
    '''
    return 1 - np.dot(a, b)


def DTW(seq_a, seq_b, b_gt_length, band_width=None):
    """
    DTW is used to find the optimal alignment path;
    Returns GT like 001110000 for each seq_a
    """
    dist_func = cosine_dist

    if band_width is None:
        path, dist = dtw_path_from_metric(seq_a.detach().cpu().numpy(),
                                          seq_b.detach().cpu().numpy(),
                                          metric=dist_func)
    else:
        path, dist = dtw_path_from_metric(seq_a.detach().cpu().numpy(),
                                          seq_b.detach().cpu().numpy(),
                                          sakoe_chiba_radius=band_width,
                                          metric=dist_func)

    with torch.no_grad():
        att_gt = torch.zeros((seq_a.shape[0], b_gt_length)).cuda()

        for i in range(len(path)):
            att_gt[path[i][0], path[i][1]] = 1

        # v2 new: allow overlap
        for i in range(seq_a.shape[0]):
            pos = (att_gt[i] == 1).nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                pos = [i, i]
            if pos[0] - 1 >= 0:
                att_gt[i, pos[0] - 1] = 1
            if pos[-1] + 1 < seq_b.shape[0]:
                att_gt[i, pos[-1] + 1] = 1

    return att_gt


class Entity_Knowledge(nn.Module):
    def __init__(self, fact_dropout, top_k_facts):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=300, out_channels=100, kernel_size=(1, 1), padding=0, bias=True)

        self.top_k_facts = top_k_facts

        self.fact_embed_drop = nn.Dropout2d(fact_dropout)

        with torch.no_grad():
            with open('./data/pkls/knowledge_rel_embed_glove.pkl', 'rb') as f:
                self.knowledge_fact_dict = pickle.load(f)

            with open('./data/pkls/vg_class_glove_embed.pkl', 'rb') as f:
                self.vg_class_glove_embed = pickle.load(f)

            self.vg_class_glove_embed[-1] = torch.zeros(self.vg_class_glove_embed[0].shape[-1])
            self.knowledge_fact_dict[-1] = torch.zeros(top_k_facts, 300)

    def forward(self, detect_labels):
        # detect_labels: batch_size, max_len, n_labels

        batch_size, max_len, n_labels = detect_labels.shape
        detect_labels = detect_labels.reshape(-1)

        with torch.no_grad():
            facts = [self.knowledge_fact_dict[int(label.item())] for _, label in enumerate(detect_labels)]
            facts = torch.stack(facts[:300], dim=0).cuda()  # n_entities, top_k_facts, 600 ([rel_entity_embed,
            # rel_embed])
            if self.top_k_facts < facts.shape[1]:
                facts = facts[:, :self.top_k_facts, :]
            n_entities = facts.shape[0]  # n_entities = batch_size * max_len * n_labels
            facts = facts.reshape(n_entities, self.top_k_facts * 2, 300, 1)
            facts = facts.permute(0, 2, 1, 3)

        x = self.conv(facts)  # (n_entities, 100, self.top_k_facts * 2, 1)
        x = x.permute(0, 2, 1, 3)  # (n_entities, self.top_k_facts * 2, 100, 1)
        x = x.reshape(batch_size * max_len, n_labels * self.top_k_facts * 2, 100)
        x = x.mean(1).reshape(batch_size, max_len, 600)

        final_embed = x  # (batch_size, max_len, 300)
        final_embed = self.fact_embed_drop(final_embed)

        return final_embed


class PanoEntityKnowledge(nn.Module):
    def __init__(self, fact_dropout, top_k_facts):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=300, out_channels=100, kernel_size=(1, 1), padding=0, bias=True)

        self.top_k_facts = top_k_facts

        self.fact_embed_drop = nn.Dropout2d(fact_dropout)

        with torch.no_grad():
            with open('./data/pkls/knowledge_rel_embed_glove.pkl', 'rb') as f:
                self.knowledge_fact_dict = pickle.load(f)
            self.knowledge_fact_dict[-1] = torch.zeros(top_k_facts, 600)

    def forward(self, detect_labels):
        # detect_labels: batch_size, max_len, n_labels

        batch_size, max_len, views = detect_labels.shape
        detect_labels = detect_labels.reshape(-1)

        with torch.no_grad():
            facts = [self.knowledge_fact_dict[int(label.item())] for _, label in enumerate(detect_labels)]
            facts = torch.stack(facts, dim=0).cuda()  # n_entities, top_k_facts, 600 ([rel_entity_embed, rel_embed])
            if self.top_k_facts < facts.shape[1]:
                facts = facts[:, :self.top_k_facts, :]
            n_entities = facts.shape[0]  # n_entities = batch_size * max_len * n_labels
            facts = facts.reshape(n_entities, self.top_k_facts*2, 300, 1)
            facts = facts.permute(0, 2, 1, 3)  # n_entities, 300, self.top_k_facts * 2, 1

        x = self.conv(facts)  # (n_entities, 100, self.top_k_facts * 2, 1)
        x = x.permute(0, 2, 1, 3)  # (n_entities, self.top_k_facts * 2, 100, 1)
        x = x.reshape(batch_size * max_len, views, self.top_k_facts*2, 100)
        x = x.mean(2).reshape(batch_size, max_len, views, 100)

        final_embed = x  # (batch_size, max_len, 300)
        final_embed = self.fact_embed_drop(final_embed)

        return final_embed
