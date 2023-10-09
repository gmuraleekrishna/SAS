__author__ = 'tylin'

from .bleu.bleu import Bleu
from r2r_src.eval_utils.cococaption.pycocoevalcap.meteor.meteor import Meteor
from r2r_src.eval_utils.cococaption.pycocoevalcap.rouge.rouge import Rouge
from .cider.cider import Cider
from r2r_src.eval_utils.cococaption.pycocoevalcap.spice.spice import Spice
from r2r_src.eval_utils.cococaption.pycocoevalcap.spice.spice_plus_action_v1 import Spice_action_v1
import networkx as nx

import sys

from r2r_src.eval_utils.cococaption.pycocoevalcap.spice.spice_plus_rooms_v1 import Spice_rooms_v1

sys.path.append("..")
from utils import load_datasets, load_nav_graphs


class COCOEvalCap:
    def __init__(self, splits, scans, tok):
        self.evalImgs = []
        self.eval = {}
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []

        for split in splits:
            for item in load_datasets([split]):
                if scans is not None and item['scan'] not in scans:
                    continue
                self.gt[str(item['path_id'])] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]

        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def evaluate(self, path2inst, metric=None):
        gts = {}
        gt_paths = {}
        res = {}
        for path_id, inst in path2inst.items():
            path_id = str(path_id)
            assert path_id in self.gt
            # There are three references
            gts[path_id] = [' '.join(self.tok.split_sentence(sent)) for sent in self.gt[path_id]['instructions']]
            gt_paths[path_id] = self.gt[path_id]['path']
            res[path_id] = [' '.join([self.tok.index_to_word[word_id] for word_id in inst])]

        # imgIds = self.params['image_id']
        # # imgIds = self.coco.getImgIds()
        # gts = {}
        # res = {}
        # for imgId in imgIds:
        #     gts[imgId] = self.coco.imgToAnns[imgId]
        #     res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        # tokenizer = PTBTokenizer()
        # gts = tokenizer.tokenize(gts)
        # res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = {
            'Bleu_1':  Bleu(1),
            'Bleu_2': Bleu(2),
            'Bleu_3': Bleu(3),
            'Bleu_4': Bleu(4),
            'METEOR': Meteor(),
            'ROUGE_L': Rouge(),
            'CIDEr': Cider(),
            'SPICE': Spice(),
            'SPICE_action_v1': Spice_action_v1(self.tok),
            'SPICE_rooms_v1': Spice_rooms_v1(self.tok),
        }


        # =================================================
        # Compute scores
        # =================================================
        for method, scorer in scorers.items():
            if method == metric:
                print('computing %s score...' % (scorer.method()))
                score, scores = scorer.compute_score(gts, res, gt_paths)
                self.setEval(score, method)
                # self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f"%(method, score))
            # self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

    def get_reward(self, uidXpred, metric="CIDEr"):
        gts = []
        preds = []
        for uid, pred in uidXpred:
            preds.append(pred)
            gts.append(self.uid2ref[uid])
        return self.langeval.eval_batch(gts, preds, metric)
