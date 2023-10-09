__author__ = 'tylin'

from collections import defaultdict

import numpy as np


class Eval:
    def __init__(self, splits, scans, tok):
        pass

    def eval(self, gt, pred):
        pass

    def eval_batch(self, gt, pred, real=True):
        assert gt.shape[0] == pred.shape[0]
        batch_size = gt.shape[0]
        result = np.zeros([batch_size], np.float32)
        for i in range(batch_size):
            result[i] = self.eval(gt, pred)
        return result


class LanguageEval(Eval):
    def __init__(self, splits, scans, tok):
        from .cococaption.pycocoevalcap.evil import COCOEvalCaption
        self.cocoEvil = COCOEvalCaption(splits, scans, tok)

    def eval_whole(self, gt, pred, **kwargs):
        import copy
        self.cocoEvil.evaluate(gt, pred, **kwargs)
        return copy.copy(self.cocoEvil.eval)

    def eval_batch(self, gt, pred, metrics=('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr',
                                           'SPICE')):
        """
        metric:
        :param gt:
        :param pred:
        :param metrics: one of [Bleu_1, ..., Bleu_4, METEOR, ROUGE_L, CIDEr]
        :return:
        """
        self.cocoEvil.evaluate(gt, pred, metrics=metrics)
        result = defaultdict(lambda x: np.zeros(len(gt), np.float32))
        for metric in metrics:
            total_ims = 0
            for i in list(self.cocoEvil.imgToEval.keys()):
                total_ims += 1
                result[metric] = self.cocoEvil.imgToEval[i][metric]
            result[metric] = result[metric] / total_ims
        return result
