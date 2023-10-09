import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import logging

from eval_utils.cococaption.pycocoevalcap.bleu.bleu import Bleu
from eval_utils.cococaption.pycocoevalcap.meteor.meteor import Meteor
from eval_utils.cococaption.pycocoevalcap.rouge.rouge import Rouge
from utils import load_datasets


def to_contiguous(tensor: Tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


class ReinforceCriterion(nn.Module):
    def __init__(self, opt, evaluator):
        super(ReinforceCriterion, self).__init__()
        self.reward_type = opt.metric
        self.bleu = None
        self.reward_func = lambda path2inst: evaluator.eval_batch(path2inst, metrics=[opt.reward_type])

    def _cal_action_loss(self, log_probs, reward, mask):
        output = - log_probs * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

    def _cal_value_loss(self, reward, baseline, mask):
        output = (reward - baseline).pow(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

    def forward(self, path2inst, seq, seq_log_probs, baseline, rewards=None):
        '''
        :param seq: (batch_size, 5, seq_length)
        :param seq_log_probs: (batch_size, 5, seq_length)
        :param baseline: (batch_size, 5, seq_length)
        :param rewards: (batch_size, 5, seq_length)
        :return:
        '''
        if rewards is None:
            # compute the reward
            batch_size = seq_log_probs.size(0)
            scores = self.reward_func(path2inst)
            if self.bleu is not None:
                rewards = [score[self.bleu] for score in scores]
            else:
                rewards = scores[self.reward_type]
            rewards = torch.FloatTensor(rewards)  # (batch_size,)
            avg_reward = rewards.mean()
        else:
            batch_size = seq_log_probs.size(0)
            avg_reward = rewards.mean()
            rewards = rewards.view(batch_size, -1, 1)

        # get the mask
        mask = (seq > 0).float()  # its size is supposed to be (batch_size, 5, seq_length)
        if mask.size(2) > 1:
            mask = torch.cat([mask.new(mask.size(0), mask.size(1), 1).fill_(1), mask[:, :, :-1]], 2).contiguous()
        else:
            mask.fill_(1)
        mask = Variable(mask)

        # compute the loss
        advantage = Variable(rewards.data - baseline.data)
        value_loss = self._cal_value_loss(rewards, baseline, mask)
        action_loss = self._cal_action_loss(seq_log_probs, advantage, mask)

        return action_loss + value_loss, avg_reward


class LanguageModelCriterion(nn.Module):
    def __init__(self, weight=0.0):
        self.weight = weight
        super(LanguageModelCriterion, self).__init__()

    def forward(self, x, target, weights=None, compute_prob=False):
        if len(target.size()) == 3:  # separate story
            x = x.view(-1, x.size(2), x.size(3))
            target = target.view(-1, target.size(2))

        seq_length = x.size(1)
        # truncate to the same size
        target = target[:, :x.size(1)]
        mask = (target > 0).float()
        mask = to_contiguous(torch.cat([Variable(mask.data.new(mask.size(0), 1).fill_(1)), mask[:, :-1]], 1))

        # reshape the variables
        x = to_contiguous(x).view(-1, x.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = mask.view(-1, 1)

        if weights is None:
            output = - x.gather(1, target) * mask
        else:
            output = - x.gather(1, target) * mask * to_contiguous(weights).view(-1, 1)

        if compute_prob:
            output = output.view(-1, seq_length)
            mask = mask.view(-1, seq_length)
            return output.sum(-1) / mask.sum(-1)

        output = torch.sum(output) / torch.sum(mask)

        entropy = -(torch.exp(x) * x).sum(-1) * mask
        entropy = torch.sum(entropy) / torch.sum(mask)

        return output + self.weight * entropy
