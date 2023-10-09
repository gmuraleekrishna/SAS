import random

import torch
import math

from torch import nn


def batch_input_sequence_by_prefix_length(input_sequence, prefix_length):
    seq_len = input_sequence.size(2)
    # Discard tokens if the sequence length is not divisible by the split length.
    new_seq_len = (seq_len // prefix_length) * prefix_length
    input_sequence = input_sequence[:, :new_seq_len]
    batch = input_sequence.view(-1, prefix_length).contiguous()
    return batch


class SequencePenaltyCriterion(nn.Module):
    def __init__(self, sequence_ngram_n=4, sequence_prefix_length=1,
                 sequence_candidate_type='repeat', reduce='sum'):
        super(SequencePenaltyCriterion, self).__init__()
        self.sequence_ngram_n = sequence_ngram_n
        self.sequence_candidate_type = sequence_candidate_type
        self.mask_p = 0.15
        self.reduce = reduce

    def forward(self, pred):
        n_batches, vocab_size, seq_len = pred.size()
        mask = None
        pred_toks = pred.argmax(dim=1)
        pred_probs = torch.log_softmax(dim=1)
        if self.sequence_candidate_type == 'repeat':
            mask = ngram_repeat_mask(pred_toks, self.sequence_ngram_n).type_as(pred_probs)
        elif self.sequence_candidate_type == 'random':
            mask = torch.bernoulli(torch.zeros_like(pred_toks, dtype=torch.float).fill_(self.mask_p))

        pred_lprobs = pred_probs.view(-1, pred_probs.size(1)).gather(1, pred_toks.view(-1, pred_probs.size(2)))
        one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
        loss = -torch.log(one_minus_probs) * mask
        if self.reduce == 'sum':
            loss = loss.sum()
            loss /= n_batches
        elif self.reduce == 'mean':
            loss = loss.mean()
            loss /= n_batches
        return loss


def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x) - n):
            ng = tuple(xl[j:j + n])
            if ng in seen:
                mask[i, j:j + n] = 1
            seen.add(ng)
    return mask


def obtain_rep_baseline_prob(target_probs, P, L, N, K):
    gt_probs = []
    mask = []
    valid_tokens = 0

    for i, x in enumerate(target_probs):
        prefix_len = P[i]
        rep_len = L[i]
        repeat_times = N[i]
        remain_tokens = K[i]
        start_sen_prob = x[prefix_len:]
        to_pelize_tokens = rep_len * (repeat_times - 1) + remain_tokens
        new_cp_probs = torch.zeros_like(x)
        new_cp_probs[prefix_len + rep_len:] = start_sen_prob[:to_pelize_tokens]
        gt_probs.append(new_cp_probs)
        this_mask = torch.zeros_like(x, dtype=torch.bool)
        this_mask[prefix_len + rep_len:] = True
        mask.append(this_mask)
        valid_tokens += to_pelize_tokens
    gt_probs = torch.stack(gt_probs, dim=0)
    mask = torch.stack(mask, dim=0)
    return gt_probs, mask, valid_tokens


class RepetitionPenaltyAccumCriterion(nn.Module):
    def __init__(self, rep_reduce_gamma, end_sentence_decoded, loss_type):
        super(RepetitionPenaltyAccumCriterion, self).__init__()
        # self.sequence_ngram_n = args.sequence_ngram_n
        # self.sequence_prefix_length = args.sequence_prefix_length
        # self.sequence_completion_length = args.sequence_completion_length
        # self.sequence_candidate_type = args.sequence_candidate_type
        # self.mask_p = args.mask_p
        self.rep_reduce_gamma = rep_reduce_gamma  # repetetion prob discount
        self.end_sentence_decoded = end_sentence_decoded
        self.loss_type = loss_type

    def re_orgnize_sentence(self, src, target_tokens):
        # First ,random pick a sentence in a batch
        # Then, get sentence length, L, repeat N times to get maximun len, calculate remains K
        # Last, return the batch of samples

        max_tokens = src.size(1)
        P = []
        L = []
        N = []
        K = []
        ALL_TOKENS = []
        TARGET_TOKENS = []
        prefixes = target_tokens

        for i, (x, y) in enumerate(zip()):
            xl = x.tolist()
            yl = y.tolist()
            xl.extend(yl)
            sentence_end_indexes = []
            for idx, token in enumerate(xl):
                if token == self.end_sentence_decoded:
                    if idx == 3:
                        sentence_end_indexes.append(4)
                    else:
                        sentence_end_indexes.append(idx)
            sentence_end_indexes.append(max_tokens)
            sentence_end_indexes.append(len(yl) + max_tokens)
            sentence_end_indexes = list(sorted(sentence_end_indexes))
            try:
                sen_idx = random.randint(1, len(sentence_end_indexes) - 2)
                last_sen_start = sentence_end_indexes[sen_idx - 1] + 1
                sen_start = sentence_end_indexes[sen_idx]
                sen_end = sentence_end_indexes[sen_idx + 1]
            except:
                return None, None, None, None, None, None
            prefix = x[last_sen_start: sen_start]
            prefix_len = sen_start - last_sen_start
            left_tokens = max_tokens - prefix_len

            x_sentence = x[sen_start: sen_end].view(1, -1)
            sen_len = sen_end - sen_start
            n, k = left_tokens // sen_len, left_tokens % sen_len
            x_sentence = x_sentence.repeat(n + 1, 1).view(-1)
            new_sentence = torch.cat([prefix, x_sentence], dim=0)
            input_sentence = new_sentence[:max_tokens]
            target_sentence = new_sentence[1:max_tokens + 1]
            assert target_sentence.size()[0] == max_tokens
            P.append(sen_start - last_sen_start)
            N.append(n)
            K.append(k)
            L.append(sen_len)
            ALL_TOKENS.append(input_sentence)
            TARGET_TOKENS.append(target_sentence)
        ALL_TOKENS = torch.stack(ALL_TOKENS, dim=0)
        TARGET_TOKENS = torch.stack(TARGET_TOKENS, dim=0)

        return ALL_TOKENS, TARGET_TOKENS, P, L, N, K

    def forward(self, src_logits, target_tokens, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_probs, src_tokens = src_logits.max(dim=1)
        src_tokens, target, P, L, N, K = self.re_orgnize_sentence(src_tokens, target_tokens)

        if src_tokens is None:
            return 0.0
        # logits = net_output[0].view(-1, net_output[0].size(-1))
        # target = model.get_targets(sample, net_output)
        # target = target.view(-1)
        loss, _, sample_size = self.compute_loss(target, src_probs, P, L, N, K, reduce=reduce)

        return loss

    def compute_loss(self, src_logits, src_probs, P, L, N, K, reduce=True):
        B, T, vocab_size = src_probs.size()
        probs = src_probs.view(B * T, vocab_size)
        target = src_logits.view(-1, 1)
        target_probs = probs.gather(1, target).view(B, T)
        gt_probs, mask, valid_tokens = obtain_rep_baseline_prob(target,
                                                                target_probs.detach(),
                                                                P, L, N, K)
        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(target_probs, gt_probs * self.rep_reduce_gamma, reduction='none')
            loss = loss.sum()
            loss = loss * 3  # loss scale is smaller than nll
            return loss, loss, valid_tokens
        elif self.loss_type == 'nl':
            one_minus_probs = torch.clamp((1.0 - torch.abs((target_probs - gt_probs * self.rep_reduce_gamma))),
                                          min=1e-20)
            loss = -torch.log(one_minus_probs) * mask
            loss = loss.sum()
            return loss, loss, valid_tokens
        elif self.loss_type == 'nl_clip':
            one_minus_probs = torch.clamp((1.0 - torch.clamp(target_probs - gt_probs * self.rep_reduce_gamma, min=0.0)),
                                          min=1e-20)
            loss = -torch.log(one_minus_probs) * mask
            loss = loss.sum()
            return loss, valid_tokens
        else:
            assert 1 == 0, 'not implemented error'


class PenaltyBuilder(object):
    """Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen

    Attributes:
        has_cov_pen (bool): Whether coverage penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting beta
            to 0 should force coverage length to be a no-op.
        has_len_pen (bool): Whether length penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting alpha
            to 1 should force length penalty to be a no-op.
        coverage_penalty (callable[[FloatTensor, float], FloatTensor]):
            Calculates the coverage penalty.
        length_penalty (callable[[int, float], float]): Calculates
            the length penalty.
    """

    def __init__(self, cov_pen, length_pen):
        self.has_cov_pen = not self._pen_is_none(cov_pen)
        self.coverage_penalty = self._coverage_penalty(cov_pen)
        self.has_len_pen = not self._pen_is_none(length_pen)
        self.length_penalty = self._length_penalty(length_pen)

    @staticmethod
    def _pen_is_none(pen):
        return pen == "none" or pen is None

    def _coverage_penalty(self, cov_pen):
        if cov_pen == "wu":
            return self.coverage_wu
        elif cov_pen == "summary":
            return self.coverage_summary
        elif self._pen_is_none(cov_pen):
            return self.coverage_none
        else:
            raise NotImplementedError("No '{:s}' coverage penalty.".format(cov_pen))

    def _length_penalty(self, length_pen):
        if length_pen == "wu":
            return self.length_wu
        elif length_pen == "avg":
            return self.length_average
        elif self._pen_is_none(length_pen):
            return self.length_none
        else:
            raise NotImplementedError("No '{:s}' length penalty.".format(length_pen))

    # Below are all the different penalty terms implemented so far.
    # Subtract coverage penalty from topk log probs.
    # Divide topk log probs by length penalty.

    def coverage_wu(self, cov, beta=0.0):
        """GNMT coverage re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        ``cov`` is expected to be sized ``(*, seq_len)``, where ``*`` is
        probably ``batch_size x beam_size`` but could be several
        dimensions like ``(batch_size, beam_size)``. If ``cov`` is attention,
        then the ``seq_len`` axis probably sums to (almost) 1.
        """

        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(-1)
        return beta * penalty

    def coverage_summary(self, cov, beta=0.0):
        """Our summary penalty."""
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
        penalty -= cov.size(-1)
        return beta * penalty

    def coverage_none(self, cov, beta=0.0):
        """Returns zero as penalty"""
        none = torch.zeros((1,), device=cov.device, dtype=torch.float)
        if cov.dim() == 3:
            none = none.unsqueeze(0)
        return none

    def length_wu(self, cur_len, alpha=0.0):
        """GNMT length re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        return ((5 + cur_len) / 6.0) ** alpha

    def length_average(self, cur_len, alpha=1.0):
        """Returns the current sequence length."""
        return cur_len ** alpha

    def length_none(self, cur_len, alpha=0.0):
        """Returns unmodified scores."""
        return 1.0


class RepetitionPenaltyLogitsProcessor(nn.Module):
    r"""
    [`LogitsProcessor`] that prevents the repetition of previous tokens through an exponential penalty. This technique
    shares some similarities with coverage mechanisms and other aimed at reducing repetition. During the text
    generation process, the probability distribution for the next token is determined using a formula that incorporates
    token scores based on their occurrence in the generated sequence. Tokens with higher scores are less likely to be
    selected. The formula can be seen in the original [paper](https://arxiv.org/pdf/1909.05858.pdf). According to the
    paper a penalty of around 1.2 yields a good balance between truthful generation and lack of repetition.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.

    """

    def __init__(self, penalty: float):
        super(RepetitionPenaltyLogitsProcessor, self).__init__()
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def forward(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores
