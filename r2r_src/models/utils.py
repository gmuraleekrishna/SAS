import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from torch import Tensor


def parse_param(param_str):
    """
    Parse param_str like 'a=1,b=2.5,c=Name' into dict
    """
    str_list = param_str.strip().split(',')
    params = {}
    for s in str_list:
        try:
            name, value = s.split('=')
        except:
            return {}
        if value.find('.') > 0:
            value = float(value)
        elif value.isdigit():
            value = int(value)
        else:
            value = str(value)
        params[name] = value

    return params


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LogitsSampling:
    def __init__(self, method: str = 'top_k', *args, **kwargs):
        if method == 'top_k':
            self.method = self.top_k_sampling
        elif method == 'top_k_p':
            self.method = self.top_k_top_p_sampling
        else:
            raise NotImplementedError
        self.top_k: int = kwargs.get('top_k', 3)
        self.top_p: float = kwargs.get('top_p', 0.9)
        self.min_tokens_to_keep: int = kwargs.get('min_tokens_to_keep', 300)
        self.filter_value: float = kwargs.get('filter_value', -1e10)

    def sample(self, logits):
        return self.method(logits)

    def top_k_sampling(self, logits: Tensor):
        """
        Masks everything but the k top entries as -infinity (1e10).
        Used to mask logits such that e^-infinity -> 0 won't contribute to the
        sum of the denominator.
        """
        # bs = logits.shape(0)
        # logits (bs, 992, 80)
        if self.top_k == 0:
            return logits
        else:
            top_k = max(self.top_k, self.min_tokens_to_keep)
            values = torch.topk(logits, top_k, dim=1)[0]
            batch_mins = values[:, -1, :].view(-1, 1, logits.shape[2]).expand_as(logits)
            # if probs:
            #     return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
            return torch.where(logits < batch_mins, torch.ones_like(logits) * self.filter_value, logits)

    def top_k_top_p_sampling(self, logits: Tensor):
        """ Filter a distribution of logits using top-top_k and/or nucleus (top-top_p) filtering
            Args:
                logits: logits distribution shape (batch size x vocabulary size)
                top_k > 0: keep only top top_k tokens with highest probability (top-top_k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.size(-2))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-top_k
            logits = self.top_k_sampling(logits)

        if self.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., self.min_tokens_to_keep:, :] = sorted_indices_to_remove[...,
                                                                         :-self.min_tokens_to_keep, :].clone()
            sorted_indices_to_remove[..., :self.min_tokens_to_keep, :] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices,
                                                                 src=sorted_indices_to_remove)
            logits[indices_to_remove] = self.filter_value
        return logits
