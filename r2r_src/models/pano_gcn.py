from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from torch import Tensor
from torch.autograd import Variable

def get_pano_affinity():
    # get the affinity diff_matrix of panorama, where edges between adjacent views are 1

    # low elevation view 0-11
    # middle elevation view 12-23
    # high elevation view 24-35

    # pano_a = np.zeros((36, 36))  # no self-connect
    pano_a = np.eye(36, dtype=float)  # self-connect

    # low elevation
    for view_id in range(0, 12):
        # up
        pano_a[view_id, view_id + 12] = 1

        # left, left-up
        if view_id == 0:
            pano_a[view_id, 11] = 1
            pano_a[view_id, 11 + 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 + 12] = 1

        # right, right-up
        if view_id == 11:
            pano_a[view_id, 0] = 1
            pano_a[view_id, 0 + 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 + 12] = 1

    # middle elevation
    for view_id in range(12, 24):
        # up
        pano_a[view_id, view_id + 12] = 1

        # down
        pano_a[view_id, view_id - 12] = 1

        # left, left-up, left-down
        if view_id == 12:
            pano_a[view_id, 23] = 1
            pano_a[view_id, 23 + 12] = 1
            pano_a[view_id, 23 - 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 + 12] = 1
            pano_a[view_id, view_id - 1 - 12] = 1

        # right, right-up, right-down
        if view_id == 23:
            pano_a[view_id, 12] = 1
            pano_a[view_id, 12 + 12] = 1
            pano_a[view_id, 12 - 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 + 12] = 1
            pano_a[view_id, view_id + 1 - 12] = 1

    # high elevation
    for view_id in range(24, 36):
        # down
        pano_a[view_id, view_id - 12] = 1

        # left, left-down
        if view_id == 24:
            pano_a[view_id, 35] = 1
            pano_a[view_id, 35 - 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 - 12] = 1

        # right, right-down
        if view_id == 35:
            pano_a[view_id, 24] = 1
            pano_a[view_id, 24 - 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 - 12] = 1

    # checking symmetry
    assert np.sum(pano_a - pano_a.T) == 0
    pano_a = cv.GaussianBlur(pano_a, (3, 3), 0, cv.BORDER_TRANSPARENT)
    pano_a[np.eye(36, dtype=int) == 1] = 1
    return pano_a


class DotProductAttention(nn.Module):
    def __init__(self, key_dimension: int) -> None:
        super().__init__()
        self.scale = torch.tensor(1.0 / (key_dimension ** 0.5))
        self.softmax = nn.Softmax(dim=2)

    def forward(
            self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Scaled dot-product attention with an optional mask.
        2X speed improvement over `torch.einsum`.
        Args:
            query: [Batch, Dk]
            key: [Batch, Dk, P]
            value: [Batch, Dv, P]
        Returns:
            tensor of dimension [Batch, Dv]
        """
        energy = torch.bmm(Q.unsqueeze(1), K)
        if mask is not None:
            energy *= mask.unsqueeze(1).float()

        attn = self.softmax(energy * self.scale)
        return torch.bmm(attn, V.permute(0, 2, 1)).squeeze(1)


class PanoAttention(nn.Module):
    def __init__(
            self,
            d_q_in: int = 512,
            d_k_in: int = 640,
            d_v_in: int = 430,
            d_qk: int = 128,
            d_v: int = 256,
            num_heads: int = 12,
            d_out: int = 512,
            normalize: bool = True,
            dropout_p: float = 0.0,
    ) -> None:
        """The residual connection of Vaswani et al is not used here. The
        residual makes sense if self-attention is being used.
        Args:
            d_q_in (int): dimension of the query vector x
            d_k_in (int): dimension of the key vector x
            d_v_in (int): dimension of the value vector x
            d_qk (int): dimension to map queries & keys to prior to attention
            d_v (int): dimension to map values to prior to attention
            num_heads (int): number of attention heads
            d_out (int): output dimension of this module (final linear layer)
        """
        super(PanoAttention, self).__init__()
        self.num_heads = num_heads
        self.normalize = normalize
        self.q_linear = nn.Linear(d_q_in, d_qk * num_heads, bias=False)
        self.k_linear = nn.Linear(d_k_in, d_qk * num_heads, bias=False)
        self.v_linear = nn.Linear(d_v_in, d_v * num_heads, bias=False)

        self.attn = DotProductAttention(d_qk)
        self.final_linear = nn.Linear(d_v * num_heads, d_out, bias=False)

        self.dropout = None
        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)

        if self.normalize:
            self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
            self.tanh = nn.Tanh()
        self.pano_a = Variable(torch.from_numpy(get_pano_affinity()).float()).cuda()
        self.pano_a[self.pano_a.eq(1)] = 0.95
        self.pano_a[self.pano_a.eq(0)] = 0.05

    def forward(self, Q, K, V, knowledge, teacher_action_view_ids, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate Q through the network.

        Q: batch x query_dim
        V: batch x seq_len x query_dim
        mask: batch x seq_len indices to be masked
        '''
        Q = self.q_linear(Q).permute(0, 2, 1).unsqueeze(1)
        K = self.k_linear(K.permute(0, 2, 1, 3)).permute(0, 1, 3, 2).view(K.size(0) * K.size(2), -1, K.size(1))
        V = self.v_linear(V.permute(0, 2, 1, 3)).permute(0, 1, 3, 2).view(V.size(0) * V.size(2), -1, V.size(1))

        Q = Q.view(Q.shape[0] * self.num_heads, Q.shape[1] // self.num_heads)
        K = K.view(
            K.shape[0] * self.num_heads,
            K.shape[1] // self.num_heads,
            K.shape[2],
        )
        V = V.view(
            V.shape[0] * self.num_heads,
            V.shape[1] // self.num_heads,
            V.shape[2],
        )

        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0)

        attended_V = self.attn(Q, K, V, mask=mask)

        attended_V = attended_V.view(
            attended_V.shape[0] // self.num_heads,
            self.num_heads,
            attended_V.shape[1],
        )

        attended_V = attended_V.view(
            attended_V.shape[0], attended_V.shape[1] * attended_V.shape[2]
        )
        attended_V = attended_V * pano_a_tea_batch
        out = self.final_linear(attended_V)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.normalize:
            out = self.layer_norm(out)
        return out

class pano_att_gcn_v5(nn.Module):
    def __init__(self, query_dim, ctx_dim, knowledge_dim):
        '''Initialize layer.'''
        super(pano_att_gcn_v5, self).__init__()
        self.knowledge_dim = 300
        self.query_dim = query_dim
        self.linear_key = nn.Linear(ctx_dim + self.knowledge_dim, query_dim, bias=False)

        self.linear_query = nn.Linear(query_dim, query_dim, bias=False)
        # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.scale = query_dim ** -0.5
        self.linear_out = nn.Linear(query_dim + ctx_dim + 100, query_dim, bias=False)
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.05

    def forward(self, h, context, detect_feats, knowledge_vector, teacher_action_view_ids , mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_query(h).unsqueeze(2)  # batch x dim x 1

        context_know = torch.cat((context, detect_feats.unsqueeze(1).expand(-1, 36, -1)), dim=2)
        ba, _, ck_dim = context_know.shape
        context_know = self.linear_key(context_know.reshape(-1, ck_dim))
        context_know = context_know.reshape(ba, 36, self.query_dim)

        # Get attention
        attn = torch.bmm(context_know, target).squeeze(2)  # batch x seq_len
        logit = attn

        # new: add scale
        attn *= self.scale

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))

        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        # gcn new
        batch, seq_len, ctx_dim = context.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch), teacher_action_view_ids, :]
            pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)

        attn3_gcn = attn3 * pano_a_tea_batch

        weighted_context = torch.bmm(attn3_gcn, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h, knowledge_vector), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn

class pano_att_gcn_v6(nn.Module):
    def __init__(self, query_dim, ctx_dim, knowledge_dim):
        '''Initialize layer.'''
        super(pano_att_gcn_v6, self).__init__()
        self.knowledge_dim = knowledge_dim
        self.query_dim = query_dim
        self.ctx_dim = ctx_dim
        self.linear_key = nn.Linear(2476, query_dim, bias=False) # 1540 # 3076
        self.hidden_size = query_dim
        self.linear_query = nn.Linear(query_dim, query_dim, bias=False)
        # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.scale = query_dim ** -0.5
        self.linear_out = nn.Linear(query_dim + ctx_dim + 100 + 6 * 2, query_dim, bias=False)
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            # self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.01

    def forward(self, Q, V, detect_feats, knowledge_vector, teacher_action_view_ids, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate Q through the network.

        Q: batch x query_dim
        V: batch x seq_len x query_dim
        mask: batch x seq_len indices to be masked
        '''
        batch_size, max_length, _ = Q.size()
        Q = Q.contiguous().view(-1, self.hidden_size)
        teacher_action_view_ids = teacher_action_view_ids.reshape(-1)
        detect_feats = detect_feats.reshape(-1, 900)
        knowledge_vector = knowledge_vector.reshape(-1, 112)

        Q = self.linear_query(Q).unsqueeze(2)  # batch x query_dim x 1
        V = V.view(batch_size * max_length, -1, self.ctx_dim)
        K = torch.cat((V, detect_feats.unsqueeze(1).expand(-1, 36, -1)), dim=2)
        ba, _, ck_dim = K.shape
        K = self.linear_key(K.reshape(-1, ck_dim))
        K = K.reshape(ba, 36, self.query_dim)

        # Get attention
        energy = torch.bmm(K, Q).squeeze(2)  # batch, seq_len
        logit = energy

        # new: add scale
        energy *= self.scale

        if mask is not None:
            # -Inf masking prior to the softmax
            energy.masked_fill_(mask, -float('inf'))

        energy = self.sm(energy)  # There will be a bug here, but it's actually a problem in torch source code.
        energy = energy.view(energy.size(0), 1, energy.size(1))  # batch, 1, 36

        # gcn new
        batch, seq_len, ctx_dim = V.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch),
                               teacher_action_view_ids, :]
            pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)  # batch*seqlen,1,36

        attn3_gcn = energy * pano_a_tea_batch  # batch*seqlen,1,36

        weighted_context = torch.bmm(attn3_gcn, V).squeeze(1)  # batch*seqlen, query_dim
        if not output_prob:
            energy = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, Q.squeeze(2), knowledge_vector), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, energy
        else:
            return weighted_context, energy


class PanoAttentionv2(nn.Module):
    def __init__(self, query_dim, ctx_dim, knowledge_dim):
        '''Initialize layer.'''
        super(PanoAttentionv2, self).__init__()
        self.knowledge_dim = knowledge_dim
        self.query_dim = query_dim
        self.ctx_dim = ctx_dim
        self.linear_key = nn.Linear(ctx_dim + self.knowledge_dim + 300, query_dim, bias=False)
        self.hidden_size = query_dim
        self.linear_query = nn.Linear(query_dim, query_dim, bias=False)
        # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.scale = query_dim ** -0.5
        self.linear_out = nn.Linear(2277, query_dim, bias=False) #741, 2277
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            # self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.01

    def forward(self, Q, V, detect_feats, knowledge_vector, structural_embeds, teacher_action_view_ids, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate Q through the network.

        Q: batch x query_dim
        V: batch x seq_len x query_dim
        mask: batch x seq_len indices to be masked
        '''
        batch_size, max_length, _ = Q.size()
        Q = Q.contiguous().view(-1, self.hidden_size)
        V = V.view(batch_size * max_length, 36, self.ctx_dim)
        teacher_action_view_ids = teacher_action_view_ids.reshape(-1)
        # detect_feats = detect_feats.reshape(-1, 900)
        knowledge_vector = knowledge_vector.reshape(-1, 36, 100)  # batch_size*seqlen, 36, 100

        Q = self.linear_query(Q).unsqueeze(2)  # batch_size*seqlen, query_dim, 1

        K = torch.cat((V, detect_feats.view(batch_size * max_length, 36, -1)), dim=2)
        ba, _, ck_dim = K.shape
        K = self.linear_key(K.reshape(-1, ck_dim))
        K = K.reshape(ba, 36, self.query_dim)

        # Get attention
        energy = torch.bmm(K, Q).squeeze(2)  # batch, seq_len
        logit = energy

        # new: add scale
        energy *= self.scale

        if mask is not None:
            # -Inf masking prior to the softmax
            energy.masked_fill_(mask, -float('inf'))

        energy = self.sm(energy)  # There will be a bug here, but it's actually a problem in torch source code.
        energy = energy.view(energy.size(0), 1, energy.size(1))  # batch, 1, 36

        # gcn new
        batch, seq_len, ctx_dim = V.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)
            pano_a_tea_batch = pano_a_tea_batch  # batch*seqlen,36

        attn = energy * pano_a_tea_batch
        weighted_context = torch.bmm(attn, V).squeeze(1)  # batch*seqlen, query_dim
        if not output_prob:
            energy = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, knowledge_vector, structural_embeds.reshape(-1, 36, 1)), 2)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, energy
        else:
            return weighted_context, energy



if __name__ == '__main__':
    model = pano_att_gcn(query_dim=512, ctx_dim=2048)

    dummy_q = torch.rand(8, 512)
    dummy_ctx = torch.rand(8, 36, 2048)

    output = model(dummy_q, dummy_ctx)
