import torch
import cv2 as cv
from torch import nn
from torch.nn import init
import numpy as np

from param import args
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


class SlotKnowledgeAttention(nn.Module):
    def __init__(self, num_slots, query_dim, ctx_dim, knowledge_dim, iters=3, eps=1e-8, hidden_dim=512,
                 drop_rate=0.4):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.knowledge_dim = knowledge_dim
        self.scale = query_dim ** -0.5
        self.feature_size = ctx_dim

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        if args.slot_share_qk:
            self.to_k = self.to_q
        else:
            self.to_k = nn.Linear(ctx_dim + self.knowledge_dim, query_dim, bias=False)

        self.to_v = nn.Linear(ctx_dim, ctx_dim)
        self.sm = nn.Softmax()
        self.gru = nn.GRUCell(ctx_dim + 100 + 6 * 2, query_dim)
        self.linear_out = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, query_dim)
        )

        self.norm_slots = nn.LayerNorm(query_dim)
        self.norm_pre_ff = nn.LayerNorm(query_dim)
        self.norm_input = nn.LayerNorm(query_dim)

        self.slot_dropout = nn.Dropout(drop_rate)
        self.input_dropout = nn.Dropout(drop_rate)
        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            # self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.01
    # def forward_slot(self, cand_feat, pano_feat, cand_mask):
    #     (b, n, d), device = *pano_feat.shape, pano_feat.device
    #
    #     # original Q as the initial slot
    #     slots = cand_feat.clone()
    #     slots[..., :-args.angle_feat_size] = self.slot_dropout(slots[..., :-args.angle_feat_size])
    #
    #     pano_feat[..., :-args.angle_feat_size] = self.norm_input(pano_feat.clone()[..., :-args.angle_feat_size])
    #     pano_feat[..., :-args.angle_feat_size] = self.input_dropout(pano_feat[..., :-args.angle_feat_size])
    #
    #     # (bs, num_ctx, hidden_size)
    #     k = self.to_k(pano_feat)
    #     v = self.to_v(pano_feat[..., :-args.angle_feat_size])
    #
    #     attn_weights = []
    #
    #     for t in range(self.iters):
    #         slots_prev = slots
    #
    #         slots[..., : -args.angle_feat_size] = self.norm_slots(slots[..., : -args.angle_feat_size].clone())
    #
    #         # (bs, num_slots, hidden_size)
    #         q = self.to_q(slots.clone())
    #
    #         # (bs, num_slots, num_ctx)
    #         dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
    #         dots.masked_fill_(cand_mask, -float('inf'))
    #         attn = dots.softmax(dim=1)
    #
    #         attn_weights.append(attn)   # for visualization
    #
    #         # (bs, num_slots, feature_size)
    #         updates = torch.einsum('bjd,bij->bid', v, attn)
    #
    #         gru_updates = self.gru(
    #             updates.reshape(-1, self.feature_size),
    #             slots_prev.clone()[..., : -args.angle_feat_size].reshape(-1, self.feature_size)
    #         )
    #         gru_updates = gru_updates.reshape(b, -1, gru_updates.shape[-1])
    #         gru_updates = gru_updates + self.linear_out(self.norm_pre_ff(gru_updates))
    #
    #         slots[..., : -args.angle_feat_size] = gru_updates.clone()
    #
    #     return slots, np.stack([a.cpu().detach().numpy() for a in attn_weights], 0)

    def forward(self, h, context, teacher_action_view_ids, detect_feats, knowledge_vector, mask=None,
                          output_tilde=True, output_prob=True):
        batch, max_len, n_views, d = context.shape

        # original Q as the initial slot
        slots = self.pano_a.unsqueeze(0).unsqueeze(0).expand(batch, max_len, -1, -1).clone()
        # slots[..., :-args.angle_feat_size] = self.slot_dropout(slots[..., :-args.angle_feat_size])
        #
        # V[..., :-args.angle_feat_size] = self.norm_input(V.clone()[..., :-args.angle_feat_size])
        # V[..., :-args.angle_feat_size] = self.input_dropout(V[..., :-args.angle_feat_size])

        context_know = torch.cat((context, detect_feats.unsqueeze(2).expand(-1, -1, 36, -1)), dim=3)
        ba, _, n_views, ck_dim = context_know.shape

        # (bs, num_ctx, hidden_size)
        k = self.to_k(context_know)
        # v = self.to_v(V[..., :-args.angle_feat_size])

        attn_weights = []

        for t in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots.clone())

            # (bs, num_slots, hidden_size)
            q = self.to_q(slots.clone())

            # (bs, num_slots, num_ctx)
            dots = torch.einsum('bid,bkjd->bkj', q, k) * self.scale
            if mask is not None:
                dots.masked_fill_(mask, -float('inf'))
            attn = self.sm(dots)

            attn_weights.append(attn)   # for visualization

            # (bs, num_slots, feature_size)
            updates = torch.einsum('bkjd,bij->bid', context, attn)
            updates = torch.cat([updates, knowledge_vector], dim=2)
            gru_updates = self.gru(
                updates.reshape(-1, updates.shape[-1]),
                slots_prev.clone().reshape(-1, h.shape[-1])
            )
            gru_updates = gru_updates.reshape(batch, -1, gru_updates.shape[-1])
            gru_updates = gru_updates + self.linear_out(self.norm_pre_ff(gru_updates))

            slots = gru_updates.clone()

        return slots, np.stack([a.cpu().detach().numpy() for a in attn_weights], 0)
