import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate Q through the network.

        Q: batch x query_dim
        V: batch x seq_len x query_dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x query_dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x query_dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class ContextOnlySoftDotAttention(nn.Module):
    '''Like SoftDot, but don't concatenat Q or perform the non-linearity transform
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim, context_dim=None):
        '''Initialize layer.'''
        super(ContextOnlySoftDotAttention, self).__init__()
        if context_dim is None:
            context_dim = dim
        self.linear_in = nn.Linear(dim, context_dim, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, context, mask=None):
        '''Propagate Q through the network.
        Q: batch x query_dim
        V: batch x seq_len x query_dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x query_dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x query_dim
        return weighted_context, attn


class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):
        '''Initialize layer.'''
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, visual_context, mask=None):
        '''Propagate Q through the network.

        Q: batch x h_dim
        visual_context: batch x v_num x v_dim
        '''
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
        context = self.linear_in_v(visual_context)  # batch x v_num x dot_dim

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num

        weighted_context = torch.bmm(
            attn3, visual_context).squeeze(1)  # batch x v_dim
        return weighted_context, attn


class EltwiseProdScoring(nn.Module):
    '''
    Linearly mapping Q and v to the same dimension, and do a elementwise
    multiplication and a linear scoring
    '''

    def __init__(self, h_dim, a_dim, dot_dim=256):
        '''Initialize layer.'''
        super(EltwiseProdScoring, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)

    def forward(self, h, all_u_t, mask=None):
        '''Propagate Q through the network.

        Q: batch x h_dim
        all_u_t: batch x a_num x a_dim
        '''
        target = self.linear_in_h(h).unsqueeze(1)  # batch x 1 x dot_dim
        context = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, context)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits


###############################################################################
# speaker models
###############################################################################


class SpeakerEncoderLSTM(nn.Module):
    def __init__(self, action_embedding_size, world_embedding_size,
                 hidden_size, dropout_ratio, bidirectional=False):
        super(SpeakerEncoderLSTM, self).__init__()
        # assert not bidirectional, 'Bidirectional is not implemented yet'
        print('Bidirectional is not implemented yet, now using default non-bidirectional EncoderLSTM')

        self.action_embedding_size = action_embedding_size
        self.word_embedding_size = world_embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.visual_attention_layer = VisualSoftDotAttention(
            hidden_size, world_embedding_size)
        self.lstm = nn.LSTMCell(
            action_embedding_size + world_embedding_size, hidden_size)
        self.encoder2decoder = nn.Linear(hidden_size, hidden_size)

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        with torch.no_grad():
            h0 = torch.zeros(batch_size, self.hidden_size).cuda()
            c0 = torch.zeros(batch_size, self.hidden_size).cuda()
        return h0, c0
        # h0 = Variable(
        #     torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        # c0 = Variable(
        #     torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        # return try_cuda(h0), try_cuda(c0)

    def _forward_one_step(self, h_0, c_0, action_embedding,
                          world_state_embedding):
        # print(action_embedding.shape, world_state_embedding.shape)
        feature, _ = self.visual_attention_layer(h_0, world_state_embedding)
        concat_input = torch.cat((action_embedding, feature), 1)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        return h_1, c_1

    def forward(self, batched_action_embeddings, world_state_embeddings):
        ''' Expects action indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        assert isinstance(batched_action_embeddings, list)
        assert isinstance(world_state_embeddings, list)
        assert len(batched_action_embeddings) == len(world_state_embeddings)
        batch_size = world_state_embeddings[0].shape[0]

        h, c = self.init_state(batch_size)
        h_list = []
        for t, (action_embedding, world_state_embedding) in enumerate(
                zip(batched_action_embeddings, world_state_embeddings)):
            h, c = self._forward_one_step(
                h, c, action_embedding, world_state_embedding)
            h_list.append(h)

        decoder_init = nn.Tanh()(self.encoder2decoder(h))

        ctx = torch.stack(h_list, dim=1)  # (batch, seq_len, hidden_size)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c  # (batch, hidden_size)


class SpeakerDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, vocab_embedding_size, hidden_size,
                 dropout_ratio, glove=None, use_input_att_feed=False, padding_idx=None):
        super(SpeakerDecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.embedding = nn.Embedding(vocab_size, vocab_embedding_size, padding_idx)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.drop = nn.Dropout(p=dropout_ratio)
        self.use_input_att_feed = use_input_att_feed
        if self.use_input_att_feed:
            print('using x attention feed in SpeakerDecoderLSTM')
            self.lstm = nn.LSTMCell(
                vocab_embedding_size + hidden_size, hidden_size)
            self.attention_layer = ContextOnlySoftDotAttention(hidden_size)
            self.output_l1 = nn.Linear(hidden_size * 2, hidden_size)
            self.tanh = nn.Tanh()
        else:
            self.lstm = nn.LSTMCell(vocab_embedding_size, hidden_size)
            self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, vocab_size)

    def forward(self, previous_word, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx_: batch x seq_len x query_dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        word_embeds = self.embedding(previous_word)
        word_embeds = word_embeds.squeeze()  # (batch, embedding_size)
        if not self.use_glove:
            word_embeds_drop = self.drop(word_embeds)
        else:
            word_embeds_drop = word_embeds

        if self.use_input_att_feed:
            h_tilde, alpha = self.attention_layer(
                self.drop(h_0), ctx, ctx_mask)
            concat_input = torch.cat((word_embeds_drop, self.drop(h_tilde)), 1)
            h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
            x = torch.cat((h_1, h_tilde), 1)
            x = self.drop(x)
            x = self.output_l1(x)
            x = self.tanh(x)
            logit = self.decoder2action(x)
        else:
            h_1, c_1 = self.lstm(word_embeds_drop, (h_0, c_0))
            h_1_drop = self.drop(h_1)
            h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
            logit = self.decoder2action(h_tilde)
        return h_1, c_1, alpha, logit
