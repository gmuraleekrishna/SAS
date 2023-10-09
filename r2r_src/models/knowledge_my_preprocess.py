import copy
import json
import os
import pickle
import re
import time
from collections import Counter
from typing import Dict, List, Union

import numpy as np
import requests
from transformers import BertTokenizer, BertModel

_Relation = Dict[str, Union[str, float]]  # {'@id': str, 'weight': float}
Relation = Dict[str, Union[str, List[_Relation]]]  # {'@id': str, 'related': List[_Relation]}

import torch
import torch.nn as nn


class FactEncoder(nn.Module):
    """
    Applies a multi-layer RNN to an x sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `Q`
        input_dropout_p (float): dropout probability for the x sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.
    """

    def __init__(self, vocab_size, max_len, embedding_size, hidden_size, input_dropout_p, dropout_p, n_layers,
                 rnn_cell='lstm',
                 bidirectional=False, variable_lengths=True, embedding=None, update_embedding=True):
        super(FactEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an x sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the x sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the x sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state Q
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths and input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True,
                                                         enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths and input_lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class SceneObjectAttention(nn.Module):
    def __init__(self, d_model=256, vis_dim=2048, fact_dim=256):
        super(SceneObjectAttention, self).__init__()
        self.q_layer = nn.Linear(vis_dim, d_model)
        self.k_layer = nn.Linear(fact_dim, d_model)
        self.v_layer = nn.Linear(fact_dim, d_model)
        self.d_model = d_model
        self.scale = self.d_model ** -0.5

    def forward(self, pano_feats, fact_embeds):
        # pano feat [batch (* path_len), 36, vis_dim]
        # fact_embeds [batch (* path_len), n_fact, fact_dim]

        vis_dim = pano_feats.shape[2]
        ba, n_fact, fact_dim = fact_embeds.shape

        pano_feats = pano_feats.reshape(ba * 36, vis_dim)
        fact_embeds = fact_embeds.reshape(ba * n_fact, fact_dim)

        q = self.q_layer(pano_feats)
        k = self.k_layer(fact_embeds)
        v = self.v_layer(fact_embeds)

        q = q.reshape(ba, 36, self.d_model)
        k = k.reshape(ba, n_fact, self.d_model)
        v = v.reshape(ba, n_fact, self.d_model)

        dots = torch.bmm(q, k.transpose(1, 2))  # ba, 36, n_fact
        dots = nn.functional.softmax(dots * self.scale, dim=2)

        output = torch.bmm(dots, v)  # ba, 36, self.d_model

        return output



def request_entity_obj():
    # entity_path = 'D:\\MyFiles\\Projects\\CKR-main\\KB\\data\\entities.txt'
    entity_path = './scripts/entities.txt'

    with open(entity_path, 'r') as entity_file:
        core_entities = [entity[:-1].split(',')[0].replace(' ', '_') for entity in entity_file]

    obj_cache = {}

    for i, entity in enumerate(core_entities):
        obj = requests.get('http://api.conceptnet.io/c/en/{}?filter=/c/en'.format(entity)).json()
        obj_cache[entity] = copy.deepcopy(obj)

        if i % 200 == 0:
            print('{}/{} finished.'.format(i, len(core_entities)))
        time.sleep(0.5)

    with open('data/pkls/obj_cache.pkl', 'wb') as f:
        pickle.dump(obj_cache, f)


# check CKR cache
def main():
    cache_path = './data/conceptnet.cache'
    cache: Dict[str, Relation] = {}  # @id -> obj

    with open(cache_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            cache[obj['@id']] = obj

    cnt = 0
    for k, v in cache.items():
        cnt += 1
        if cnt == 1600:
            print(k)
            print(v)
            print(type(v))
            print(v['related'])
            print(type(v['related']), len(v['related']))
            break

    # exit(0)

    embedding_path = './data/embedding.cache'

    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    print(type(embeddings))
    print(embeddings.shape)

    entity_path = './scripts/entities.txt'

    with open(entity_path, 'r') as entity_file:
        core_entities = [entity[:-1].split(',')[0] for entity in entity_file]

    print(core_entities)
    print(len(core_entities))


def test_fact_encoder():
    with open('data/pkls/knowledge_rel_embed.pkl', 'rb') as f:
        knowledge_dict_new, embedding_matrix = pickle.load(f)

    model = FactEncoder(vocab_size=embedding_matrix.shape[0],
                        max_len=20, embedding_size=768, hidden_size=256,
                        input_dropout_p=0.3, dropout_p=0, n_layers=1,
                        rnn_cell='gru', bidirectional=False, variable_lengths=True,
                        embedding=embedding_matrix, update_embedding=False)

    test_obj_idx = [1, 432, 555, 123, 1234]

    for obj_idx in test_obj_idx:
        fact, lengths = knowledge_dict_new[obj_idx]
        output, hidden = model(fact, lengths)
        print(output.shape, hidden.shape)


def process_entity_knowledge():
    with open('data/pkls/obj_cache.pkl', 'rb') as f:
        obj_cache = pickle.load(f)

    print(len(obj_cache))

    entity_path = './scripts/entities.txt'

    with open(entity_path, 'r') as entity_file:
        core_entities = [entity[:-1].split(',')[0].replace(' ', '_') for entity in entity_file]

    knowledge_cnt = 0
    knowledge_dict = {}  # i_entity -> [[knowledge_i_1], ..., [knowledge_i_n]]

    for i, entity in enumerate(core_entities):
        obj = obj_cache[entity]
        edges = obj['edges']
        knowledge_entity = set()

        for edge in edges:
            if 'language' in edge['start'] and 'language' in edge['end']:
                if edge['start']['language'] != 'en' or edge['end']['language'] != 'en':
                    continue

            st_label = edge['start']['label'].replace('_', ' ').replace('-', ' ').lower()
            ed_label = edge['end']['label'].replace('_', ' ').replace('-', ' ').lower()
            entity_label = entity.replace('_', ' ').replace('-', ' ').lower()

            # fix matching bug
            if ed_label == 'tile a floor':
                ed_label = 'tile floor'

            edge_type = -1
            if st_label == entity_label:
                edge_type = 0
            elif ed_label == entity_label:
                edge_type = 1
            elif st_label in entity_label:
                edge_type = 0
            elif ed_label in entity_label:
                edge_type = 1
            else:
                print(entity_label, ',', st_label, ',', ed_label)

            tmp = []  # [rel_entity, rel]
            if edge_type == 0:  # st_label is the core entity, then add the ed_label
                tmp.append(ed_label)
            else:
                tmp.append(st_label)

            rel = re.findall(r'([A-Z][a-z]*)', edge['rel']['label'])
            rel = [word.lower() for word in rel]
            tmp.append(" ".join(rel))

            knowledge_entity.add(' '.join(tmp))
            knowledge_cnt += 1
            # print('{} {} {}'.format(edge['start']['label'], edge['rel']['label'], edge['end']['label']))
            # print(len(obj['related']))
            # print(obj['related'][0])
            # break

        knowledge_dict[i] = knowledge_entity
    # print(glove_embeds['<unk>'])

    with open('data/pkls/knowledge.pkl', 'wb') as f:
        pickle.dump(knowledge_dict, f)


def process_entity_knowledge_glove():
    with open('data/pkls/obj_cache.pkl', 'rb') as f:
        obj_cache = pickle.load(f)

    print(len(obj_cache))

    entity_path = './scripts/entities.txt'

    with open(entity_path, 'r') as entity_file:
        core_entities = [entity[:-1].split(',')[0].replace(' ', '_') for entity in entity_file]

    knowledge_cnt = 0
    knowledge_dict = {}  # i_entity -> [[knowledge_i_1], ..., [knowledge_i_n]]

    glove_embeds = {'<unk>': 'na'}

    for i, entity in enumerate(core_entities):
        obj = obj_cache[entity]
        edges = obj['edges']
        knowledge_entity = []

        for edge in edges:
            if 'language' in edge['start'] and 'language' in edge['end']:
                if edge['start']['language'] != 'en' or edge['end']['language'] != 'en':
                    continue

            st_label = edge['start']['label'].replace('_', ' ').replace('-', ' ').lower()
            ed_label = edge['end']['label'].replace('_', ' ').replace('-', ' ').lower()
            entity_label = entity.replace('_', ' ').replace('-', ' ').lower()

            # fix matching bug
            if ed_label == 'tile a floor':
                ed_label = 'tile floor'

            edge_type = -1
            if st_label == entity_label:
                edge_type = 0
            elif ed_label == entity_label:
                edge_type = 1
            elif st_label.find(entity_label) > -1:
                edge_type = 0
            elif ed_label.find(entity_label) > -1:
                edge_type = 1
            else:
                print(entity_label, ',', st_label, ',', ed_label)

            tmp = []  # [rel_entity, rel]
            if edge_type == 0:  # st_label is the core entity, then add the ed_label
                tmp.append(ed_label.split(' '))
            else:
                tmp.append(st_label.split(' '))

            rel = re.findall('([A-Z][a-z]*)', edge['rel']['label'])
            rel = [word.lower() for word in rel]
            tmp.append(rel)

            # save vocabulary for later
            for knowledge_component in tmp:
                for word in knowledge_component:
                    if word not in glove_embeds:
                        glove_embeds[word] = 'na'

            knowledge_entity.append(tmp)
            knowledge_cnt += 1
            # print('{} {} {}'.format(edge['start']['label'], edge['rel']['label'], edge['end']['label']))
            # print(len(obj['related']))
            # print(obj['related'][0])
            # break

        knowledge_dict[i] = knowledge_entity

    embedding_path = './data/glove.840B.300d.txt'
    # embedding_path = '/data/haitian/data/KB/data/glove.840B.300d.txt'
    emb_size = 300
    emb_cnt = 0

    with open(embedding_path, 'r', encoding='utf-8') as emb_file:
        for line_no, line in enumerate(emb_file):
            line = line.split(' ')
            assert len(line) == emb_size + 1, f"Invalid embedding in file {embedding_path} line {line_no}."
            now_word = line[0]
            if now_word not in glove_embeds:
                continue
            glove_embeds[now_word] = np.array([float(e) for e in line[1:]])
            emb_cnt += 1

    print('embed found: {}/{}.'.format(emb_cnt, len(glove_embeds)))

    unk_cnt = 0
    for k, v in glove_embeds.items():
        if v == 'na':
            glove_embeds[k] = glove_embeds['<unk>']
            unk_cnt += 1

    print(unk_cnt, unk_cnt + emb_cnt)

    with open('./data/pkls/knowledge_plus_embed_glove.pkl', 'wb') as f:
        pickle.dump([knowledge_dict, glove_embeds], f)


def build_fact_vocab_glove():
    with open('data/pkls/knowledge_plus_embed_glove.pkl', 'rb') as f:
        knowledge_dict, glove_embeds = pickle.load(f)

    num_fact = 6
    knowledge_dict_new = {}
    for k, v in knowledge_dict.items():
        relation_embeds = []
        for relation in v:
            # last word of rel_entity
            rel_entity_embed = glove_embeds[relation[0][-1]]
            if rel_entity_embed == 'na':
                rel_entity_embed = np.zeros(300)
            # average word embedding of relation words
            rel = np.zeros(300)
            for rel_word in relation[1]:
                rel += glove_embeds[rel_word]
            if len(relation[1]) > 0:
                rel_embed = rel / len(relation[1])

            relation_embeds.append(torch.from_numpy(np.concatenate((rel_entity_embed, rel_embed), axis=0)).float())

        if len(relation_embeds) == 0:
            relation_embeds.append(torch.zeros(600))

        relation_embeds = torch.stack(relation_embeds, dim=0)
        if len(relation_embeds) > num_fact:
            choice = torch.randperm(len(relation_embeds))
            relation_embeds = relation_embeds[choice[:num_fact], :]
        elif len(relation_embeds) < num_fact:
            relation_embeds = torch.cat((relation_embeds, torch.zeros((num_fact - len(relation_embeds), 600))), dim=0)

        assert relation_embeds.shape[0] == num_fact
        assert relation_embeds.shape[1] == 600

        knowledge_dict_new[k] = relation_embeds

    with open('data/pkls/knowledge_rel_embed_glove.pkl', 'wb') as f:
        pickle.dump(knowledge_dict_new, f)


def build_fact_vocab_bert():
    with open('knowledge_plus_embed.pkl', 'rb') as f:
        knowledge_dict, glove_embeds = pickle.load(f)

    start_vocab = ['<PAD>', '<BOS>', '<EOS>']

    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    for k, v in knowledge_dict.items():
        for relation in v:
            count.update(relation)
    vocab = start_vocab[:]
    for word, num in count.most_common():
        vocab.append(word)

    embedding_matrix = torch.empty((len(vocab), 300))
    word_to_idx = {}

    for idx, word in enumerate(vocab):
        embedding_matrix[idx] = torch.from_numpy(glove_embeds[word])
        word_to_idx[word] = idx

    embedding_matrix = embedding_matrix.float()

    max_len = 20

    knowledge_dict_new = {}
    for k, v in knowledge_dict.items():
        relations = []
        lengths = []
        for relation in v:
            padded_relation = [word_to_idx['<BOS>']]
            padded_relation.extend([word_to_idx[word] for word in relation])
            padded_relation.append(word_to_idx['<EOS>'])

            if len(padded_relation) < max_len:
                lengths.append(len(padded_relation))
                padded_relation.extend([word_to_idx['<PAD>'] for _ in range(max_len - len(padded_relation))])
            elif len(padded_relation) > max_len:
                lengths.append(max_len)
                padded_relation = padded_relation[:max_len]
                padded_relation[-1] = word_to_idx['<EOS>']

            relations.append(torch.LongTensor(padded_relation))
        if len(relations) == 0:
            relations = torch.zeros((1, max_len))
            lengths.append(1)
        else:
            relations = torch.stack(relations, dim=0)

        knowledge_dict_new[k] = [relations, lengths]

    with open('data/pkls/knowledge_rel_embed_bert.pkl', 'wb') as f:
        pickle.dump([knowledge_dict_new, embedding_matrix], f)


if __name__ == '__main__':
    # main()
    # request_entity_obj()
    # process_entity_knowledge_glove()
    # build_fact_vocab_bert()
    build_fact_vocab_glove()
