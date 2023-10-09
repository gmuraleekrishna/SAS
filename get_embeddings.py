import pickle

import numpy as np

from r2r_src.utils import read_vocab, Tokenizer

vocab = read_vocab('./tasks/R2R/data/train_vocab.txt')
tok = Tokenizer(vocab=vocab, encoding_length=80)

emb_size = 300
glove_embeds = np.zeros((tok.vocab_size(), emb_size))
emb_cnt = 0

with open('./data/glove.840B.300d.txt', 'r', encoding='utf-8') as emb_file:
    for line_no, line in enumerate(emb_file):
        line = line.split(' ')
        assert len(line) == emb_size + 1, f"Invalid embedding GloVe line {line_no}."
        now_word = line[0]
        if now_word not in tok.word_to_index.keys():
            continue
        glove_embeds[tok.word_to_index[now_word], :] = np.array([float(e) for e in line[1:]])
        emb_cnt += 1

np.save('./data/pkls/glove_vocab_embedding.npy', glove_embeds)
