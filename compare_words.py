import json
from copy import copy
from typing import List, Dict, Set

import numpy as np
import stanza
from tqdm import tqdm

# nlp = stanza.Pipeline('en', processors='tokenize,lemma')
# with open('./data/R2R/annotations/R2R_test_enc.json') as test_jsn:
#     test_data = json.load(test_jsn)
# with open('./data/R2R/annotations/R2R_train_enc.json') as train_jsn:
#     train_data = json.load(train_jsn)
# with open('./data/R2R/annotations/R2R_val_seen_enc.json') as val_seen_jsn:
#     val_seen_data = json.load(val_seen_jsn)
# with open('./data/R2R/annotations/R2R_val_unseen_enc.json') as val_unseen_jsn:
#     val_unseen_data = json.load(val_unseen_jsn)
#
#
# def get_words(episodes: List[Dict]) -> List:
#     unique_words = []
#     for epi in tqdm(episodes):
#         for inst in epi['instructions']:
#             doc = nlp(inst)
#             for sent in doc.sentences:
#                 unique_words.extend([word.text.lower() for word in sent.words])
#     unique_words = list(set(unique_words))
#     return unique_words
#
#
# test_words = get_words(test_data)
# train_words = get_words(train_data)
# val_unseen_words = get_words(val_unseen_data)
# val_seen_words = get_words(val_seen_data)
#
# with open('all_unique_words.json', 'w') as jsn:
#     json.dump({'splits': {
#         'test': test_words,
#         'train': train_words,
#         'val_unseen': val_unseen_words,
#         'val_seen': val_seen_words
#     }}, jsn)

with open('all_unique_words.json', 'r') as jsn:
    uniques = json.load(jsn)
diff_matrix = np.zeros((4, 4), dtype=int)
sym_diff_matrix = np.zeros((4, 4), dtype=int)
comm_matrix = np.zeros((4, 4), dtype=int)

sym_diff_words = {}
diff_words = {}
comm_words = {}
for i, (spl1, spl1_words) in enumerate(uniques['splits'].items()):
    for j, (spl2, spl2_words) in enumerate(uniques['splits'].items()):
        # if spl1 != spl2:
        diff = set(spl1_words).difference(set(spl2_words))
        sym_diff = set(spl1_words).symmetric_difference(set(spl2_words))
        inter = set(spl1_words).intersection(set(spl2_words))

        diff_words[f"{spl1}_{spl2}"] = list(diff)
        sym_diff_words[f"{spl1}_{spl2}"] = list(sym_diff)
        comm_words[f"{spl1}_{spl2}"] = list(inter)

        diff_matrix[i, j] = len(diff)
        sym_diff_matrix[i, j] = len(sym_diff)
        comm_matrix[i, j] = len(inter)

with open('all_unique_words.json', 'w') as jsn:
    uniques = copy(uniques)
    uniques['intersection'] = comm_words
    uniques['symmetric_difference'] = sym_diff_words
    uniques['difference'] = diff_words

    uniques['diff_matrix'] = diff_matrix.tolist()
    uniques['sym_diff_matrix'] = sym_diff_matrix.tolist()
    uniques['comm_matrix'] = comm_matrix.tolist()
    json.dump(uniques, jsn)

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20,
                     "font.family": "serif",
                     'font.serif': 'Linux Libertine O',
                     'font.weight': 'bold',
                     "savefig.dpi": 600})

axis_names = ["".join([_.title() for _ in w.split('_')]) for w in uniques['splits'].keys()]

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('Difference (X-Y)')
ax.set_xticks(np.arange(4), axis_names)
ax.set_yticks(np.arange(4), axis_names, rotation=45, ha='right')
ax.set_xlabel("Y")
ax.set_ylabel("X")
plt.imshow(diff_matrix)
plt.colorbar()
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('Symmetric Difference $(X-Y) \\cup (Y-X)$')
ax.set_xticks(np.arange(4), axis_names)
ax.set_yticks(np.arange(4), axis_names, rotation=45, ha='right')
ax.set_xlabel("Y")
ax.set_ylabel("X")
plt.imshow(sym_diff_matrix)
plt.colorbar()
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('Common $(X \\cup Y)$')
ax.set_xticks(np.arange(4), axis_names)
ax.set_yticks(np.arange(4), axis_names,  rotation=45, ha='right')
ax.set_xlabel("Y")
ax.set_ylabel("X")
plt.imshow(comm_matrix)
plt.colorbar()
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('Symmetric Ratio $ \\frac{(X-Y)\cup (Y-X)}{X\cap Y} $')
ax.set_xticks(np.arange(4), axis_names)
ax.set_yticks(np.arange(4), axis_names, rotation=45, ha='right')
plt.imshow(sym_diff_matrix / comm_matrix)
ax.set_xlabel("Y")
ax.set_ylabel("X")
plt.colorbar()
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title('Difference Ratio $ \\frac{X-Y}{X\cap Y} $')
ax.set_xticks(np.arange(4), axis_names)
ax.set_yticks(np.arange(4), axis_names, rotation=45, ha='right')
plt.imshow(diff_matrix / comm_matrix)
ax.set_xlabel("Y")
ax.set_ylabel("X")
plt.colorbar()
plt.show()
#############################
# plt.rcParams.update({'font.size': 22})
# fig = plt.figure(figsize=(20, 1))
# ax = fig.add_subplot(321)
# ax.set_title('Difference {X-Y}')
# ax.set_xticks(np.arange(4), list(uniques['splits'].keys()), rotation = 45)
# ax.set_yticks(np.arange(4), list(uniques['splits'].keys()))
# im = ax.imshow(diff_matrix)
# plt.colorbar(im, ax=ax)
#
# ax = fig.add_subplot(322)
# ax.set_title('Symmetric Difference {(X-Y)|(Y-X)}')
# ax.set_xticks(np.arange(4), list(uniques['splits'].keys()), rotation = 45)
# ax.set_yticks(np.arange(4), list(uniques['splits'].keys()))
# im = ax.imshow(sym_diff_matrix)
# plt.colorbar(im, ax=ax)
#
# ax = fig.add_subplot(323)
# ax.set_title('Common {X&Y}')
# ax.set_xticks(np.arange(4), list(uniques['splits'].keys()), rotation = 45)
# ax.set_yticks(np.arange(4), list(uniques['splits'].keys()))
# im = ax.imshow(comm_matrix)
# plt.colorbar(im, ax=ax)
#
# ax = fig.add_subplot(324)
# ax.set_title('Symmetric Ratio {((Y-X)|(Y-X))/(X&Y)} ')
# ax.set_xticks(np.arange(4), list(uniques['splits'].keys()), rotation = 45)
# ax.set_yticks(np.arange(4), list(uniques['splits'].keys()))
# im = ax.imshow(sym_diff_matrix / comm_matrix)
# plt.colorbar(im, ax=ax)
#
# ax = fig.add_subplot(325)
# ax.set_title('Difference Ratio {(X-Y)/(X&Y)} ')
# ax.set_xticks(np.arange(4), list(uniques['splits'].keys()), rotation = 45)
# ax.set_yticks(np.arange(4), list(uniques['splits'].keys()))
# im = ax.imshow(diff_matrix / comm_matrix)
# plt.colorbar(im, ax=ax)
# plt.subplots_adjust(bottom=0.15)
# plt.show()
