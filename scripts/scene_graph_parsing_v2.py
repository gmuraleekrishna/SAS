import json
import copy
from collections import defaultdict

import yaml
import pickle
import spacy

nlp = spacy.load('en_core_web_lg')

import sng_parser


def check_verb(token):
    """Check verb type given spacy token"""
    if token.pos_ == 'VERB':
        indirect_object = False
        direct_object = False
        for item in token.children:
            if item.dep_ == "iobj" or item.dep_ == "pobj":
                indirect_object = True
            if item.dep_ == "dobj" or item.dep_ == "dative":
                direct_object = True
        if indirect_object and direct_object:
            return 'DITRANVERB'
        elif direct_object and not indirect_object:
            return 'TRANVERB'
        elif not direct_object and not indirect_object:
            return 'INTRANVERB'
        else:
            return 'VERB'
    else:
        return token.pos_


def get_elem(arr, ind, ret_fail=False):
    try:
        return arr[ind]
    except IndexError:
        return ret_fail


def main():
    split = 'val_seen' #, 'val_unseen'
    source = f'./datasets/FGR2R/FGR2R_{split}.json'
    target = './datasets/OBJ_FGR2R/data/OBJ_FGR2R_{}.json'.format(split)

    all_objects = dict()
    all_objects_target = './data/OBJ_FGR2R/data/all_objects_{}.pkl'.format(split)
    all_action_target = './data/OBJ_FGR2R/data/all_action_{}.json'.format(split)
    all_action = defaultdict(lambda: 0)
    with open(source, 'r') as f_:
        data = json.load(f_)

    new_data = copy.deepcopy(data)

    total_length = len(data)
    map_pairs = {
        'TRANVERB': ['ADV'],
        'INTRANVERB': ['ADP', 'ADV'],
        'NOUN': ['INTJ', 'DET', 'INTRANVERB']
    }
    for idx, item in enumerate(data):
        # for subins objects parsing
        instr_list = yaml.safe_load(item['new_instructions'])
        obj_list = []  # contains objects in all 3 instructions
        act_list = []  # contains actions in all 3 instructions
        obj_origin_list = []  # original words in the sentence
        for instr in instr_list:
            ins_obj = []
            ins_obj_ori = []
            inst_act = []
            for sub_instr in instr:
                sub_instr_sentence = ' '.join([word for word in sub_instr])
                # print(sub_instr_sentence)
                graph = sng_parser.parse(sub_instr_sentence)
                subins_obj = []
                subins_obj_ori = []

                tokens = nlp(sub_instr_sentence)
                pos_tags = [check_verb(t) for t in tokens]
                for i in range(len(tokens)):
                    tok_type_curr = get_elem(pos_tags, i, '')
                    if tok_type_curr in map_pairs.keys():
                        if i + 1 < len(tokens):
                            tok_type_next = get_elem(pos_tags, i + 1, '')
                            if tok_type_next in map_pairs[tok_type_curr]:
                                inst_act.append(f"{tokens[i]} {tokens[i + 1]}")
                                all_action[f"{tokens[i]} {tokens[i + 1]}"] += 1
                        elif tok_type_curr != 'NOUN':
                            inst_act.append(str(tokens[i]))
                            all_action[str(tokens[i])] += 1
                    if tok_type_curr == 'TRANVERB':
                        if get_elem(pos_tags, i + 2, '') == 'NOUN':
                            inst_act.append(f"{tokens[i]} {tokens[i + 2]}")
                            all_action[f"{tokens[i]} {tokens[i + 2]}"] += 1

                for entity in graph['entities']:
                    obj_word = str(entity['lemma_head'])
                    # obj_word = str(entity['head'])
                    obj_word_ori = str(entity['span'])

                    subins_obj.append(obj_word)
                    subins_obj_ori.append(obj_word_ori)

                    # add to dictionary of all object words
                    all_objects[obj_word] = 0

                ins_obj.append(subins_obj)
                ins_obj_ori.append(subins_obj_ori)

            obj_list.append(ins_obj)
            act_list.append(inst_act)
            obj_origin_list.append(ins_obj_ori)

        # merge into the data dictionary
        new_data[idx]['subins_obj_lemma_list'] = str(obj_list)
        new_data[idx]['subins_obj_ori_list'] = str(obj_origin_list)
        new_data[idx]['subins_act_list'] = str(act_list)

        # for instruction objects parsing
        instr_list = item['instructions']
        obj_list = []
        obj_origin_list = []
        for instr in instr_list:
            graph = sng_parser.parse(instr)

            ins_obj = []
            ins_obj_ori = []
            inst_act = []

            tokens = nlp(instr)
            pos_tags = [check_verb(t) for t in tokens]
            for i in range(len(tokens)):
                tok_type_curr = get_elem(pos_tags, i, '')
                if tok_type_curr in map_pairs.keys():
                    if i + 1 < len(tokens):
                        tok_type_next = get_elem(pos_tags, i + 1, '')
                        if tok_type_next in map_pairs[tok_type_curr]:
                            inst_act.append(f"{tokens[i]} {tokens[i + 1]}")
                            all_action[f"{tokens[i]} {tokens[i + 1]}"] += 1
                    elif tok_type_curr != 'NOUN':
                        inst_act.append(str(tokens[i]))
                        all_action[str(tokens[i])] += 1
                if tok_type_curr == 'TRANVERB':
                    if get_elem(pos_tags, i + 2, '') == 'NOUN':
                        inst_act.append(f"{tokens[i]} {tokens[i + 2]}")
                        all_action[f"{tokens[i]} {tokens[i + 2]}"] += 1

            for entity in graph['entities']:
                obj_word = str(entity['lemma_head'])
                # obj_word = str(entity['head'])
                obj_word_ori = str(entity['span'])

                ins_obj.append(obj_word)
                ins_obj_ori.append(obj_word_ori)

            obj_list.append(ins_obj)
            act_list.append(inst_act)
            obj_origin_list.append(ins_obj_ori)

        new_data[idx]['ins_obj_lemma_list'] = str(obj_list)
        new_data[idx]['ins_obj_ori_list'] = str(obj_origin_list)
        new_data[idx]['ins_act_list'] = str(act_list)

        if idx > 0 and idx % 200 == 0:
            print('{}/{} finished.'.format(idx, total_length))

    with open(target, 'w') as file_:
        json.dump(new_data, file_, ensure_ascii=False, indent=4)
    with open(all_action_target, 'w') as file_:
        json.dump(all_action, file_, ensure_ascii=False, indent=4)
    #
    # with open(all_objects_target, 'wb') as f:
    #     pickle.dump(all_objects, f)


if __name__ == '__main__':
    main()
