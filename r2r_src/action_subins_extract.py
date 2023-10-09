import stanza
import json
import copy
import yaml
import pickle

# directional
directional = dict()  # left 0, right 1, around 2
for ins in ['turn right', 'turn to the right', 'make a right', 'veer right']:
    directional[ins] = 0
for ins in ['turn left', 'turn to the left', 'make a left', 'veer left']:
    directional[ins] = 1
for ins in ['turn around', 'turn 180 degrees', 'make a 180 degree turn', 'veer around']:
    directional[ins] = 2


def main():
    split = 'train'  # 'test', 'val_seen', 'val_unseen'
    source = '../tasks/FGR2R/data/FGR2R_{}.json'.format(split)
    target = '../tasks/OBJ_FGR2R/data/OBJVGv3_FGR2R_{}.json'.format(split)

    with open(source, 'r') as f_:
        data = json.load(f_)

    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma')

    total_length = len(data)

    all_verbs = dict()

    for idx, item in enumerate(data):
        instr_list = eval(item['new_instructions'])
        verb_list = []  # contains objects in all 3 instructions
        for instr in instr_list:
            ins_verb = []
            for sub_instr in instr:
                sub_instr_sentence = ' '.join([word for word in sub_instr])
                # print(sub_instr_sentence)
                doc = nlp(sub_instr_sentence)

                subins_verbs = []

                for sent in doc.sentences:
                    for word in sent.words:
                        # print(word, word.upos)
                        if word.upos == 'VERB':
                            subins_verbs.append(word.lemma)

                            if word.lemma not in all_verbs:
                                all_verbs[word.lemma] = 0

        if idx % 100 == 0 and idx > 0:
            print('{}/{} finished.'.format(idx, total_length))

    for k, _ in all_verbs.items():
        print(k)

    with open(target, 'w') as file_:
        json.dump(new_data, file_, ensure_ascii=False, indent=4)
    #
    # with open(all_objects_target, 'wb') as f:
    #     pickle.dump(all_objects, f)


if __name__ == '__main__':
    main()
