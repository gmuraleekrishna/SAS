import torch

import os
# from speaker import Speaker
from model_factory import get_model

from utils import read_vocab, write_vocab, build_vocab, Tokenizer, read_img_features
from env import R2RBatch
from agent import Seq2SeqAgent
from param import args

import warnings

warnings.filterwarnings("ignore")

from r2r_src.eval_utils.cococaption.pycocoevalcap.evil import COCOEvalCap

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'
CLIP32_FEATURES = 'img_features/CLIP-ViT-B-32-views.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES

if args.features == 'clip32':
    features = CLIP32_FEATURES

if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + ext

feedback_method = args.feedback  # teacher or sample

print(args)


def eval_speaker(train_env, tok, val_envs={}):
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    # speaker = Speaker(train_env, listner, tok)
    speaker = get_model(args, train_env, listner, tok)

    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    scores = {}

    # Evaluation
    for env_name, (env, evaluator) in val_envs.items():
        if 'train' in env_name:  # Ignore the large training set for the efficiency
            continue

        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid()
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[str(path_id)]['instructions'])

        # bleu_score, precisions = evaluator.bleu_score(path2inst)
        # print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))
        # print('With %s env the bleu is %0.4f' % (env_name, bleu_score))

        # new coco metrics
        evaluator.evaluate(path2inst)
        scores[env_name] = {k: v for k, v in evaluator.eval.items()}

    for env_name in ['val_seen', 'val_unseen']:
        eval = scores[env_name]
        print("............ Eval Results %s ............." % env_name)
        for method, score in eval.items():
            print("%s: %0.3f" % (method, score))


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), TRAINVAL_VOCAB)


def val():
    ''' Evaluate on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        # val_env_names.append('train')

    if not args.beam:
        val_env_names.append("train")

    # val_envs = OrderedDict(
    #     ((split,
    #       (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
    #        Evaluation([split], featurized_scans, tok))
    #       )
    #      for split in val_env_names
    #      )
    # )

    # new, COCO metrics
    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           COCOEvalCap([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'speaker':
        eval_speaker(train_env, tok, val_envs=val_envs)
    else:
        assert False


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        val()
    else:
        assert False
