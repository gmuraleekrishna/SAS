import json
import logging
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn

import utils
from agent import Seq2SeqAgent
from env_object import R2RBatch
# from env import R2RBatch
from eval import Evaluation
from model_factory import get_model
from param import args
from criterion import to_contiguous, LanguageModelCriterion, ReinforceCriterion, set_lr
from models.reward import CNNRewardModel, GRURewardModel
from utils import read_vocab, write_vocab, build_vocab, Tokenizer, timeSince, read_img_features, ReDirectSTD, \
    HiddenPrints
from log_utils import Logger

warnings.filterwarnings("ignore")

from eval_utils.cococaption.pycocoevalcap.evil import COCOEvalCaption

from tensorboardX import SummaryWriter

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

ReDirectSTD(os.path.join(args.log_dir, 'log.txt'), 'stdout', False)

print(args)


class Flag:
    def __init__(self, D_iters, G_iters, always=None):
        self.D_iters = D_iters
        self.G_iters = G_iters

        self.flag = "Disc"
        self.iters = self.D_iters
        self.curr = 0
        self.always = always

    def inc(self):
        self.curr += 1
        if self.curr >= self.iters and self.always is None:
            if self.flag == "Disc":
                self.flag = "Gen"
                self.iters = self.G_iters
            elif self.flag == "Gen":
                self.flag = "Disc"
                self.iters = self.D_iters
            self.curr = 0


def train_gan_speaker(train_env, tok, n_iters, val_envs=None, aug_env=None, augment_every=100):
    logger = Logger(args)
    if val_envs is None:
        val_envs = {}
    flag = Flag(D_iters=args.D_iter, G_iters=args.G_iter, always=args.always)
    writer = SummaryWriter(logdir=args.log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = get_model(args, train_env, listner, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)
    if args.discriminator_type == 'cnn':
        disc = CNNRewardModel(tok.vocab_size(), args.angle_feat_size + args.feature_size, args.featdropout,
                              args.activation)
    else:
        disc = GRURewardModel(tok.vocab_size(), args.angle_feat_size + args.feature_size, args.featdropout,
                              args.activation)
    disc = disc.cuda()
    disc_optimizer = args.optimizer(disc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_evaluator = val_envs['train'][1]
    rl_crit = ReinforceCriterion(args, train_evaluator)
    bad_valid = 0
    speaker.env = train_env
    for epoch in range(args.start_iter, n_iters, args.log_every):
        interval = min(args.log_every, n_iters - epoch)
        idx_tobe = epoch + interval

        print("Epoch: %05d" % epoch)
        # Evaluation
        for itr in range(interval):
            logger.iteration += 1
            speaker.zero_grad_optimizers()
            disc_optimizer.zero_grad()
            obs = speaker.env.reset()
            start = time.time()
            flag.inc()
            if flag.flag == "Disc":
                speaker.eval()
                disc.train()

                if args.disc_decoding == 'sample':
                    seq, seq_log_probs, baseline = speaker.sample(obs, sample_max=False, rl_training=True, pad=True,
                                                                  tf=False)
                elif args.disc_decoding == 'greedy':
                    seq, seq_log_probs, baseline = speaker.sample(obs, sample_max=True, rl_training=True, pad=True,
                                                                  tf=False)
            else:
                speaker.train()
                disc.eval()
                seq, seq_log_probs, baseline, tf_loss = speaker.sample(obs, sample_max=False, rl_training=True,
                                                                       ml_training=True,
                                                                       pad=True,
                                                                       tf=True,
                                                                       auxiliary=(itr % args.hparams[
                                                                           'warmup_iter']) == 0)

            seq = Variable(seq).cuda()
            mask = (seq > 0).float()
            mask = to_contiguous(torch.cat([Variable(mask.data.new(mask.size(0), mask.size(1), 1).fill_(1)),
                                            mask[:, :, :-1]], 2))
            normed_seq_log_probs = (seq_log_probs * mask).sum(-1) / mask.sum(-1)
            feature = torch.stack([torch.from_numpy(ob['feature']) for ob in obs]).cuda()
            # seq = speaker.decoder.inst_embed(seq.squeeze()).detach()
            gen_score = disc(seq, feature.view(feature.size(0), -1), mask.sum(-1).squeeze(1).int().cpu())

            if flag.flag == "Disc":
                target = torch.stack([torch.from_numpy(ob['instr_encoding']) for ob in obs]).cuda()
                # target = speaker.decoder.inst_embed(target).detach()
                lens = (target > 0).sum(-1).squeeze().int().cpu()
                gt_score = disc(target.unsqueeze(1), feature.view(feature.size(0), -1), lens)
                loss = -torch.sum(gt_score) + torch.sum(gen_score)

                avg_pos_score = torch.mean(gt_score)
                avg_neg_score = torch.mean(gen_score)
                if logger.iteration % 5 == 0:
                    logging.info(f"pos reward {avg_pos_score.item():.5f} neg reward {avg_neg_score.item():.5f}")
            else:
                rewards = Variable(gen_score.data - 0. * normed_seq_log_probs.data)
                path2inst = {ob['path_id']: seq[i][0].cpu().numpy().tolist() for i, ob in enumerate(obs)}
                loss, avg_score = rl_crit(path2inst, seq.data, seq_log_probs, baseline, rewards)
                avg_pos_score = torch.mean(gen_score)
                logging.info(f"average reward: {avg_score.item():.5f} average IRL score: {avg_pos_score.item():.5f}")

            if flag.flag == "Disc":
                loss.backward()
                nn.utils.clip_grad_norm(disc.parameters(), 40, norm_type=2)
                disc_optimizer.step()
            else:
                logging.info(f"RL loss {loss.item():.5f} TF loss {tf_loss.item():.5f}")
                loss = args.rl_weight * loss + (1 - args.rl_weight) * tf_loss
                loss.backward()
                speaker.step_optimizer()

            train_loss = loss.item()

            # Write the training loss summary
            if logger.iteration % args.log_every == 0:
                # logger.log_training(idx, train_loss, args.lr)
                logging.info(f"Epoch {epoch} Iter {itr}, Train {flag.flag}, loss = {train_loss:.5f}, "
                             f"time used = {time.time() - start:.3f}s")

                if args.always is None:
                    for env_name, (env, evaluator) in val_envs.items():
                        if 'train' in env_name:  # Ignore the large training set for the efficiency
                            continue

                        print("............ Evaluating %s ............." % env_name)
                        speaker.env = env
                        path2inst, val_loss, word_accu, sent_accu = speaker.valid()
                        metrics = {
                            'word_accu': word_accu,
                            'sent_accu': sent_accu
                        }
                        lang_metrics = evaluator.eval_batch(path2inst, metrics=('Bleu_1', 'Bleu_2', 'Bleu_3',
                                                                                'Bleu_4', 'ROUGE_L', 'CIDEr', 'SPICE'))
                        metrics.update(lang_metrics)
                        path_id = next(iter(path2inst.keys()))
                        inference = tok.decode_sentence(path2inst[path_id])
                        gt = evaluator.gt[str(path_id)]['instructions']
                        prediction = {
                            'path_id': path_id,
                            'inference': inference,
                            'gt': gt
                        }
                        print("Inference: ", inference)
                        print("GT: ", gt)

                        if args.metric == 'XE':
                            score = -val_loss
                        else:
                            score = metrics[args.metric]
                        logging.info({k: v.mean() for k, v in metrics.items()})
                        logger.log_checkpoint(epoch, itr, val_loss, metrics, prediction, args, speaker, split=env_name)
                        # halve the learning rate if not improving for a long time
                        if env_name == 'val_unseen':
                            if logger.best_val_scores[env_name] > score:
                                bad_valid += 1
                                if bad_valid >= 10:
                                    args.lr = args.lr / 5.0
                                    logging.info(f"halve learning rate to {args.lr}")
                                    checkpoint_path = os.path.join(logger.log_dir, f'model-best_{env_name}.pth')
                                    speaker.load(checkpoint_path)
                                    set_lr(speaker.decoder_optimizer, args.lr)  # set the decayed rate
                                    set_lr(speaker.encoder_optimizer, args.lr)  # set the decayed rate
                                    bad_valid = 0
                                    logging.info(f"bad valid : {bad_valid}")
                            else:
                                logging.info(f"achieving best {env_name} {args.metric} score: {score}")
                                bad_valid = 0
                else:
                    torch.save(disc.state_dict(), os.path.join(logger.log_dir, 'disc-model.pth'))
        if (epoch + 1) % augment_every == 0:
            speaker.env = aug_env
        else:
            speaker.env = train_env


def train_speaker_rl(train_env, tok, n_iters, val_envs=None):
    if val_envs is None:
        val_envs = {}
    writer = SummaryWriter(logdir=args.log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = get_model(args, train_env, listner, tok)
    pretrain_iters = -100 if args.self_critical else 500  # No pretrain for self_critical
    assert pretrain_iters % args.log_every == 0
    best_score = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 9595)
    major_metric = args.metric
    reward_func = lambda batch, path2inst: val_envs['train'][1].eval_batch(path2inst, metrics=[major_metric])
    for idx in range(args.start_iter, n_iters, args.log_every):
        interval = min(args.log_every, n_iters - idx)
        idx_tobe = idx + interval

        print("Iter: %05d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name:  # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()  # The dict here is to avoid multiple evaluation for one path
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            avg_length = utils.average_length(path2inst)
            evaluator.evaluate(path2inst, no_metrics=('METEOR', 'SPICE_action_v1'))
            name2score = evaluator.eval
            major_score = name2score[major_metric]
            score_string = " "
            for score_name, score in name2score.items():
                writer.add_scalar("lang_score/%s/%s" % (score_name, env_name), score, idx)
                score_string += "%s_%s: %0.4f " % (env_name, score_name, score)

            # Tensorboard log
            writer.add_scalar("loss/%s" % env_name, loss, idx)
            writer.add_scalar("word_accu/%s" % env_name, word_accu, idx)
            writer.add_scalar("sent_accu/%s" % env_name, sent_accu, idx)
            writer.add_scalar("avg_length/%s" % env_name, avg_length, idx)

            # Save the model according to the bleu score
            if major_score > best_score[env_name]:
                best_score[env_name] = major_score
                print('Save the model with %s env %s %0.4f' % (env_name, major_metric, major_score))
                speaker.save(os.path.join(args.log_dir, 'state_dict', 'best_%s_bleu' % env_name), idx_tobe)

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s env loss %0.4f' % (env_name, loss))
                speaker.save(os.path.join(args.log_dir, 'state_dict', 'best_%s_loss' % env_name), idx_tobe)

            # Screen print out
            print(score_string)
        print()

        # Train for log_every interval
        speaker.env = train_env

        log_dict = defaultdict(lambda: 0)

        if idx_tobe <= pretrain_iters:
            ml_weight, policy_weight, baseline_weight = 1., 0., 3.
            log_dict = speaker.rl_train(reward_func, interval, ml_weight=ml_weight, policy_weight=policy_weight,
                                        baseline_weight=baseline_weight)
            if idx_tobe == pretrain_iters:
                speaker.save(os.path.join(args.log_dir, 'state_dict',
                                          'pretrain_iter%d_%0.3f_%0.3f_%0.3f' % (
                                              idx_tobe, ml_weight, policy_weight, baseline_weight)), idx_tobe)
        else:
            rl_log = speaker.rl_train(reward_func, interval, ml_weight=0.05, policy_weight=1., baseline_weight=.5,
                                      entropy_weight=args.entropy,
                                      self_critical=args.self_critical
                                      )
            # rl_log = speaker.rl_train(reward_func, interval, ml_weight=1., policy_weight=0., baseline_weight=0., entropy_weight=args.entropy,
            #                           self_critical=args.self_critical
            #                           )
            for key, value in rl_log.items():
                log_dict[key + "/score_rl"] += value

        train_log_str = "Iter %05d, " % idx_tobe
        for name, value in log_dict.items():
            writer.add_scalar(name, value / interval, idx_tobe)
            train_log_str += "%s: %0.4f  " % (name, value / interval)
        print(train_log_str)


def train_speaker(train_env, tok, n_iters, val_envs=None, aug_env=None, augment_every=2000):
    if val_envs is None:
        val_envs = {}
    writer = SummaryWriter(logdir=args.log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    # speaker = Speaker(train_env, listner, tok)
    speaker = get_model(args, train_env, listner, tok)

    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)
    if args.fast_train:
        args.log_every = 40

    scores = {}

    best_spice = {}
    best_spice_iter = {}

    for idx in range(args.start_iter, n_iters, args.log_every):
        interval = min(args.log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train_tf(interval)  # Train interval iters

        if aug_env and idx % augment_every == 0:
            speaker.env = aug_env
            speaker.train_tf(interval)  # Train interval iters
        print()
        print("Iter: %d" % idx)

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

            with HiddenPrints():
                evaluator.evaluate(path2inst)
                scores[env_name] = {k: v for k, v in evaluator.eval.items()}

            # print("............ Evaluating %s ............." % env_name)
            # speaker.env = env
            # path2inst, loss, word_accu, sent_accu = speaker.valid()
            # path_id = next(iter(path2inst.keys()))
            # print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            # print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            # bleu_score, precisions = evaluator.bleu_score(path2inst)

            # # Tensorboard log
            # writer.add_scalar("bleu/%s" % env_name, bleu_score, idx)
            writer.add_scalar("loss/%s" % env_name, loss, idx)
            writer.add_scalar("word_accu/%s" % env_name, word_accu, idx)
            writer.add_scalar("sent_accu/%s" % env_name, sent_accu, idx)
            # writer.add_scalar("bleu4/%s" % env_name, precisions[3], idx)

            # Save the model according to the bleu score
            if env_name not in best_spice:
                best_spice[env_name] = 0
                best_spice_iter[env_name] = 0
            if scores[env_name]['SPICE'] > best_spice[env_name]:
                best_spice[env_name] = scores[env_name]['SPICE']
                best_spice_iter[env_name] = idx
                print('Save the model with %s BEST env SPICE %0.4f' % (env_name, scores[env_name]['SPICE']))
                speaker.save(os.path.join(args.log_dir, 'state_dict', 'best_%s_spice' % env_name), idx)

            # if loss < best_loss[env_name]:
            #     best_loss[env_name] = loss
            #     print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
            #     speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))
            #
            # Screen print out
            # print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))

        for env_name in ['val_seen', 'val_unseen']:
            eval_score = scores[env_name]
            print("............ Eval Results %s ............." % env_name)
            for method, score in eval_score.items():
                print("%s: %0.3f" % (method, score))
            print(f"Best SPICE yet: {best_spice[env_name]:0.3f} @ {best_spice_iter[env_name]}")


def train(train_env, tok, n_iters, val_envs=None, aug_env=None):
    if val_envs is None:
        val_envs = {}
    writer = SummaryWriter(logdir=args.log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = None
    if args.self_train:
        speaker = get_model(args, train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)

    start_iter = args.start_iter
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    best_val = {'val_seen': {"accu": 0., "state": "", 'update': False},
                'val_unseen': {"accu": 0., "state": "", 'update': False}}
    if args.fast_train:
        args.log_every = 40
    for idx in range(start_iter, n_iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, n_iters - idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:  # The default training process
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)  # Train interval iters
        else:
            if args.accumulate_grad:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    args.ml_weight = 0.2
                    listner.accumulate_gradient(feedback_method)
                    listner.env = aug_env

                    # Train with Back Translation
                    args.ml_weight = 0.6  # Sem-Configuration
                    listner.accumulate_gradient(feedback_method, speaker=speaker)
                    listner.optim_step()
            else:
                for _ in range(interval // 2):
                    # Train with GT data
                    listner.env = train_env
                    args.ml_weight = 0.2
                    listner.train(1, feedback=feedback_method)

                    # Train with Back Translation
                    listner.env = aug_env
                    args.ml_weight = 0.6
                    listner.train(1, feedback=feedback_method, speaker=speaker)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total  # / length / args.batchSize
        entropy = sum(listner.logs['entropy']) / total  # / length / args.batchSize
        predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/unsupervised", predict_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("total_actions", total)
        print("max_length", length)

        # Run validation
        loss_str = ""
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20  # 20 * 64 = 1280

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric, val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.3f' % (metric, val)

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                    iter, float(iter) / n_iters * 100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

        if iter % 50000 == 0:
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(args.log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def beam_valid(train_env, tok, val_envs={}):
    listener = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = get_model(args, train_env, listener, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    print("Loaded the listener model at iter % d" % listener.load(args.load))

    final_log = ""
    for env_name, (env, evaluator) in val_envs.items():
        listener.logs = defaultdict(list)
        listener.env = env

        listener.beam_search_test(speaker)
        results = listener.results

        def cal_score(x, alpha, avg_speaker, avg_listener):
            speaker_score = sum(x["speaker_scores"]) * alpha
            if avg_speaker:
                speaker_score /= len(x["speaker_scores"])
            # normalizer = sum(math.log(top_k) for top_k in x['listener_actions'])
            normalizer = 0.
            listener_score = (sum(x["listener_scores"]) + normalizer) * (1 - alpha)
            if avg_listener:
                listener_score /= len(x["listener_scores"])
            return speaker_score + listener_score

        if args.param_search:
            # Search for the best speaker / listener ratio
            interval = 0.01
            logs = []
            for avg_speaker in [False, True]:
                for avg_listener in [False, True]:
                    for alpha in np.arange(0, 1 + interval, interval):
                        result_for_eval = []
                        for key in results:
                            result_for_eval.append({
                                "instr_id": key,
                                "trajectory": max(results[key]['paths'],
                                                  key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                                  )['trajectory']
                            })
                        score_summary, _ = evaluator.score(result_for_eval)
                        for metric, val in score_summary.items():
                            if metric in ['success_rate']:
                                print(
                                    "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                                    (avg_speaker, avg_listener, alpha, val))
                                logs.append((avg_speaker, avg_listener, alpha, val))
            tmp_result = "Env Name %s\n" % (env_name) + \
                         "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f\n" % max(
                logs, key=lambda x: x[3])
            print(tmp_result)
            # print("Env Name %s" % (env_name))
            # print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
            #       max(logs, key=lambda x: x[3]))
            final_log += tmp_result
            print()
        else:
            avg_speaker = True
            avg_listener = True
            alpha = args.alpha

            result_for_eval = []
            for key in results:
                result_for_eval.append({
                    "instr_id": key,
                    "trajectory": [(vp, 0, 0) for vp in results[key]['dijk_path']] + \
                                  max(results[key]['paths'],
                                      key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                      )['trajectory']
                })
            # result_for_eval = utils.add_exploration(result_for_eval)
            score_summary, _ = evaluator.score(result_for_eval)

            if env_name != 'test':
                loss_str = "Env Name: %s" % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                              (avg_speaker, avg_listener, alpha, val))
                    loss_str += ",%s: %0.4f " % (metric, val)
                print(loss_str)
            print()

            if args.submit:
                json.dump(
                    result_for_eval,
                    open(os.path.join(args.log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    print(final_log)


def setup():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        # val_env_names.append('train')

    if not args.beam:
        val_env_names.append("train")

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok, name='train')
    aug_env = None
    if args.aug:
        aug_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=[args.aug], tokenizer=tok, name='aug')
    # new, COCO metrics
    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok, name=split),
           COCOEvalCaption([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, val_envs=val_envs)
        else:
            valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'rlspeaker':
        train_speaker_rl(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'ganspeaker':
        train_gan_speaker(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env,
                          augment_every=args.augment_every)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env,
                      augment_every=args.augment_every)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False


def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))


def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                       splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener', 'ganspeaker']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False
