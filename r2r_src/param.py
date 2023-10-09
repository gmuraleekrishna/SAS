import argparse
import os
import torch


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--iters', type=int, default=100000)
        self.parser.add_argument('--start_iter', type=int, default=0)
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--train', type=str, default='speaker')
        self.parser.add_argument('--method', type=str, default='sf')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max x instruction")
        self.parser.add_argument('--maxDecode', type=int, default=80, help="max x instruction")
        self.parser.add_argument('--maxAction', type=int, default=20, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=64)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim", action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--load", type=str, default=None)
        self.parser.add_argument("--baseline", type=str, default='linear')

        # More Paths from
        self.parser.add_argument("--aug", default='./data/R2R/annotations/augment__trainv2.json')
        self.parser.add_argument("--augment_every", default=15, type=int)
        self.parser.add_argument("--log_every", type=int, default=100)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
        self.parser.add_argument("--features", type=str, default='imagenet')

        self.parser.add_argument("--metric", type=str, default='SPICE')
        self.parser.add_argument("--rl_weight", type=float, default=0.2)
        self.parser.add_argument("--entropy", default=0., type=float)
        self.parser.add_argument("--sameInBatch", dest="same_in_batch", action="store_const", default=False, const=True)
        self.parser.add_argument("--normalizeReward", dest='normalize_reward', action='store_const', default=False, const=True)

        # Env Dropout Param
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # SSL configuration
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        self.parser.add_argument("--sample_top_k", action='store_true')
        self.parser.add_argument("--sample_top_p", action='store_true')

        self.parser.add_argument("--self_critical", action='store_true', default=False)
        self.parser.add_argument("--grad_baseline", action='store_true')

        self.parser.add_argument("--reward_type", type=str, default='SPICE')
        self.parser.add_argument("--discriminator_type", type=str, default='cnn', help='cnn or gru')
        self.parser.add_argument("--disc_decoding", type=str, default='sample')
        self.parser.add_argument('--always', type=str, default=None, help='always train one model, no alternating '                                                          'training')
        self.parser.add_argument('--D_iter', type=int, default=50, help='Discriminator update iterations')
        self.parser.add_argument('--G_iter', type=int, default=50, help='Generator update iterations')
        self.parser.add_argument('--activation', type=str, default="sign",
                            help='the last activation function of the reward model: sign | tahn')
        self.parser.add_argument("--slot_share_qk", action='store_true')

        self.parser.add_argument("--slot_dropout", type=float, default=0, help='dropout rate for slot attention')
        self.parser.add_argument("--slot_ignore_end", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_noise", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_residual", action="store_const", default=False, const=True)
        self.parser.add_argument("--slot_local_mask", action="store_const", default=False, const=True)
        self.parser.add_argument('--slot_local_mask_h', type=int, default=3, help='local mask horizontal span')
        self.parser.add_argument('--slot_local_mask_v', type=int, default=3, help='local mask vertical span')

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--seed', type=int, default=1)
        self.parser.add_argument('--feedback', type=str, default='sample',
                                 help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                                 help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=512)
        self.parser.add_argument('--aemb', type=int, default=64)
        self.parser.add_argument('--proj', type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)

        self.parser.add_argument("--bidir", type=bool, default=True)    # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--encoder", type=str, default="pano")    # pano

        self.parser.add_argument("--trial", action="store_true", default=False)

        self.parser.add_argument("--resume_from", action="store_true", default=False)
        # dis_shift
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        # Configuring models
        self.parser.add_argument("--model_name", type=str, default='Speaker_sas', help='model name')
        self.parser.add_argument("--hparams", type=str, default='', help='model hyper-parameters')

        self.parser.add_argument("--train_small", action='store_true', default=False)

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False


param = Param()
args = param.args
args.TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
args.TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.CANDIDATE_FEATURES = 'img_features/ResNet-152-candidate.tsv'
args.features_fast = 'img_features/ResNet-152-imagenet-fast.tsv'
args.log_dir = 'logs/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join(args.log_dir, "debug.log"), 'w')
