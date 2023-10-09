from models.utils import parse_param
from r2r_src.models.speaker import Speaker as SASpeaker
from speaker import Speaker


def get_model(args, train_env, listner, tok):
    # Parse model-specific hyper parameters
    hparams = parse_param(args.hparams)

    if args.method == 'sas':
        args.hparams = hparams
        return SASpeaker(train_env, listner, tok, args)
    elif args.method == 'sf':
        args.hparams = hparams
        return Speaker(train_env, listner, tok)
    return None
