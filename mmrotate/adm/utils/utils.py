import random

import numpy as np
import torch


def fix_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def strfy(t):
    if isinstance(t, dict):
        return '{' + ','.join(['"{}":{}'.format(key, strfy(t[key])) for key in t]) + '}'
    elif isinstance(t, list) or isinstance(t, np.ndarray):
        return '[' + ','.join([strfy(el) for el in t]) + ']'
    elif isinstance(t, float) or isinstance(t, np.float32):
        return '{:.4f}'.format(t)
    else:
        raise ValueError('unknown type: ' + str(type(t)))


def common_argparser():
    import warnings; warnings.simplefilter('ignore')
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark settings.")
    parser.add_argument("--benchmark", type=str, default="nyuv2_pspnet")
    parser.add_argument("--round", type=int, default=1, help="[1..10]")
    parser.add_argument("--balancer", type=str, default="amtl", help="balancer type")
    parser.add_argument("--scale-heads", type=bool, default=False, help="scale heads or not")
    parser.add_argument("--data-path", type=str, default="/home/caomingxiang/MTL-master/NYU/", help="path to the dataset")
    parser.add_argument("--output-path", type=str, default="./logs/", help="path to store the results")
    parser.add_argument("--compute-cnumber", action='store_true', default=False,
                        help="Log training statistics such as condition number, "
                             "gradient magnitude similarity, cosine distance. "
                             "Warning: computationally expensive")

    parser.add_argument("--train-batch", type=int, default=2, help="train batch size")
    parser.add_argument("--test-batch", type=int, default=2, help="test batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="encoder learning rate")
    parser.add_argument("--epochs", type=int, default=60, help="number of epochs")
    parser.add_argument("--gpu", type=int, default=6, help="gpu id")
    parser.add_argument("--load-state", type=str, help='checkpoint path to load')
    parser.add_argument("--eval-only", action='store_true', help='perform only evaluation and exit then')
    return parser
