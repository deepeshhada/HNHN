"""
    Utilities for the framework.
"""
import pandas as pd
import numpy as np
import os
import random
import argparse
import torch
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt

import pickle
import warnings
import sklearn.metrics
warnings.filterwarnings('ignore')

res_dir = 'results'
data_dir = 'data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser()
    #    
    parser.add_argument("--verbose", dest='verbose', action='store_const', default=False, const=True, help='Print out verbose info during optimization')
    parser.add_argument("--exp_wt", dest='use_exp_wt', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--do_svd", action='store_const', default=False, const=True, help='use svd')
    parser.add_argument("--method", type=str, default='hypergcn', help='which baseline method')
    parser.add_argument("--kfold", default=3, type=int, help='for k-fold cross validation')
    parser.add_argument("--predict_edge", action='store_const', default=False, const=True, help='whether to predict edges')
    parser.add_argument("--edge_linear", action='store_const', default=False, const=True, help='linerity')
    parser.add_argument("--alpha_e", default=-0.1, type=float, help='alpha')
    parser.add_argument("--alpha_v", default=-0.1, type=float, help='alpha')
    parser.add_argument("--dropout_p", default=0.3, type=float, help='dropout')
    parser.add_argument("--n_layers", default=1, type=int, help='number of layers')
    parser.add_argument("--seed", default=42, type=int, help='seed for reproducibility')
    parser.add_argument("--top_k", default=10, type=int, help='top_k predictions for HR and NDCG')
    parser.add_argument("--embed_dim", default=800, type=int, help='user, item and list embedding sizes')
    parser.add_argument("--learning_rate", default=0.004, type=float, help='learning rate')
    parser.add_argument("--n_epoch", default=600, type=int, help='number of epochs to train for')
    parser.add_argument("--batch_size", default=2048, type=int, help='batch size of train data loader')
    parser.add_argument("--test_batch_size", default=101, type=int, help='batch size of test data loader')
    parser.add_argument("--dataset_name", type=str, default='MusicalInstruments', help='dataset name')
    parser.add_argument("--num_ng", type=int, default=4, help="Number of negative samples for training set")
    parser.add_argument("--num_ng_test", type=int, default=100, help="Number of negative samples for test set")

    opt = parser.parse_args()
    # if opt.predict_edge and opt.dataset_name != 'citeseer':
    #     raise Exception('edge prediction not currently supported on {}'.format(opt.dataset_name))
    return opt


def readlines(path):
    with open(path, 'r') as f:
        return f.readlines()
    

def get_label_percent(dataset_name):
    if dataset_name == 'cora':
        return .052
    elif dataset_name == 'citeseer':
        return .15 
    elif dataset_name == 'dblp':
        return .04
    else:
        return .15
    
    
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
