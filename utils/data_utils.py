import pandas as pd
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle
import warnings
import sklearn.metrics


res_dir = 'results'
data_dir = 'data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_embeddings(ids, out_dim, mode="user"):
    embeddings = nn.Embedding(len(ids), out_dim)
    embed_dict = dict()
    for idx, node_id in enumerate(ids):
        embed_dict[node_id] = embeddings(torch.LongTensor([idx])).detach().squeeze().numpy()

    with open('./data/generic/' + mode + '_embeddings.pkl', 'wb') as f:
        pickle.dump(embed_dict, f, pickle.HIGHEST_PROTOCOL)


def read_embeddings(mode="user"):
    with open('./data/generic/' + mode + '_embeddings.pkl', 'rb') as f:
        embed_dict = pickle.load(f)
    return embed_dict
