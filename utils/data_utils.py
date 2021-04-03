import pandas as pd
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import random
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


class HNHNData(object):
    """
        Construct Dataset for HNHN
    """

    def __init__(self, args, ratings):
        random.seed(args.seed)
        self.args = args
        self.ratings = ratings
        self.num_ng = args.num_ng
        self.num_ng_test = args.num_ng_test
        self.batch_size = args.test_batch_size

        self.preprocess_ratings = self._reindex(self.ratings)

        self.user_pool = set(self.ratings['user_id'].unique())
        self.item_pool = set(self.ratings['item_id'].unique())

        self.train_ratings, self.test_ratings = self._leave_one_out(self.preprocess_ratings)
        self.negatives = self._negative_sampling(self.preprocess_ratings)

    def _reindex(self, ratings):
        """
            Process dataset to reindex userID and itemID, also set rating as binary feedback
        """
        user_list = list(ratings['user_id'].drop_duplicates())
        user2id = {w: i for i, w in enumerate(user_list)}

        item_list = list(ratings['item_id'].drop_duplicates())
        item2id = {w: i for i, w in enumerate(item_list)}

        ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
        ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
        ratings['rating'] = ratings['rating'].apply(lambda x: float(x > 0))
        return ratings

    def _leave_one_out(self, ratings):
        ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
        test = ratings.loc[ratings['rank_latest'] == 1]
        train = ratings.loc[ratings['rank_latest'] > 1]
        assert train['user_id'].nunique() == test['user_id'].nunique(), 'Not Match Train User with Test User'
        return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

    def _negative_sampling(self, ratings):
        interact_status = (
            ratings.groupby('user_id')['item_id']
                .apply(set)
                .reset_index()
                .rename(columns={'item_id': 'interacted_items'}))
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(
            lambda x: random.sample(x, self.num_ng_test))
        return interact_status[['user_id', 'negative_items', 'negative_samples']]

    def save_train_instance(self):
        save_path = os.path.join('data', self.args.dataset_name, 'train.csv')
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'negative_items']], on='user_id')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))
        for row in train_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in range(self.num_ng):
                users.append(int(row.user_id))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = RatingDataset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        df = pd.DataFrame(columns=('users', 'items', 'ratings'))
        df['users'], df['items'], df['ratings'] = dataset.user_list, dataset.item_list, dataset.rating_list
        df.to_csv(save_path, sep='\t', index=False, header=False)
        return df

    def save_test_instance(self):
        save_path = os.path.join('data', self.args.dataset_name, 'test.csv')
        users, items, ratings = [], [], []
        test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'negative_samples']], on='user_id')
        for row in test_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in getattr(row, 'negative_samples'):
                users.append(int(row.user_id))
                items.append(int(i))
                ratings.append(float(0))
        dataset = RatingDataset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        df = pd.DataFrame(columns=('users', 'items', 'ratings'))
        df['users'], df['items'], df['ratings'] = dataset.user_list, dataset.item_list, dataset.rating_list
        df.to_csv(save_path, sep='\t', index=False, header=False)
        return df


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, user_list, item_list, rating_list):
        super(RatingDataset, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item = self.item_list[idx]
        rating = self.rating_list[idx]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )


def save_embeddings(args, ids, mode):
    save_path = os.path.join('data', args.dataset_name, mode + '_embeddings.pkl')
    embeddings = nn.Embedding(len(ids), args.embed_dim)
    embed_dict = dict()

    for idx, node_id in enumerate(ids):
        embed_dict[node_id] = embeddings(torch.LongTensor([idx])).detach().squeeze().numpy()

    with open(save_path, 'wb') as f:
        pickle.dump(embed_dict, f, pickle.HIGHEST_PROTOCOL)


def read_embeddings(args, mode):
    load_path = os.path.join('data', args.dataset_name, mode + '_embeddings.pkl')
    with open(load_path, 'rb') as f:
        embed_dict = pickle.load(f)
    return embed_dict


def load_data_dict(root_path):
    n_edges = torch.load(os.path.join(root_path, 'n_author.pth'))
    n_nodes = torch.load(os.path.join(root_path, 'n_paper.pth'))
    classes = torch.load(os.path.join(root_path, 'classes.pth'))
    edge_classes = torch.load(os.path.join(root_path, 'author_classes.pth'))
    node_edge = torch.load(os.path.join(root_path, 'paper_author.pth'))
    edge_node = torch.load(os.path.join(root_path, 'author_paper.pth'))
    nodewt = torch.load(os.path.join(root_path, 'paperwt.pth'))
    edgewt = torch.load(os.path.join(root_path, 'authorwt.pth'))
    X = torch.load(os.path.join(root_path, 'paper_X.pth'))
    # edge_X = torch.load(os.path.join(root_path, 'author_X.pth'))
    edge_X = np.load(os.path.join(root_path, 'author_X.npy'))
    train_len = torch.load(os.path.join(root_path, 'train_len.pth'))
    test_len = torch.load(os.path.join(root_path, 'test_len.pth'))
    test_loader = torch.load(os.path.join(root_path, 'test_loader.pth'))
    cls2idx = torch.load(os.path.join(root_path, 'user_item_cls_map.pth'))
    train_negatives = torch.load(os.path.join(root_path, 'train_negatives.pth'))

    data_dict = {
        'n_author': n_edges,
        'n_paper': n_nodes,
        'classes': classes,
        'author_classes': edge_classes,
        'paper_author': node_edge,
        'author_paper': edge_node,
        'paperwt': nodewt,
        'authorwt': edgewt,
        'paper_X': X,
        'author_X': edge_X,
        'train_len': train_len,
        'test_len': test_len,
        'test_loader': test_loader,
        'user_item_cls_map': cls2idx,
        'train_negatives': train_negatives
    }

    return data_dict

