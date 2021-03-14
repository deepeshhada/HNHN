"""
	Processing data
"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import random
from collections import defaultdict
import re
import sklearn
import sklearn.feature_extraction as feat_extract
from sklearn.decomposition import TruncatedSVD
from utils import utils, data_utils


def save_splits(args):
	data_path = os.path.join("data", args.dataset_name, args.dataset_name + ".csv")
	dataset = pd.read_csv(
		data_path,
		names=['user_id', 'item_id', 'rating', 'timestamp'],
		engine='python',
		header=None
	)

	print('constructing train and test datasets and saving splits...')
	data = data_utils.HNHNData(args, dataset)
	train_df = data.save_train_instance()
	test_df = data.save_test_instance()

	print('saving user, item and list embeddings...')
	num_users = train_df['users'].nunique()
	num_items = train_df['items'].nunique()
	num_lists = num_items

	user_ids = np.arange(num_users)
	item_ids = np.arange(num_users, num_users + num_items)
	list_ids = np.arange(num_users + num_items, num_users + num_items + num_lists)

	data_utils.save_embeddings(args, user_ids, mode='user')
	data_utils.save_embeddings(args, item_ids, mode='item')
	data_utils.save_embeddings(args, list_ids, mode='list')


def load_preprocessed(args):
	"""
		reads preprocessed dataset, saves combined train df, test df and test_loader.
		:returns
			len(train_set), len(test_set), test_loader
	"""
	train_data_path = os.path.join('data', args.dataset_name, 'train.csv')
	test_data_path = os.path.join('data', args.dataset_name, 'test.csv')
	df_save_path = os.path.join('data', args.dataset_name, args.dataset_name + '.graph')

	train_df = pd.read_csv(train_data_path, sep='\t', header=None)
	test_df = pd.read_csv(test_data_path, sep='\t', header=None)
	df = train_df.append(test_df)

	train_df.columns = ['user_id', 'item_id', 'rating']
	test_df.columns = ['user_id', 'item_id', 'rating']
	df.columns = ['user_id', 'item_id', 'rating']
	num_users = train_df['user_id'].nunique()
	num_items = train_df['item_id'].nunique()

	df['list_id'] = df['item_id']

	df['item_id'] = df['item_id'] + num_users
	df['list_id'] = df['list_id'] + num_users + num_items

	df = df[['user_id', 'item_id', 'list_id', 'rating']]

	df.to_csv(df_save_path, sep='\t', index=False, header=False)

	print('saved graph df!')

	test_df = df[len(train_df):]
	del test_df['list_id']
	test_df.columns = ['users', 'items', 'ratings']
	users = torch.Tensor(test_df['users'])
	items = torch.Tensor(test_df['items'])
	ratings = torch.Tensor(test_df['ratings'])

	test_loader = DataLoader(
		dataset=TensorDataset(users, items, ratings),
		batch_size=args.test_batch_size,
		shuffle=False
	)

	return len(train_df), len(test_df), test_loader


def process_generic_edge(args):
	"""
		Processing Citeseer data including the edges. Include
		citeseer.content of form <paper_id> <word_attributes> + <class_label>
		citeseer.cites of form <id of cited paper> <id of citing paper>
		paper_citing, cited papers are hypernodes, citing papers are hyperedges.

		Each row is a hyperedge. Users+Items are hypernodes.
		Associate class with each hyperedge (captured in author classes)
		author/citing = edge, paper = node

		node2edge: for each hypernode v, a set of hyperedges e1, ..., ek incident on it.
		edge2node: hypernodes connected by each hyperedge e. In our case, each hyperedge connects a user and an item.
			Therefore, dim(e) = 2 for each e in E.

	"""
	train_len, test_len, test_loader = load_preprocessed(args)
	data_path = os.path.join('data', args.dataset_name, args.dataset_name + '.graph')

	feat_dim = args.embed_dim
	user_embeddings = data_utils.read_embeddings(args, mode="user")
	item_embeddings = data_utils.read_embeddings(args, mode="item")
	list_embeddings = data_utils.read_embeddings(args, mode="list")

	node2edge = defaultdict(set)
	edge2node = defaultdict(set)

	print('building node2edge and edge2node relationships...')

	with open(data_path, 'r') as f:
		lines = f.readlines()
		edge_idx = 0
		for line in lines:
			line = line.strip().split('\t')
			try:
				node2edge[line[0]].add(str(edge_idx))
				node2edge[line[1]].add(str(edge_idx))
				node2edge[line[2]].add(str(edge_idx))
				# hyperedge connects which nodes - add line[0] and line[1] to each edge id
				edge2node[str(edge_idx)].add(line[0])
				edge2node[str(edge_idx)].add(line[1])
				edge2node[str(edge_idx)].add(line[2])
				edge_idx += 1
			except ValueError as e:
				continue

	id2node_idx = {}
	id2edge_idx = {}
	node_edge = []  # node
	edge_node = []  # edge
	nodewt = torch.zeros(len(node2edge))
	edgewt = torch.zeros(len(edge2node))

	print('assigning weights to nodes and edges...')

	for edge, nodes in edge2node.items():
		if len(nodes) == 1:
			# remove self loops: shouldn't be valid in our case.
			continue
		edge_idx = id2edge_idx[edge] if edge in id2edge_idx else len(id2edge_idx)
		id2edge_idx[edge] = edge_idx
		for node in nodes:
			node_idx = id2node_idx[node] if node in id2node_idx else len(id2node_idx)
			id2node_idx[node] = node_idx
			node_edge.append([node_idx, edge_idx])
			edge_node.append([edge_idx, node_idx])
			edgewt[edge_idx] += 1
			nodewt[node_idx] += 1

	n_nodes = len(id2node_idx.keys())
	n_edges = len(id2edge_idx.keys())
	nodewt = nodewt[:n_nodes]
	edgewt = edgewt[:n_edges]
	X = np.zeros((n_nodes, feat_dim))
	edge_X = np.zeros((n_edges, feat_dim))
	classes = [0] * n_nodes
	edge_classes = [0] * n_edges
	node_idx_set = set()
	edge_idx_set = set()

	print('preparing feature vectors and adding nodes and edges to sets...')

	with open(data_path, 'r') as f:
		lines = f.readlines()
		edge_id = 0
		for line in lines:
			line = line.strip().split('\t')
			user_id = line[0]
			item_id = line[1]
			list_id = line[2]

			user_x = torch.FloatTensor(user_embeddings[int(user_id)])
			item_x = torch.FloatTensor(item_embeddings[int(item_id)])
			list_x = torch.FloatTensor(list_embeddings[int(list_id)])

			if user_id in id2node_idx:
				idx = id2node_idx[user_id]
				node_idx_set.add(idx)
				X[idx] = user_x
				classes[idx] = 'user'
			if item_id in id2node_idx:
				idx = id2node_idx[item_id]
				node_idx_set.add(idx)
				X[idx] = item_x
				classes[idx] = 'item'
			if list_id in id2node_idx:
				idx = id2node_idx[list_id]
				node_idx_set.add(idx)
				X[idx] = list_x
				classes[idx] = 'list'
			if str(edge_id) in id2edge_idx:
				idx = id2edge_idx[str(edge_id)]
				edge_idx_set.add(idx)
				edge_X[idx] = torch.rand(feat_dim)  # TODO: save the edge representations?
				edge_classes[idx] = line[-1]
				edge_id += 1

		print('node idx set: {}'.format(len(node_idx_set)))
		print('edge idx set: {}'.format(len(edge_idx_set)))

	print('total number of nodes (users amd items) = {}'.format(len(X)))
	cls2idx = {}
	for cls in set(classes):
		cls2idx[cls] = cls2idx[cls] if cls in cls2idx else len(cls2idx)
	classes = [cls2idx[c] for c in classes]

	edge_classes = [int(float(c)) for c in edge_classes]
	print('saving dataset...')
	root_save_path = os.path.join('data', args.dataset_name, 'data_dict')
	if not os.path.exists(root_save_path):
		os.mkdir(root_save_path)
	save_data_dict(root_save_path, 'n_author', n_edges)
	save_data_dict(root_save_path, 'n_paper', n_nodes)
	save_data_dict(root_save_path, 'classes', classes)
	save_data_dict(root_save_path, 'author_classes', edge_classes)
	save_data_dict(root_save_path, 'paper_author', node_edge)
	save_data_dict(root_save_path, 'author_paper', edge_node)
	save_data_dict(root_save_path, 'paperwt', nodewt)
	save_data_dict(root_save_path, 'authorwt', edgewt)
	save_data_dict(root_save_path, 'paper_X', X)
	# save_data_dict(root_save_path, 'author_X', edge_X)
	save_data_dict(root_save_path, 'train_len', train_len)
	save_data_dict(root_save_path, 'test_len', test_len)
	save_data_dict(root_save_path, 'test_loader', test_loader)
	save_data_dict(root_save_path, 'user_item_cls_map', cls2idx)
	np.save(os.path.join(root_save_path, 'author_X.npy'), edge_X)

	print('Saved dataset at "{}"'.format(root_save_path))


def save_data_dict(root_path, key, value):
	torch.save(value, os.path.join(root_path, key + '.pth'))


if __name__ == '__main__':
	args = utils.parse_args()

	print(f'seeding for reproducibility at {args.seed}...')
	utils.seed_everything(args.seed)
	save_splits(args)
	process_generic_edge(args)
