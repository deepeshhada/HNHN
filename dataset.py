from collections import defaultdict
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GraphDataset(Dataset):
	def __init__(self, args, negatives=None):
		self.node_X = args.v
		self.edge_X = args.e
		self.v_weight = args.v_weight
		self.positive_vidx = args.vidx
		self.negatives = args.train_negatives
		self.num_ng = args.num_ng

		self.v_reg_sum = args.v_reg_sum
		self.v_reg_wt = args.v_weight ** args.alpha_v  # shape = (3229); to be used for e_reg_sum in __get_item__

		self.args = args

	def __len__(self):
		return self.edge_X.shape[0]

	def __getitem__(self, index):
		vidx = self.positive_vidx[index:index+3]
		negatives = self.get_negative_instances(index)
		vidx = torch.cat((vidx, negatives))

		edge_x = self.node_X[vidx].view(-1, 3, 300).sum(dim=1)

		v_reg_weight = self.v_reg_wt[vidx]
		# edge_x = self.edge_X[index].unsqueeze(0)
		# negative_x = torch.rand((self.num_ng, 300))
		# edge_x = torch.cat((edge_x, negative_x))

		return vidx, v_reg_weight, edge_x

	def get_negative_instances(self, index):
		return self.negatives[index:index+self.num_ng].view(-1)


class Collate:
	def __init__(self, args):
		self.args = args
		self.num_ng = args.num_ng
		self.v_reg_sum = args.v_reg_sum
		self.e_reg_wt = (1 / 3) ** args.alpha_e  # shape = 1

	def __call__(self, batch):
		batch_size = len(batch) * (self.num_ng + 1)
		eidx = torch.from_numpy(np.arange(batch_size)).repeat_interleave(3).long()
		e_reg_weight = (torch.ones(batch_size*3) * self.e_reg_wt).unsqueeze(-1)
		labels = torch.zeros(self.num_ng + 1)
		labels[0] = 1
		labels = labels.repeat(len(batch))

		vidx = batch[0][0]
		v_reg_weight = batch[0][1]
		edge_x = batch[0][2]
		for idx in range(1, len(batch)):
			vidx = torch.cat((vidx, batch[idx][0]))
			v_reg_weight = torch.cat((v_reg_weight, batch[idx][1]))
			edge_x = torch.cat((edge_x, batch[idx][2]))

		e_reg_sum = v_reg_weight.reshape((-1, 3)).sum(dim=1).unsqueeze(-1)

		return {
			"e": edge_x,
			"vidx": vidx.long(),
			"eidx": eidx,
			"v_reg_weight": v_reg_weight,
			"e_reg_weight": e_reg_weight,
			"v_reg_sum": self.v_reg_sum,
			"e_reg_sum": e_reg_sum,
			"labels": labels
		}


class GraphTestDataset(Dataset):
	def __init__(self, args):
		data_path = os.path.join('data', args.dataset_name, args.dataset_name + '.graph')
		train_len = torch.load(os.path.join('data', args.dataset_name, "data_dict/train_len.pth"))
		test_points = []
		labels = []
		with open(data_path, 'r') as f:
			lines = f.readlines()
			for i, line in enumerate(lines):
				if i < train_len:
					continue
				line = line.split('\t')
				test_points.append((int(line[0]), int(line[1]), int(line[2])))
				labels.append(float(line[3]))

		self.vidx = torch.tensor(test_points).view(-1)
		self.labels = torch.tensor(labels)
		self.node_X = args.v
		self.edge_X = torch.rand((self.labels.shape[0], 300))
		self.v_weight = args.v_weight
		self.v_reg_wt = args.v_weight ** args.alpha_v
		self.args = args

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, index):
		vidx = self.vidx[index:index+3]
		v_reg_weight = self.v_reg_wt[vidx]
		# edge_x = self.edge_X[index].unsqueeze(0)
		edge_x = self.node_X[vidx].view(-1, 3, 300).sum(dim=1)
		label = self.labels[index]

		return vidx, v_reg_weight, edge_x, label.view(1)  # can put view in init


class CollateTest:
	def __init__(self, args):
		self.args = args
		self.num_ng = args.num_ng
		self.v_reg_sum = args.v_reg_sum
		self.e_reg_wt = (1 / 3) ** args.alpha_e

	def __call__(self, batch):
		batch_size = len(batch)
		eidx = torch.from_numpy(np.arange(batch_size)).repeat_interleave(3).long()
		e_reg_weight = (torch.ones(batch_size*3) * self.e_reg_wt).unsqueeze(-1)

		vidx = batch[0][0]
		v_reg_weight = batch[0][1]
		edge_x = batch[0][2]
		labels = batch[0][3]
		for idx in range(1, len(batch)):
			vidx = torch.cat((vidx, batch[idx][0]))
			v_reg_weight = torch.cat((v_reg_weight, batch[idx][1]))
			edge_x = torch.cat((edge_x, batch[idx][2]))
			labels = torch.cat((labels, batch[idx][3]))

		e_reg_sum = v_reg_weight.reshape((-1, 3)).sum(dim=1).unsqueeze(-1)

		return {
			"e": edge_x,
			"vidx": vidx.long(),
			"eidx": eidx,
			"v_reg_weight": v_reg_weight,
			"e_reg_weight": e_reg_weight,
			"v_reg_sum": self.v_reg_sum,
			"e_reg_sum": e_reg_sum,
			"labels": labels
		}
