import os
import sys
import time
from collections import defaultdict
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from utils import utils, evaluate, data_utils

device = utils.device


class HyperMod(nn.Module):
	def __init__(self, input_dim, vidx, eidx, nv, ne, v_weight, e_weight, args, is_last=False, use_edge_lin=False):
		super(HyperMod, self).__init__()
		self.args = args
		self.eidx = eidx
		self.vidx = vidx
		self.v_weight = v_weight
		self.e_weight = e_weight
		self.nv, self.ne = args.nv, args.ne

		self.W_v2e = Parameter(torch.randn(args.n_hidden, args.n_hidden))
		self.W_e2v = Parameter(torch.randn(args.n_hidden, args.n_hidden))
		self.b_v = Parameter(torch.zeros(args.n_hidden))
		self.b_e = Parameter(torch.zeros(args.n_hidden))
		self.is_last_mod = is_last
		self.use_edge_lin = use_edge_lin
		if is_last and self.use_edge_lin:
			self.edge_lin = torch.nn.Linear(args.n_hidden, args.final_edge_dim)

	def forward(self, v, e, batch_idx):
		bsz = e.shape[0]

		start_idx = batch_idx * bsz
		end_idx = (batch_idx + 1) * bsz
		start_idx_3 = batch_idx * bsz * 3
		end_idx_3 = (batch_idx + 1) * bsz * 3

		if args.edge_linear:
			ve = torch.matmul(v, self.W_v2e) + self.b_v
		else:
			ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)
		# weigh ve according to how many edges a vertex is connected to
		v_fac = 4 if args.predict_edge else 1
		v = v * self.v_weight * v_fac

		eidx = self.eidx.unsqueeze(-1).expand(-1, self.args.n_hidden)

		e = e.clone()
		ve = (ve * self.v_weight)[self.args.paper_author[:, 0]][start_idx_3:end_idx_3]
		ve *= args.v_reg_weight[start_idx_3:end_idx_3]
		e = e.scatter_add(src=ve, index=eidx[:bsz*3], dim=0)
		e /= args.e_reg_sum[start_idx:end_idx]
		ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)

		vidx = self.vidx.unsqueeze(-1).expand(-1, self.args.n_hidden)
		# ev_vtx = (ev * self.e_weight[start_idx:end_idx])[self.args.paper_author[:, 1][start_idx_3:end_idx_3]]
		ev_vtx = (ev * self.e_weight[start_idx:end_idx])[self.args.paper_author[:, 1][:bsz*3]]
		ev_vtx *= args.e_reg_weight[start_idx_3:end_idx_3]
		v = v.scatter_add(src=ev_vtx, index=vidx[:bsz*3], dim=0)
		v /= args.v_reg_sum
		if not self.is_last_mod:
			v = F.dropout(v, args.dropout_p)
		if self.is_last_mod and self.use_edge_lin:
			ev_edge = (ev * torch.exp(self.e_weight) / np.exp(2))[self.args.paper_author[:, 1]]
			v2 = torch.zeros_like(v)
			v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
			v2 = self.edge_lin(v2)
			v = torch.cat([v, v2], -1)
		return v, e


class Hypergraph(nn.Module):
	"""
	Hypergraph class, uses weights for vertex-edge and edge-vertex incidence matrix.
	One large graph.
	"""

	def __init__(self, vidx, eidx, nv, ne, v_weight, e_weight, args):
		"""
		vidx: idx tensor of elements to select, shape (ne, max_n),
		shifted by 1 to account for 0th elem (which is 0)
		eidx has shape (nv, max n)..
		"""
		super(Hypergraph, self).__init__()
		self.args = args
		self.hypermods = []

		is_first = True
		for i in range(args.n_layers):
			is_last = True if i == args.n_layers - 1 else False
			self.hypermods.append(
				HyperMod(args.input_dim if is_first else args.n_hidden, vidx, eidx, nv, ne, v_weight, e_weight, args, is_last=is_last))
			is_first = False

		if args.predict_edge:
			self.edge_lin = torch.nn.Linear(args.input_dim, args.n_hidden)

		self.vtx_lin = torch.nn.Linear(args.input_dim, args.n_hidden)
		# self.cls = nn.Linear(args.n_hidden, args.n_cls)
		self.affine_output = nn.Linear(args.n_hidden, 1)
		self.logistic = nn.Sigmoid()

	def to_device(self, device):
		self.to(device)
		for mod in self.hypermods:
			mod.to(device)
		return self

	def all_params(self):
		params = []
		for mod in self.hypermods:
			params.extend(mod.parameters())
		return params

	def forward(self, v, e, batch_idx):
		"""
			Take initial embeddings from the select labeled data.
			Return predicted cls.
		"""
		v = self.vtx_lin(v)
		if self.args.predict_edge:
			e = self.edge_lin(e)
		for mod in self.hypermods:
			v, e = mod(v, e, batch_idx)

		logits = self.affine_output(e)
		pred = self.logistic(logits)  # pred is the implicit rating in (0, 1)
		return v, e, pred


class Hypertrain:
	def __init__(self, args):
		self.loss_fn = nn.BCELoss()  # consider logits

		self.hypergraph = Hypergraph(args.vidx, args.eidx, args.nv, args.ne, args.v_weight, args.e_weight, args)
		self.optim = optim.Adam(self.hypergraph.all_params(), lr=.04)

		milestones = [100 * i for i in range(1, 4)]  # [100, 200, 300]
		self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=milestones, gamma=0.51)
		self.args = args

	def train(self, v, e, label_idx, labels):
		self.hypergraph = self.hypergraph.to_device(device)
		v_init = v
		e_init = e

		# create dataloader here from v, e, labels and label_idx
		# in each batch, all nodes must be present.
		# batching must be done only for the hyperedges
		train_dataset = TensorDataset(e_init[label_idx], labels)
		train_loader = DataLoader(
			dataset=train_dataset,
			batch_size=self.args.batch_size,
			shuffle=False
		)

		test_dataset = TensorDataset(e_init[self.args.val_idx], self.args.val_labels)
		test_loader = DataLoader(
			dataset=test_dataset,
			batch_size=self.args.batch_size,
			shuffle=False
		)

		print(f"train_loader len: {len(train_loader)}")

		best_err = sys.maxsize

		for epoch in tqdm(range(self.args.n_epoch)):
			args.cur_epoch = epoch
			epoch_losses = []
			for batch_idx, data in enumerate(train_loader):
				self.optim.zero_grad()
				batch_e, batch_labels = [item.to(device) for item in data]

				v, e, pred = self.hypergraph(v_init, batch_e, batch_idx)
				loss = self.loss_fn(pred.squeeze(), batch_labels.float())
				epoch_losses.append(loss.item())

				loss.backward()
				self.optim.step()
				self.scheduler.step()
			epoch_losses = np.array(epoch_losses)

			print(f'train loss: {np.mean(epoch_losses)}')
			test_err = self.eval(v_init, test_loader)
			print("test loss:", test_err)
			# if test_err < best_err:
			# 	best_err = test_err

		# return pred_all, loss, best_err

	def eval(self, v_init, test_loader):
		preds = []
		for batch_idx, data in tqdm(enumerate(test_loader), position=0, leave=False):
			batch_e, batch_labels = [item.to(device) for item in data]

			v, e, pred = self.hypergraph(v_init, batch_e, batch_idx)
			preds.extend(pred.cpu().detach().tolist())

		# preds = all_pred[self.args.val_idx].squeeze()
		preds = torch.Tensor(preds).squeeze()
		tgt = self.args.val_labels
		fn = nn.BCELoss()
		loss = fn(preds, tgt.float())
		HR, NDCG = evaluate.metrics(self.args, preds)
		print(f"HR@{args.top_k}: {HR} | NDCG@{args.top_k}: {NDCG}")

		return loss.item()


def train(args):
	"""
		args.vidx, args.eidx, args.nv, args.ne, args = s
		args.e_weight = s
		args.v_weight = s
		label_idx, labels = s
	"""
	# args.e = torch.randn(args.ne, args.n_hidden)
	if args.predict_edge:
		args.e = args.edge_X
	else:
		args.e = torch.zeros(args.ne, args.n_hidden)
	# args.v = torch.randn(self.args.nv, args.n_hidden)
	hypertrain = Hypertrain(args)
	hypertrain.train(args.v, args.e, args.label_idx, args.labels)

	# pred_all, loss, test_err = hypertrain.train(args.v, args.e, args.label_idx, args.labels)
	return test_err


def gen_data(args, data_dict, flip_edge_node=False, do_val=False):
	"""
		Retrieve and process data, can be used generically for any dataset with predefined data format, eg cora, citeseer, etc.
		flip_edge_node: whether to flip edge and node in case of relation prediction.
	"""
	# data_dict = torch.load(data_path)
	paper_author = torch.LongTensor(data_dict['paper_author']).to(device)
	author_paper = torch.LongTensor(data_dict['author_paper']).to(device)
	n_author = data_dict['n_author']  # num users
	n_paper = data_dict['n_paper']  # num items
	classes = data_dict['classes']  # [0, 1]
	paper_X = data_dict['paper_X']  # item features

	train_len = data_dict['train_len']
	test_len = data_dict['test_len']
	test_loader = data_dict['test_loader']

	if args.predict_edge:  # 'author_X' in data_dict:
		# edge representations
		author_X = data_dict['author_X']
		author_classes = data_dict['author_classes']

	paperwt = data_dict['paperwt']
	authorwt = data_dict['authorwt']
	# n_cls = data_dict['n_cls']
	cls_l = list(set(classes))

	args.edge_X = torch.from_numpy(author_X).to(torch.float32)
	args.edge_classes = torch.LongTensor(author_classes).to(device)

	cls2int = {k: i for (i, k) in enumerate(cls_l)}
	classes = [cls2int[c] for c in classes]
	args.input_dim = paper_X.shape[-1]  # 300 if args.dataset_name == 'citeseer' else 300
	args.n_hidden = 800 if args.predict_edge else 400
	args.final_edge_dim = 100
	args.n_epoch = 140 if args.n_layers == 1 else 230  # 130 #120
	args.ne = n_author
	args.nv = n_paper
	ne = args.ne
	nv = args.nv
	args.n_cls = len(cls_l)

	# n_labels = max(1, math.ceil(nv * utils.get_label_percent(args.dataset_name)))  # TODO: compare
	# args.all_labels = torch.LongTensor(classes)  # TODO: compare
	# args.label_idx = torch.from_numpy(np.random.choice(nv, size=(n_labels,), replace=False)).to(torch.int64)  # TODO: compare
	# n_labels = max(1, math.ceil(ne * utils.get_label_percent(args.dataset_name)))
	# n_labels = ne
	# args.all_labels = torch.LongTensor(args.edge_classes)
	# args.label_idx = torch.from_numpy(np.random.choice(ne, size=(n_labels,), replace=False)).to(torch.int64)
	#
	# print('getting validation indices...')
	# val_idx = torch.from_numpy(
	# 	np.random.choice(len(args.label_idx), size=math.ceil(ne * utils.get_label_percent(args.dataset_name)),
	# 	                 replace=False))
	# args.val_idx = args.label_idx[val_idx.long()]
	# args.val_labels = args.all_labels[args.val_idx].to(device)

	n_labels = ne
	args.all_labels = torch.cuda.LongTensor(args.edge_classes) if torch.cuda.is_available() else torch.LongTensor(args.edge_classes)
	args.label_idx = torch.from_numpy(np.arange(n_labels)).to(torch.int64)

	print('\ngetting validation indices...')
	val_idx = torch.from_numpy(np.arange(start=train_len, stop=train_len+test_len))
	args.val_idx = args.label_idx[val_idx.long()]
	args.val_labels = args.all_labels[args.val_idx]

	ones = torch.ones(len(args.label_idx))
	ones[args.val_idx] = -1

	args.label_idx = args.label_idx[ones > -1]
	args.labels = args.all_labels[args.label_idx]
	args.all_labels = args.all_labels

	args.test_loader = test_loader

	if isinstance(paper_X, np.ndarray):
		args.v = torch.from_numpy(paper_X.astype(np.float32))
	else:
		args.v = torch.from_numpy(np.array(paper_X.astype(np.float32).todense()))

	args.vidx = paper_author[:, 0]
	args.eidx = paper_author[:, 1]
	args.paper_author = paper_author
	args.v_weight = torch.Tensor([(1 / w if w > 0 else 1) for w in paperwt]).unsqueeze(-1)  # torch.ones((nv, 1)) / 2 #####
	args.e_weight = torch.Tensor([(1 / w if w > 0 else 1) for w in authorwt]).unsqueeze(-1)  # 1)) / 2 #####torch.ones(ne, 1) / 3
	assert len(args.v_weight) == nv and len(args.e_weight) == ne

	paper2sum = defaultdict(list)
	author2sum = defaultdict(list)
	e_reg_weight = torch.zeros(len(paper_author))  ###
	v_reg_weight = torch.zeros(len(paper_author))  ###
	# a switch to determine whether to have wt in exponent or base
	use_exp_wt = args.use_exp_wt  # True #False
	for i, (paper_idx, author_idx) in enumerate(paper_author.tolist()):
		e_wt = args.e_weight[author_idx]
		e_reg_wt = torch.exp(args.alpha_e * e_wt) if use_exp_wt else e_wt ** args.alpha_e
		e_reg_weight[i] = e_reg_wt
		paper2sum[paper_idx].append(e_reg_wt)  ###

		v_wt = args.v_weight[paper_idx]
		v_reg_wt = torch.exp(args.alpha_v * v_wt) if use_exp_wt else v_wt ** args.alpha_v
		v_reg_weight[i] = v_reg_wt
		author2sum[author_idx].append(v_reg_wt)  ###
	# """
	v_reg_sum = torch.zeros(nv)
	e_reg_sum = torch.zeros(ne)
	for paper_idx, wt_l in paper2sum.items():
		v_reg_sum[paper_idx] = sum(wt_l)
	for author_idx, wt_l in author2sum.items():
		e_reg_sum[author_idx] = sum(wt_l)

	# pdb.set_trace()
	# this is used in denominator only
	e_reg_sum[e_reg_sum == 0] = 1
	v_reg_sum[v_reg_sum == 0] = 1
	args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(device)
	args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(device)
	args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(device)
	args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(device)
	print('dataset processed into tensors')
	return args


def select_params(data_dict, args):
	# find best hyperparameters with by splitting training set into train + validation set
	best_err = sys.maxsize
	best_err_std = sys.maxsize
	best_alpha_v = sys.maxsize
	best_alpha_e = sys.maxsize
	print('ARGS {}'.format(args))
	mean_err_l = []
	mean_err_std_l = []
	time_l = []
	time_std_l = []
	args.kfold = 1  # 5
	args.kfold = 5  # 5
	kfold = args.kfold
	a_list = range(-1, 2, 1)
	a_list1 = []
	test_alpha = False  # True
	for av in a_list:
		if test_alpha:
			a_list1.append([0, av])  # test alpha
		else:
			for ae in a_list:
				a_list1.append([av, ae])
	sys.stdout.write('alpha beta list {}'.format(a_list1))
	for av, ae in a_list1:
		args.alpha_v = av / 10
		args.alpha_e = ae / 10

		args = gen_data(args, data_dict=data_dict, do_val=True)

		time0 = time.time()
		test_err = train(args)
		time_ar = time.time() - time0
		err_ar = test_err
		sys.stdout.write(' Validation err {}\t'.format(test_err))

		mean_err = err_ar.mean()
		err_std = err_ar.std()
		mean_err_l.append(mean_err)
		mean_err_std_l.append(err_std)
		dur = time_ar.mean()
		time_l.append(dur)
		time_std_l.append(time_ar.std())

		sys.stdout.write('\n ~~~Mean VAL err {}+-{} for alpha {} {} time {}~~~\n'.format(np.round(mean_err, 2), np.round(err_std, 2), args.alpha_v, args.alpha_e, dur))
		if mean_err < best_err:
			best_err = mean_err
			best_err_std = err_std
			best_alpha_v = args.alpha_v
			best_alpha_e = args.alpha_e
			best_time = np.round(dur, 3)
			best_time_std = time_ar.std()
	print('mean validation errs {} mean err std {}'.format(mean_err_l, mean_err_std_l))
	print('best err {}+-{} best alpha_v {} alpha_e {} for dataset {}'.format(np.round(best_err * 100, 2), np.round(best_err_std * 100, 2), best_alpha_v, best_alpha_e, args.dataset_name))
	print('best validation ACC {}+-{} time {}+-{}'.format(np.round((1 - best_err) * 100, 2), np.round(best_err_std * 100, 2), best_time, best_time_std))
	return best_alpha_v, best_alpha_e


if __name__ == '__main__':
	args = utils.parse_args()
	dataset_name = args.dataset_name
	data_path = os.path.join('data', args.dataset_name, args.dataset_name + '.pt')
	data_dict = data_utils.load_data_dict(args)

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	select_params(data_dict, args)
	print('Done!')
