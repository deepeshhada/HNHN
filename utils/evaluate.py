import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0


def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(args, test_preds):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	test_preds = test_preds.to(device)
	preds_dataset = TensorDataset(test_preds)
	pred_loader = DataLoader(
		dataset=preds_dataset,
		batch_size=args.test_batch_size,
		shuffle=False
	)
	loader_iter = iter(pred_loader)

	HR, NDCG = [], []

	for user, item, label in args.test_loader:
		item = item.to(device)
		predictions = next(loader_iter)[0]

		_, indices = torch.topk(predictions, args.top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		ng_item = item[0].item()  # leave one-out evaluation has only one item per user
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))

	return np.mean(HR), np.mean(NDCG)