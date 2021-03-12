import os
import argparse
import numpy as np
import pandas as pd


def process(args):
	ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

	df = pd.read_json(os.path.join(ROOT_PATH, args.dataset_name + '.json'), lines=True)
	df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
	df.columns = ['userId', 'itemId', 'rating', 'timestamp']
	df['userId'] = df['userId'].astype('category').cat.codes
	df['itemId'] = df['itemId'].astype('category').cat.codes

	outdir = os.path.join(ROOT_PATH, args.dataset_name)
	if not os.path.exists(outdir):
		os.mkdir(outdir)

	df.to_csv(os.path.join(outdir, args.dataset_name + '.csv'), index=False, header=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="InstantVideo",
		choices=("InstantVideo", "MusicalInstruments"),
		help="Name of the Amazon Dataset."
	)

	process(parser.parse_args())
