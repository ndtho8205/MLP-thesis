import os
import sys

from mlp import MLP

# dir & path
DIR = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(DIR, '../'))

DATA_PATH = os.path.join(DIR, './data')
MLP_MODEL_PATH = os.path.join(DIR, './models')


def main(args):
    if args.train:
        train()
    if args.predict:
        print('Predicting...')


def train():
    mlp = MLP(MLP_MODEL_PATH)
    mlp.fit(
        os.path.join(DATA_PATH, './x_train_512.pkl'), os.path.join(DATA_PATH, './y_train_512.pkl'),
        os.path.join(DATA_PATH, './x_test_512.pkl'), os.path.join(DATA_PATH, './y_test_512.pkl'))


def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Realtime classify face is Known or Unknown.')

    parser.add_argument(
        '--train', dest='train', action='store_const', help='Start training.', const=True, default=False)
    parser.add_argument(
        '--predict', dest='predict', action='store_const', help='Start predicting.', const=True, default=False)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_args())
