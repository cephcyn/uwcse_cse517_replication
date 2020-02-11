import argparse

parser = argparse.ArgumentParser(
    description='Process Reddit post CSV, build and score clusters.')
parser.add_argument('--model',
                    help='The model type to use for clustering (default: word2vec)',
                    default='word2vec',
                    choices=['word2vec', 'lda', 'bert'])

args = parser.parse_args()

print("Using model", args.model)
