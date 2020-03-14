import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_source", required=False, default='data/final_proj_data_preprocessed.json', help="File (CSV) to source experiment samples from")
parser.add_argument("--num_posts", type=int, required=True, help="Number of posts to sample")
args = parser.parse_args()
print('reading from:', args.csv_source)
print('experiment sample size:', args.num_posts)

# Split our post sampling into a sample of a smaller / limited size
# Code adapted from sampling_for_experiments.ipynb, data/bert_title_text.ipynb
# Originally primarily written by Regina Cheng

print(f'sample_experiment_data({args.csv_source}, {args.num_posts}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
t0 = time.process_time()

all = pd.read_csv(args.csv_source)

d_all = all[['id', 'author', 'title', 'selftext']]

d_sample = d_all.sample(args.num_posts)
d_sample.to_csv(f'data/data_sample_{args.num_posts}.csv')

def csv_to_dict(csv_file):
    d = pd.read_csv(csv_file)
    size = len(d)
    author = d.author.to_list( )
    post_id = d.id.to_list( )
    text = d.selftext.to_list( )
    title = d.title.to_list( )
    data = dict(zip(post_id,zip(title, text)))
    file_name = f'data/bert_title_text{size}.pickle'
    with open(file_name, 'wb') as handle:
        pickle.dump(data,handle)

csv_file = f'data/data_sample_{args.num_posts}.csv'
csv_to_dict(csv_file)

print(f'sample_experiment_data({args.csv_source}, {args.num_posts}) ELAPSED(s)', time.process_time() - t0)
print('sample_experiment_data ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
