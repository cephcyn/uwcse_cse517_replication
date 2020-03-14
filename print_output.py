import pickle
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", required=True, help="Name of the experiment we want to get data for")
args = parser.parse_args()
print('experiment name:', args.experiment_name)

# Get our output/scores/logs into a nice CSV file that we can process elsewhere
# Code adapted from read_output_logs.ipynb, data/bert_title_text.ipynb
# Originally primarily written by Joyce Zhou

# Print out the experiment cluster scores

with open(f'outputs/{args.experiment_name}_scores.pickle', 'rb') as handle:
    scores = pickle.load(handle)

# scores levels:
# 0: {'sas', 'jaccard'}
# 1: {'bert', top_tfidf', 'top_bow', 'w2v_weighted', 'w2v_sif'}

df_scores = pd.DataFrame(columns=[
    'sas(bert)',
    'sas(top_tfidf)',
    'sas(top_bow)',
    'sas(w2v_weighted)',
    'sas(w2v_sif)',
    'jaccard(bert)',
    'jaccard(top_tfidf)',
    'jaccard(top_bow)',
    'jaccard(w2v_weighted)',
    'jaccard(w2v_sif)'
])
# There should be an equal number of elems in each score list...
for i in range(len(scores['sas']['bert'])):
    # build the row
    row = {}
    for score_type in ['sas', 'jaccard']:
        for model_type in ['bert', 'top_tfidf', 'top_bow', 'w2v_weighted', 'w2v_sif']:
            row[f'{score_type}({model_type})'] = scores[score_type][model_type][i]
    df_scores = df_scores.append(row, ignore_index=True)

df_scores.to_csv(f'outputs/{args.experiment_name}_scores.csv')

# Print out the experiment runtimes

import pickle
import pandas as pd

df_times = pd.DataFrame(columns=[
    'func_name',
    'time_elapsed(s)'
])

with open(f'outputs/{args.experiment_name}_log.txt') as f:
    for line in f:
        if 'ELAPSED(s)' in line:
            line = line.strip()
            pieces = line.split('ELAPSED(s)')
            curr_func = pieces[0]
            time_elapsed = pieces[1]
            df_times = df_times.append({
                'func_name': curr_func,
                'time_elapsed(s)': time_elapsed
            }, ignore_index=True)

df_times.to_csv(f'outputs/{args.experiment_name}_times.csv')
