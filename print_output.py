import pickle
import pandas as pd
import altair as alt
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

df_scores=None
df_scores = pd.DataFrame(columns=[
    'scoreType',
    'score'
])
for score_type in ['sas', 'jaccard']:
    for model_type in ['bert', 'top_tfidf', 'top_bow', 'w2v_weighted', 'w2v_sif']:
        for i in range(len(scores[score_type][model_type])):
            row = {
                'scoreType': f'{score_type}({model_type})',
                'score': scores[score_type][model_type][i]
            }
            df_scores = df_scores.append(row, ignore_index=True)

# Old schema: good for google docs, bad for Altair
# df_scores = pd.DataFrame(columns=[
#     'sas(bert)',
#     'sas(top_tfidf)',
#     'sas(top_bow)',
#     'sas(w2v_weighted)',
#     'sas(w2v_sif)',
#     'jaccard(bert)',
#     'jaccard(top_tfidf)',
#     'jaccard(top_bow)',
#     'jaccard(w2v_weighted)',
#     'jaccard(w2v_sif)'
# ])
# # There should be an equal number of elems in each score list...
# for i in range(len(scores['sas']['bert'])):
#     # build the row
#     row = {}
#     for score_type in ['sas', 'jaccard']:
#         for model_type in ['bert', 'top_tfidf', 'top_bow', 'w2v_weighted', 'w2v_sif']:
#             row[f'{score_type}({model_type})'] = scores[score_type][model_type][i]
#     df_scores = df_scores.append(row, ignore_index=True)

df_scores.to_csv(f'outputs/{args.experiment_name}_scores.csv')

# Get a summary of mean, min, max, stdev in table format
df_scores_agg = df_scores.groupby(['scoreType']).agg({
    'score': ['mean', 'min', 'max', 'std']
})
df_scores_agg.columns = ['score_mean', 'score_min', 'score_max', 'score_stdev']
df_scores_agg = df_scores_agg.reset_index()
df_scores_agg.to_csv(f'outputs/{args.experiment_name}_scores_agg.csv')

# Print out the experiment runtimes
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

# Get the SAS graph
sas_data = df_scores.loc[df_scores['scoreType'].str.contains('sas(', regex=False)]

scale = alt.Scale(zero=False)
stdev_err = alt.Chart(sas_data).mark_errorbar(extent='stdev').encode(
  x=alt.X('scoreType:N'),
  y=alt.Y(
        'score:Q',
        scale=scale
    ),
)
bars = alt.Chart(
    sas_data,
    title='SAS scores for prob. clust. vs baselines'
).mark_bar().encode(
    x=alt.X(
        'scoreType:N',
        axis=alt.Axis(title='')
    ),
    y=alt.Y(
        'score:Q',
        aggregate='mean',
        scale=scale
    ),
).properties(
    width=300,
    height=200
)

(bars + stdev_err).save(f'outputs/{args.experiment_name}_graph_sas.png', webdriver='firefox')

# Get the Jaccard graph
jaccard_data = df_scores.loc[df_scores['scoreType'].str.contains('jaccard(', regex=False)]

scale = alt.Scale(zero=False)
stdev_err = alt.Chart(jaccard_data).mark_errorbar(extent='stdev').encode(
  x=alt.X('scoreType:N'),
  y=alt.Y(
        'score:Q',
        scale=scale
    ),
)
bars = alt.Chart(
    jaccard_data,
    title='Jaccard scores for prob. clust. vs baselines'
).mark_bar().encode(
    x=alt.X(
        'scoreType:N',
        axis=alt.Axis(title='')
    ),
    y=alt.Y(
        'score:Q',
        aggregate='mean',
        scale=scale
    ),
).properties(
    width=300,
    height=200
)

(bars + stdev_err).save(f'outputs/{args.experiment_name}_graph_jaccard.png', webdriver='firefox')
