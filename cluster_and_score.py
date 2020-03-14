import random
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import json
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", required=True, help="Name of the experiment we want to get embeds for")
parser.add_argument("--num_clusters", required=True, type=int, help="The number of clusters to make")
parser.add_argument("--num_loops", required=False, type=int, default=100, help="The number of clusters to make")
args = parser.parse_args()
print('experiment name:', args.experiment_name)
print('number of clusters:', args.num_clusters)
print('number of loops:', args.num_loops)

# Build clusters based on a similarity matrix/dict.
# Code adapted from reference_clusters.ipynb
# Originally primarily written by Regina Cheng

def similarity_clustering(similarity_dict, m, n):
    clusters = {};
    unselected_posts = similarity_dict.copy()
    post_keys = list(unselected_posts.keys())
    unselected_keys = list(unselected_posts.keys())
    cluster_size = int(np.ceil(n / m))
    # print(cluster_size)
    while len(unselected_posts) != 0:
        selected_post = random.choice(unselected_keys)
        # labeling the selected row
        emb_dict = dict(zip(post_keys, unselected_posts[selected_post]))
        # only sort the unselected columns
        sim = {k: emb_dict[k] for k in unselected_keys}
        sim_sort = [k for k in sorted(sim.items(), key=lambda item: item[1])][::-1]
        cluster_size = int(np.ceil(n / m))
        try:
            sim_most = sim_sort[0:cluster_size]
        except:
            sim_most = sim_sort[0:end]
        clusters[selected_post] = sim_most
        # deleted the selected rows from the unselected
        for p in sim_most:
            del unselected_posts[p[0]]
        unselected_keys = list(unselected_posts.keys())
        # print(cluster_size)
    return clusters

def clust_any_ref(setname, embedname, numClusters):
    # Load the parse
    with open(f'partials/{setname}_parse.pickle', 'rb') as handle:
        parsed = pickle.load(handle)
    # Load the embed / topic matrix
    with open(f'partials/{setname}_embed_{embedname}.pickle', 'rb') as handle:
        sen_emb = pickle.load(handle)

    print(f'clust_any_ref({setname}, {embedname}, {numClusters}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    print('clustering dataset:', setname, '; embeds:', embedname)
    t0 = time.process_time()
    numTotalPosts = len(parsed)
    d = pd.DataFrame(sen_emb).transpose()
    sim_mat = cosine_similarity(d)

    post = list(d.index)
    post_emb = dict(zip(post, sim_mat))

    cluster = similarity_clustering(post_emb, numClusters, numTotalPosts)
    # print(cluster)

    with open(f'partials/{setname}_{numClusters}_clust_{embedname}.pickle', 'wb') as handle:
        pickle.dump(cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Transform clusters into a post_id:cluster_id dict
    transformed_cluster = {}
    clust_num = 0
    for key in cluster.keys():
        for post in cluster[key]:
            transformed_cluster[post[0]] = clust_num
        clust_num += 1
    # print(transformed_cluster)

    print(f'clust_any_ref({setname}, {embedname}, {numClusters}) ELAPSED(s)', time.process_time() - t0)

    with open(f'partials/{setname}_{numClusters}_clustdict_{embedname}.pickle', 'wb') as handle:
        pickle.dump(transformed_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('clust_any_ref ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    return

# Build clusters directly from the BERT similarity table
# Code adapted from reference_clusters.ipynb
# Originally primarily written by Regina Cheng, later modified by Joyce Zhou

def clust_any_bert(setname, embedname, numClusters):
    # Load the parse
    with open(f'partials/{setname}_parse.pickle', 'rb') as handle:
        parsed = pickle.load(handle)
    # Load the similarity matrix
    with open(f'partials/{setname}_matrix_bert.pickle', 'rb') as handle:
        post_emb = pickle.load(handle)

    print(f'clust_any_bert({setname}, {embedname}, {numClusters}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    print('clustering dataset:', setname, '; embeds:', embedname)
    t0 = time.process_time()
    numTotalPosts = len(parsed)

    cluster = similarity_clustering(post_emb, numClusters, numTotalPosts)
    # print(cluster)

    with open(f'partials/{setname}_{numClusters}_clust_{embedname}.pickle', 'wb') as handle:
        pickle.dump(cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Transform clusters into a post_id:cluster_id dict
    transformed_cluster = {}
    clust_num = 0
    for key in cluster.keys():
        for post in cluster[key]:
            transformed_cluster[post[0]] = clust_num
        clust_num += 1
    # print(transformed_cluster)

    print(f'clust_any_bert({setname}, {embedname}, {numClusters}) ELAPSED(s)', time.process_time() - t0)

    with open(f'partials/{setname}_{numClusters}_clustdict_{embedname}.pickle', 'wb') as handle:
        pickle.dump(transformed_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('clust_any_bert ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    return

# Score clusters with Same-Author-Score (SAS)
# Code adapted from reference_clusters.ipynb
# Originally primarily written by Joyce Zhou

def score_sas(setname, embedname):
    # Read clusters
    with open(f'partials/{setname}_clust_{embedname}.pickle', 'rb') as handle:
        clusters = pickle.load(handle)
    with open(f'partials/{setname}_clustdict_{embedname}.pickle', 'rb') as handle:
        clustdict = pickle.load(handle)
    # Read author list
    with open(f'partials/{setname}_parse_authors.pickle', 'rb') as handle:
        authors = pickle.load(handle)

    print(f'score_sas({setname}, {embedname}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    print('scoring dataset:', setname, '; embeds:', embedname)
    t0 = time.process_time()

    num_clust_pair = 0
    num_total_pair = 0

    for auth in authors:
        # authors[auth] is a list of post IDs made by author 'auth'
        if len(authors[auth]) < 2:
            continue
        for pair in itertools.product(authors[auth],authors[auth]):
            if pair[0] == pair[1]:
                continue
            num_total_pair += 1
            if clustdict[pair[0]] == clustdict[pair[1]]:
                num_clust_pair += 1

    score_sas = (num_clust_pair / num_total_pair) - (1/len(clusters))
    print(f'score_sas({setname}, {embedname}) ELAPSED(s)', time.process_time() - t0)
    print('score_sas ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    return score_sas

# Score clusters with Jaccard score
# Code adapted from reference_clusters.ipynb
# Originally primarily written by Joyce Zhou

def score_jaccard(setname, embedname):
    # Read post data
    with open(f'partials/{setname}_parse.pickle', 'rb') as handle:
        parsed = pickle.load(handle)
    # Read clusters
    with open(f'partials/{setname}_clust_{embedname}.pickle', 'rb') as handle:
        clusters = pickle.load(handle)
    with open(f'partials/{setname}_clustdict_{embedname}.pickle', 'rb') as handle:
        clustdict = pickle.load(handle)
    # Read author list
    with open(f'partials/{setname}_parse_authors.pickle', 'rb') as handle:
        authors = pickle.load(handle)
    # Read author subreddits
    with open(f'data/authorsubs.json', 'r') as fp:
        sub_mappings = json.load(fp)

    print(f'score_jaccard({setname}, {embedname}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    print('scoring dataset:', setname, '; embeds:', embedname)
    t0 = time.process_time()

    # Transform parsed into something more usable for Jaccard
    metadata = {}
    for p in parsed:
        metadata[p['post_id']] = p

    # Set up constants
    target_sub = 'Advice'
    default_subs = {
        'comment': [target_sub],
        'submission': [target_sub]
    }

    intersect_sum = 0
    for clustkey in clusters.keys():
        ids_in_clust = [i[0] for i in clusters[clustkey]]
        for pair in itertools.product(ids_in_clust,ids_in_clust):
            a0 = metadata[pair[0]]['author']
            a1 = metadata[pair[1]]['author']
            a0_subs = sub_mappings[a0] if (a0 in sub_mappings) else default_subs
            a1_subs = sub_mappings[a1] if (a1 in sub_mappings) else default_subs
            # Check for "throwaways"
            a0_subs_total = set(a0_subs['comment'] + a0_subs['submission'])
            a1_subs_total = set(a1_subs['comment'] + a1_subs['submission'])
            if len(a0_subs_total) == 1:
                continue
            if len(a1_subs_total) == 1:
                continue
            # New formulation: use set of subreddits an author has ever interacted with
            intersect_sum += (
                len(a0_subs_total.intersection(a1_subs_total))
                /
                len(a0_subs_total.union(a1_subs_total))
            )
            # Original paper formulation: this fails if neither author has ever commented on a sub
#             comment_subscore = (
#                 len(set(a0_subs['comment']).intersection(set(a1_subs['comment'])))
#                 /
#                 len(set(a0_subs['comment']).union(set(a1_subs['comment'])))
#             )
#             submits_subscore = (
#                 len(set(a0_subs['submission']).intersection(set(a1_subs['submission'])))
#                 /
#                 len(set(a0_subs['submission']).union(set(a1_subs['submission'])))
#             )
#             intersect_sum += 0.5 * (comment_subscore + submits_subscore)
    score_jaccard = intersect_sum * len(clusters) / (len(clustdict) ** 2)
    print(f'score_jaccard({setname}, {embedname}) ELAPSED(s)', time.process_time() - t0)
    print('score_jaccard ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    return score_jaccard

# Repeatedly cluster based on the similarity matrices / embeddings we have
# and score each clustering by SAS and Jaccard, then save those scores.
# Code adapted from reference_clusters.ipynb
# Originally primarily written by Joyce Zhou

setname = args.experiment_name
num_clusters = args.num_clusters
num_loops = args.num_loops

scores = {
    'sas': {
        'bert': [],
        'top_tfidf': [],
        'top_bow': [],
        'w2v_weighted': [],
        'w2v_sif': [],
    },
    'jaccard': {
        'bert': [],
        'top_tfidf': [],
        'top_bow': [],
        'w2v_weighted': [],
        'w2v_sif': [],
    }
}

print(f'score_loop({setname}_{num_clusters}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
t0 = time.process_time()

# Do it for the reference models
embed_types = ['top_tfidf', 'top_bow', 'w2v_weighted', 'w2v_sif']
for i in range(num_loops):
    for embed_name in embed_types:
        clust_any_ref(setname, embed_name, num_clusters)
    for embed_name in embed_types:
        scores['sas'][embed_name].append(score_sas(f'{setname}_{num_clusters}', embed_name))
        scores['jaccard'][embed_name].append(score_jaccard(f'{setname}_{num_clusters}', embed_name))

# Do it for BERT
for i in range(num_loops):
    clust_any_bert(setname, 'bert', num_clusters)
    scores['sas']['bert'].append(score_sas(f'{setname}_{num_clusters}', 'bert'))
    scores['jaccard']['bert'].append(score_jaccard(f'{setname}_{num_clusters}', 'bert'))

# Save scores
print(f'score_loop({setname}_{num_clusters}) ELAPSED(s)', time.process_time() - t0)
print('score_loop ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
with open(f'outputs/{setname}_{num_clusters}_scores.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
