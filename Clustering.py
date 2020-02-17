import pickle
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

with open('sample1000_emb.pickle', 'rb') as handle:
    sen_emb = pickle.load(handle)

d = pd.DataFrame(sen_emb).transpose()
sim_mat = cosine_similarity(d)

post = d.index.to_list()
post_emb = dict(zip(post, sim_mat))


def Similarity_clustering(similarity_dict, m, n):
    clusters = {};
    unselected_posts = similarity_dict.copy()
    post_keys = list(unselected_posts.keys())
    unselected_keys = list(unselected_posts.keys())
    cluster_size = int(np.ceil(n / m))
    while len(unselected_posts) != 0:
        selected_post = random.choice(unselected_keys)
        # labeling the selected row
        emb_dict = dict(zip(post_keys, unselected_posts[selected_post]))
        # only sort the unselected columns
        sim = {k: emb_dict[k] for k in unselected_keys}
        sim_sort = [k for k in sorted(sim.items(), key=lambda item: item[1])][::-1]
        sim_most = sim_sort[0:cluster_size]
        clusters[selected_post] = sim_most
        # deleted the selected rows from the unselected
        for p in sim_most:
            del unselected_posts[p[0]]
        unselected_keys = list(unselected_posts.keys())
        cluster_size = int(np.floor(n / m))
    return (clusters)


cluster = Similarity_clustering(post_emb, 6, 1000)
print(cluster)