from collections import Counter
import pickle
import numpy as np
from sklearn.decomposition import PCA
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", required=True, help="Name of the experiment we want to save parse for")
args = parser.parse_args()
print('experiment name:', args.experiment_name)

# Write Word2Vec embeddings for the given experiment.
# Code adapted from reference_clusters.ipynb
# Originally primarily written by Regina Cheng

def embed_w2v(setname, model=None):
    # Load the parse
    with open(f'partials/{setname}_parse.pickle', 'rb') as handle:
        parsed = pickle.load(handle)

    # Load Google's pre-trained Word2Vec model.
    if model == None:
        model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)

    print(f'embed_w2v({setname}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    t0 = time.process_time()
    # Build weighted embeddings
    weighted_emb = {}
    for i in range(len(parsed)):
        counts = Counter(parsed[i]['selftext'])
        freq = pd.DataFrame.from_dict(counts, orient='index').reset_index()
        freq = freq.rename(columns={'index': 'word', 0: 'freq'})
        # Weight by inverse relative frequency
        freq['inv_rfreq'] = freq['freq'].sum()/freq['freq']
        unknowns = []
        emb_dict = {}
        for w in freq['word']:
            try:
                emb = model[w]
                emb_dict.update({w:emb})
            except:
                unknowns.append(w)
        emb_value = pd.DataFrame(emb_dict).transpose().reset_index()
        emb_value = emb_value.rename(columns={'index': 'word'})
        emb_value_list = list(emb_value.iloc[:, 1:301].mul(freq['inv_rfreq'], axis = 0).sum())
        weighted_emb.update({parsed[i]['post_id']:emb_value_list})
    # Build SIF (remove first principal component)
    pca = PCA()
    ids = [key for (key, val) in list(weighted_emb.items())]
    weighted_matrix = np.array([val for (key, val) in list(weighted_emb.items())])
    # calculate PCA projections
    pca_matrix = pca.fit_transform(weighted_matrix)
    # calculate p-component that we need to subtract
    pca_adjust = [[emb[0] * c for c in pca.components_[0]] for emb in pca_matrix.tolist()]
    # drop p-component
    sif_matrix = [[i - j for i, j in zip(emb, pc)] for emb, pc in zip(weighted_matrix.tolist(), pca_adjust)]
    # convert back to dict format
    sif_emb = dict(zip(ids, sif_matrix))
    print(f'embed_w2v({setname}) ELAPSED(s)', time.process_time() - t0)
    with open(f'partials/{setname}_embed_w2v_weighted.pickle', 'wb') as handle:
        pickle.dump(weighted_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'partials/{setname}_embed_w2v_sif.pickle', 'wb') as handle:
        pickle.dump(sif_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('embed_w2v ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    return

model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
embed_w2v(args.experiment_name, model=model)
