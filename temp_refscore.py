import gensim
import numpy as np 
import pandas as pd
import pickle
import csv
import csv
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import pickle
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import models
import pickle
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import itertools
import pickle
import json
import itertools
import time

def parse_reddit_csv(filename, setname, lemmatizer=None, tokenizer=None):
    if lemmatizer == None:
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
    if tokenizer == None:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')

    print("Reading from", filename)
    csv_cols = []
    authors = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Remove numbers, punctuation
            row['selftext'] = re.sub(r'\d+', '', row['selftext'])
            row['title'] = re.sub(r'\d+', '', row['title'])
            # Tokenize the post text (selftext) and post title
            post_tokens = tokenizer.tokenize(row['selftext'])
            title_tokens = tokenizer.tokenize(row['title'])
            # Filter out stopwords
            post_tokens = [w for w in post_tokens if not w in stop_words]
            title_tokens = [w for w in title_tokens if not w in stop_words]
            # Lemmatize the post text (reduce words to word stems i.e. cats->cat, liked->like)
            post_tokens = [lemmatizer.lemmatize(w, 'n') for w in post_tokens]
            post_tokens = [lemmatizer.lemmatize(w, 'v') for w in post_tokens]
            title_tokens = [lemmatizer.lemmatize(w, 'n') for w in title_tokens]
            title_tokens = [lemmatizer.lemmatize(w, 'v') for w in title_tokens]
            csv_cols.append({'author': row['author'],
                             'selftext': post_tokens,
                             'title': title_tokens,
                             'post_id': row['id']})
            # Add author mapping
            if row['author'] not in authors:
                authors[row['author']] = []
            authors[row['author']].append(row['id'])
    with open(f'partials/{setname}_parse.pickle', 'wb') as handle:
        pickle.dump(csv_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'partials/{setname}_parse_authors.pickle', 'wb') as handle:
        pickle.dump(authors, handle, protocol=pickle.HIGHEST_PROTOCOL)

def embed_w2v(setname, model=None):
    # Load the parse
    with open(f'partials/{setname}_parse.pickle', 'rb') as handle:
        parsed = pickle.load(handle)

    # Load Google's pre-trained Word2Vec model.
    if model == None:
        model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)

    # Build embeddings
    sen_emb = {}
    for i in range(len(parsed)):
        counts = Counter(parsed[i]['selftext']).items()
        freq = pd.DataFrame(counts)
        freq = freq.rename(columns={0: "word", 1: 'freq'})
        # Weight by inverse relative frequency
        freq['inv_rfreq'] = freq['freq'].sum()/freq['freq']
        unknowns = []
        emb_dict = {}
        for w in freq['word'].to_list():
            try:
                emb = model[w]
                emb_dict.update({w:emb})
            except:
                unknowns.append(w)
        emb_value = pd.DataFrame(emb_dict).transpose().reset_index()
        emb_value = emb_value.rename(columns={'index': "word"})
        emb_value_list = emb_value.iloc[:, 1:301].mul(freq['inv_rfreq'], axis = 0).sum().to_list()
        sen_emb.update({parsed[i]['post_id']:emb_value_list})
        # TODO: implement SiF version
    with open(f'{setname}_embed_w2v_weighted.pickle', 'wb') as handle:
        pickle.dump(sen_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_topics(dictionary, corpus):
    # Train LDA model, get model & topic vectors
    # Set training parameters.
    num_topics = 30
    chunksize = 100
    passes = 20
    iterations = 400
    eval_every = 100  # None = Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    # Get basic evaluation
    top_topics = model.top_topics(corpus) #, num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    # Get topic vectors
    all_topics = model.get_document_topics(corpus, per_word_topics=True)
    all_topics = [(doc_topics, word_topics, word_phis) for doc_topics, word_topics, word_phis in all_topics]
    sen_top = {}
    for i in range(len(parsed)):
        # These are in the same order as the documents themselves.
        doc_topics, word_topics, phi_values = all_topics[i]
        # Generate the topic VECTOR not just list of topics
        doc_topic_vector = [0] * num_topics
        for topic in doc_topics:
            doc_topic_vector[topic[0]] = topic[1]
        sen_top.update({parsed[i]['post_id']:doc_topic_vector})

    return model, sen_top

def embed_lda(setname):
    # Load the parse
    with open(f'partials/{setname}_parse.pickle', 'rb') as handle:
        parsed = pickle.load(handle)

    # Create a dictionary representation of the documents.
    dictionary = Dictionary([parsed[i]['selftext'] for i in range(len(parsed))])
    print(dictionary)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(parsed[i]['selftext']) for i in range(len(parsed))]
    # for doc in corpus:
    #     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

    # TF-IDF (term freq, inverse document freq) representation
    tfidf = models.TfidfModel(corpus)
    # for doc in tfidf[corpus]:
    #     print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

    # Get bow data
    print("Generating topics for BOW...")
    model_bow, sen_top_bow = get_topics(dictionary, corpus)

    # Save bow data
    with open(f'partials/{setname}_model_top_bow.pickle', 'wb') as handle:
        pickle.dump(model_bow, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'partials/{setname}_embed_top_bow.pickle', 'wb') as handle:
        pickle.dump(sen_top_bow, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Get tfidf data
    print("Generating topics for TFIDF...")
    model_tfidf, sen_top_tfidf = get_topics(dictionary, tfidf[corpus])

    # Save tfidf data
    with open(f'partials/{setname}_model_top_tfidf.pickle', 'wb') as handle:
        pickle.dump(model_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'partials/{setname}_embed_top_tfidf.pickle', 'wb') as handle:
        pickle.dump(sen_top_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)

def similarity_clustering(similarity_dict, m, n):
    # TODO this clustering still has a bug, Regina should update it at some point
    # specifically, it creates 1 more cluster than asked for
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
    return clusters

def clust_any(setname, embedname, numClusters):
    # Load the parse
    with open(f'partials/{setname}_parse.pickle', 'rb') as handle:
        parsed = pickle.load(handle)
    # Load the embed / topic matrix
    with open(f'partials/{setname}_embed_{embedname}.pickle', 'rb') as handle:
        sen_emb = pickle.load(handle)

    numTotalPosts = len(parsed)
    d = pd.DataFrame(sen_emb).transpose()
    sim_mat = cosine_similarity(d)

    post = d.index.to_list()
    post_emb = dict(zip(post, sim_mat))

    cluster = similarity_clustering(post_emb, numClusters, numTotalPosts)
    # print(cluster)

    with open(f'partials/{setname}_clust_{embedname}.pickle', 'wb') as handle:
        pickle.dump(cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Transform clusters into a post_id:cluster_id dict
    transformed_cluster = {}
    clust_num = 0
    for key in cluster.keys():
        for post in cluster[key]:
            transformed_cluster[post[0]] = clust_num
        clust_num += 1
    # print(transformed_cluster)

    with open(f'partials/{setname}_clustdict_{embedname}.pickle', 'wb') as handle:
        pickle.dump(transformed_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)

def score_sas(setname, embedname):
    # Read clusters
    with open(f'partials/{setname}_clust_{embedname}.pickle', 'rb') as handle:
        clusters = pickle.load(handle)
    with open(f'partials/{setname}_clustdict_{embedname}.pickle', 'rb') as handle:
        clustdict = pickle.load(handle)
    # Read author list
    with open(f'partials/{setname}_parse_authors.pickle', 'rb') as handle:
        authors = pickle.load(handle)

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
    return score_sas

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
    return score_jaccard

# Load constants / large loading items
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)

# Load basic data
setname = 'sample1000'
parse_reddit_csv('data/final_proj_data_preprocessed_1000sample.csv', setname,
                lemmatizer=lemmatizer, tokenizer=tokenizer)

# Generate embeddings for reference models
embed_w2v('sample1000', model=model)
embed_lda('sample1000')

# Cluster and score in loops
embed_types = ['top_tfidf', 'top_bow', 'w2v_weighted', 's2v_sif']
scores = {
    'sas': {
        'top_tfidf': [],
        'top_bow': [],
        'w2v_weighted': [],
        's2v_sif': [],
    },
    'jaccard': {
        'top_tfidf': [],
        'top_bow': [],
        'w2v_weighted': [],
        's2v_sif': [],
    }
}
for i in range(100):
    for embed_name in embed_types:
        clust_any('sample1000', embed_name, 6)
    for embed_name in embed_types:
        scores['sas'][embed_name].append(score_sas('sample1000', embed_name))
        scores['jaccard'][embed_name].append(score_jaccard('sample1000', embed_name))

# Save scores
time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())
with open(f'outputs/{setname}_scores_{time_string}.pickle', 'wb') as handle:
    pickle.dump(transformed_cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)


