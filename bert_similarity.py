import pickle as pickle
import random
import operator
from itertools import combinations
import time
import timeit
import numpy as np
import pandas as pd
from torch import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", required=True, help="Name of the experiment we want to save parse for")
args = parser.parse_args()
print('experiment name:', args.experiment_name)

# Generate BERT similarity matrix
# Code adapted from bert_sim_table.ipynb
# Originally primarily written by Regina Cheng

model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Bert NextSentencePrediction

def bert_nsp(seq_A, seq_B):
    encoded = tokenizer.encode_plus(seq_A, text_pair=seq_B, return_tensors='pt',max_length=10000)
    try:
        seq_relationship_logits = model(**encoded)[0]
        probs = softmax(seq_relationship_logits, dim=1)
        similarity = probs.detach().numpy()[0][0]
    except:
        similarity = 0
    return similarity

# Functions for generating Bert Similarity Tables

def One_prob_clustering(posts, m,n,t):
    unselected_posts = posts.copy()
    clusters = {}
    while len(unselected_posts)> 0:
        select_post = random.choice(unselected_posts)
        sim_dict = {}
        for p in unselected_posts:
            similarity = bert_nsp(t[select_post][0], t[p][1])
            sim_dict[p] = similarity
        sorted_sim_list = [key for key,value in sorted(sim_dict.items(), key=operator.itemgetter(1),reverse=True)]
        try:
            most_similar = sorted_sim_list[0:int(np.ceil(n/m))]
        except:
            most_similar = sorted_sim_list[0:end]
        clusters[select_post] = most_similar
        for post in most_similar:
            unselected_posts.remove(post)
    return clusters

def Merge_multiple_prob_clustering(p, m, posts,n,t):
    similarity_table = {}
    for i in posts:
        similarity_table[i] = np.zeros(n)
    for i in range(p):
        one_probabilistic_clustering = One_prob_clustering(posts, m, n,t)
        for j in one_probabilistic_clustering.keys():
            one_cluster =  [j] + one_probabilistic_clustering[j]
            all_similar_pairs = combinations(one_cluster, 2)
            for k in all_similar_pairs:
                row_index = posts.index(k[1])
                similarity_table[k[0]][row_index] += 1
    return similarity_table

with open(f'data/bert_title_text{args.experiment_name}.pickle', 'rb') as handle:
    d = pickle.load(handle)

posts = list(d.keys())
n = len(posts)
p = 5
m = 30 # hyperparameter from the paper
t = d.copy()

print(f'bert_similarity({args.experiment_name}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
t0 = time.process_time()

start = timeit.default_timer()
similarity_table = Merge_multiple_prob_clustering(p, m, posts,n,t)
stop = timeit.default_timer()

with open(f'partials/{args.experiment_name}_matrix_bert.pickle', 'wb') as handle:
    pickle.dump(similarity_table, handle)

print(f'bert_similarity({args.experiment_name}) ELAPSED (s)', time.process_time() - t0)
print('bert_similarity ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
print('bert_similarity time (by timeit): ', stop - start)
