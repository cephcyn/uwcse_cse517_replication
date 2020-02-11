import parse

parsed = parse.parse_reddit_csv('data/final_proj_data_preprocessed_1000sample.csv')
print(parsed)

# Compute word2vec post embeddings (using both selftext and title)
# TODO: do the below
# The first (thereafter called W2VWeighted) is calculated by weighing the
# contribution of each word embedding by the inverse of its relative frequency
# to the final sentence embedding.
# In doing so, the contributions of the most common words are minimized.
# The second (thereafter called W2V-SIF) is calculated by first taking the
# weighed sentence embedding before removing the first principal component from it.
# Sanjeev Arora, Yingyu Liang, and Tengyu Ma. 2017.
# A simple but tough-to-beat baseline for sentence embeddings. In ICLR.

# Compute LDA post embeddings (using both selftext and title)
# TODO:
# A Bag of Words (BoW) corpus was obtained before a term frequency-inverse
# document frequency (TF-IDF) corpus was derived from it. Topic modeling was
# then performed on both the BoW corpus (thereafter LDA-BoW) and
# TF-IDF corpus (thereafter LDA-TFIDF) with the number of topics set to 30,
# in line with the number of clusters used. The document-topic mapping of
# each post is then used for computing cosine similarities with all other posts
# note: using gensim?
