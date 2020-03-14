from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import models
import pickle
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", required=True, help="Name of the experiment we want to save parse for")
args = parser.parse_args()
print('experiment name:', args.experiment_name)

def get_topics(dictionary, corpus, parsed):
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

    print(f'embed_lda({setname}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    t0 = time.process_time()
    # Create a dictionary representation of the documents.
    dictionary = Dictionary([parsed[i]['selftext'] for i in range(len(parsed))])
    # print(dictionary)

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
    model_bow, sen_top_bow = get_topics(dictionary, corpus, parsed)

    # Get tfidf data
    print("Generating topics for TFIDF...")
    model_tfidf, sen_top_tfidf = get_topics(dictionary, tfidf[corpus], parsed)

    print(f'embed_lda({setname}) ELAPSED (s)', time.process_time() - t0)

    # Save bow data
    with open(f'partials/{setname}_model_top_bow.pickle', 'wb') as handle:
        pickle.dump(model_bow, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'partials/{setname}_embed_top_bow.pickle', 'wb') as handle:
        pickle.dump(sen_top_bow, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save tfidf data
    with open(f'partials/{setname}_model_top_tfidf.pickle', 'wb') as handle:
        pickle.dump(model_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'partials/{setname}_embed_top_tfidf.pickle', 'wb') as handle:
        pickle.dump(sen_top_tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('embed_lda ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    return

embed_lda(args.experiment_name)
