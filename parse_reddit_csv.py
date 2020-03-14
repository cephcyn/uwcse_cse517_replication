import csv
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import re
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_file_name", required=True, help="Filename of the Reddit-scrape-data CSV we want to parse")
parser.add_argument("--experiment_name", required=True, help="Name of the experiment we want to save parse for")
args = parser.parse_args()
print('reading from:', args.csv_file_name)
print('experiment name:', args.experiment_name)
print('  (outputting to:', f'partials/{args.experiment_name}_parse.pickle', ')')
print('  (outputting to:', f'partials/{args.experiment_name}_parse_authors.pickle', ')')

# Tokenize, lemmatize, and build an author mapping for the collected sample CSV
# Code adapted from reference_clusters.ipynb
# Originally primarily written by Joyce Zhou

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("cats"))

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

def parse_reddit_csv(filename, setname, stop_words=None, lemmatizer=None, tokenizer=None):
    if lemmatizer == None:
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
    if stop_words == None or tokenizer == None:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')

    print(f'parse_reddit_csv({filename}, {setname}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    t0 = time.process_time()
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
    print(f'parse_reddit_csv({filename}, {setname}) ELAPSED(s)', time.process_time() - t0)
    with open(f'partials/{setname}_parse.pickle', 'wb') as handle:
        pickle.dump(csv_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'partials/{setname}_parse_authors.pickle', 'wb') as handle:
        pickle.dump(authors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('parse_reddit_csv ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    return

# parse_reddit_csv('data/final_proj_data_preprocessed_1000sample.csv', '1000')
parse_reddit_csv(args.csv_file_name, args.experiment_name)
