import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("cats"))

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def parse_reddit_csv(filename):
    print("Reading from", filename)
    csv_cols = []
    frequencies = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Tokenize the post text (selftext) and post title
            post_tokens = word_tokenize(row['selftext'])
            title_tokens = word_tokenize(row['title'])
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
                             'title': title_tokens})
            # TODO need to collect frequencies of words in the entire corpus
            # TODO update frequencies mapping from word->count and also get a sum
    return csv_cols, frequencies
