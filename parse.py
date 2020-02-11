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

with open('data/final_proj_data_preprocessed_1000sample.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # print(row['author'], row['selftext'], row['title'])
        # Tokenize the post and post title combined
        post_tokens = word_tokenize(row['selftext']) + word_tokenize(row['title'])
        # Filter out stopwords
        post_tokens = [w for w in post_tokens if not w in stop_words]
        # Lemmatize the post text (reduce words to word stems i.e. cats->cat, liked->like)
        post_tokens = [lemmatizer.lemmatize(w, 'n') for w in post_tokens]
        post_tokens = [lemmatizer.lemmatize(w, 'v') for w in post_tokens]
        print(post_tokens)
        break
