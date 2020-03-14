import urllib.request, json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subreddit", required=False, default='Advice', help="Subreddit to scrape posts from")
parser.add_argument("--output", required=False, default='data/final_proj_data.csv', help="file (CSV) to save raw data to")
parser.add_argument("--output_preprocess", required=False, default='data/final_proj_data_preprocessed.csv', help="file (CSV) to save preprocessed data to")
args = parser.parse_args()
print('scraping posts from subreddit:', args.subreddit)
print('outputting raw data CSV to:', args.output)
print('outputting preprocessed data CSV to:', args.output_preprocess)

# Collect a lot of posts from the subreddit we requested
# Code adapted from scraper_posts.ipynb
# Originally primarily written by Regina Cheng

print(f'scrape_post_data({args.csv_file_name}, {args.output}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
t0 = time.process_time()

with urllib.request.urlopen(f"https://api.pushshift.io/reddit/search/submission/?subreddit={args.subreddit}&size=1&sort=asc&before=30d") as url:
    data = json.loads(url.read().decode())
data = data['data']
df_sub = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
created_utc_first_sub = df_sub.tail(1)['created_utc']
print(created_utc_first_sub)

with urllib.request.urlopen(f"https://api.pushshift.io/reddit/search/submission/?subreddit={args.subreddit}&size=1000&sort=desc&before=30d") as url:
    data = json.loads(url.read().decode())
data = data['data']
df_sub = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
created_utc_now_sub = df_sub.tail(1)['created_utc']

for i in range(10):
    with urllib.request.urlopen(f"https://api.pushshift.io/reddit/search/submission/?subreddit={args.subreddit}&size=1000&sort=desc&before=%d"%df_sub.tail(1)['created_utc']) as url:
        data = json.loads(url.read().decode())
    data = data['data']
    df_new_sub = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
    df_sub = df_sub.append(df_new_sub)
    created_utc_now_sub = df_sub.tail(1)['created_utc']
    print(created_utc_now_sub)
    print(df_sub.shape)

df_sub.to_csv(args.output)

# Preprocess the post data that we scraped, also filter out posts with low scores
# Code adapted from data_preprocessing.ipynb
# Originally primarily written by Regina Cheng

d = pd.read_csv(args.output)
d = d[d.selftext != '[removed]']
d = d[d.selftext.notnull()]
d = d[d.selftext != '[deleted]']
d = d[d.score >= 3]
d = d[d.selftext.str.len()>=5]
d.to_csv(args.output_preprocess)

print(f'scrape_post_data({args.csv_file_name}, {args.output}) ELAPSED(s)', time.process_time() - t0)
print('scrape_post_data ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
