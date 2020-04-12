import urllib.request, json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import json
import time
import praw
from praw.models import User
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_file_name", required=True, help="Filename of the Reddit-scrape-data CSV we want author data for")
parser.add_argument("--output", required=False, default='data/authorsubs.json', help="Filename to save to")
args = parser.parse_args()
print(' reading from:', args.csv_file_name)
print('outputting to:', args.output)

# Collect comment & submission subreddit data for authors in our sample
# This version is designed to add onto preexisting CSV
# Code adapted from scraper_authors.ipynb
# Originally primarily written by Joyce Zhou

print(f'scrape_author_data_praw({args.csv_file_name}, {args.output}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
t0 = time.process_time()

df_posts = pd.read_csv(args.csv_file_name)

reddit = praw.Reddit(client_id='',
                     client_secret='',
                     user_agent='',
                     username='',
                     password='')

# Build subreddit mappings
# THIS SCRIPT REQUIRES THE FILE TO EXIST ALREADY CONTAINING PLAINTEXT '{}'
with open(args.output, 'r') as fp:
    sub_mappings = json.load(fp)

for username in set(df_posts['author']):
    try:
        if username not in sub_mappings:
            person = reddit.redditor(username)
            contribs = User.karma(person)
            df_comment_set = []
            df_submission_set = []
            for k in contribs.keys():
                df_submission_set.append(k.display_name)
                if contribs[k]['comment_karma'] is not 0:
                    df_comment_set.append(k.display_name)
            sub_mappings[username] = {
                'comment': df_comment_set,
                'submission': df_submission_set
            }
            print('newly read', username)
        else:
            print('already read', username)
    except:
        print('failed to read', username)
    finally:
        # save what we have so far if the last read attempt failed
        with open(args.output, 'w') as fp:
            json.dump(sub_mappings, fp)

#     print('got subreddits for', username)

with open(args.output, 'w') as fp:
    json.dump(sub_mappings, fp)

print(f'scrape_author_data_praw({args.csv_file_name}, {args.output}) ELAPSED(s)', time.process_time() - t0)
print('scrape_author_data_praw ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
