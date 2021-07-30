import urllib.request, json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import json
import time
from datetime import datetime, date
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_file_name", required=True, help="Filename of the Reddit-scrape-data CSV we want author data for")
parser.add_argument("--before_date", required=False, help="Only include data before this date (format YYYY/MM/DD)")
parser.add_argument("--output", required=False, default='data/authorsubs.json', help="Filename to save to")
args = parser.parse_args()
print(' reading from:', args.csv_file_name)
print('  date cutoff:', args.before_date)
print('outputting to:', args.output)

# Collect comment & submission subreddit data for authors in our sample
# This version is designed to add onto preexisting JSON
# so the output file NEEDS to already exist and contain a plaintext {}
# Code adapted from scraper_authors.ipynb
# Originally primarily written by Joyce Zhou

print(f'scrape_author_data({args.csv_file_name}, {args.before_date}, {args.output}) START:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
t0 = time.process_time()

df_posts = pd.read_csv(args.csv_file_name)

# calculate the before epoch point
before_epoch = time.mktime(datetime.today().timetuple())
if (args.before_date is not None):
    before_input = [int(i) for i in args.before_date.split('/')]
    before_epoch = time.mktime(date(before_input[0], before_input[1], before_input[2]).timetuple())
before_epoch = int(before_epoch)
print(before_epoch)
print(f"https://api.pushshift.io/reddit/search/comment/?author=cephcyn&before={before_epoch}&sort=asc")

def getAllType(username, contentType):
    n_posts = 500
    with urllib.request.urlopen(f"https://api.pushshift.io/reddit/search/{contentType}/?author={username}&before={before_epoch}&sort=asc&size={n_posts}") as url:
        data = json.loads(url.read().decode())
        data = data['data']
    df_content = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
    if len(df_content) == 0:
        return df_content
    created_utc_last = df_content.tail(1)['created_utc'].copy().reset_index()
    created_utc_last = created_utc_last['created_utc'][0]
    while len(data) > 0:
        with urllib.request.urlopen(f"https://api.pushshift.io/reddit/search/{contentType}/?author={username}&before={before_epoch}&sort=asc&size={n_posts}&after={created_utc_last}") as url:
            data = json.loads(url.read().decode())
            data = data['data']
        df_content = df_content.append(pd.DataFrame.from_dict(json_normalize(data), orient='columns'))
        created_utc_last = df_content.tail(1)['created_utc'].copy().reset_index()
        created_utc_last = created_utc_last['created_utc'][0]
    return df_content

# Build subreddit mappings
# THIS SCRIPT REQUIRES THE FILE TO EXIST ALREADY CONTAINING PLAINTEXT '{}'
with open(args.output, 'r') as fp:
    sub_mappings = json.load(fp)

for username in set(df_posts['author']):
    try:
        if username not in sub_mappings:
            df_comment = getAllType(username, 'comment').reset_index()
            df_submission = getAllType(username, 'submission').reset_index()
            df_comment_set = list(set(df_comment['subreddit'])) if len(df_comment) > 0 else []
            df_submission_set = list(set(df_submission['subreddit'])) if len(df_submission) > 0 else []
            sub_mappings[username] = {
                'comment': df_comment_set,
                'submission': df_submission_set
            }
            print('newly read', username)
        else:
            print('already read', username)
    except Exception as e:
        print('failed to read', username)
        print('reason:', e)
    finally:
        # save what we have so far if the last read attempt failed
        with open(args.output, 'w') as fp:
            json.dump(sub_mappings, fp)

#     print('got subreddits for', username)

with open(args.output, 'w') as fp:
    json.dump(sub_mappings, fp)

print(f'scrape_author_data({args.csv_file_name}, {args.before_date}, {args.output}) ELAPSED(s)', time.process_time() - t0)
print('scrape_author_data ENDED:', time.strftime("%Y%m%d-%H%M%S", time.localtime()))
