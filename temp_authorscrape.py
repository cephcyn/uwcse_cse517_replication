import urllib.request, json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import json

version = 'preprocessed_1000sample'
df_posts = pd.read_csv(f'data/data_sample_13200.csv')

def getAllType(username, contentType):
    n_posts = 500
    with urllib.request.urlopen(f"https://api.pushshift.io/reddit/search/{contentType}/?author={username}&sort=asc&size={n_posts}") as url:
        data = json.loads(url.read().decode())
        data = data['data']
    df_content = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
    if len(df_content) == 0:
        return df_content
    created_utc_last = df_content.tail(1)['created_utc'].copy().reset_index()
    created_utc_last = created_utc_last['created_utc'][0]
    while len(data) > 0:
        with urllib.request.urlopen(f"https://api.pushshift.io/reddit/search/{contentType}/?author={username}&sort=asc&size={n_posts}&after={created_utc_last}") as url:
            data = json.loads(url.read().decode())
            data = data['data']
        df_content = df_content.append(pd.DataFrame.from_dict(json_normalize(data), orient='columns'))
        created_utc_last = df_content.tail(1)['created_utc'].copy().reset_index()
        created_utc_last = created_utc_last['created_utc'][0]
    return df_content

# Build subreddit mappings
sub_mappings = {}

# Save preliminary total mappings file
with open(f'data/authorsubs.json', 'w') as fp:
    json.dump(sub_mappings, fp)

for username in set(df_posts['author']):
    with open(f'data/authorsubs.json', 'r') as fp:
        sub_mappings = json.load(fp)
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
    except:
        print('failed to read', username)
    finally:
        # save what we have so far if the last read attempt failed
        with open(f'data/authorsubs.json', 'w') as fp:
            json.dump(sub_mappings, fp)
        
#     print('got subreddits for', username)

with open(f'data/authorsubs.json', 'w') as fp:
    json.dump(sub_mappings, fp)
