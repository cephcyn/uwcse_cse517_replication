# Run to obtain at least 25k posts from each of the listed subreddits.

subreddits=(confession trueoffmychest confidence socialanxiety anxiety socialskills)

for sub in "${subreddits[@]}"
do
  echo "starting scraping subreddit: ${sub}"
  python3 scrape_from_subreddit.py --subreddit ${sub} \
                                   --output data/expanded_${sub}.csv \
                                   --output_preprocess data/expanded_${sub}_preprocessed.csv \
                                   --post_count 25000
  echo "finished scraping subreddit: ${sub}"
done
