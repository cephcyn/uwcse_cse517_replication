# Number of posts to use in the experiment
# we ran with 1000, 2200, 4400, 8800, 13200
EXPERIMENT_SIZE=1000 # Also doubles as the experiment name
NUM_CLUSTERS=6
NUM_LOOPS=100

# Before running this script!!!!!
# STEP 1: make sure you have python dependencies installed
  # Python environment needs:
  # pandas, matplotlib, numpy, sklearn, gensim, nltk, torch, transformers,
  # altair, selenium
  # also need to have installed:
  # geckodriver
# STEP 2: download the word2vec model (unfortunately can't do this in bash)
  # See model/getModel.sh for notes on how to download the word2vec model

# Note that we ran our experiments primarily from the Jupyter notebooks, so this
# may be buggy.
# Each Python script contains a credit to the Jupyter notebook that it is from,
# so you can run through the notebooks individually

# Create the timing log
touch outputs
touch outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt

# get super large collection of all the posts we'll be using from r/Advice
# and preprocess and filter that data
# This only *needs* to be done once ever, but it should be fine if you repeat
python3 scrape_from_subreddit.py >> outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt

# break that collection down into smaller experiment sizes
python3 experiment_sampling.py --num_posts ${EXPERIMENT_SIZE} >> outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt

# get author data for all of the posts in our sample
echo {} > data/authorsubs_${EXPERIMENT_SIZE}.json
python3 scrape_author_data_praw.py --csv_file_name data/data_sample_${EXPERIMENT_SIZE}.csv --output data/authorsubs_${EXPERIMENT_SIZE}.json >> outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt &

# Parse the experiment sample data into something more helpful
python3 parse_reddit_csv.py --csv_file_name data/data_sample_${EXPERIMENT_SIZE}.csv --experiment_name ${EXPERIMENT_SIZE} >> outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt

# NOTE: for all embed systems, we use the same topic/cluster count hyperparam
# (30) regardless of the number of final clusters produced.

# Perform BERT similarity table generation
python3 bert_similarity.py --experiment_name ${EXPERIMENT_SIZE} >> outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt &

# Perform W2V embed generation
python3 embed_w2v.py --experiment_name ${EXPERIMENT_SIZE} >> outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt &

# Perform LDA embed generation
python3 embed_lda.py --experiment_name ${EXPERIMENT_SIZE} >> outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt &

# we need to have author data, embeds, and BERT done for the following steps...
wait

# Perform clustering and scoring for W2V, LDA, BERT
python3 cluster_and_score.py --experiment_name ${EXPERIMENT_SIZE} --num_clusters ${NUM_CLUSTERS} --num_loops ${NUM_LOOPS} >> outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_log.txt

# Produce output files that we can more easily analyze
python3 print_output.py --experiment_name ${EXPERIMENT_SIZE}_${NUM_CLUSTERS}

echo Scores saved to outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_scores.csv
echo Times saved to outputs/${EXPERIMENT_SIZE}_${NUM_CLUSTERS}_times.csv
