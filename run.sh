# Number of posts to use in the experiment
# we ran with 1000, 2200, 4400, 8800, 13200
EXPERIMENT_SIZE=1000

# Before running script: make sure you have python dependencies installed
# environment needs:
# pandas, matplotlib, numpy

# Note that we ran our experiments primarily from the Jupyter notebooks, so this
# may be buggy.
# Each Python script contains a credit to the Jupyter notebook that it is from,
# so you can run the notebooks in order individually

# Create the timing log
touch outputs
touch outputs/${EXPERIMENT_SIZE}_log.txt

# get super large collection of all the posts we'll be using from r/Advice
# and preprocess and filter that data
# This only *needs* to be done once ever, but it should be fine if you repeat
python3 scrape_from_subreddit.py  >> outputs/${EXPERIMENT_SIZE}_log.txt

# break that collection down into smaller experiment sizes
python3 experiment_sampling.py --num_posts ${EXPERIMENT_SIZE}  >> outputs/${EXPERIMENT_SIZE}_log.txt

# get author data for all of the posts in our sample ...
python3 scrape_author_data.py --csv_file_name data/data_sample_${EXPERIMENT_SIZE}.csv --output data/authorsubs${EXPERIMENT_SIZE}.json >> outputs/${EXPERIMENT_SIZE}_log.txt &

# Perform W2V, LDA embed generation

# Perform BERT similarity table generation

# Perform clustering and scoring for W2V, LDA, BERT

# Produce output files that we can more easily analyze
