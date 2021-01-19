rm outputs/8800_20_log.txt &
rm outputs/8800_40_log.txt &
rm outputs/8800_50_log.txt &
rm outputs/8800_60_log.txt &
rm outputs/8800_70_log.txt &
rm outputs/8800_80_log.txt &
rm outputs/8800_90_log.txt &
rm outputs/8800_100_log.txt &

wait

touch outputs/8800_20_log.txt &
touch outputs/8800_40_log.txt &
touch outputs/8800_50_log.txt &
touch outputs/8800_60_log.txt &
touch outputs/8800_70_log.txt &
touch outputs/8800_80_log.txt &
touch outputs/8800_90_log.txt &
touch outputs/8800_100_log.txt &

wait

# python3 scrape_author_data_praw.py --csv_file_name data/data_sample_17600.csv --output data/authorsubs_17600.json >> outputs/17600_30_log.txt &

# wait

# python3 parse_reddit_csv.py --csv_file_name data/data_sample_17600.csv --experiment_name 17600 >> outputs/17600_30_log.txt

# wait

# python3 embed_w2v.py --experiment_name 17600 >> outputs/17600_30_log.txt &
# python3 embed_lda.py --experiment_name 17600 >> outputs/17600_30_log.txt &

# wait

python3 cluster_and_score.py --experiment_name 8800 --num_clusters 20 --num_loops 100 >> outputs/8800_20_log.txt
# python3 print_output.py --experiment_name 8800_20

python3 cluster_and_score.py --experiment_name 8800 --num_clusters 40 --num_loops 100 >> outputs/8800_40_log.txt
# python3 print_output.py --experiment_name 8800_40

python3 cluster_and_score.py --experiment_name 8800 --num_clusters 50 --num_loops 100 >> outputs/8800_50_log.txt
# python3 print_output.py --experiment_name 8800_50

python3 cluster_and_score.py --experiment_name 8800 --num_clusters 60 --num_loops 100 >> outputs/8800_60_log.txt
# python3 print_output.py --experiment_name 8800_60

python3 cluster_and_score.py --experiment_name 8800 --num_clusters 70 --num_loops 100 >> outputs/8800_70_log.txt
# python3 print_output.py --experiment_name 8800_70

python3 cluster_and_score.py --experiment_name 8800 --num_clusters 80 --num_loops 100 >> outputs/8800_80_log.txt
# python3 print_output.py --experiment_name 8800_80

python3 cluster_and_score.py --experiment_name 8800 --num_clusters 90 --num_loops 100 >> outputs/8800_90_log.txt
# python3 print_output.py --experiment_name 8800_90

python3 cluster_and_score.py --experiment_name 8800 --num_clusters 100 --num_loops 100 >> outputs/8800_100_log.txt
# python3 print_output.py --experiment_name 8800_100


