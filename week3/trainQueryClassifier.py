# Create labeled queries dataset with category pruning to satisfy minimum queries for each category
python week3/create_labeled_queries.py --min_queries 1000

# Shuffle the dataset and save
shuf /workspace/datasets/fasttext/labeled_queries.txt --random-source=<(seq 99999) > /workspace/datasets/fasttext/shuffled_labeled_queries.txt

# Split train and test dataset to train Query Classifier
head -10000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt > /workspace/datasets/fasttext/queries_training_data.txt
tail -10000 /workspace/datasets/fasttext/shuffled_labeled_queries.txt > /workspace/datasets/fasttext/queries_test_data.txt

# Train the classifier
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/queries_training_data.txt -output /workspace/datasets/fasttext/query_classifier -lr 1.0 -epoch 25 -wordNgrams 2

# Test the classifer on test dataset (Following gives the P@1 = 0.964 and R@1=0.964)
~/fastText-0.9.2/fasttext test /workspace/datasets/fasttext/query_classifier.bin /workspace/datasets/fasttext/queries_test_data.txt
