python week2/createContentTrainingData.py --output /workspace/datasets/fasttext/pruned_labeled_products.txt --min_products 500

shuf /workspace/datasets/fasttext/pruned_labeled_products.txt --random-source=<(seq 99999) > /workspace/datasets/fasttext/shuffled_pruned_labeled_products.txt

# Normalize the text to remove punctuations, special characters, lowercasing, extra spa
cat /workspace/datasets/fasttext/shuffled_pruned_labeled_products.txt |sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]_]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_pruned_labeled_products.txt

# Split train and test dataset
head -10000 /workspace/datasets/fasttext/normalized_pruned_labeled_products.txt > /workspace/datasets/fasttext/normalized_pruned_training_data.txt
tail -10000 /workspace/datasets/fasttext/normalized_pruned_labeled_products.txt > /workspace/datasets/fasttext/normalized_pruned_test_data.txt

# Train the classifier
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/normalized_pruned_training_data.txt -output /workspace/datasets/fasttext/product_classifier_pruned_normalized_ngrams -lr 1.0 -epoch 25 -wordNgrams 2

# Test the classifer on test dataset (Following gives the P@1 = 0.964 and R@1=0.964)
~/fastText-0.9.2/fasttext test /workspace/datasets/fasttext/product_classifier_pruned_normalized_ngrams.bin /workspace/datasets/fasttext/normalized_pruned_test_data.txt
