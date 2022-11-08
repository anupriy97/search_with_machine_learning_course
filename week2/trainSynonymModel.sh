# Prepare title file just containing Name field to be used further for Synonym model
cut -d' ' -f2- /workspace/datasets/fasttext/shuffled_labeled_products.txt > /workspace/datasets/fasttext/titles.txt

# Preprocess the title file to remove punctuations, lower case, remove special characters, extra spaces, etc.
cat /workspace/datasets/fasttext/titles.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_titles.txt

# Train the synonym model using fasttext skipgram algorithm
~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/normalized_titles.txt -output /workspace/datasets/fasttext/title_model_normalized_epoch_5_minCount_20 -epoch 5 -minCount 20

# Extract top 1000 words from normalized/preprocessed title file to be used to generate synonym file using createSynonyms.py
cat /workspace/datasets/fasttext/normalized_titles.txt | tr " " "\n" | grep "...." | sort | uniq -c | sort -nr | head -1000 | grep -oE '[^ ]+$' > /workspace/datasets/fasttext/top_words.txt
