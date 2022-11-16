import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns=['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
def normalize_query(query):
    query_l = query.lower()
    query_s = re.sub(r'[^a-zA-Z0-9]+', ' ', query_l)
    query_norm = ' '.join([stemmer.stem(token) for token in query_s.split(' ')])
    return query_norm

queries_df['query'] = queries_df['query'].apply(normalize_query)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
def satisfy_min_query_criteria(df, threshold):
    df_gp = df.groupby(['category']).agg({'query': 'count'}).reset_index().sort_values(['query'], ascending=[True])
    df_gp_below_threshold = df_gp[df_gp['query'] < threshold]

    if len(df_gp_below_threshold) > 0:
        return False
    
    return True

def get_category_with_min_queries(df):
    df_gp = df.groupby(['category']).agg({'query': 'count'}).reset_index()
    df_gp = df_gp.sort_values(['query'], ascending=[True]).reset_index(drop=True)
    min_queries_cat = df_gp['category'][0]
    cnt = df_gp['query'][0]
    return min_queries_cat, cnt

while not satisfy_min_query_criteria(queries_df, min_queries):
    # Find the category with minimum number of queries
    min_queries_cat, cnt = get_category_with_min_queries(queries_df)

    # Find the parent of the least query category
    min_queries_cat_parent = parents_df[parents_df['category'] == min_queries_cat]['parent'].values[0]

    # Update the category of least query category with it's parent
    queries_df.loc[queries_df['category'] == min_queries_cat, 'category'] = min_queries_cat_parent

    print(f"Updating category {min_queries_cat} with {cnt} queries to it's parent category {min_queries_cat_parent}.")

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
