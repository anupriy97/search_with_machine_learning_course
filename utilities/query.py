# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import fileinput
import logging
import re
import fasttext

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, boosts=None, sort="_score", sortDir="desc", size=10, source=None, synonyms=False):
    if synonyms:
        nameField = "name.synonyms"
    else:
        nameField = "name"
    
    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [

                        ],
                        "should": [  #
                            {
                                "match": {
                                    nameField: {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": ["name^10", "name.hyphens^10", "shortDescription^5",
                                               "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                               "categoryPath"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        }
    }
    
    if boosts is not None:
        query_obj['query']["function_score"]["query"]["bool"]["should"].extend(boosts)
    
    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj


def normalize_query(query):
    query_l = query.lower()
    query_s = re.sub(r'[^a-zA-Z0-9]+', ' ', query_l)
    query_norm = ' '.join([stemmer.stem(token) for token in query_s.split(' ')])
    return query_norm


def classify_query(query, model, k=10, threshold=0.5):
    # Normalize the query
    query_norm = normalize_query(query)
    
    # Predict the category of the normalized query
    prediction = model.predict(query_norm, k=k)
    
    # Retuning just the Top category if score is above 0.5 or combination of top categories till sum of scores get above 0.5
    query_cats = []
    cum_sum = 0

    for i in range(k):
        cum_sum += prediction[1][i]

        curr_query_cat = prediction[0][i].replace('__label__', '')

        if cum_sum >= threshold:
            query_cats.append(curr_query_cat)
            break
        
        query_cats.append(curr_query_cat)

    if cum_sum >= threshold:
        print(query_cats, cum_sum)
        return query_cats
    
    return None


def search(client, user_query, index="bbuy_products", sort="_score", sortDir="desc", synonyms=False, model=None,
           query_category_filter=True, query_category_boost=0):
    #### W3: classify the query
    #### W3: create filters and boosts
    # Note: you may also want to modify the `create_query` method above
    filters = None
    boosts = None
    
    if model is not None:
        query_cats = classify_query(user_query, model)
        logging.info(query_cats)

        if query_category_filter:
            filters = []
            filters.append({
                "terms": {
                    "categoryPathIds": query_cats
                }
            })

        if query_category_boost != 0:
            boosts = []
            boosts.append({
                "terms": {
                    "categoryPathIds": query_cats,
                    "boost": query_category_boost
                }
            })

    query_obj = create_query(user_query, click_prior_query=None, filters=filters, boosts=boosts, sort=sort,
                             sortDir=sortDir, source=["name", "shortDescription"], synonyms=synonyms)
    logging.info(query_obj)
    response = client.search(query_obj, index=index)
    if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
        hits = response['hits']['hits']
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")

    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument('--user',
                         help='The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin')
    general.add_argument("-q", '--query', default="nespresso",
                         help='Query to search using Opensearch')
    general.add_argument('--synonyms', action='store_true', help='Use this flag to do query using synonyms')
    general.add_argument('--model', default='/workspace/datasets/fasttext/query_classifier_lr_0.5_epoch_25_ngrams_2.bin', help='Location of the model file to classify query into category')
    general.add_argument('--query_category_filter', action='store_true', help='Use this flag to filter results using query category predicted by Query Classifier model')
    general.add_argument('--query_category_boost', type=int, default=0, help='Use this argument to boost results with query category predicted by Query Classifier model (Default is 0 i.e. no boosting)')

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    host = args.host
    port = args.port
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    index_name = args.index
    synonyms = args.synonyms
    query = args.query
    model_file = args.model
    query_category_filter = args.query_category_filter
    query_category_boost = args.query_category_boost

    # query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    # print(query_prompt)
    # for line in fileinput.input():
    #     query = line.rstrip()
    #     if query == "Exit":
    #         break
    #     search(client=opensearch, user_query=query, index=index_name, synonyms=args.synonyms)

    #     print(query_prompt)

    # Load Query Classifier model using fasttext
    query_classifier_model = fasttext.load_model(model_file)

    search(client=opensearch, user_query=query, index=index_name, synonyms=synonyms, model=query_classifier_model,
           query_category_filter=query_category_filter, query_category_boost=query_category_boost)

    