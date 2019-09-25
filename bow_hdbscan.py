import gzip
import ijson
import os
import pandas as pd
from sklearn.metrics import pairwise_distances
import csv
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import hdbscan
import re
import math

DATADIR = os.getenv("DATADIR")
clean_content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
content = pd.read_pickle(clean_content_path)
content['combined_text_embedding'] = ''
content['title_embedding'] = ''

def get_content_for_taxon(content, taxon_id):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon_id]['content_id'])
    return content[content['content_id'].isin(content_ids_for_taxon)];

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        if re.match("^[A-Za-z ]", item):
            stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_limit_length(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    tokens = tokens[:100]
    stems = stem_tokens(tokens, stemmer)
    return stems

def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def find_best_clustering(corpus, tokenizer):
    max_features = 300
    lowest_percentage_unclassified_items = float('infinity')
    cluster_size_lowest_percentage_unclassified_items = -1
    lowest_number_items_per_topic = float('infinity')
    cluster_size_lowest_number_items_per_topic = -1
    for min_cluster_size in list(range(2,10)):
        clusterer, percentage_unclassified, number_items_per_topic = cluster(corpus, max_features, min_cluster_size, tokenizer)
        if percentage_unclassified < lowest_percentage_unclassified_items:
            cluster_size_lowest_percentage_unclassified_items = min_cluster_size
            lowest_percentage_unclassified_items = percentage_unclassified
        if number_items_per_topic < lowest_number_items_per_topic:
            cluster_size_lowest_number_items_per_topic = min_cluster_size
            lowest_number_items_per_topic = number_items_per_topic
    min_cluster_sizes = [cluster_size_lowest_percentage_unclassified_items, cluster_size_lowest_number_items_per_topic]
    average_best_cluster_size = sum(min_cluster_sizes) / len(min_cluster_sizes)
    best_min_cluster_size = math.floor(average_best_cluster_size)
    clustering, percentage_unclassified, number_items_per_topic = cluster(corpus, max_features, best_min_cluster_size, tokenizer)
    return clustering

def cluster(corpus, max_features, min_cluster_size, tokenizer):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, analyzer='word', stop_words='english', max_features=max_features, ngram_range=(1,4)  )
    X = vectorizer.fit_transform(corpus).toarray()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=1, cluster_selection_method='leaf')
    clusterer.fit(X)
    total_number_items = len(corpus)
    number_unclassified_items = list(clusterer.labels_).count(-1)
    number_classified_items = total_number_items - number_unclassified_items
    percentage_unclassified = number_unclassified_items / total_number_items
    number_items_per_topic = number_classified_items / (max(clusterer.labels_) + 1)
    return (clusterer, percentage_unclassified, number_items_per_topic);

# Universal credit
content_for_taxon = get_content_for_taxon(content, "62fcbba5-3a75-4d15-85a6-d8a80b03d57c")
corpus = content_for_taxon['combined_text'].to_list()
clusterer = find_best_clustering(corpus, tokenize_limit_length)

output = pd.DataFrame()
output['title'] = content_for_taxon['title'].to_list()
output['combined_text'] = content_for_taxon['combined_text'].to_list()
output['bottom_label'] = clusterer.labels_
top_label_corpus = {}
top_label_corpus_mappings = {}
for index, row in output.iterrows():
    if(row['bottom_label'] == -1):
        print(f"Ignoring: {row['bottom_label']}")
        continue
    if row['bottom_label'] not in top_label_corpus:
        top_label_corpus[row['bottom_label']] = ""
    top_label_corpus[row['bottom_label']] += row['combined_text']
corpus_mappings = list(top_label_corpus.keys())
corpus = list(top_label_corpus.values())
clusterer = find_best_clustering(corpus, tokenize)
top_labels = clusterer.labels_
output['top_label'] = -2
for (index, bottom_level_label) in enumerate(corpus_mappings):
    top_level_label = top_labels[index]
    for row_index, row in output.iterrows():
        if row['bottom_label'] == bottom_level_label:
            output.at[row_index, 'top_label'] = top_level_label
# print("Most important words are:")
# print(vectorizer.get_feature_names())
output = output.sort_values('top_label')
output.to_csv(f"hdbscan5.csv")
