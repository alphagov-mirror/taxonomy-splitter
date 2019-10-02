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
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def split_content(content):
    article = content.split(". ")
    sentences = []
    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(content, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step 1 - Read text anc split it
    sentences = split_content(content)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    if(len(ranked_sentence) < top_n):
        top_n = len(ranked_sentence)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    # Step 5 - Offcourse, output the summarize texr
    summarised_text = ". ".join(summarize_text)
    print("Summarize Text: \n", summarised_text)
    return summarised_text

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


DATADIR = os.getenv("DATADIR")
clean_content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
content = pd.read_pickle(clean_content_path)
content['combined_text_embedding'] = ''
content['title_embedding'] = ''

# Universal credit
content_for_taxon = get_content_for_taxon(content, "357110bb-cbc5-4708-9711-1b26e6c63e86")
corpus = content_for_taxon['combined_text'].to_list()
corpus = [generate_summary(content, top_n=5) for content in corpus]
clusterer = find_best_clustering(corpus, tokenize)

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
output = output.sort_values(['top_label', 'bottom_label'])
output.to_csv(f"hdbscan5.csv")
